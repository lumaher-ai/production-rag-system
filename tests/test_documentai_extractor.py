"""Document AI as a parser: the layout tree in, this system's segments out.

Three failure classes live here, and all three are quiet.

**Page numbers.** The synchronous API takes 15 pages at a time, so a long PDF is
cut into shards and each shard's page numbers come back starting at 1. Forget to
add the offset back and every citation past page 15 points at the wrong page —
which no test of "did it extract text" would catch.

**Structure.** Headings become the ``section`` a chunk falls under, and a table
that flattens into a wall of numbers is a table nobody can retrieve a row from.

**Money.** Sharding decides how many billable calls a document costs, and the
page budget is the only thing between a mis-sized upload and a four-figure
invoice. Both are asserted on call counts, not on outputs.
"""

import pytest
from google.api_core import exceptions as gexc

from production_rag.config import get_settings
from production_rag.exceptions import (
    ConnectorNotConfiguredError,
    SourceFetchError,
    ValidationError,
)
from production_rag.ingestion import document_ai
from production_rag.ingestion.document_ai import (
    DOCAI_EXTRACTOR_VERSION,
    ONLINE_MAX_PAGES,
    DocumentAIExtractor,
)
from tests._documentai import FakeDocumentAIClient, document, list_block, table_block, text_block
from tests._file_builders import make_pdf

PDF_MIME = "application/pdf"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _configured(**overrides):
    return get_settings().model_copy(
        update={
            "ocr_enabled": True,
            "documentai_project_id": "test-project",
            "documentai_processor_id": "abc123",
            "documentai_service_account_file": "/tmp/does-not-matter.json",
            **overrides,
        }
    )


def _extractor(client: FakeDocumentAIClient, **overrides) -> DocumentAIExtractor:
    """An extractor with its client pre-injected, so no credentials are read."""
    extractor = DocumentAIExtractor(_configured(**overrides))
    extractor._client = client
    return extractor


# ─── Block tree → segments ───


async def test_headings_become_sections_and_stay_in_the_body() -> None:
    """Same choice the Markdown and DOCX loaders make, for the same reason.

    A chunk about termination still has to match "termination clause" when the
    heading is the only place that word appears.
    """
    client = FakeDocumentAIClient(
        document(
            [
                text_block("Introduction", "heading-1"),
                text_block("Opening remarks."),
                text_block("Termination", "heading-2"),
                text_block("Either party may terminate."),
            ]
        )
    )

    segments = await _extractor(client).extract(b"x", "deck.pptx", XLSX_MIME)

    assert [s.section for s in segments] == ["Introduction", "Termination"]
    assert "Introduction" in segments[0].text
    assert "Opening remarks." in segments[0].text
    assert "Either party may terminate." in segments[1].text


async def test_title_blocks_also_open_a_section() -> None:
    client = FakeDocumentAIClient(
        document([text_block("Annual Report", "title"), text_block("Body.")])
    )
    segments = await _extractor(client).extract(b"x", "r.pptx", XLSX_MIME)

    assert segments[0].section == "Annual Report"


async def test_tables_render_as_markdown_rows() -> None:
    """Row-per-line, because the chunker splits on newlines before characters.

    Flattened cells would scatter a row across chunk boundaries, which is the
    difference between a retrievable row and a handful of orphaned numbers.
    """
    client = FakeDocumentAIClient(
        document(
            [
                table_block(
                    header=["Region", "Revenue"],
                    rows=[["North", "1,200"], ["South", "980"]],
                    caption="Q3 by region",
                )
            ]
        )
    )

    segments = await _extractor(client).extract(b"x", "book.xlsx", XLSX_MIME)
    text = segments[0].text

    assert "Q3 by region" in text
    assert "| Region | Revenue |" in text
    assert "| --- | --- |" in text
    assert "| North | 1,200 |" in text
    # One row per line: a chunk boundary can fall between rows, never inside one.
    assert "| South | 980 |" in text.splitlines()[-1]


async def test_lists_render_as_bullets() -> None:
    client = FakeDocumentAIClient(document([list_block(["First point", "Second point"])]))
    segments = await _extractor(client).extract(b"x", "d.pptx", XLSX_MIME)

    assert "- First point" in segments[0].text
    assert "- Second point" in segments[0].text


async def test_nested_blocks_are_walked() -> None:
    """Layout Parser nests body blocks under their heading; missing them loses text."""
    client = FakeDocumentAIClient(
        document(
            [
                text_block(
                    "Outer",
                    "heading-1",
                    blocks=[text_block("Nested body text.")],
                )
            ]
        )
    )

    segments = await _extractor(client).extract(b"x", "d.pptx", XLSX_MIME)

    assert "Nested body text." in segments[0].text
    assert segments[0].section == "Outer"


async def test_an_empty_layout_yields_no_segments() -> None:
    segments = await _extractor(FakeDocumentAIClient(document([]))).extract(
        b"x", "d.pptx", XLSX_MIME
    )
    assert segments == []


# ─── Sharding and page offsets ───


async def test_a_long_pdf_is_sharded_at_the_api_limit() -> None:
    pdf = make_pdf([f"Page {i}." for i in range(40)])
    client = FakeDocumentAIClient(document([text_block("body")]))

    await _extractor(client, documentai_batch_threshold_pages=100).extract(
        pdf, "long.pdf", PDF_MIME
    )

    # 40 pages -> 15 + 15 + 10, not one oversized call and not forty tiny ones.
    assert client.shard_page_counts == [ONLINE_MAX_PAGES, ONLINE_MAX_PAGES, 10]


async def test_shard_page_numbers_are_offset_back_into_the_whole_document() -> None:
    """The quiet one: without the offset, every citation past page 15 is wrong."""
    pdf = make_pdf([f"Page {i}." for i in range(40)])
    # Each shard reports page 1, exactly as the API does.
    client = FakeDocumentAIClient(
        per_shard=[
            document([text_block("first shard", page=1)]),
            document([text_block("second shard", page=1)]),
            document([text_block("third shard", page=1)]),
        ]
    )

    segments = await _extractor(client, documentai_batch_threshold_pages=100).extract(
        pdf, "long.pdf", PDF_MIME
    )

    assert [s.page for s in segments] == [1, 16, 31]


async def test_a_short_pdf_is_one_call() -> None:
    pdf = make_pdf(["Only page."])
    client = FakeDocumentAIClient(document([text_block("body")]))

    await _extractor(client).extract(pdf, "short.pdf", PDF_MIME)

    assert len(client.requests) == 1


async def test_non_pdf_formats_are_never_sharded() -> None:
    """You cannot cut an OOXML container on a page boundary."""
    client = FakeDocumentAIClient(document([text_block("cells")]))

    await _extractor(client).extract(b"x" * 5000, "book.xlsx", XLSX_MIME)

    assert len(client.requests) == 1


# ─── The page budget ───


async def test_the_page_budget_is_enforced_before_any_billable_call() -> None:
    pdf = make_pdf([f"Page {i}." for i in range(40)])
    client = FakeDocumentAIClient(document([text_block("body")]))

    with pytest.raises(ValidationError) as excinfo:
        await _extractor(
            client, documentai_max_pages=10, documentai_batch_threshold_pages=100
        ).extract(pdf, "long.pdf", PDF_MIME)

    assert "40 pages" in excinfo.value.detail
    assert "DOCUMENTAI_MAX_PAGES" in excinfo.value.detail
    # The point: refused before spending anything, not after.
    assert client.requests == []


async def test_batch_without_a_bucket_fails_before_spending() -> None:
    pdf = make_pdf([f"Page {i}." for i in range(40)])
    client = FakeDocumentAIClient(document([text_block("body")]))

    with pytest.raises(ConnectorNotConfiguredError) as excinfo:
        await _extractor(
            client, documentai_batch_threshold_pages=10, documentai_gcs_bucket=""
        ).extract(pdf, "long.pdf", PDF_MIME)

    assert "DOCUMENTAI_GCS_BUCKET" in excinfo.value.detail
    assert client.requests == []


# ─── Error mapping decides what gets retried ───


@pytest.mark.parametrize(
    ("google_error", "expected", "status"),
    [
        (gexc.ResourceExhausted("429"), SourceFetchError, 502),
        (gexc.ServiceUnavailable("503"), SourceFetchError, 502),
        (gexc.DeadlineExceeded("timeout"), SourceFetchError, 502),
        (gexc.PermissionDenied("nope"), ConnectorNotConfiguredError, 503),
        (gexc.Unauthenticated("nope"), ConnectorNotConfiguredError, 503),
        (gexc.NotFound("no such processor"), ConnectorNotConfiguredError, 503),
        (gexc.InvalidArgument("bad pdf"), ValidationError, 422),
    ],
)
async def test_google_errors_map_onto_the_local_taxonomy(
    google_error, expected, status
) -> None:
    """The mapping is what decides retry: a 429 comes back, a bad file does not."""
    client = FakeDocumentAIClient(error=google_error)

    with pytest.raises(expected) as excinfo:
        await _extractor(client).extract(b"x", "d.pptx", XLSX_MIME)

    assert excinfo.value.status_code == status


# ─── Configuration ───


def test_the_processor_version_is_pinned_in_the_resource_name() -> None:
    """Naming the processor alone would follow whatever version becomes default."""
    client = FakeDocumentAIClient()
    name = _extractor(client)._processor_name()

    assert name.endswith(
        f"/processorVersions/{get_settings().documentai_processor_version}"
    )
    assert "/processors/abc123/" in name


def test_missing_credentials_are_a_configuration_error_not_a_crash() -> None:
    settings = get_settings().model_copy(
        update={"ocr_enabled": True, "documentai_service_account_file": "",
                "google_service_account_file": ""}
    )
    with pytest.raises(ConnectorNotConfiguredError) as excinfo:
        document_ai._build_client(settings)

    assert "SERVICE_ACCOUNT_FILE" in excinfo.value.detail
    assert excinfo.value.status_code == 503


def test_the_extractor_advertises_its_mapping_version() -> None:
    """Stored on every document it produces, so old parses stay findable."""
    assert DocumentAIExtractor.version == DOCAI_EXTRACTOR_VERSION
    assert DOCAI_EXTRACTOR_VERSION != "local"


def test_chunking_is_never_requested_from_the_service() -> None:
    """Chunk boundaries are pinned by CHUNKER_VERSION and stay local.

    Asking for chunks would hand a silent input to every stored vector — and the
    validity of the resume cursor — to a remote, versioned service.
    """
    options = document_ai._process_options()

    assert not options.layout_config.chunking_config.chunk_size
    # Annotations are Gemini-written and non-deterministic; same objection.
    assert options.layout_config.enable_image_annotation is False
    assert options.layout_config.enable_table_annotation is False


def test_the_regional_endpoint_is_used(monkeypatch) -> None:
    """A regional processor is not reachable on the global default endpoint."""
    captured = {}

    class _Credentials:
        @staticmethod
        def from_service_account_file(path, scopes):
            return object()

    def _fake_async_client(credentials, client_options):
        captured["endpoint"] = client_options.api_endpoint
        return object()

    monkeypatch.setattr(
        document_ai.documentai, "DocumentProcessorServiceAsyncClient", _fake_async_client
    )
    import google.oauth2.service_account as sa

    monkeypatch.setattr(sa.Credentials, "from_service_account_file",
                        _Credentials.from_service_account_file)

    document_ai._build_client(_configured(documentai_location="eu"))

    assert captured["endpoint"] == "eu-documentai.googleapis.com"
