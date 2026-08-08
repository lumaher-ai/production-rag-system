"""When Document AI gets called — which is the cost model, not a detail.

Layout Parser bills roughly $10 per 1,000 pages. A policy that reaches for it
one step too eagerly is not a slightly slower pipeline; it is a bill that scales
with the corpus, paid to get an answer the local parser already had.

So the assertions here are mostly about a call *not* happening. That is
deliberate: "did we extract text" passes whether or not the expensive path ran,
and a regression that starts routing every PDF through OCR would be invisible to
any test that only looks at the output.

The escalation ladder, in one line: local parser, quality gate, and OCR only if
the gate says the local answer was not real — or if there was no local parser to
try, which is the case for spreadsheets and presentations.
"""

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.config import get_settings
from production_rag.exceptions import LowTextYieldError, UnsupportedFileTypeError
from production_rag.ingestion.document_ai import DOCAI_EXTRACTOR_VERSION
from production_rag.ingestion.extraction import extract_segments
from production_rag.ingestion.loaders import ExtractedSegment
from production_rag.ingestion.quality import METHOD_LOCAL
from production_rag.models.document import Document
from production_rag.models.ingestion_job import IngestionJob
from tests._file_builders import make_pdf, make_scanned_pdf
from tests._jobs import drain_expecting_failure, drain_jobs

module_loop = pytest.mark.asyncio(loop_scope="module")

PDF_MIME = "application/pdf"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


class SpyExtractor:
    """Counts calls, because the call is the thing that costs money."""

    version = DOCAI_EXTRACTOR_VERSION

    def __init__(self, segments: list[ExtractedSegment] | None = None) -> None:
        self.segments = segments if segments is not None else []
        self.calls: list[tuple[str | None, str]] = []

    async def extract(self, content, filename, mime_type):
        self.calls.append((filename, mime_type))
        return self.segments


def _ocr_on(**overrides):
    return get_settings().model_copy(
        update={
            "ocr_enabled": True,
            "documentai_project_id": "p",
            "documentai_processor_id": "abc",
            "documentai_service_account_file": "/tmp/key.json",
            **overrides,
        }
    )


def _dense(pages: int) -> list[ExtractedSegment]:
    return [
        ExtractedSegment(text="Recovered prose. " * 40, page=i + 1) for i in range(pages)
    ]


# ─── A document that parses fine never reaches the expensive path ───


async def test_a_readable_pdf_does_not_call_ocr() -> None:
    ocr = SpyExtractor()
    result = await extract_segments(
        make_pdf(["Page one.", "Page two."]), "report.pdf", PDF_MIME, _ocr_on(), ocr=ocr
    )

    assert ocr.calls == []
    assert result.report.method == METHOD_LOCAL
    assert result.used_ocr is False


async def test_markdown_never_reaches_ocr_however_short() -> None:
    """No pages, no scanner — there is nothing here OCR could improve."""
    ocr = SpyExtractor()
    result = await extract_segments(
        b"# Note\n\nShort.", "n.md", "text/markdown", _ocr_on(), ocr=ocr
    )

    assert ocr.calls == []
    assert result.report.method == METHOD_LOCAL


# ─── A scan escalates, once ───


async def test_a_scanned_pdf_escalates_and_the_ocr_result_wins() -> None:
    ocr = SpyExtractor(segments=_dense(20))
    result = await extract_segments(
        make_scanned_pdf(20, text_pages=1), "scan.pdf", PDF_MIME, _ocr_on(), ocr=ocr
    )

    assert ocr.calls == [("scan.pdf", PDF_MIME)]
    assert result.used_ocr is True
    assert result.report.method == DOCAI_EXTRACTOR_VERSION
    assert len(result.segments) == 20


async def test_a_scan_with_ocr_off_is_rejected_with_a_pointer_to_the_fix() -> None:
    with pytest.raises(LowTextYieldError) as excinfo:
        await extract_segments(
            make_scanned_pdf(20, text_pages=1), "scan.pdf", PDF_MIME, get_settings()
        )

    assert "OCR is disabled" in excinfo.value.detail
    assert "OCR_ENABLED=true" in excinfo.value.detail


async def test_ocr_enabled_but_unconfigured_says_so_specifically() -> None:
    """"Enabled" and "addressed" are different problems with different fixes."""
    settings = get_settings().model_copy(update={"ocr_enabled": True})

    with pytest.raises(LowTextYieldError) as excinfo:
        await extract_segments(
            make_scanned_pdf(20, text_pages=1), "scan.pdf", PDF_MIME, settings
        )

    assert "not configured" in excinfo.value.detail
    assert "DOCUMENTAI_PROCESSOR_ID" in excinfo.value.detail


async def test_ocr_output_faces_the_same_gate() -> None:
    """A scan of blank paper is still blank after OCR; OCR is not an exemption."""
    ocr = SpyExtractor(segments=[ExtractedSegment(text="12", page=1)])

    with pytest.raises(LowTextYieldError) as excinfo:
        await extract_segments(
            make_scanned_pdf(20, text_pages=1), "scan.pdf", PDF_MIME, _ocr_on(), ocr=ocr
        )

    assert len(ocr.calls) == 1  # tried once, not retried in a loop
    assert "OCR was applied and still produced too little" in excinfo.value.detail


async def test_ocr_errors_propagate_rather_than_becoming_quality_failures() -> None:
    """A Google outage and a blank scan both end in "no text". Only one retries."""

    class Broken(SpyExtractor):
        async def extract(self, content, filename, mime_type):
            from production_rag.exceptions import SourceFetchError

            raise SourceFetchError("Document AI is temporarily unavailable")

    from production_rag.exceptions import SourceFetchError

    with pytest.raises(SourceFetchError):
        await extract_segments(
            make_scanned_pdf(20, text_pages=1), "scan.pdf", PDF_MIME, _ocr_on(), ocr=Broken()
        )


# ─── Formats with no local parser go straight out ───


async def test_a_spreadsheet_goes_directly_to_ocr() -> None:
    ocr = SpyExtractor(segments=_dense(1))
    result = await extract_segments(b"PK\x03\x04", "book.xlsx", XLSX_MIME, _ocr_on(), ocr=ocr)

    assert ocr.calls == [("book.xlsx", XLSX_MIME)]
    assert result.report.method == DOCAI_EXTRACTOR_VERSION


async def test_a_presentation_goes_directly_to_ocr() -> None:
    ocr = SpyExtractor(segments=_dense(1))
    await extract_segments(b"PK\x03\x04", "deck.pptx", PPTX_MIME, _ocr_on(), ocr=ocr)

    assert ocr.calls == [("deck.pptx", PPTX_MIME)]


async def test_a_spreadsheet_without_ocr_is_an_unsupported_type() -> None:
    with pytest.raises(UnsupportedFileTypeError) as excinfo:
        await extract_segments(b"PK\x03\x04", "book.xlsx", XLSX_MIME, get_settings())

    assert "XLSX" in excinfo.value.detail
    assert "OCR_ENABLED=true" in excinfo.value.detail


async def test_a_genuinely_unknown_format_is_rejected_even_with_ocr_on() -> None:
    """OCR widens the supported set; it does not make it unbounded."""
    with pytest.raises(UnsupportedFileTypeError):
        await extract_segments(b"\x00\x01", "archive.zip", "application/zip", _ocr_on())


# ─── The endpoint's answer depends on deployment state ───


async def _auth_token(client: AsyncClient, email: str) -> str:
    await client.post(
        "/auth/signup",
        json={"name": "Uploader", "email": email, "password": "securepass123"},
    )
    login = await client.post(
        "/auth/login", json={"email": email, "password": "securepass123"}
    )
    return login.json()["access_token"]


@module_loop
async def test_uploading_a_spreadsheet_is_415_when_ocr_is_unconfigured(
    pg_async_client: AsyncClient, job_queue
) -> None:
    """Better than a 202 followed by a job that fails a second later."""
    token = await _auth_token(pg_async_client, "xlsx-off@example.com")

    response = await pg_async_client.post(
        "/documents/upload",
        files={"file": ("book.xlsx", b"PK\x03\x04", XLSX_MIME)},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 415
    assert "OCR_ENABLED=true" in response.json()["detail"]
    assert job_queue.enqueued == []


# ─── The extraction cache ───


@module_loop
async def test_a_retry_does_not_pay_for_ocr_twice(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue, monkeypatch
) -> None:
    """The whole reason the extraction is cached on the job row.

    A remote extraction is both expensive and not guaranteed byte-identical
    across calls — and the resume cursor is only valid because re-deriving the
    chunk list yields the same list. Caching answers both at once.
    """
    ocr = SpyExtractor(segments=_dense(20))
    settings = _ocr_on()
    monkeypatch.setattr(
        "production_rag.ingestion.extraction._default_ocr_extractor", lambda _s: ocr
    )

    token = await _auth_token(pg_async_client, "ocr-cache@example.com")
    response = await pg_async_client.post(
        "/documents/upload",
        files={"file": ("scan.pdf", make_scanned_pdf(20, text_pages=1), PDF_MIME)},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 202
    job_id = job_queue.enqueued[0]

    await drain_jobs(pg_session, job_queue, settings=settings)
    assert len(ocr.calls) == 1

    document = (
        await pg_session.execute(
            select(Document).where(Document.source == response.json()["source"])
        )
    ).scalar_one()
    # Recorded on the document, so "which of my documents came out of an OCR
    # engine?" is a metadata filter rather than an archaeology project.
    assert document.doc_metadata["extraction_method"] == DOCAI_EXTRACTOR_VERSION

    job = (
        await pg_session.execute(select(IngestionJob).where(IngestionJob.id == job_id))
    ).scalar_one()
    # Released on success alongside the payload — both exist only to make a
    # *retry* cheap, and there is no retry coming.
    assert job.extracted_segments is None


@module_loop
async def test_the_cache_survives_a_failed_attempt(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue, monkeypatch
) -> None:
    """Extraction succeeded, a later stage failed: attempt two must not re-buy it."""
    ocr = SpyExtractor(segments=_dense(20))
    settings = _ocr_on()
    monkeypatch.setattr(
        "production_rag.ingestion.extraction._default_ocr_extractor", lambda _s: ocr
    )

    token = await _auth_token(pg_async_client, "ocr-resume@example.com")
    await pg_async_client.post(
        "/documents/upload",
        files={"file": ("scan.pdf", make_scanned_pdf(20, text_pages=1), PDF_MIME)},
        headers={"Authorization": f"Bearer {token}"},
    )
    job_id = job_queue.enqueued[0]

    # Fail after extraction, at the embedding stage.
    from unittest.mock import AsyncMock

    from production_rag.llm.embedding_service import EmbeddingService

    broken = AsyncMock(spec=EmbeddingService)
    broken.embed_batch.side_effect = RuntimeError("embedding provider is down")
    broken.model = "text-embedding-3-small"

    job = await drain_expecting_failure(
        pg_session, job_queue, settings=settings, embeddings=broken
    )
    assert len(ocr.calls) == 1
    assert job.extracted_segments is not None
    assert job.extracted_segments["method"] == DOCAI_EXTRACTOR_VERSION
    assert len(job.extracted_segments["segments"]) == 20

    # Attempt two reuses the cached extraction rather than calling out again.
    job_queue.enqueued.append(job_id)
    await drain_jobs(pg_session, job_queue, settings=settings)

    assert len(ocr.calls) == 1, "the second attempt must not re-bill Document AI"
