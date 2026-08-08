"""Google Document AI as a fallback extractor — a parser, not a chunker.

Reached only when the local parsers cannot do the job: a PDF that came back
below the quality gate (a scan), or a format this process has no loader for
(XLSX, XLSM, PPTX). Everything else stays local, because local is free,
synchronous, deterministic, and needs no credentials.

**Which processor, and why.** ``LAYOUT_PARSER_PROCESSOR`` is the only Document
AI processor that reads OOXML at all — "HTML and OOXML support are only
available with layout parser" — so choosing it is what makes spreadsheets and
presentations ingestible, not just scans. It returns a ``document_layout.blocks``
tree of typed blocks (heading, paragraph, table, list) with per-block page
spans, which is the same shape ``DocxLoader`` derives from Word heading styles:
text, a page, and the section it falls under.

**We deliberately do not use its chunker.** Layout Parser will happily return
``chunked_document.chunks``, and taking them would hand chunk boundaries to a
remote, versioned, partly-generative service. Chunking here is pinned by
``CHUNKER_VERSION`` and is load-bearing twice over: it is a silent input to
every stored vector, and the resume cursor is only meaningful because
re-deriving the chunk list produces the identical list. A boundary that can
shift under us breaks both, and nothing downstream could detect it.
``enable_image_annotation`` / ``enable_table_annotation`` — Gemini-written
descriptions of figures and tables — are off for the same reason. They are the
obvious next lever for a figure-heavy corpus, and turning them on should be a
deliberate decision with a version bump, not a default.

**Two transports, one seam.** The synchronous API caps at 15 pages and 20 MB per
call, so a PDF is sharded client-side and the shards are processed
concurrently, with each shard's page numbers offset back into the whole. Past
``documentai_batch_threshold_pages`` that becomes dozens of round trips, so the
document goes through Cloud Storage and one batch operation instead. Both paths
return the same list of segments; callers cannot tell which ran.
"""

import asyncio
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from uuid import uuid4

from google.api_core import exceptions as gexc
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PyPdfError

from production_rag.config import Settings
from production_rag.exceptions import (
    ConnectorNotConfiguredError,
    SourceFetchError,
    ValidationError,
)
from production_rag.ingestion.loaders import MAX_SECTION_LEN, ExtractedSegment
from production_rag.logging_config import get_logger

logger = get_logger(__name__)

# Versions the block → segment mapping below, and is stored on every document
# this extractor produced. Bump it when the mapping changes, so "which documents
# were parsed under the old rules?" stays answerable with a query.
DOCAI_EXTRACTOR_VERSION = "docai-layout-v1"

# API limits, not deployment configuration — they belong to Google, and a
# setting would only let an operator configure their way into a 400.
ONLINE_MAX_PAGES = 15
ONLINE_MAX_BYTES = 20 * 1024 * 1024

PDF_MIME = "application/pdf"

# Document AI needs project-wide scope; the Drive connector's drive.readonly is
# not sufficient, which is why credentials are built here rather than shared.
CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# Block types that open a new section. Layout Parser emits "heading-1".."heading-6"
# and "title"; matching on the prefix keeps a new heading level from silently
# becoming body text.
_HEADING_TYPES = re.compile(r"^(heading|title)", re.IGNORECASE)

# Failures where trying again could plausibly help. Everything else is a
# configuration or input problem that a retry repeats at full price.
_RETRYABLE = (
    gexc.ResourceExhausted,
    gexc.ServiceUnavailable,
    gexc.DeadlineExceeded,
    gexc.InternalServerError,
    gexc.Aborted,
)


@dataclass(frozen=True, slots=True)
class _Shard:
    """A slice of a document small enough for one synchronous call."""

    content: bytes
    # 0-based index of this shard's first page within the whole document. Added
    # back onto every page number the API reports, which are shard-local.
    page_offset: int


class DocumentAIExtractor:
    """Extracts text and structure from bytes Document AI can read."""

    version = DOCAI_EXTRACTOR_VERSION

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None

    # ─── Entry point ───

    async def extract(
        self, content: bytes, filename: str | None, mime_type: str
    ) -> list[ExtractedSegment]:
        """Parse a document into segments, choosing a transport by size."""
        page_count = _pdf_page_count(content) if mime_type == PDF_MIME else None
        self._assert_within_budget(page_count, filename)

        use_batch = self._needs_batch(content, page_count, mime_type)
        logger.info(
            "documentai_extract_started",
            filename=filename,
            mime_type=mime_type,
            page_count=page_count,
            transport="batch" if use_batch else "online",
        )

        if use_batch:
            documents = [await self._batch_process(content, filename, mime_type)]
            shards = [_Shard(content=content, page_offset=0)]
        else:
            shards = _shard(content, mime_type)
            documents = await self._process_shards(shards, filename, mime_type)

        segments: list[ExtractedSegment] = []
        for shard, document in zip(shards, documents, strict=True):
            segments.extend(_segments_from(document, page_offset=shard.page_offset))

        logger.info(
            "documentai_extract_completed",
            filename=filename,
            shards=len(shards),
            segments=len(segments),
        )
        return segments

    # ─── Transport choice ───

    def _needs_batch(
        self, content: bytes, page_count: int | None, mime_type: str
    ) -> bool:
        """Whether this document is too big for the synchronous API.

        A non-PDF cannot be sharded — you would have to rewrite the OOXML — so
        one that exceeds the inline limits has only the batch path. For a PDF,
        the threshold is about round trips rather than possibility: fifteen pages
        at a time is fine for a report and absurd for a book.
        """
        if mime_type != PDF_MIME:
            return len(content) > ONLINE_MAX_BYTES
        if page_count is None:
            return False
        return page_count > self._settings.documentai_batch_threshold_pages

    def _assert_within_budget(self, page_count: int | None, filename: str | None) -> None:
        """Refuse an oversized document *before* the first billable call.

        A cost guard, not a technical limit. Retries are automatic, so an
        unbounded scan is billed once per attempt.
        """
        limit = self._settings.documentai_max_pages
        if page_count is not None and page_count > limit:
            raise ValidationError(
                f"'{filename}' has {page_count} pages, above the OCR limit of "
                f"{limit}. Raise DOCUMENTAI_MAX_PAGES to process it — at roughly "
                f"$10 per 1,000 pages, that is a deliberate spend."
            )

    # ─── Online path ───

    async def _process_shards(
        self, shards: list[_Shard], filename: str | None, mime_type: str
    ) -> list[Any]:
        """Process every shard concurrently, bounded by the configured limit.

        Bounded because the online quota is 120 requests per minute per
        processor: a 500-page document is 34 shards, and firing all of them at
        once would spend the whole project's budget on one upload.
        """
        semaphore = asyncio.Semaphore(max(1, self._settings.documentai_max_concurrency))

        async def run(shard: _Shard) -> Any:
            async with semaphore:
                return await self._process_online(shard.content, filename, mime_type)

        return list(await asyncio.gather(*(run(shard) for shard in shards)))

    async def _process_online(
        self, content: bytes, filename: str | None, mime_type: str
    ) -> Any:
        client = await self._get_client()
        request = documentai.ProcessRequest(
            name=self._processor_name(),
            raw_document=documentai.RawDocument(
                content=content, mime_type=mime_type, display_name=filename or "document"
            ),
            process_options=_process_options(),
        )
        result = await _call(client.process_document, request=request)
        return result.document

    # ─── Batch path ───

    async def _batch_process(
        self, content: bytes, filename: str | None, mime_type: str
    ) -> Any:
        """Process one large document through Cloud Storage.

        Staging is per-job and cleaned up in a ``finally``: the bucket is a
        transport, not storage, and a document that failed halfway should not
        leave a copy of itself behind. A lifecycle rule on the bucket is the
        backstop for the case where this process dies between the two.
        """
        bucket = self._settings.documentai_gcs_bucket
        if not bucket:
            raise ConnectorNotConfiguredError(
                f"'{filename}' needs Document AI batch processing (it is past "
                f"DOCUMENTAI_BATCH_THRESHOLD_PAGES), which stages through Cloud "
                f"Storage. Set DOCUMENTAI_GCS_BUCKET."
            )

        run_id = uuid4().hex
        prefix = f"{self._settings.documentai_gcs_prefix}/{run_id}"
        input_blob = f"{prefix}/input/{filename or 'document'}"
        output_prefix = f"{prefix}/output/"

        storage = _storage_client(self._settings.documentai_credentials_path)
        try:
            await asyncio.to_thread(
                _upload, storage, bucket, input_blob, content, mime_type
            )

            client = await self._get_client()
            request = documentai.BatchProcessRequest(
                name=self._processor_name(),
                input_documents=documentai.BatchDocumentsInputConfig(
                    gcs_documents=documentai.GcsDocuments(
                        documents=[
                            documentai.GcsDocument(
                                gcs_uri=f"gs://{bucket}/{input_blob}", mime_type=mime_type
                            )
                        ]
                    )
                ),
                document_output_config=documentai.DocumentOutputConfig(
                    gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
                        gcs_uri=f"gs://{bucket}/{output_prefix}"
                    )
                ),
                process_options=_process_options(),
            )
            operation = await _call(client.batch_process_documents, request=request)
            await _call(
                operation.result, timeout=self._settings.documentai_batch_timeout_seconds
            )

            return await asyncio.to_thread(
                _read_batch_output, storage, bucket, output_prefix, filename
            )
        finally:
            await asyncio.to_thread(_delete_prefix, storage, bucket, prefix)

    # ─── Client and credentials ───

    def _processor_name(self) -> str:
        """The pinned processor version's full resource name.

        ``processor_version_path``, not ``processor_path``: naming the processor
        alone resolves to whatever version is currently default, which is the
        thing this system cannot allow to move on its own.
        """
        client = self._client
        assert client is not None  # only called after _get_client
        return str(
            client.processor_version_path(
                self._settings.documentai_project_id,
                self._settings.documentai_location,
                self._settings.documentai_processor_id,
                self._settings.documentai_processor_version,
            )
        )

    async def _get_client(self) -> Any:
        """Build the client once, splitting the blocking half from the bound half.

        The key file is read in a worker thread — it is blocking I/O and has no
        business on the event loop. The client itself is then constructed *here*,
        on the loop, and that split is load-bearing rather than tidy:
        ``grpc.aio`` binds its channel to the running event loop at construction
        time, so building the whole client inside ``to_thread`` raises
        "There is no current event loop in thread 'asyncio_0'" before a single
        request is sent.
        """
        if self._client is None:
            credentials = await asyncio.to_thread(_load_credentials, self._settings)
            self._client = _build_client(self._settings, credentials)
        return self._client


# ─── Client construction ───


def _load_credentials(settings: Settings) -> Any:
    """Read the service-account key. Blocking; call from a thread.

    Explicit credentials rather than Application Default Credentials, matching
    the Drive connector: this system's Google access is always a key file named
    in configuration, so a misconfigured deployment fails with "that path does
    not exist" instead of silently picking up whatever identity the host happens
    to carry.
    """
    from google.oauth2 import service_account

    path = settings.documentai_credentials_path
    if not path:
        raise ConnectorNotConfiguredError(
            "Document AI is enabled but has no credentials. Set "
            "DOCUMENTAI_SERVICE_ACCOUNT_FILE (or GOOGLE_SERVICE_ACCOUNT_FILE) to "
            "a service-account JSON key with roles/documentai.apiUser."
        )
    if not settings.documentai_project_id or not settings.documentai_processor_id:
        raise ConnectorNotConfiguredError(
            "Document AI is enabled but not addressed. Set DOCUMENTAI_PROJECT_ID "
            "and DOCUMENTAI_PROCESSOR_ID — see docs/document-ai-setup.md."
        )

    try:
        return service_account.Credentials.from_service_account_file(
            path, scopes=[CLOUD_PLATFORM_SCOPE]
        )
    except (OSError, ValueError) as exc:
        raise ConnectorNotConfiguredError(
            f"Could not read the Document AI service-account key at '{path}': {exc}"
        ) from exc


def _build_client(settings: Settings, credentials: Any) -> Any:
    """Construct the async client. **Must run on the event loop** — see ``_get_client``."""
    # The regional endpoint is mandatory. A processor lives in one location, and
    # the global default endpoint answers requests for it with a confusing
    # not-found rather than a redirect.
    location = settings.documentai_location
    return documentai.DocumentProcessorServiceAsyncClient(
        credentials=credentials,
        client_options=ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com"),
    )


def _storage_client(credentials_path: str) -> Any:
    import google.cloud.storage  # type: ignore[import-untyped]
    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=[CLOUD_PLATFORM_SCOPE]
    )
    return google.cloud.storage.Client(
        credentials=credentials, project=credentials.project_id
    )


# ─── Cloud Storage helpers (the SDK is synchronous; callers thread them) ───


def _upload(storage: Any, bucket: str, name: str, content: bytes, mime_type: str) -> None:
    storage.bucket(bucket).blob(name).upload_from_string(content, content_type=mime_type)


def _read_batch_output(
    storage: Any, bucket: str, output_prefix: str, filename: str | None
) -> Any:
    """Reassemble the Document the batch operation wrote.

    Document AI shards its output: one input file can produce several JSON
    objects, and they are only in document order if you sort by name. Read in
    that order and concatenate their block lists — the blocks already carry
    absolute page spans, so nothing needs re-offsetting.
    """
    blobs = sorted(
        (b for b in storage.list_blobs(bucket, prefix=output_prefix) if b.name.endswith(".json")),
        key=lambda b: b.name,
    )
    if not blobs:
        raise SourceFetchError(
            f"Document AI reported success for '{filename}' but wrote no output to "
            f"gs://{bucket}/{output_prefix}."
        )

    merged = documentai.Document()
    for blob in blobs:
        shard = documentai.Document.from_json(
            blob.download_as_bytes(), ignore_unknown_fields=True
        )
        merged.document_layout.blocks.extend(shard.document_layout.blocks)
    return merged


def _delete_prefix(storage: Any, bucket: str, prefix: str) -> None:
    """Best-effort cleanup. A leaked object is a cost, not a correctness bug."""
    try:
        for blob in storage.list_blobs(bucket, prefix=prefix):
            blob.delete()
    except Exception as exc:  # noqa: BLE001 - cleanup must not mask the real error
        logger.warning("documentai_cleanup_failed", prefix=prefix, error=str(exc))


# ─── Requests ───


def _process_options() -> Any:
    """Layout parsing with chunking and annotation left off.

    ``chunking_config`` is unset, so no ``chunked_document`` comes back and the
    response is the layout tree alone — see the module docstring for why owning
    chunk boundaries locally is not negotiable.
    """
    return documentai.ProcessOptions(
        layout_config=documentai.ProcessOptions.LayoutConfig(
            return_images=False,
            return_bounding_boxes=False,
            enable_image_annotation=False,
            enable_table_annotation=False,
        )
    )


async def _call(func: Any, **kwargs: Any) -> Any:
    """Invoke a Google API call, translating its errors into this system's.

    The mapping is what decides whether a failure is retried, so it is a
    correctness concern rather than cosmetics: a 429 should come back, and a
    malformed document should not.
    """
    try:
        result = func(**kwargs)
        return await result if asyncio.iscoroutine(result) else result
    except _RETRYABLE as exc:
        raise SourceFetchError(f"Document AI is temporarily unavailable: {exc}") from exc
    except (gexc.PermissionDenied, gexc.Unauthenticated) as exc:
        raise ConnectorNotConfiguredError(
            f"Document AI rejected these credentials: {exc}. The service account "
            f"needs roles/documentai.apiUser on the processor's project."
        ) from exc
    except gexc.NotFound as exc:
        raise ConnectorNotConfiguredError(
            f"Document AI processor not found: {exc}. Check DOCUMENTAI_PROJECT_ID, "
            f"DOCUMENTAI_LOCATION, DOCUMENTAI_PROCESSOR_ID and "
            f"DOCUMENTAI_PROCESSOR_VERSION."
        ) from exc
    except gexc.InvalidArgument as exc:
        raise ValidationError(f"Document AI could not process this document: {exc}") from exc


# ─── Sharding ───


def _pdf_page_count(content: bytes) -> int | None:
    try:
        return len(PdfReader(BytesIO(content)).pages)
    except (PyPdfError, ValueError, OSError):
        return None


def _shard(content: bytes, mime_type: str) -> list[_Shard]:
    """Split a document into pieces the synchronous API will accept.

    Only PDFs can be split — an OOXML file has no page boundary you can cut on
    without rewriting the container — so everything else is one shard and either
    fits or has already been routed to batch.
    """
    if mime_type != PDF_MIME:
        return [_Shard(content=content, page_offset=0)]

    try:
        reader = PdfReader(BytesIO(content))
        total = len(reader.pages)
    except (PyPdfError, ValueError, OSError):
        # Unreadable as a PDF here means unreadable everywhere; let the API say
        # so rather than inventing a second, competing error message.
        return [_Shard(content=content, page_offset=0)]

    if total <= ONLINE_MAX_PAGES and len(content) <= ONLINE_MAX_BYTES:
        return [_Shard(content=content, page_offset=0)]

    shards: list[_Shard] = []
    for start in range(0, total, ONLINE_MAX_PAGES):
        stop = min(start + ONLINE_MAX_PAGES, total)
        shards.extend(_split_range(reader, start, stop))
    return shards


def _split_range(reader: PdfReader, start: int, stop: int) -> list[_Shard]:
    """Write pages [start, stop) as one shard, halving if it is still too large.

    The page cap and the byte cap are independent: fifteen pages of text are a
    few hundred KB, and fifteen pages of scanned colour images can be well past
    20 MB. Halving recursively converges on whatever mix this document has,
    down to a single page — beyond which there is nothing left to split and the
    API's own error is the honest answer.
    """
    payload = _write_pages(reader, start, stop)
    if len(payload) <= ONLINE_MAX_BYTES or stop - start == 1:
        return [_Shard(content=payload, page_offset=start)]

    middle = start + (stop - start) // 2
    return _split_range(reader, start, middle) + _split_range(reader, middle, stop)


def _write_pages(reader: PdfReader, start: int, stop: int) -> bytes:
    writer = PdfWriter()
    for index in range(start, stop):
        writer.add_page(reader.pages[index])
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


# ─── Response mapping ───


def _segments_from(document: Any, *, page_offset: int) -> list[ExtractedSegment]:
    """Walk the layout tree into segments, the way DocxLoader walks headings.

    Headings both open a section and stay in the body text, so retrieval still
    sees the words in the heading — the same choice the Markdown and DOCX
    loaders make, and the reason a chunk about "Termination" still matches the
    query "termination clause" when the section name is the only place that word
    appears.

    Page numbers arrive shard-local and are shifted by ``page_offset`` into the
    whole document's numbering.
    """
    segments: list[ExtractedSegment] = []
    state = _WalkState(page_offset=page_offset)
    for block in document.document_layout.blocks:
        _walk(block, state, segments)
    state.flush(segments)
    return segments


@dataclass
class _WalkState:
    page_offset: int
    section: str | None = None
    buffer: list[str] = None  # type: ignore[assignment]
    page: int | None = None

    def __post_init__(self) -> None:
        self.buffer = []

    def flush(self, segments: list[ExtractedSegment]) -> None:
        body = "\n\n".join(part for part in self.buffer if part.strip()).strip()
        self.buffer = []
        if body:
            segments.append(
                ExtractedSegment(text=body, page=self.page, section=self.section)
            )


def _walk(block: Any, state: _WalkState, segments: list[ExtractedSegment]) -> None:
    page = _page_of(block, state.page_offset)

    # A page boundary closes the current segment, exactly as a heading does.
    # Without this, a run of paragraphs spanning pages 1-3 becomes one segment
    # tagged with page 1, every chunk built from it inherits that number, and the
    # citation for text on page 3 points at page 1. PdfLoader gets this right for
    # free by emitting one segment per page; this walker has to do it explicitly,
    # and the failure is silent — the text is correct, only the provenance lies.
    if page is not None and state.page is not None and page != state.page:
        state.flush(segments)
    if page is not None:
        state.page = page

    text_block = block.text_block
    if text_block and text_block.text:
        if _HEADING_TYPES.match(text_block.type_ or ""):
            # A new heading closes the previous section's segment, so the section
            # label on each segment is the heading it actually falls under.
            state.flush(segments)
            state.section = text_block.text.strip()[:MAX_SECTION_LEN] or None
            state.page = page
        state.buffer.append(text_block.text)

    for nested in getattr(text_block, "blocks", None) or ():
        _walk(nested, state, segments)

    if block.table_block:
        rendered = _render_table(block.table_block)
        if rendered:
            state.buffer.append(rendered)

    if block.list_block:
        rendered = _render_list(block.list_block)
        if rendered:
            state.buffer.append(rendered)


def _page_of(block: Any, offset: int) -> int | None:
    span = block.page_span
    if span is None or not span.page_start:
        return None
    return int(span.page_start) + offset


def _render_table(table: Any) -> str:
    """Render a table as Markdown.

    Markdown rather than a flattened cell list because the chunker splits on
    "\\n\\n" then "\\n": a row stays on one line and therefore stays intact
    inside a chunk, which is the difference between a retrievable row and a
    scattering of numbers.
    """
    rows = [_render_row(row) for row in table.header_rows]
    if rows:
        # The separator is what makes it a table rather than three lines that
        # happen to contain pipes.
        columns = max(len(row.cells) for row in table.header_rows)
        rows.append("| " + " | ".join(["---"] * columns) + " |")
    rows.extend(_render_row(row) for row in table.body_rows)
    body = "\n".join(row for row in rows if row)
    caption = (table.caption or "").strip()
    return f"{caption}\n{body}".strip() if caption else body


def _render_row(row: Any) -> str:
    cells = [_cell_text(cell) for cell in row.cells]
    return "| " + " | ".join(cells) + " |" if cells else ""


def _cell_text(cell: Any) -> str:
    """Flatten a cell's nested blocks to one line.

    A cell can contain paragraphs and even nested tables; newlines inside it
    would break the row apart, so everything collapses to spaces.
    """
    parts: list[str] = []
    for block in cell.blocks:
        if block.text_block and block.text_block.text:
            parts.append(block.text_block.text)
    return " ".join(" ".join(parts).split())


def _render_list(list_block: Any) -> str:
    lines: list[str] = []
    for entry in list_block.list_entries:
        for block in entry.blocks:
            if block.text_block and block.text_block.text:
                lines.append(f"- {block.text_block.text.strip()}")
    return "\n".join(lines)
