"""File loaders for document ingestion.

Each loader turns raw file bytes into a list of ``ExtractedSegment`` objects.
Splitting into segments (rather than one flat string) lets us keep the structure
we already have while parsing — PDF page numbers, Markdown/DOCX/HTML headings —
which downstream ingestion maps onto the ``page``/``section`` columns of each
chunk for citation provenance.

Loaders are wired through a registry (MIME type -> loader). Adding a new format
means writing one loader and registering it here — no dispatch chain to edit.
"""

import re
from io import BytesIO
from pathlib import Path
from typing import Protocol

from docx import Document as DocxDocument
from markdownify import markdownify
from pydantic import BaseModel
from pypdf import PdfReader

from production_rag.exceptions import UnsupportedFileTypeError, ValidationError
from production_rag.logging_config import get_logger

logger = get_logger(__name__)

# section is persisted to a String(255) column — headings longer than this are truncated.
MAX_SECTION_LEN = 255

# ATX-style Markdown heading: 1-6 '#', text, optional trailing '#'.
_HEADING_RE = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$")

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
XLSM_MIME = "application/vnd.ms-excel.sheet.macroenabled.12"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

# Formats this process cannot parse itself — there is no loader in the registry
# below for them, and a remote extractor (Document AI) is the only way in. Kept
# separate from the registry rather than faked into it: every loader here is
# synchronous and free, and a network call that bills per page is neither.
DOCAI_ONLY_MIMES: frozenset[str] = frozenset({XLSX_MIME, XLSM_MIME, PPTX_MIME})

# Everything the remote extractor can read. Wider than DOCAI_ONLY_MIMES: a PDF
# has a local loader *and* an OCR fallback for when that loader comes back
# empty. Images are absent deliberately — Document AI reads them, but this
# system has no notion of a document that is one picture.
OCR_CAPABLE_MIMES: frozenset[str] = DOCAI_ONLY_MIMES | frozenset(
    {"application/pdf", DOCX_MIME, "text/html"}
)


class ExtractedSegment(BaseModel):
    """A contiguous span of text with optional structural provenance."""

    text: str
    page: int | None = None  # 1-based PDF page number
    section: str | None = None  # Markdown/DOCX/HTML heading the span falls under


class FileLoader(Protocol):
    def extract(self, content: bytes, filename: str) -> list[ExtractedSegment]:
        """Extract text segments from file bytes."""
        ...


def _split_markdown_by_headings(text: str) -> list[ExtractedSegment]:
    """Split Markdown into one segment per heading region.

    Text before the first heading becomes a section-less segment; each heading
    starts a new segment tagged with that heading, and the heading line is kept
    in the segment body so retrieval still sees it.
    """
    segments: list[ExtractedSegment] = []
    current_section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        body = "".join(buffer).strip()
        if body:
            segments.append(ExtractedSegment(text=body, section=current_section))

    for line in text.splitlines(keepends=True):
        match = _HEADING_RE.match(line)
        if match:
            flush()
            buffer = [line]
            current_section = match.group(1).strip()[:MAX_SECTION_LEN]
        else:
            buffer.append(line)
    flush()
    return segments


class PdfLoader:
    def extract(self, content: bytes, filename: str) -> list[ExtractedSegment]:
        reader = PdfReader(BytesIO(content))
        segments: list[ExtractedSegment] = []
        for index, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:  # skip blank/scanned pages rather than storing empty spans
                segments.append(ExtractedSegment(text=text, page=index + 1))
        return segments


class DocxLoader:
    def extract(self, content: bytes, filename: str) -> list[ExtractedSegment]:
        document = DocxDocument(BytesIO(content))
        segments: list[ExtractedSegment] = []
        current_section: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            body = "\n".join(buffer).strip()
            if body:
                segments.append(ExtractedSegment(text=body, section=current_section))

        for paragraph in document.paragraphs:
            style = (paragraph.style.name if paragraph.style else "") or ""
            para_text = paragraph.text
            if style.startswith("Heading") or style == "Title":
                flush()
                buffer = []
                current_section = para_text.strip()[:MAX_SECTION_LEN] or None
                if para_text.strip():
                    buffer.append(para_text)
            elif para_text.strip():
                buffer.append(para_text)
        flush()
        return segments


class HtmlLoader:
    def extract(self, content: bytes, filename: str) -> list[ExtractedSegment]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        # Convert to Markdown so we can reuse the heading splitter for section
        # provenance. ATX style ('## H') is required for the heading regex — the
        # markdownify default emits setext underlines for h1/h2.
        markdown = markdownify(str(soup), heading_style="ATX")
        return _split_markdown_by_headings(markdown)


class PlainTextLoader:
    def extract(self, content: bytes, filename: str) -> list[ExtractedSegment]:
        text = content.decode("utf-8", errors="replace").strip()
        return [ExtractedSegment(text=text)] if text else []


class MarkdownLoader:
    def extract(self, content: bytes, filename: str) -> list[ExtractedSegment]:
        text = content.decode("utf-8", errors="replace")
        return _split_markdown_by_headings(text)


LOADER_REGISTRY: dict[str, FileLoader] = {
    "application/pdf": PdfLoader(),
    DOCX_MIME: DocxLoader(),
    "text/html": HtmlLoader(),
    "text/plain": PlainTextLoader(),
    "text/markdown": MarkdownLoader(),
}

# Filename extension -> canonical MIME. Extension is the primary dispatch signal
# because clients frequently send a generic/wrong content_type (e.g.
# application/octet-stream); content_type is only the fallback.
EXTENSION_TO_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": DOCX_MIME,
    ".html": "text/html",
    ".htm": "text/html",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    # No loader below; reachable only through Document AI.
    ".xlsx": XLSX_MIME,
    ".xlsm": XLSM_MIME,
    ".pptx": PPTX_MIME,
}

# What the 415 message offers, split by what it costs to support.
SUPPORTED_FORMATS = "PDF, DOCX, HTML, TXT, Markdown"
OCR_ONLY_FORMATS = "XLSX, XLSM, PPTX"


def resolve_mime(filename: str | None, content_type: str | None) -> str | None:
    """Canonical MIME for a file, or ``None`` if neither signal identifies one.

    The extension wins over the declared ``content_type`` — see the note on
    ``EXTENSION_TO_MIME`` — and the declared type is normalized before lookup so
    ``text/plain; charset=utf-8`` matches ``text/plain``.

    Split out from ``resolve_loader`` because the answer is useful on its own:
    it is stored as ``metadata.mime_type``, where "which format was this" is a
    fact worth filtering on even though the loader that consumed it is not.
    """
    ext = Path(filename).suffix.lower() if filename else ""
    mime = EXTENSION_TO_MIME.get(ext)
    if mime is not None:
        return mime
    if content_type:
        return content_type.split(";")[0].strip().lower()  # drop '; charset=...'
    return None


def resolve_loader(filename: str | None, content_type: str | None) -> FileLoader | None:
    """Pick a loader for a file, or ``None`` if the format is unsupported.

    Public so the upload endpoint can reject an unsupported type *before*
    queueing a job — a synchronous 415 is far more useful than a job that fails
    a second later with the same message.
    """
    mime = resolve_mime(filename, content_type)
    return LOADER_REGISTRY.get(mime) if mime is not None else None


def is_ingestible(
    filename: str | None,
    content_type: str | None,
    *,
    ocr_available: bool,
) -> bool:
    """Whether this system can ingest the file at all, right now.

    Wider than ``resolve_loader``: a spreadsheet has no local loader but is
    ingestible when Document AI is configured. Answering that here rather than
    at the endpoint keeps the "what can we read" question in one module, and
    makes it depend on *deployment* state — the same upload is a 202 on a
    deployment with OCR credentials and a 415 on one without.
    """
    mime = resolve_mime(filename, content_type)
    if mime is None:
        return False
    if mime in LOADER_REGISTRY:
        return True
    return ocr_available and mime in DOCAI_ONLY_MIMES


def unsupported_type_message(filename: str | None, *, ocr_available: bool) -> str:
    """The 415 body: what is supported, and what would make more supported."""
    base = f"Unsupported file type for '{filename}'. Supported: {SUPPORTED_FORMATS}"
    if ocr_available:
        return f"{base}, {OCR_ONLY_FORMATS}."
    return (
        f"{base}. {OCR_ONLY_FORMATS} require Document AI OCR — set OCR_ENABLED=true "
        f"and DOCUMENTAI_PROCESSOR_ID to enable them."
    )


def extract_local(
    content: bytes,
    filename: str | None,
    content_type: str | None,
) -> list[ExtractedSegment]:
    """Parse with the local loader, returning **possibly zero** segments.

    An empty result is a *measurement*, not an error. A PDF that yields nothing
    at all is the strongest available signal that it is a scan — every page an
    image — and that is precisely the document a remote extractor exists to
    rescue. Raising here would make the most common scanned document
    unreachable by the fallback built for it, which is exactly the bug this
    split fixes.

    Deciding what an empty result *means* belongs to the caller, which knows
    whether OCR is available. ``load_file`` is the strict wrapper for callers
    that have no such option.

    Still raises ``UnsupportedFileTypeError`` (415): "no loader for this format"
    is genuinely an error and no measurement can follow it.

    Local only, by design — free, synchronous, deterministic. Formats that need
    a remote extractor are routed by ``ingestion.extraction``.
    """
    loader = resolve_loader(filename, content_type)
    if loader is None:
        logger.warning("upload_unsupported_type", filename=filename, content_type=content_type)
        raise UnsupportedFileTypeError(unsupported_type_message(filename, ocr_available=False))

    segments = [seg for seg in loader.extract(content, filename or "") if seg.text.strip()]
    logger.info(
        "file_extracted",
        filename=filename,
        loader=type(loader).__name__,
        segment_count=len(segments),
    )
    return segments


def load_file(
    content: bytes,
    filename: str | None,
    content_type: str | None,
) -> list[ExtractedSegment]:
    """``extract_local``, but an empty result is an error.

    Raises ``UnsupportedFileTypeError`` (415) for unknown formats and
    ``ValidationError`` (422) when nothing extractable is found. For callers
    with no fallback to escalate to, which is every caller except the
    extraction orchestrator.
    """
    segments = extract_local(content, filename, content_type)
    if not segments:
        raise ValidationError("No extractable text found in the uploaded file.")
    return segments
