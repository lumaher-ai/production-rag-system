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
}


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


def load_file(
    content: bytes,
    filename: str | None,
    content_type: str | None,
) -> list[ExtractedSegment]:
    """Dispatch to the right loader and return non-empty text segments.

    Raises ``UnsupportedFileTypeError`` (415) for unknown formats and
    ``ValidationError`` (422) when nothing extractable is found.
    """
    loader = resolve_loader(filename, content_type)
    if loader is None:
        logger.warning("upload_unsupported_type", filename=filename, content_type=content_type)
        raise UnsupportedFileTypeError(
            f"Unsupported file type for '{filename}'. Supported: PDF, DOCX, HTML, TXT, Markdown."
        )

    segments = [seg for seg in loader.extract(content, filename or "") if seg.text.strip()]
    if not segments:
        raise ValidationError("No extractable text found in the uploaded file.")

    logger.info(
        "file_extracted",
        filename=filename,
        loader=type(loader).__name__,
        segment_count=len(segments),
    )
    return segments
