"""Ingestion quality gate: reject documents that parsed to almost nothing.

A scanned PDF is the failure this module exists for. `pypdf` reads its page
tree happily and returns an empty string for every page, so nothing raises —
the document ingests, produces a handful of chunks or none, and quietly becomes
a hole in the corpus. Nothing downstream can detect it: a retrieval that misses
looks exactly like a question the corpus does not answer.

**Silent partial ingestion is worse than a rejected upload**, because a rejected
upload is a message and a hole in recall is not. So the gate errs toward the
loud failure, and the error carries the numbers that justify it — how many
characters came out, across how many pages, against what threshold — so the
person reading it can tell "this is a scan" from "your threshold is wrong".

**The denominator is the load-bearing detail.** ``PdfLoader`` skips pages that
yield no text (loaders.py), so the number of *segments* is the number of pages
that worked, not the number of pages there are. Dividing by it would compute the
character density of exactly the pages that were fine and report a healthy
number for a document that is 99% images. The page count therefore comes from
the PDF's own page tree, independently of extraction.

The gate is deliberately PDF-only. "Characters per page" is a statement about a
paginated, possibly-scanned rendering; for Markdown, HTML or plain text there is
no page and no scanner, and the "no non-blank segments" check in
``loaders.load_file`` already covers the only failure that can occur there.
"""

from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from production_rag.exceptions import LowTextYieldError
from production_rag.ingestion.loaders import ExtractedSegment
from production_rag.logging_config import get_logger

logger = get_logger(__name__)

PDF_MIME = "application/pdf"

# Extraction methods, recorded on the report and persisted with the document so
# "the local parser gave up" is distinguishable from "OCR ran and still failed",
# and so a corpus can be audited for text that came out of an OCR engine.
METHOD_LOCAL = "local"
# A re-index that rebuilt its segments from ``Document.content`` rather than
# from the original bytes. Neither parsed nor OCR'd — and worth saying so,
# because that path is also where page/section provenance is known to be lost.
METHOD_STORED_BODY = "stored-body"

# A page of real prose runs 1,500-3,000 characters. 50 is far below anything a
# text PDF produces and far above the stray artefacts a scanner leaves behind
# (a page number, a header stamped by the scanning software), so the threshold
# separates the two populations without sitting near either.
DEFAULT_MIN_CHARS_PER_PAGE = 50


@dataclass(frozen=True, slots=True)
class ExtractionReport:
    """What extraction actually produced, in the terms the gate judges it by."""

    method: str
    char_count: int
    segment_count: int
    # None when pagination is not a property of the format (Markdown, HTML,
    # plain text). Distinct from 0, which would mean "a PDF with no pages".
    page_count: int | None = None
    pages_with_text: int | None = None

    @property
    def chars_per_page(self) -> float | None:
        """Characters extracted per page of the *source*, not per page that worked."""
        if not self.page_count:
            return None
        return self.char_count / self.page_count

    def as_diagnostics(self) -> dict[str, object]:
        """The flat form stored on a dead-letter row and logged."""
        density = self.chars_per_page
        return {
            "extraction_method": self.method,
            "char_count": self.char_count,
            "segment_count": self.segment_count,
            "page_count": self.page_count,
            "pages_with_text": self.pages_with_text,
            "chars_per_page": round(density, 2) if density is not None else None,
        }


def pdf_page_count(content: bytes) -> int | None:
    """Pages in a PDF, or ``None`` if it cannot be read as one.

    Cheap: `pypdf` resolves this from the page tree and never touches the
    content streams, so this does not repeat the work ``PdfLoader`` already did.

    Returns ``None`` rather than raising on a malformed file. A PDF that cannot
    be opened is a parse failure, and it will surface as one from the loader —
    the gate should not be the thing that reports it, with a different error.
    """
    try:
        return len(PdfReader(BytesIO(content)).pages)
    except (PyPdfError, ValueError, OSError) as exc:
        logger.warning("pdf_page_count_failed", error=str(exc))
        return None


def assess(
    segments: list[ExtractedSegment],
    *,
    content: bytes,
    mime_type: str | None,
    method: str,
) -> ExtractionReport:
    """Measure an extraction result. Never raises — judging is ``enforce``'s job.

    Split from ``enforce`` so the numbers exist whether or not they fail: a
    passing document still logs its density, which is what makes "documents with
    anomalously low chars-per-page" answerable before one of them trips the gate.
    """
    char_count = sum(len(segment.text) for segment in segments)

    if mime_type != PDF_MIME:
        return ExtractionReport(
            method=method,
            char_count=char_count,
            segment_count=len(segments),
        )

    page_count = pdf_page_count(content)
    # Segments carry 1-based page numbers; a page appearing at all means it
    # produced text, since blank segments are dropped before this point.
    pages_with_text = len({s.page for s in segments if s.page is not None})

    return ExtractionReport(
        method=method,
        char_count=char_count,
        segment_count=len(segments),
        page_count=page_count,
        pages_with_text=pages_with_text,
    )


def passes(report: ExtractionReport, *, min_chars_per_page: int) -> bool:
    """Whether this report clears the gate.

    A report with no page count always passes: for a format without pages there
    is no density to judge, and "it has some text" was already established by
    ``load_file``.

    Callers branch on this rather than catching ``rejection``'s exception,
    because "too thin, try OCR" is control flow, not an error.
    """
    density = report.chars_per_page
    return density is None or density >= min_chars_per_page


def rejection(
    report: ExtractionReport,
    *,
    min_chars_per_page: int = DEFAULT_MIN_CHARS_PER_PAGE,
    remedy: str = "",
) -> LowTextYieldError:
    """Build the rejection for a report that failed ``passes``.

    Returns rather than raises so the caller's control flow stays visible at the
    call site — and so the message can be assembled in one place while the
    decision to give up is made in another.

    ``remedy`` says what the caller can do about it. This module cannot know:
    whether OCR is off, unconfigured, or was already tried is the orchestrator's
    knowledge, and threading it in beats guessing here.
    """
    # Only reachable for a report that failed ``passes``, which implies a page
    # count — but a 0.0 fallback keeps a misuse from turning into a TypeError
    # inside the error path, which is the worst place to raise a second error.
    density = report.chars_per_page or 0.0
    coverage = ""
    if report.pages_with_text is not None and report.page_count:
        coverage = (
            f" Only {report.pages_with_text} of {report.page_count} pages "
            f"produced any text."
        )

    return LowTextYieldError(
        f"This document yielded {density:.0f} characters per page across "
        f"{report.page_count} pages, below the minimum of {min_chars_per_page}."
        f"{coverage} It appears to be scanned or image-only."
        f"{(' ' + remedy) if remedy else ''}",
        report=report,
    )
