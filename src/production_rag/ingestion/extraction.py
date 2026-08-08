"""Bytes to segments: the one place that decides *how* a document gets read.

Three things used to be tangled together at the call site — pick a parser, run
it, hope the result is real. This module separates them into a policy:

    local parser  →  quality gate  →  (remote extractor)  →  quality gate

Local parsing runs first for every format that has a loader, because it is free,
synchronous, deterministic, and works with no credentials. Document AI runs only
when local parsing cannot do the job at all (spreadsheets, presentations) or has
demonstrably failed to (a scanned PDF that came back nearly empty). That order
is the cost model: at roughly $10 per 1,000 pages, an OCR call that a local
parser could have avoided is money spent to get a worse answer more slowly.

**The gate runs on the OCR output too.** OCR is a second attempt, not an
exemption — a scan of blank paper should still be rejected, and the rejection
should say that OCR was tried, so nobody goes looking for a configuration
problem that is not there.
"""

from dataclasses import dataclass
from typing import Protocol

from production_rag.config import Settings
from production_rag.exceptions import LowTextYieldError, UnsupportedFileTypeError
from production_rag.ingestion import quality
from production_rag.ingestion.loaders import (
    DOCAI_ONLY_MIMES,
    LOADER_REGISTRY,
    OCR_CAPABLE_MIMES,
    ExtractedSegment,
    extract_local,
    resolve_mime,
    unsupported_type_message,
)
from production_rag.ingestion.quality import METHOD_LOCAL, ExtractionReport
from production_rag.logging_config import get_logger

logger = get_logger(__name__)


class OcrExtractor(Protocol):
    """A remote extractor. Async because it is a network call, unlike a loader."""

    version: str

    async def extract(
        self, content: bytes, filename: str | None, mime_type: str
    ) -> list[ExtractedSegment]: ...


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    segments: list[ExtractedSegment]
    report: ExtractionReport

    @property
    def used_ocr(self) -> bool:
        return self.report.method != METHOD_LOCAL


def _remedy(settings: Settings, *, ocr_attempted: bool) -> str:
    """What the caller can do about a rejected document, if anything."""
    if ocr_attempted:
        return "Document AI OCR was applied and still produced too little text."
    if not settings.ocr_enabled:
        return (
            "OCR is disabled on this deployment — set OCR_ENABLED=true to ingest "
            "scanned documents."
        )
    return (
        "OCR is enabled but not configured — set DOCUMENTAI_PROJECT_ID and "
        "DOCUMENTAI_PROCESSOR_ID to ingest scanned documents."
    )


def _default_ocr_extractor(settings: Settings) -> OcrExtractor | None:
    """Build the configured remote extractor, or ``None`` if there isn't one.

    Imported lazily so that ``google-cloud-documentai`` is only loaded by a
    deployment that actually uses it, and so an installation without the package
    degrades to "OCR unavailable" rather than failing to import ingestion.
    """
    if not settings.ocr_available:
        return None
    try:
        from production_rag.ingestion.document_ai import DocumentAIExtractor
    except ImportError:  # pragma: no cover - dependency is declared
        logger.warning("ocr_dependency_missing", detail="google-cloud-documentai not installed")
        return None
    return DocumentAIExtractor(settings)


async def extract_segments(
    content: bytes,
    filename: str | None,
    content_type: str | None,
    settings: Settings,
    ocr: OcrExtractor | None = None,
) -> ExtractionResult:
    """Parse a document's bytes into segments, escalating to OCR if needed.

    ``ocr`` is injectable so tests can assert on *whether it was called*, which
    is the behaviour that costs money and therefore the behaviour worth pinning.
    Left unset, the configured extractor is built from settings.

    Raises ``UnsupportedFileTypeError`` (415) when nothing can read the format
    and ``LowTextYieldError`` (422) when everything that could read it produced
    too little to be a document.
    """
    mime = resolve_mime(filename, content_type)
    extractor = ocr if ocr is not None else _default_ocr_extractor(settings)
    threshold = settings.ingestion_min_chars_per_page

    if mime is None or (mime not in LOADER_REGISTRY and mime not in DOCAI_ONLY_MIMES):
        raise UnsupportedFileTypeError(
            unsupported_type_message(filename, ocr_available=settings.ocr_available)
        )

    # ─── Formats with no local loader go straight out to the extractor ───
    if mime not in LOADER_REGISTRY:
        if extractor is None:
            raise UnsupportedFileTypeError(
                unsupported_type_message(filename, ocr_available=False)
            )
        return await _extract_with_ocr(
            extractor, content, filename, mime, settings, threshold, local_report=None
        )

    # ─── Local first ───
    # `extract_local`, not `load_file`: a document that yields *nothing* must
    # reach the escalation below rather than raising here. A fully scanned PDF —
    # every page an image — is the single most common thing OCR exists to
    # rescue, and it is exactly the case that produces zero segments. Raising on
    # empty would make the fallback unreachable by the documents it was built
    # for, while still working for the half-scanned ones.
    segments = extract_local(content, filename, content_type)
    report = quality.assess(segments, content=content, mime_type=mime, method=METHOD_LOCAL)
    logger.info("extraction_assessed", filename=filename, **report.as_diagnostics())

    if segments and quality.passes(report, min_chars_per_page=threshold):
        return ExtractionResult(segments=segments, report=report)

    # ─── Local parsing produced nothing, or nearly nothing — escalate if we can ───
    if extractor is not None and mime in OCR_CAPABLE_MIMES:
        logger.info(
            "extraction_escalating_to_ocr",
            filename=filename,
            chars_per_page=report.chars_per_page,
            segments_found=len(segments),
            threshold=threshold,
        )
        return await _extract_with_ocr(
            extractor, content, filename, mime, settings, threshold, local_report=report
        )

    if not segments:
        # No density to quote — there is no text at all. Say what that usually
        # means and what would fix it, rather than the bare "nothing found" this
        # used to raise.
        raise LowTextYieldError(
            f"No extractable text found in '{filename}'. Every page is an image, "
            f"which is what a scanned document looks like to a PDF parser. "
            f"{_remedy(settings, ocr_attempted=False)}",
            report=report,
        )

    raise quality.rejection(
        report,
        min_chars_per_page=threshold,
        remedy=_remedy(settings, ocr_attempted=False),
    )


async def _extract_with_ocr(
    extractor: OcrExtractor,
    content: bytes,
    filename: str | None,
    mime: str,
    settings: Settings,
    threshold: int,
    local_report: ExtractionReport | None,
) -> ExtractionResult:
    """Run the remote extractor and hold its output to the same standard.

    Errors from the extractor propagate unchanged rather than collapsing into a
    quality failure. A Document AI outage and a scanned page of blank paper both
    end with "no text", but only one of them should be retried, and only one of
    them is the user's problem.
    """
    segments = await extractor.extract(content, filename, mime)
    report = quality.assess(
        segments, content=content, mime_type=mime, method=extractor.version
    )
    logger.info(
        "extraction_assessed",
        filename=filename,
        recovered_from_chars_per_page=(
            local_report.chars_per_page if local_report is not None else None
        ),
        **report.as_diagnostics(),
    )

    if not segments:
        # Distinct from the density gate: the extractor read the file and found
        # no content at all, which the gate cannot express for a non-paginated
        # format (page_count is None, so chars_per_page is None, so it passes).
        raise LowTextYieldError(
            f"Document AI read '{filename}' but found no text in it.", report=report
        )

    if not quality.passes(report, min_chars_per_page=threshold):
        raise quality.rejection(
            report,
            min_chars_per_page=threshold,
            remedy=_remedy(settings, ocr_attempted=True),
        )
    return ExtractionResult(segments=segments, report=report)
