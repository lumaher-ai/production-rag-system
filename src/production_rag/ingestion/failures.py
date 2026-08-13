"""Why an ingestion failed, as a value rather than a sentence.

Failures were recorded as ``f"{type(exc).__name__}: {exc}"`` on the job row and
nothing else. That string is good for a human reading one job and useless for
the question that actually matters operationally — *what is failing, and how
often* — because you cannot GROUP BY a message that embeds a filename.

So every failure is classified into two dimensions and a flag:

  * ``reason`` — what went wrong, in terms an operator can count.
  * ``stage`` — where in the pipeline it happened, which is what tells "the
    origin is down" apart from "the embedding provider is down".
  * ``retryable`` — whether attempting the identical work again could plausibly
    produce a different answer.

**``retryable`` is the one with teeth.** A scanned PDF rejected by the quality
gate will be rejected identically on attempt two and attempt three; retrying it
burns a worker slot, and once OCR is in the path it burns money. So a
non-retryable failure is terminal on the first attempt and the queue is told not
to bother.

Classification is by exception type against the taxonomy in ``exceptions.py``
rather than by an attribute each raise site has to remember to set. Raise sites
forget; a type cannot.
"""

import enum
from dataclasses import dataclass
from typing import Any

from production_rag.exceptions import (
    ConnectorNotConfiguredError,
    FileTooLargeError,
    InvalidSourceURIError,
    LowTextYieldError,
    NotFoundError,
    SourceFetchError,
    UnsupportedFileTypeError,
    ValidationError,
)


class FailureReason(enum.StrEnum):
    """What went wrong, at the granularity worth counting.

    Deliberately coarse. Every value here is a distinct thing an operator would
    *do* something different about — chase an upstream origin, fix credentials,
    tell a user their file is a scan, or look at a stack trace.
    """

    UNSUPPORTED_TYPE = "unsupported_type"
    LOW_TEXT_YIELD = "low_text_yield"
    PARSE_ERROR = "parse_error"
    OCR_NOT_CONFIGURED = "ocr_not_configured"
    OCR_FAILED = "ocr_failed"
    FETCH_FAILED = "fetch_failed"
    INVALID_SOURCE = "invalid_source"
    TOO_LARGE = "too_large"
    EMBEDDING_FAILED = "embedding_failed"
    NOT_FOUND = "not_found"
    INTERNAL = "internal"


class FailureStage(enum.StrEnum):
    """Where in the pipeline it happened."""

    FETCH = "fetch"
    PARSE = "parse"
    QUALITY_GATE = "quality_gate"
    OCR = "ocr"
    CHUNK = "chunk"
    EMBED = "embed"
    PERSIST = "persist"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FailureClassification:
    reason: FailureReason
    stage: FailureStage
    retryable: bool


# Most of the taxonomy maps one-to-one. Ordered most-specific first, because
# several of these are subclasses of each other (InvalidSourceURIError and
# LowTextYieldError are both ValidationError) and the first match wins.
_BY_TYPE: tuple[tuple[type[Exception], FailureClassification], ...] = (
    (
        LowTextYieldError,
        FailureClassification(
            FailureReason.LOW_TEXT_YIELD, FailureStage.QUALITY_GATE, retryable=False
        ),
    ),
    (
        InvalidSourceURIError,
        FailureClassification(FailureReason.INVALID_SOURCE, FailureStage.FETCH, retryable=False),
    ),
    (
        UnsupportedFileTypeError,
        FailureClassification(FailureReason.UNSUPPORTED_TYPE, FailureStage.PARSE, retryable=False),
    ),
    (
        FileTooLargeError,
        FailureClassification(FailureReason.TOO_LARGE, FailureStage.FETCH, retryable=False),
    ),
    (
        # The capability exists but the deployment has not been given credentials.
        # Retrying cannot conjure them, and three attempts only delays the
        # operator noticing.
        ConnectorNotConfiguredError,
        FailureClassification(FailureReason.OCR_NOT_CONFIGURED, FailureStage.OCR, retryable=False),
    ),
    (
        # Upstream: DNS, a 404 from the origin, a rate limit, a blocked address.
        # The one class of failure where trying again is the *right* answer.
        SourceFetchError,
        FailureClassification(FailureReason.FETCH_FAILED, FailureStage.FETCH, retryable=True),
    ),
    (
        NotFoundError,
        FailureClassification(FailureReason.NOT_FOUND, FailureStage.PERSIST, retryable=False),
    ),
    (
        # Anything else that reached 422: the input is wrong, and it will be
        # equally wrong next time.
        ValidationError,
        FailureClassification(FailureReason.PARSE_ERROR, FailureStage.PARSE, retryable=False),
    ),
)

# An unrecognised exception is retried. That is the conservative default: an
# unclassified failure is most likely a transient infrastructure problem (a
# dropped connection, a provider hiccup) rather than a permanent property of the
# document, and wrongly retrying costs one attempt while wrongly giving up loses
# a document.
_UNKNOWN = FailureClassification(FailureReason.INTERNAL, FailureStage.UNKNOWN, retryable=True)


def classify(exc: BaseException) -> FailureClassification:
    """Map an exception to a countable failure reason."""
    for exc_type, classification in _BY_TYPE:
        if isinstance(exc, exc_type):
            return classification
    return _UNKNOWN


def diagnostics(exc: BaseException) -> dict[str, Any]:
    """Measurements worth keeping alongside the failure, if the exception has any.

    Only the quality gate carries any today — the characters-per-page numbers
    that justified its rejection. Storing them is what turns "count of documents
    with anomalously low chars-per-page" from a re-parse into a query.
    """
    report = getattr(exc, "report", None)
    if report is None:
        return {}
    as_diagnostics = getattr(report, "as_diagnostics", None)
    return as_diagnostics() if callable(as_diagnostics) else {}
