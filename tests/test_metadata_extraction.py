"""Extractive metadata, and the filter validation that guards it.

These are not string-parsing tests. Extracted metadata becomes the predicate a
filtered query runs against, so an extractor that is *confidently wrong* makes a
document invisible to the filter that should have matched it — and nothing about
that failure looks like an error. The properties asserted here are therefore
mostly about restraint: absent rather than guessed, deterministic rather than
plausible, and unchanged by when the extractor happens to run.
"""

import pytest

from production_rag.exceptions import ValidationError
from production_rag.ingestion.metadata import (
    METADATA_VERSION,
    UNKNOWN_DOC_TYPE,
    detect_doc_type,
    detect_language,
    extract_document_date,
    extract_metadata,
    validate_metadata_filter,
)

SPANISH_CONTRACT = """CONTRATO DE PRESTACIÓN DE SERVICIOS PROFESIONALES
Fecha: 15 de marzo de 2024

De una parte, la empresa ACME S.L., y de otra parte el contratista independiente.
Las partes acuerdan las siguientes cláusulas relativas a la prestación del
servicio. La cláusula de rescisión se regirá por la legislación vigente y ambas
partes aceptan las obligaciones descritas en el presente contrato.
"""

ENGLISH_INVOICE = """INVOICE
Invoice Number: 2024-0091
Date: March 15, 2024

Bill to: Acme Corporation, 100 Market Street.
Payment terms: net 30 days. Subtotal 1,200.00. VAT 21%. Amount due 1,452.00 USD.
Please remit payment to the account listed below within the stated terms.
"""


# ─── Language ───


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (SPANISH_CONTRACT, "es"),
        (ENGLISH_INVOICE, "en"),
    ],
)
def test_detect_language_identifies_the_document_language(text: str, expected: str) -> None:
    assert detect_language(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "hola",  # below the minimum length: too little evidence to judge
        "SKU-1  SKU-2  SKU-3  4471  9928  1120  7734  0091  8823  4410  99",  # no prose
    ],
)
def test_detect_language_returns_none_rather_than_guessing(text: str) -> None:
    """Unknown must stay unknown.

    A wrong language does not degrade a filter gracefully — it excludes the
    document from the filter that should have matched it, silently.
    """
    assert detect_language(text) is None


def test_detect_language_is_deterministic_across_calls() -> None:
    """langdetect randomizes inference unless its seed is pinned.

    Without the pin the same file re-ingested twice could land in two different
    language buckets, so this asserts the pin is in force — not that langdetect
    happens to be stable.
    """
    assert len({detect_language(SPANISH_CONTRACT) for _ in range(10)}) == 1


# ─── Document date ───


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Informe revisado el 2024-03-15 por el comité.", "2024-03-15"),  # ISO
        ("Firmado el 15 de marzo de 2024 en Madrid.", "2024-03-15"),  # es, long form
        ("Firmado el 15 de marzo 2024 en Madrid.", "2024-03-15"),  # es, no second 'de'
        ("Dated March 15, 2024 and effective immediately.", "2024-03-15"),  # en, month-first
        ("Dated Mar 15 2024 and effective immediately.", "2024-03-15"),  # en, abbreviated
        ("Signed 15 March 2024 in Madrid.", "2024-03-15"),  # en, day-first
    ],
)
def test_extract_document_date_parses_unambiguous_formats(text: str, expected: str) -> None:
    assert extract_document_date(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Reference 03/04/2024 on file.",  # 3 April or 4 March — unresolvable, so skipped
        "Reference 2024-13-45 on file.",  # matches the ISO shape, is not a date
        "Signed 30 February 2024 in Madrid.",  # ditto for the long form
        "Order 1823-99-01 shipped.",  # year outside the accepted window
        "Case 2400-01-01 archived.",  # ditto, upper bound
        "No date appears anywhere in this sentence.",
    ],
)
def test_extract_document_date_returns_none_for_ambiguous_or_invalid(text: str) -> None:
    """A wrong date is worse than a missing one.

    All-numeric formats are the important case here: ``03/04/2024`` is 3 April to
    most of the world and 4 March to the US, and nothing in the text resolves it.
    Refusing to parse them is a decision, and this test is what fixes it.
    """
    assert extract_document_date(text) is None


def test_extract_document_date_prefers_the_earliest_occurrence() -> None:
    """The header date is the document's; later ones cite other things."""
    text = "Contrato de 15 de marzo de 2024.\n\nDeroga el acuerdo de 2 de enero de 2020."
    assert extract_document_date(text) == "2024-03-15"


def test_extract_document_date_does_not_depend_on_when_it_runs() -> None:
    """The accepted year window is fixed, not relative to the current year.

    An extractor whose output changes with the calendar would make a re-ingest of
    an unchanged file produce different metadata next January.
    """
    assert extract_document_date("Effective 2099-12-31.") == "2099-12-31"


# ─── Document type ───


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (SPANISH_CONTRACT, "contract"),
        (ENGLISH_INVOICE, "invoice"),
    ],
)
def test_detect_doc_type_classifies_on_content_not_format(text: str, expected: str) -> None:
    """doc_type is what the document *is*; mime_type is what the file is.

    A contract is a contract whether it arrived as PDF or DOCX, and "all my
    contracts" is the filter people actually want.
    """
    assert detect_doc_type(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "short",
        # Ordinary prose with no markers of any class. Must not be forced into
        # the nearest bucket just because something has to be returned.
        "El gato subió al tejado y se quedó mirando la calle durante toda la tarde, "
        "sin prestar atención a nadie que pasara por debajo del alero.",
    ],
)
def test_detect_doc_type_falls_back_to_other(text: str) -> None:
    assert detect_doc_type(text) == UNKNOWN_DOC_TYPE


def test_detect_doc_type_requires_more_than_one_marker() -> None:
    """One suggestive word is not a classification.

    'Agreement' appears in plenty of documents that are not contracts; counting
    distinct markers is what stops a single word from labelling a whole document.
    """
    text = "This agreement was mentioned in passing during the meeting, though " * 6
    assert detect_doc_type(text) == UNKNOWN_DOC_TYPE


# ─── Assembly ───


def test_extract_metadata_returns_the_expected_document() -> None:
    assert extract_metadata(SPANISH_CONTRACT, mime_type="application/pdf") == {
        "doc_type": "contract",
        "language": "es",
        "document_date": "2024-03-15",
        "mime_type": "application/pdf",
        "extractor_version": METADATA_VERSION,
    }


def test_extract_metadata_omits_undetermined_keys_rather_than_storing_null() -> None:
    """Absent and null are different to JSONB containment.

    ``metadata @> '{"language": null}'`` matches a stored null but not an absent
    key, so storing nulls would build a filter that matches documents precisely
    because detection failed on them.
    """
    metadata = extract_metadata("1 2 3", mime_type=None)

    assert "language" not in metadata
    assert "document_date" not in metadata
    assert "mime_type" not in metadata
    # doc_type is always present: "we looked and it matched nothing" is itself a
    # filterable answer, unlike a detector that could not run.
    assert metadata["doc_type"] == UNKNOWN_DOC_TYPE


def test_extract_metadata_always_stamps_the_extractor_version() -> None:
    """The version travels inside the document, which is what makes 'which rows
    predate the current extractor?' a query instead of a migration."""
    assert extract_metadata("", mime_type=None)["extractor_version"] == METADATA_VERSION


def test_extract_metadata_is_deterministic() -> None:
    """Re-ingesting unchanged content must produce byte-identical metadata.

    The ingestion path compares the stored ``extractor_version`` to decide whether
    a refresh is needed; if extraction drifted between runs, unchanged documents
    would churn their metadata forever.
    """
    first = extract_metadata(SPANISH_CONTRACT, mime_type="application/pdf")
    second = extract_metadata(SPANISH_CONTRACT, mime_type="application/pdf")
    assert first == second


# ─── Filter validation ───


@pytest.mark.parametrize(
    "filters",
    [
        {"language": "es"},
        {"language": "es", "doc_type": "contract"},
        {"page_count": 12},  # numbers pass: metadata keys are not a fixed set
        {"confidential": True},
    ],
)
def test_validate_metadata_filter_accepts_flat_scalar_maps(filters: dict) -> None:
    assert validate_metadata_filter(filters) == filters


@pytest.mark.parametrize("filters", [None, {}])
def test_validate_metadata_filter_treats_empty_as_no_filter(filters: dict | None) -> None:
    """An empty containment matches every row, so the two really are equivalent."""
    assert validate_metadata_filter(filters) is None


@pytest.mark.parametrize(
    "filters",
    [
        {"meta": {"language": "es"}},  # nested containment means something surprising
        {"tags": ["a", "b"]},  # inside an array, @> means "contains", not "equals"
        {"language": None},  # would match documents *because* detection failed
        {"": "es"},
        {"language": "e" * 500},
        dict.fromkeys((f"k{i}" for i in range(11)), "v"),  # over the key cap
    ],
)
def test_validate_metadata_filter_rejects_anything_beyond_flat_scalars(filters: dict) -> None:
    """The accepted surface is deliberately small.

    Ranges and OR are not expressible here and are not pretended to be — better a
    422 than a filter that quietly does something other than what it reads like.
    """
    with pytest.raises(ValidationError):
        validate_metadata_filter(filters)
