"""Normalization rules, and the properties the ingestion pipeline depends on.

Two properties matter beyond the individual rules:

  - **idempotence** — a re-index re-normalizes already-normalized text, so a
    non-idempotent rule would make the content hash unstable across passes and
    every re-index would look like an edit;
  - **hash convergence** — inputs differing only in normalized-away characters
    must produce the same ``content_hash``, which is what makes a re-upload that
    differs only in typography a genuine no-op instead of a re-embed.
"""

import pytest

from production_rag.ingestion.idempotency import content_hash
from production_rag.ingestion.loaders import ExtractedSegment
from production_rag.ingestion.normalize import (
    NORMALIZER_VERSION,
    normalize_segments,
    normalize_text,
)

# ─── Rule 1: NFKC ───


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ﬁle", "file"),  # ligature — the headline retrieval win
        ("ﬂow", "flow"),
        ("Ｈｅｌｌｏ", "Hello"),  # full-width → half-width
        ("①②", "12"),  # circled digits
        ("nb space", "nb space"),  # NBSP → plain space
        ("Ⅳ", "IV"),  # roman numeral
    ],
)
def test_nfkc_folds_compatibility_characters(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_nfkc_is_lossy_for_math_and_that_is_documented() -> None:
    """Pinning the accepted trade-off, so a change to it is a deliberate act.

    NFKC flattens compatibility characters used for notation. This test exists to
    make that visible rather than to endorse it — if the corpus ever becomes
    math-heavy, this is the assertion that should fail and force a new
    NORMALIZER_VERSION.
    """
    assert normalize_text("x²") == "x2"
    assert normalize_text("½") == "1⁄2"
    assert normalize_text("𝐀𝐁") == "AB"


# ─── Rule 2: invisibles and control characters ───


@pytest.mark.parametrize(
    "invisible",
    [
        "​",  # zero-width space
        "‌",  # zero-width non-joiner
        "‍",  # zero-width joiner
        "⁠",  # word joiner
        "﻿",  # BOM / zero-width no-break space
        "­",  # soft hyphen, left inside words by PDF extractors
    ],
)
def test_invisible_characters_are_stripped(invisible: str) -> None:
    """NFKC does not remove these; left in, each becomes an embedded token."""
    assert normalize_text(f"re{invisible}trieval") == "retrieval"


def test_control_characters_are_stripped_but_tabs_and_newlines_survive() -> None:
    assert normalize_text("a\x0cb") == "ab"  # form feed
    assert normalize_text("a\x0bb") == "ab"  # vertical tab
    assert normalize_text("a\tb") == "a b"  # tab is whitespace, collapsed
    assert normalize_text("a\nb") == "a\nb"  # newline is structure, kept


# ─── Rules 3-6: whitespace ───


def test_line_endings_are_unified() -> None:
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"


def test_intra_line_runs_collapse_but_paragraphs_survive() -> None:
    # Column layout in PDFs produces long space runs.
    assert normalize_text("word     spaced") == "word spaced"
    # "\n\n" is the recursive splitter's top separator — collapsing it would
    # change how every document chunks.
    assert normalize_text("para one\n\npara two") == "para one\n\npara two"


def test_trailing_whitespace_per_line_is_removed() -> None:
    assert normalize_text("line one   \n   line two") == "line one\nline two"


def test_blank_runs_are_capped_at_one_blank_line() -> None:
    assert normalize_text("a\n\n\n\n\n\nb") == "a\n\nb"


def test_document_is_trimmed() -> None:
    assert normalize_text("\n\n  content  \n\n") == "content"


@pytest.mark.parametrize("empty", ["", "   ", "\n\n\n", "​﻿", "\t \r\n"])
def test_whitespace_only_input_normalizes_to_empty(empty: str) -> None:
    assert normalize_text(empty) == ""


# ─── Properties the pipeline relies on ───


@pytest.mark.parametrize(
    "raw",
    [
        "ﬁle​ name   with\r\nmixed\t\twhitespace\n\n\n\nand paragraphs",
        "Ｈｅｌｌｏ　ｗｏｒｌｄ",
        "plain ascii already normalized",
        "",
    ],
)
def test_normalization_is_idempotent(raw: str) -> None:
    once = normalize_text(raw)
    assert normalize_text(once) == once


def test_inputs_differing_only_in_typography_hash_identically() -> None:
    """Why the hash is the precise detector: same meaning, same identity.

    A re-upload of a document that differs only in ligatures or whitespace is a
    genuine no-op — the content hash converges, so the idempotency gate skips it
    without spending on embeddings.
    """
    a = normalize_text("The ﬁle  contains\r\nﬂow control.")
    b = normalize_text("The file contains\nflow control.")

    assert a == b
    assert content_hash(a) == content_hash(b)


def test_genuinely_different_text_still_hashes_differently() -> None:
    """The converse — normalization must not collapse real differences."""
    assert content_hash(normalize_text("flow control")) != content_hash(
        normalize_text("flow controls")
    )


# ─── Segment handling ───


def test_segments_are_normalized_individually_preserving_provenance() -> None:
    segments = [
        ExtractedSegment(text="ﬁrst  page", page=1),
        ExtractedSegment(text="second\r\npage", page=2, section="Intro"),
    ]

    result = normalize_segments(segments)

    assert [s.text for s in result] == ["first page", "second\npage"]
    # Provenance must survive — it is what page/section columns are populated from.
    assert [s.page for s in result] == [1, 2]
    assert result[1].section == "Intro"


def test_segments_that_normalize_to_empty_are_dropped() -> None:
    """An empty chunk would be embedded and retrieved as a meaningless result."""
    segments = [
        ExtractedSegment(text="real content", page=1),
        ExtractedSegment(text="   ​\n\n ", page=2),
        ExtractedSegment(text="more content", page=3),
    ]

    result = normalize_segments(segments)

    assert [s.page for s in result] == [1, 3]


def test_normalize_segments_does_not_mutate_its_input() -> None:
    """Callers may still need the originals; normalization returns new objects."""
    original = ExtractedSegment(text="ﬁle", page=1)

    normalize_segments([original])

    assert original.text == "ﬁle"


# ─── The version itself ───


def test_normalizer_version_is_set() -> None:
    """A blank version would silently disable the staleness gate."""
    assert NORMALIZER_VERSION
    assert NORMALIZER_VERSION != "none"  # reserved for pre-normalization rows
