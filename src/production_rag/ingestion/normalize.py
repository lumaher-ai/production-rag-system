"""Mechanical text normalization, applied before embedding.

Loader output carries artifacts that mean nothing semantically but everything to
an embedding model: ligatures from PDF fonts (``ﬁle``), non-breaking spaces,
zero-width joiners, ``\\r\\n`` line endings, and runs of spaces left by column
layout. Each becomes a token. That noise degrades *every* query against the
document, forever, and no reranker recovers it — which is why this runs at
ingestion rather than at retrieval.

**Normalization rules are a silent input to every stored vector.** Change them
without recording which rules produced a row and the idempotency gate reports
"unchanged" while serving vectors computed under rules that no longer apply,
mixing incompatible embeddings in one index with nothing raising an error. So
``NORMALIZER_VERSION`` sits alongside ``chunker_version`` and ``embedding_model``
in every idempotency key. **Bump it whenever any rule below changes.**

Note that ``content_hash`` is computed over *normalized* text, so a rule change
that alters a document already changes its hash and trips the gate on its own.
The version is not redundant with that, for three reasons the hash cannot cover:

  - the **query side is never hashed** — a normalized question is not stored, so
    the version is the only thing that can invalidate cached answers when the
    rules move;
  - **targeted re-processing** ("which documents predate the fix?") is an indexed
    column lookup, not a full-corpus re-normalize-and-diff;
  - hashing raw bytes instead of normalized text is a tempting future
    optimization, and the version is what stands between that refactor and
    silently mixed embeddings.

**NFKC is lossy, deliberately.** It folds compatibility characters, so ``x²``
becomes ``x2``, ``½`` becomes ``1⁄2``, and mathematical alphanumerics ``𝐀``
become ``A``. That is accepted here: the retrieval win (a search for ``file``
matching a PDF's ``ﬁle`` ligature) outweighs math fidelity for a prose-heavy
corpus. A corpus of scientific notation would want NFC instead — that would be a
different normalizer, and therefore a different version string.
"""

import re
import unicodedata

from production_rag.ingestion.loaders import ExtractedSegment

# Bump on ANY change to the rules below. Part of every idempotency key, so a
# change here forces affected content to be re-normalized and re-embedded
# instead of being served from vectors computed under the old rules.
NORMALIZER_VERSION = "nfkc-ws-v1"

# Zero-width and formatting characters that survive NFKC but carry no meaning
# for retrieval: ZWSP, ZWNJ, ZWJ, word joiner, BOM/ZWNBSP, and the soft hyphen
# that PDF extractors leave inside words broken across lines.
_INVISIBLE = dict.fromkeys(
    map(ord, "​‌‍⁠﻿­"),
    None,
)

# Runs of spaces/tabs *within* a line. Newlines are excluded so paragraph
# structure survives — the recursive splitter uses "\n\n" as its top separator,
# so collapsing it would change how every document chunks.
_INLINE_WS = re.compile(r"[^\S\n]+")

# Whitespace hugging a newline, including trailing space at end of line.
_AROUND_NEWLINE = re.compile(r"[^\S\n]*\n[^\S\n]*")

# Three or more newlines — i.e. more than one blank line.
_BLANK_RUN = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Apply the mechanical normalization rules, in order.

    Idempotent: ``normalize_text(normalize_text(x)) == normalize_text(x)``. That
    matters because a re-index re-normalizes text that is already normalized, and
    a non-idempotent rule would make the content hash unstable across passes.
    """
    if not text:
        return ""

    # 1. Unicode compatibility composition. Folds ligatures, full-width forms,
    #    circled/superscript digits, and maps NBSP (U+00A0) to a plain space.
    text = unicodedata.normalize("NFKC", text)

    # 2. Line endings: CRLF and lone CR both become LF. This MUST precede the
    #    control-character strip below — CR is itself a control character, so
    #    stripping first would delete a lone CR and silently lose that line
    #    break instead of converting it.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Drop invisibles NFKC leaves behind, then any remaining control
    #    characters except tab and newline (category Cc/Cf: stray form feeds,
    #    vertical tabs, and bidi marks all show up in real PDF extractions).
    text = text.translate(_INVISIBLE)
    text = "".join(
        ch for ch in text if ch in "\n\t" or unicodedata.category(ch) not in ("Cc", "Cf")
    )

    # 4. Collapse intra-line whitespace runs, and strip whitespace around
    #    newlines so no line carries trailing spaces.
    text = _INLINE_WS.sub(" ", text)
    text = _AROUND_NEWLINE.sub("\n", text)

    # 5. Cap blank runs at a single blank line. Paragraph breaks are meaningful
    #    to the splitter; six of them are not.
    text = _BLANK_RUN.sub("\n\n", text)

    # 6. Trim the document as a whole.
    return text.strip()


def normalize_segments(segments: list[ExtractedSegment]) -> list[ExtractedSegment]:
    """Normalize each segment, dropping any that becomes empty.

    Segments are normalized individually so their ``page``/``section``
    provenance is preserved. A segment of nothing but whitespace or invisible
    characters normalizes to ``""`` and is dropped rather than becoming an empty
    chunk that would be embedded and retrieved as a meaningless result.
    """
    normalized: list[ExtractedSegment] = []
    for segment in segments:
        text = normalize_text(segment.text)
        if text:
            normalized.append(segment.model_copy(update={"text": text}))
    return normalized
