"""Extractive document metadata: language, document date, document type.

Retrieval that can only rank by similarity cannot answer "the *Spanish*
contracts" — it can only answer "the passages that read like contracts", which
is a different and worse question. This module produces the small set of facts a
filter needs, derived from the document's own text at ingestion time.

**Everything here is extractive: regex and heuristics over text the pipeline
already has.** No model call, no network, no per-document cost that scales with
corpus size. That ceiling is deliberate — model-derived metadata (NER, topic
classification, LLM summaries) is a real option, but it should be bought only
when a filter or an eval failure proves the heuristics are insufficient.

**Absent beats wrong.** Every extractor returns ``None`` rather than a guess when
its evidence is thin, and ``extract_metadata`` omits null keys from the document
instead of storing ``null``. A wrong ``language`` does not degrade a filter
gracefully — it makes the document invisible to the filter that should have
matched it, and visible to one that should not. Silence is recoverable; a
confident error is not.

**Where this metadata lives is the point.** It goes into one ``metadata`` JSONB
column with a GIN index, not into typed columns. Which fields matter is not yet
known, and a typed column per field means an ALTER TABLE per field forever. The
version below is stored *inside* the document, so the question "which rows
predate the current extractor?" is answerable without a migration:

    SELECT id FROM documents
    WHERE metadata->>'extractor_version' IS DISTINCT FROM 'extractive-v1';

**METADATA_VERSION is NOT part of the ingestion idempotency gate**, unlike
``NORMALIZER_VERSION`` / ``CHUNKER_VERSION`` / ``embedding_model``. Those three
are silent inputs to a stored *vector*: change one and the embeddings mean
something different. This metadata is derived from content and never reaches the
embedding model, so putting it in the gate would force a full corpus re-embed
every time a keyword is added to a list below. Instead, unchanged content whose
``extractor_version`` has moved gets its metadata refreshed in place — see
``IngestionService._persist``. **Bump the version on any rule change below** so
that refresh can find the affected rows.
"""

import re
from datetime import date
from typing import Any

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

from production_rag.exceptions import ValidationError

# langdetect's inference is randomized and returns different answers for the same
# input across processes unless the seed is pinned. This module's output feeds
# stored metadata and query filters, so non-determinism would mean the same file
# re-ingested twice could land in two different language buckets.
DetectorFactory.seed = 0

# Bump on ANY change to the rules below — the marker tables, the date patterns,
# the thresholds, or the shape of the returned document.
METADATA_VERSION = "extractive-v1"

# Extractors read a bounded prefix, not the whole document. A 200-page PDF and a
# one-page memo should cost the same to classify, and the signals these rules look
# for (title, date line, opening boilerplate) are front-loaded by construction.
_SAMPLE_CHARS = 4000

# Below this, there is not enough text for any of these heuristics to mean
# anything, and langdetect in particular becomes a coin flip on short strings.
_MIN_CHARS = 40


# ─── Language ───

# langdetect only raises on input with *no* usable features. Given a little — a
# table of SKUs, a page of figures — it answers, and it answers confidently: a
# column of part numbers comes back as Croatian at p=0.86. So two guards sit
# around it, one on each side.
#
# Input side: how much of the sample is actually letters. Prose in any script runs
# well above 0.7 (CJK characters are alphabetic to `str.isalpha`); reference tables
# and numeric layouts land near 0.15.
_MIN_ALPHA_RATIO = 0.5
# Output side: langdetect's own probability for its top candidate. Real prose
# comes back at ~0.9999; text it is really only guessing at splits its mass across
# candidates and lands well below this.
_MIN_LANGUAGE_PROB = 0.90


def detect_language(text: str) -> str | None:
    """ISO 639-1 code for the document's language, or ``None`` if undetermined.

    ``None`` is a real answer, returned for text too short to judge, for input
    that is mostly not prose (a page of figures, a table of SKUs), and for input
    the detector itself is unsure about. Callers must treat it as "unknown", never
    as "not Spanish".
    """
    sample = text[:_SAMPLE_CHARS].strip()
    if len(sample) < _MIN_CHARS:
        return None

    alpha_ratio = sum(char.isalpha() for char in sample) / len(sample)
    if alpha_ratio < _MIN_ALPHA_RATIO:
        return None

    try:
        candidates = detect_langs(sample)
    except LangDetectException:
        # Raised when the input has no features to detect on. Not an error here —
        # it is the same "unknown" the guards above return.
        return None

    if not candidates or candidates[0].prob < _MIN_LANGUAGE_PROB:
        return None
    return str(candidates[0].lang)


# ─── Document date ───

_MONTHS: dict[str, int] = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

# Each pattern names its groups y/m/d so one loop can handle all three orderings.
_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # ISO 8601: 2024-03-15. Unambiguous by definition, so it wins by being first.
    re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b"),
    # Spanish long form: "15 de marzo de 2024" / "15 de marzo 2024".
    re.compile(
        rf"\b(?P<d>\d{{1,2}})\s+de\s+(?P<m>{_MONTH_ALT})\s+(?:de\s+)?(?P<y>\d{{4}})\b",
        re.IGNORECASE,
    ),
    # English month-first: "March 15, 2024" / "Mar 15 2024".
    re.compile(rf"\b(?P<m>{_MONTH_ALT})\.?\s+(?P<d>\d{{1,2}}),?\s+(?P<y>\d{{4}})\b", re.IGNORECASE),
    # English/European day-first: "15 March 2024".
    re.compile(rf"\b(?P<d>\d{{1,2}})\s+(?P<m>{_MONTH_ALT})\.?,?\s+(?P<y>\d{{4}})\b", re.IGNORECASE),
)

# A document dated outside this window is a parse artifact, not a date — page
# numbers and reference codes produce four-digit runs constantly. The bounds are
# constants rather than "the current year" on purpose: an extractor whose output
# depends on when it ran is not reproducible, and re-ingesting an unchanged file
# next January must not change its stored metadata.
_MIN_YEAR = 1900
_MAX_YEAR = 2100


def extract_document_date(text: str) -> str | None:
    """The document's own date as ``YYYY-MM-DD``, or ``None`` if not found.

    This is the date written *in* the document — the date of the contract, the
    invoice, the report — which is a different fact from ``created_at`` (when the
    row was written) and is the one a user means by "documents from 2024".

    The earliest-occurring match wins: document dates appear in the header,
    while dates further in are more likely to be references to other things.

    **All-numeric formats are deliberately not parsed.** ``03/04/2024`` is 3 April
    to most of the world and 4 March to the US, and nothing in the text resolves
    it. Since a wrong date is worse than a missing one — it silently excludes the
    document from the filter that should have matched it — these are skipped.
    Documents needing them should carry an explicit ISO date.
    """
    sample = text[:_SAMPLE_CHARS]

    best: tuple[int, date] | None = None
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(sample):
            raw_month = match.group("m")
            month = int(raw_month) if raw_month.isdigit() else _MONTHS[raw_month.lower()]
            year = int(match.group("y"))
            if not _MIN_YEAR <= year <= _MAX_YEAR:
                continue
            try:
                parsed = date(year, month, int(match.group("d")))
            except ValueError:
                continue  # e.g. 2024-13-45 or 30 February — matched the shape, is not a date
            if best is None or match.start() < best[0]:
                best = (match.start(), parsed)

    return best[1].isoformat() if best is not None else None


# ─── Document type ───

# Markers are matched against lowercased text. A "strong" marker is one that
# essentially only appears in its class (an invoice number label, a contract
# recital); a normal marker is suggestive but common enough that two distinct
# ones are required. Counting *distinct* markers rather than occurrences is what
# keeps a single repeated word from classifying a whole document.
_STRONG_MARKERS: dict[str, tuple[str, ...]] = {
    "invoice": ("invoice number", "invoice no", "número de factura", "numero de factura"),
    "contract": ("hereinafter referred to as", "witnesseth", "en fe de lo cual", "de una parte"),
    "resume": ("curriculum vitae", "work experience", "experiencia laboral"),
    "policy": ("privacy policy", "política de privacidad", "politica de privacidad"),
    "manual": ("user manual", "manual de usuario", "installation guide"),
    "email": ("-----original message-----", "mensaje original"),
    "report": ("executive summary", "resumen ejecutivo"),
}

_MARKERS: dict[str, tuple[str, ...]] = {
    "invoice": (
        "invoice",
        "factura",
        "subtotal",
        "amount due",
        "total a pagar",
        "bill to",
        "facturar a",
        "tax id",
        "iva",
        "vat",
        "payment terms",
        "condiciones de pago",
    ),
    "contract": (
        "agreement",
        "contrato",
        "the parties",
        "las partes",
        "clause",
        "cláusula",
        "clausula",
        "shall be governed by",
        "se regirá por",
        "termination",
        "rescisión",
        "rescision",
        "obligations",
        "obligaciones",
    ),
    "resume": (
        "resume",
        "education",
        "formación académica",
        "formacion academica",
        "skills",
        "habilidades",
        "references available",
        "professional summary",
        "perfil profesional",
    ),
    "policy": (
        "policy",
        "política",
        "politica",
        "must comply",
        "debe cumplir",
        "scope",
        "alcance",
        "effective date",
        "fecha de vigencia",
        "applies to all",
    ),
    "manual": (
        "step 1",
        "paso 1",
        "troubleshooting",
        "solución de problemas",
        "solucion de problemas",
        "requirements",
        "requisitos",
        "getting started",
    ),
    "email": ("subject:", "asunto:", "from:", "de:", "to:", "para:", "cc:", "sent:"),
    "report": (
        "abstract",
        "resumen",
        "introduction",
        "introducción",
        "introduccion",
        "conclusions",
        "conclusiones",
        "methodology",
        "metodología",
        "metodologia",
        "figure 1",
        "figura 1",
        "findings",
        "hallazgos",
    ),
}

# Tie-break order, most specific class first. Fixed so the result never depends on
# dict iteration or on which class happened to be scored first.
_TYPE_PRIORITY: tuple[str, ...] = (
    "invoice",
    "contract",
    "resume",
    "email",
    "policy",
    "manual",
    "report",
)

_MIN_MARKERS = 2

UNKNOWN_DOC_TYPE = "other"


def detect_doc_type(text: str) -> str:
    """Coarse semantic document class, or ``"other"``.

    This is *what the document is*, which is not what ``mime_type`` records —
    that is what the file **format** is. A contract is a contract whether it
    arrives as PDF or DOCX, and "all my contracts" is the filter people actually
    want. Both are stored, as separate keys.

    Returns ``"other"`` rather than ``None`` because "we looked and it matched
    nothing" is itself a filterable answer, unlike language where the extractor
    can genuinely fail to run.
    """
    sample = text[:_SAMPLE_CHARS].lower()
    if len(sample.strip()) < _MIN_CHARS:
        return UNKNOWN_DOC_TYPE

    best_type = UNKNOWN_DOC_TYPE
    best_score = 0
    for doc_type in _TYPE_PRIORITY:
        if any(marker in sample for marker in _STRONG_MARKERS.get(doc_type, ())):
            score = _MIN_MARKERS
        else:
            score = sum(1 for marker in _MARKERS[doc_type] if marker in sample)
        # Strict > keeps _TYPE_PRIORITY as the tie-break: an earlier, more
        # specific class holds the lead against a later class with an equal score.
        if score >= _MIN_MARKERS and score > best_score:
            best_type, best_score = doc_type, score

    return best_type


# ─── Assembly ───


def extract_metadata(text: str, *, mime_type: str | None = None) -> dict[str, Any]:
    """Build a document's metadata document from its normalized text.

    Pure and deterministic: the same text and MIME always produce the same dict,
    which is what lets a re-ingest be compared against what is stored rather than
    blindly overwriting it.

    Keys whose extractor returned ``None`` are **omitted**, never stored as
    ``null``. JSONB containment distinguishes the two — ``metadata @>
    '{"language": null}'`` matches a stored null but not an absent key — so
    storing nulls would create a filter that matches documents precisely because
    detection failed on them. Omitting also keeps the GIN index small.

    Takes text, not bytes: it must run on the *normalized* body, so the extractor
    sees the same string the embedding model does rather than the loader's
    ligatures and stray whitespace.
    """
    metadata: dict[str, Any] = {
        "doc_type": detect_doc_type(text),
        "extractor_version": METADATA_VERSION,
    }

    language = detect_language(text)
    if language is not None:
        metadata["language"] = language

    document_date = extract_document_date(text)
    if document_date is not None:
        metadata["document_date"] = document_date

    if mime_type:
        metadata["mime_type"] = mime_type

    return metadata


# ─── Filter validation ───

# Containment is evaluated against a flat object, so the filter must be one.
_FILTER_SCALARS = (str, int, float, bool)
_MAX_FILTER_KEYS = 10
_MAX_FILTER_VALUE_LEN = 256


def validate_metadata_filter(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Check a caller-supplied retrieval filter, or raise ``ValidationError`` (422).

    The filter is applied as JSONB containment (``metadata @> :filter``), which
    accepts arbitrary nesting — but nested containment has semantics most callers
    do not expect (inside an array it means "contains these elements", not
    "equals"), so anything beyond a flat map of scalars is rejected rather than
    silently doing something surprising. This is a deliberately small surface;
    ranges and OR are not expressible and are not pretended to be.

    ``None`` and ``{}`` both pass through as "no filter" — an empty containment
    matches every row, so the two are genuinely equivalent here.
    """
    if not filters:
        return None

    if len(filters) > _MAX_FILTER_KEYS:
        raise ValidationError(
            f"Metadata filter accepts at most {_MAX_FILTER_KEYS} keys, got {len(filters)}."
        )

    for key, value in filters.items():
        if not isinstance(key, str) or not key:
            raise ValidationError("Metadata filter keys must be non-empty strings.")
        # bool is a subclass of int, so it is already covered by _FILTER_SCALARS;
        # listing it explicitly above is for readability, not behaviour.
        if value is None or not isinstance(value, _FILTER_SCALARS):
            raise ValidationError(
                f"Metadata filter value for '{key}' must be a string, number, or boolean — "
                "nested objects and arrays are not supported."
            )
        if isinstance(value, str) and len(value) > _MAX_FILTER_VALUE_LEN:
            raise ValidationError(
                f"Metadata filter value for '{key}' exceeds {_MAX_FILTER_VALUE_LEN} characters."
            )

    return filters
