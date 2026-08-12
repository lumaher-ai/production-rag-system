"""Quality gates — what stands between a language model's output and a dataset.

Every function here is pure: a record in, a verdict out, no database, no LLM, no
clock. That is deliberate and it is what makes the most important code in this
package testable without a container or an API key.

The governing principle: **the prompt is a hint, the gate is the guarantee.**
Telling a model to copy a snippet character-for-character measurably improves
compliance and does not enforce anything. A dataset whose "ground truth" is a
paraphrase the model believed it was quoting is not a measurement instrument —
it is a random number generator with provenance metadata. So every citation is
checked against the text it claims to come from, and the ones that fail are
discarded rather than repaired.
"""

import re
from collections.abc import Sequence

from production_rag.ingestion.normalize import normalize_text

# Phrases that presuppose the very context retrieval is supposed to *find*.
# A question like "what does the passage above say about X?" is unanswerable by
# any retrieval system, not because the system is bad but because the question
# names no searchable subject. These are the single most common defect in
# LLM-generated eval sets and they inflate every failure metric. Spanish is
# included because the metadata extractor detects and the corpus admits it.
SELF_REFERENTIAL = (
    "according to the document",
    "according to the context",
    "according to the passage",
    "according to the text",
    "according to the excerpt",
    "in the passage",
    "in the context",
    "in the text above",
    "in the excerpt",
    "the text above",
    "the passage above",
    "this passage",
    "this excerpt",
    "this chunk",
    "this section",
    "the given text",
    "as mentioned above",
    "as stated above",
    "as described above",
    "the document says",
    "the document states",
    "según el documento",
    "según el texto",
    "según el pasaje",
    "en el pasaje",
    "en el texto anterior",
    "el texto anterior",
)

# Trimmed to words that carry no topical signal. Kept small on purpose: an
# aggressive stopword list makes two questions about different subjects look
# identical to the shingle comparison and silently deletes good items.
STOPWORDS = frozenset(
    [
        "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
        "by", "for", "with", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "do", "does", "did", "what", "which", "who", "whom",
        "whose", "when", "where", "why", "how", "this", "that", "these",
        "those", "it", "its",
    ]
)

MIN_QUESTION_CHARS = 15
MAX_QUESTION_CHARS = 300
MIN_SNIPPET_CHARS = 20
MAX_SNIPPET_CHARS = 400
SHINGLE_SIZE = 3
DUPLICATE_JACCARD = 0.7
LEAK_RUN_WORDS = 8

_WORDS = re.compile(r"[0-9a-zà-ÿ_.\-]+", re.IGNORECASE)


# ─── Snippet verification: the load-bearing gate ───


def _collapse_with_offsets(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to single spaces, remembering where each char came from."""
    chars: list[str] = []
    offsets: list[int] = []
    in_whitespace = False
    for position, char in enumerate(text):
        if char.isspace():
            if not in_whitespace:
                chars.append(" ")
                offsets.append(position)
                in_whitespace = True
        else:
            chars.append(char)
            offsets.append(position)
            in_whitespace = False
    return "".join(chars), offsets


def recover_snippet(snippet: str, chunk_content: str) -> str | None:
    """The exact span of ``chunk_content`` the model was citing, or ``None``.

    Both sides go through ``normalize_text`` first, and skipping that would fail
    nearly every snippet from a PDF- or Unicode-bearing chunk for reasons that
    have nothing to do with the model: the corpus is stored NFKC-folded with
    whitespace collapsed, so a ligature or a non-breaking space in the model's
    reply would read as a fabricated citation when it is a faithful copy of what
    the model was shown.

    Two passes. An exact substring match is the answer when it exists. Failing
    that, both sides are compared with whitespace runs collapsed — models reflow
    lines even when told not to — and on a match the **chunk's own text is
    returned**, not the model's. That distinction is the whole point: the stored
    snippet must be a span that exists in the corpus, so that a later reader can
    find it, not a reconstruction that merely resembles one.

    Anything beyond whitespace — a changed word, an expanded abbreviation, an
    added ellipsis — is a miss, deliberately. Fuzzy matching here would admit
    paraphrased "quotations", and a citation is exact or it is not a citation.
    """
    chunk = normalize_text(chunk_content)
    snip = normalize_text(snippet)
    if not snip or not chunk:
        return None
    if snip in chunk:
        return snip

    collapsed_chunk, offsets = _collapse_with_offsets(chunk)
    collapsed_snip = re.sub(r"\s+", " ", snip).strip()
    if not collapsed_snip:
        return None
    position = collapsed_chunk.find(collapsed_snip)
    if position < 0:
        return None
    start = offsets[position]
    end = offsets[position + len(collapsed_snip) - 1] + 1
    return chunk[start:end]


def verify_snippet(snippet: str, chunk_content: str) -> bool:
    """Whether the cited span really occurs in the chunk it cites."""
    return recover_snippet(snippet, chunk_content) is not None


def find_overlap_chunks(
    snippet: str,
    neighbours: Sequence[tuple[int, str]],
) -> list[int]:
    """Neighbouring chunk indices that contain this snippet too.

    ``CHUNK_OVERLAP`` is 200 characters, so a span near a chunk boundary is
    genuinely present in the adjacent chunk as well — the same text, stored
    twice, at two indices. Counting only the chunk the generator happened to be
    shown would score a retriever as *wrong* for returning the neighbour holding
    the identical sentence, which is a defect built into the ruler rather than a
    property of the system being measured. Those indices become secondary gold.
    """
    return [index for index, content in neighbours if verify_snippet(snippet, content)]


# ─── Question-shape gates ───


def is_self_referential(question: str) -> bool:
    lowered = question.casefold()
    return any(phrase in lowered for phrase in SELF_REFERENTIAL)


def has_valid_length(question: str) -> bool:
    return MIN_QUESTION_CHARS <= len(question.strip()) <= MAX_QUESTION_CHARS


def _words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORDS.finditer(text)]


def leaks_answer(question: str, snippet: str) -> bool:
    """Whether the question quotes a long run of its own answer.

    A question carrying eight consecutive words of the snippet does not test
    retrieval; it tests string matching, and it will score well under any
    retriever for the wrong reason. Eight is chosen to sit above the length of
    an ordinary shared clause and below that of a copied sentence.
    """
    question_words = _words(question)
    snippet_words = _words(snippet)
    if len(question_words) < LEAK_RUN_WORDS or len(snippet_words) < LEAK_RUN_WORDS:
        return False
    runs = {
        tuple(snippet_words[i : i + LEAK_RUN_WORDS])
        for i in range(len(snippet_words) - LEAK_RUN_WORDS + 1)
    }
    return any(
        tuple(question_words[i : i + LEAK_RUN_WORDS]) in runs
        for i in range(len(question_words) - LEAK_RUN_WORDS + 1)
    )


def contains_exact_term(term: str, chunk_content: str) -> bool:
    """Whether an exact-term question's term is literally present in its chunk.

    Without this the exact-term stratum quietly becomes a second paraphrase
    stratum — the model invents a plausible-sounding identifier, the question
    reads fine, and the slice stops measuring the lexical-retrieval gap it
    exists to expose (decisions E2 and E5).
    """
    return normalize_text(term).casefold() in normalize_text(chunk_content).casefold()


# ─── Deduplication ───


def shingles(question: str, size: int = SHINGLE_SIZE) -> set[tuple[str, ...]]:
    words = [word for word in _words(question) if word not in STOPWORDS]
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def deduplicate(
    questions: Sequence[tuple[str, str]],
    threshold: float = DUPLICATE_JACCARD,
) -> tuple[list[str], dict[str, str]]:
    """Reduce near-duplicate questions to one each. Input is ``(qid, question)``.

    Returns the surviving qids and a ``{dropped_qid: kept_qid}`` map so the
    rejected file can say what each casualty duplicated.

    **The survivor is the lexicographically smaller qid, never "the first one
    seen".** Order-dependence here would mean that re-running the gate over a
    re-ordered silver file yields a different dataset, which quietly destroys
    the resumability the rest of this package is built on — the file would stop
    being a function of its inputs. qids are content-derived, so this rule is
    stable across machines and runs.

    Shingle overlap catches the common case (the same question generated twice
    from two overlapping chunks) for free and deterministically. It will not
    catch a true paraphrase pair such as "What is NFKC?" / "Define NFKC";
    embedding-cosine dedup is the complement and belongs in the caller, which
    has an EmbeddingService and this module deliberately does not.
    """
    ordered = sorted(questions, key=lambda pair: pair[0])
    kept: list[tuple[str, set[tuple[str, ...]]]] = []
    survivors: list[str] = []
    duplicates: dict[str, str] = {}

    for qid, question in ordered:
        fingerprint = shingles(question)
        clash = next(
            (other_qid for other_qid, other in kept if jaccard(fingerprint, other) >= threshold),
            None,
        )
        if clash is not None:
            duplicates[qid] = clash
            continue
        kept.append((qid, fingerprint))
        survivors.append(qid)

    return survivors, duplicates
