"""Choosing what to generate from. Pure, seeded, and deliberately not exhaustive.

150 questions at ~3.5 usable per chunk after gating needs roughly 60 chunks, not
the whole corpus. So this samples. The two rules below are both there to stop
the sample from being an accident of the corpus's shape rather than a view of
it.
"""

import random
from collections.abc import Sequence
from dataclasses import dataclass

# A prose chunk at CHUNK_SIZE=1000 characters runs 200-260 tokens. Anything much
# under that is a heading, a table row, a run of links, or a code fence — text
# that produces questions about formatting rather than about content.
MIN_CHUNK_TOKENS = 60
# A guard against a chunk that is one enormous unbroken code block: the model
# will mine it for identifiers and the whole exact-term slice ends up about
# variable names in a single listing.
MAX_CHUNK_TOKENS = 400
# Chunks i and i+1 literally share CHUNK_OVERLAP=200 characters of text, so a
# "multi-hop" question over neighbours is usually answerable from one of them.
# Three indices apart puts ~2400 characters between the two spans.
MIN_MULTI_HOP_GAP = 3


@dataclass(frozen=True, slots=True)
class ChunkRef:
    """The identity and size of a chunk, without its text or its vector."""

    document_key: str
    chunk_index: int
    token_count: int
    document_title: str = ""


def eligible(
    chunks: Sequence[ChunkRef],
    min_tokens: int = MIN_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[ChunkRef]:
    """Chunks worth spending a generation call on, in stable order."""
    return sorted(
        (chunk for chunk in chunks if min_tokens <= chunk.token_count <= max_tokens),
        key=lambda chunk: (chunk.document_key, chunk.chunk_index),
    )


def sample_chunks(
    chunks: Sequence[ChunkRef],
    target: int,
    seed: int,
    min_tokens: int = MIN_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[ChunkRef]:
    """Pick ``target`` chunks without letting one long document win.

    Documents are visited round-robin in sorted order, each contributing the
    next chunk from its own seeded shuffle. A document with 110 chunks and one
    with 12 therefore contribute at comparable rates rather than in proportion
    to their length. That matters concretely here: the decisions document is
    roughly six times the README, and proportional sampling would hand it six
    times the questions, turning a corpus-wide dataset into a single-document
    one whose Recall numbers describe one writing style.

    Deterministic for a given seed — ``random.Random(seed)``, never the global
    RNG, over explicitly sorted input. Re-running with the same seed samples the
    same chunks, which is what lets an interrupted run resume rather than
    diverge. (Note the honest limit of that guarantee: the *sample* is
    reproducible; the model's replies are not, even at temperature 0.)

    Returns fewer than ``target`` rather than repeating a chunk when the corpus
    is too small — a duplicated chunk would produce duplicate questions that the
    dedup gate then deletes, which is a slow way to buy nothing.
    """
    pool = eligible(chunks, min_tokens=min_tokens, max_tokens=max_tokens)
    if target <= 0 or not pool:
        return []

    rng = random.Random(seed)
    by_document: dict[str, list[ChunkRef]] = {}
    for chunk in pool:
        by_document.setdefault(chunk.document_key, []).append(chunk)
    for chunk_list in by_document.values():
        rng.shuffle(chunk_list)

    sources = sorted(by_document)
    chosen: list[ChunkRef] = []
    while len(chosen) < target:
        progressed = False
        for source in sources:
            if not by_document[source]:
                continue
            chosen.append(by_document[source].pop())
            progressed = True
            if len(chosen) >= target:
                break
        if not progressed:
            break

    return chosen


def sample_pairs(
    chunks: Sequence[ChunkRef],
    target: int,
    seed: int,
    min_gap: int = MIN_MULTI_HOP_GAP,
    min_tokens: int = MIN_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
) -> list[tuple[ChunkRef, ChunkRef]]:
    """Pick non-adjacent same-document chunk pairs for multi-hop generation.

    ``min_gap`` is the whole design. Without it the generator is handed two
    chunks that share 200 characters and asked for a question needing both; it
    obliges, and the result is a single-hop question labelled multi-hop, which
    scores as a retrieval failure when the system behaves correctly. Fake
    multi-hops are worse than no multi-hops.

    Round-robin across documents for the same reason as ``sample_chunks``, and
    no chunk is used in more than one pair — reusing one would correlate two
    supposedly independent questions and make the stratum's variance look
    smaller than it is.
    """
    pool = eligible(chunks, min_tokens=min_tokens, max_tokens=max_tokens)
    if target <= 0 or not pool:
        return []

    rng = random.Random(seed)
    by_document: dict[str, list[ChunkRef]] = {}
    for chunk in pool:
        by_document.setdefault(chunk.document_key, []).append(chunk)

    candidates: dict[str, list[tuple[ChunkRef, ChunkRef]]] = {}
    for source, chunk_list in by_document.items():
        pairs = [
            (left, right)
            for i, left in enumerate(chunk_list)
            for right in chunk_list[i + 1 :]
            if abs(left.chunk_index - right.chunk_index) >= min_gap
        ]
        rng.shuffle(pairs)
        candidates[source] = pairs

    sources = sorted(candidates)
    chosen: list[tuple[ChunkRef, ChunkRef]] = []
    used: set[tuple[str, int]] = set()
    while len(chosen) < target:
        progressed = False
        for source in sources:
            while candidates[source]:
                left, right = candidates[source].pop()
                left_key = (left.document_key, left.chunk_index)
                right_key = (right.document_key, right.chunk_index)
                if left_key in used or right_key in used:
                    continue
                used.add(left_key)
                used.add(right_key)
                chosen.append((left, right))
                progressed = True
                break
            if len(chosen) >= target:
                break
        if not progressed:
            break

    return chosen
