"""Proving the dataset is an instrument.

    "The dataset is the instrument. Its own quality check is inter-rater
    agreement on a sample and confirmation that a trivial baseline does *not*
    score well on it."

A dataset a random retriever scores well on is not measuring retrieval. That
sounds obvious and is the single most common way a home-grown eval set turns out
to be worthless: questions generic enough that any five chunks from the corpus
contain something arguably relevant. The check is cheap, it runs before anyone
spends an afternoon auditing, and its failure is a finding about the *dataset*,
not about the retriever.

Each baseline states its expected score before it runs. A measured number that
matches a prediction made in advance is evidence; one that merely looks low is
a vibe.
"""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from production_rag.eval.metrics import GoldKey, mean_recall_at_k
from production_rag.eval.schema import EvalRecord

# Above this, the dataset is not discriminating and the numbers measured against
# it would be noise wearing a decimal point.
MAX_TRIVIAL_RECALL = 0.10
# Real retrieval must clear the best trivial baseline by this margin. A smaller
# gap means either retrieval is broken or the questions are not answerable in
# practice — and you need to know which BEFORE publishing a metric.
MIN_MARGIN_OVER_TRIVIAL = 0.40


@dataclass(frozen=True, slots=True)
class BaselineResult:
    name: str
    expected: float | None
    measured: float | None
    description: str


def _score(
    records: Sequence[EvalRecord],
    pick: "Sequence[GoldKey] | None",
    per_question: "Mapping[str, Sequence[GoldKey]] | None",
    k: int,
) -> float | None:
    retrieved: dict[str, Sequence[GoldKey]] = (
        dict(per_question)
        if per_question is not None
        else {record.qid: (pick or []) for record in records}
    )
    return mean_recall_at_k(records, retrieved, k)


def random_chunk_baseline(
    records: Sequence[EvalRecord],
    corpus_keys: Sequence[GoldKey],
    k: int,
    seed: int,
) -> BaselineResult:
    """Return k chunks drawn uniformly at random, per question.

    Expected Recall@k is ``k / N``: each of a question's gold chunks has a
    ``k/N`` chance of being among the draw, and recall averages that over the
    gold set. With ~200 chunks and k=10 that is about 0.05.
    """
    rng = random.Random(seed)
    total = len(corpus_keys)
    per_question = {record.qid: rng.sample(list(corpus_keys), min(k, total)) for record in records}
    return BaselineResult(
        name="random-chunk",
        expected=(k / total) if total else None,
        measured=_score(records, None, per_question, k),
        description=f"{k} chunks drawn at random from {total}",
    )


def first_k_baseline(
    records: Sequence[EvalRecord],
    corpus_keys: Sequence[GoldKey],
    k: int,
    seed: int,
) -> BaselineResult:
    """Return the opening chunks of a randomly chosen document.

    Catches a corpus whose answers all cluster at the front of documents — a
    real hazard with structured docs, where the first chunks are abstracts and
    tables of contents that happen to mention everything.
    """
    rng = random.Random(seed)
    documents = sorted({key for key, _ in corpus_keys})
    per_question: dict[str, Sequence[GoldKey]] = {}
    for record in records:
        document = rng.choice(documents) if documents else ""
        per_question[record.qid] = [key for key in sorted(corpus_keys) if key[0] == document][:k]
    return BaselineResult(
        name="first-k",
        expected=None,
        measured=_score(records, None, per_question, k),
        description=f"first {k} chunks of a random document",
    )


def largest_document_baseline(
    records: Sequence[EvalRecord],
    corpus_keys: Sequence[GoldKey],
    k: int,
) -> BaselineResult:
    """Return the first k chunks of the biggest document, for every question.

    This is the check that proves the round-robin sampler worked. A dataset
    accidentally dominated by one long document scores well here, and that is
    precisely the failure ``sampling.sample_chunks`` exists to prevent — so a
    high number is a bug report against the sampler, not against retrieval.
    """
    counts: dict[str, int] = {}
    for key, _ in corpus_keys:
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return BaselineResult("largest-document", None, None, "no corpus")
    biggest = max(sorted(counts), key=lambda key: counts[key])
    pick = [key for key in sorted(corpus_keys) if key[0] == biggest][:k]
    return BaselineResult(
        name="largest-document",
        expected=None,
        measured=_score(records, pick, None, k),
        description=f"first {k} chunks of '{biggest}' ({counts[biggest]} chunks)",
    )


async def real_retrieval_baseline(
    records: Sequence[EvalRecord],
    document_service: object,
    user_id: UUID,
    key_for: "object",
    k: int,
) -> BaselineResult:
    """Actual retrieval, for the comparison the pass criterion needs.

    ``use_cache=False`` so this measures retrieval rather than a warm answer
    cache — the reason ``retrieve()`` exists as a separate, LLM-free method at
    all.
    """
    per_question: dict[str, Sequence[GoldKey]] = {}
    for record in records:
        if not record.answerable:
            continue
        scored = await document_service.retrieve(  # type: ignore[attr-defined]
            question=record.question, user_id=user_id, top_k=k, use_cache=False
        )
        per_question[record.qid] = [
            (key_for(chunk.chunk.document_id) or "?", chunk.chunk.chunk_index)  # type: ignore[operator]
            for chunk in scored
        ]
    return BaselineResult(
        name="real-retrieval",
        expected=None,
        measured=_score(records, None, per_question, k),
        description=f"DocumentService.retrieve(top_k={k}, use_cache=False)",
    )


def verdict(trivial: Sequence[BaselineResult], real: BaselineResult | None) -> list[str]:
    """The pass criterion, stated as assertions rather than left to the reader."""
    problems: list[str] = []
    best_trivial = max(
        (result.measured for result in trivial if result.measured is not None), default=None
    )
    if best_trivial is None:
        return ["no trivial baseline produced a score — the dataset has no answerable records"]

    if best_trivial > MAX_TRIVIAL_RECALL:
        problems.append(
            f"a trivial baseline scored Recall@k = {best_trivial:.3f}, above the "
            f"{MAX_TRIVIAL_RECALL:.2f} ceiling. The questions are too generic or the "
            f"corpus is too small: this dataset does not discriminate, and any metric "
            f"measured against it is noise."
        )
    if real is not None and real.measured is not None:
        margin = real.measured - best_trivial
        if margin < MIN_MARGIN_OVER_TRIVIAL:
            problems.append(
                f"real retrieval beat the best trivial baseline by only {margin:.3f} "
                f"(need {MIN_MARGIN_OVER_TRIVIAL:.2f}). Either retrieval is broken or the "
                f"questions are not answerable in practice — find out which before "
                f"publishing a number."
            )
    return problems
