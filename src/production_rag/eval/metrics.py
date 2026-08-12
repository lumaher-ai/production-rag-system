"""Recall@k — and nothing else, on purpose.

This file is a deliberate down-payment on decision G2, not an attempt at it.
``baseline.py`` genuinely cannot run without a recall function, and G1's Proof
clause requires the baseline; nDCG, MRR and Precision@5 are not needed to show
that the dataset discriminates and belong in G2, where they can be designed
against real measurements rather than guessed at now.

One rule here is easy to get catastrophically wrong, so it is enforced rather
than documented: **unanswerable records are excluded from the denominator.**
With ``gold == []`` recall is ``0/0``, not ``0``. Averaging in a zero for every
negative would make every retrieval number in the report look about twenty per
cent worse than it is — a bug that produces plausible output and would survive
review indefinitely.
"""

from collections.abc import Sequence

from production_rag.eval.schema import EvalRecord

GoldKey = tuple[str, int]


def recall_at_k(record: EvalRecord, retrieved: Sequence[GoldKey], k: int) -> float | None:
    """Fraction of this question's gold chunks found in the top ``k``.

    ``None`` for an unanswerable record — it has no gold, so it has no recall,
    and a caller must skip it rather than score it. Returning ``None`` instead
    of ``0.0`` is what makes that impossible to get wrong silently.

    Partial credit, so a two-gold multi-hop question retrieving one chunk scores
    0.5. A hit on an ``overlap`` gold entry counts: that chunk holds the same
    text at a different index, and penalising the retriever for returning it
    would be measuring the chunker's overlap setting, not retrieval.
    """
    if not record.answerable or not record.gold:
        return None
    top = set(retrieved[:k])
    wanted = set(record.all_gold_keys)
    primaries = set(record.primary_gold_keys) or wanted
    # Score against primaries, but let an overlap hit satisfy its primary: the
    # two entries name the same text.
    satisfied = sum(
        1
        for primary in primaries
        if primary in top or _overlap_satisfies(record, primary, top)
    )
    return satisfied / len(primaries)


def _overlap_satisfies(record: EvalRecord, primary: GoldKey, top: set[GoldKey]) -> bool:
    """Whether an overlap chunk carrying the same snippet as ``primary`` was retrieved."""
    snippet = next(
        (
            gold.snippet
            for gold in record.gold
            if (gold.document_key, gold.chunk_index) == primary
        ),
        None,
    )
    if snippet is None:
        return False
    return any(
        gold.role == "overlap"
        and gold.snippet == snippet
        and (gold.document_key, gold.chunk_index) in top
        for gold in record.gold
    )


def all_gold_at_k(record: EvalRecord, retrieved: Sequence[GoldKey], k: int) -> float | None:
    """1.0 only if *every* primary gold chunk is in the top ``k``.

    The strict counterpart to ``recall_at_k``, and the number that actually
    predicts whether generation can succeed: a two-hop answer missing one hop is
    a hallucination waiting to happen, not a half-right answer.

    For single-gold strata this equals ``recall_at_k`` by construction, so
    reporting both costs nothing and the multi-hop row is the only place they
    diverge. That divergence is itself informative and belongs in the report.
    """
    value = recall_at_k(record, retrieved, k)
    if value is None:
        return None
    return 1.0 if value >= 1.0 else 0.0


def mean_recall_at_k(
    records: Sequence[EvalRecord],
    retrieved_by_qid: dict[str, Sequence[GoldKey]],
    k: int,
) -> float | None:
    """Mean Recall@k over the answerable records only.

    ``None`` when no record in the slice is answerable — an all-negative slice
    has an undefined recall, and reporting 0.0 for it would read as "retrieval
    found nothing" rather than "this question does not ask about retrieval".
    """
    scores = [
        score
        for record in records
        if (score := recall_at_k(record, retrieved_by_qid.get(record.qid, []), k)) is not None
    ]
    return sum(scores) / len(scores) if scores else None


def refusal_rate(records: Sequence[EvalRecord], refused_by_qid: dict[str, bool]) -> float | None:
    """Share of unanswerable questions the system correctly declined to answer.

    The other half of the instrument. Paired with false-abstention rate over the
    answerable slice it is exactly E6's Proof clause — *"Abstention rate, and
    false-abstention rate on questions known to be answerable. Both matter — a
    system that refuses everything scores perfectly on faithfulness."*
    """
    negatives = [record for record in records if not record.answerable]
    if not negatives:
        return None
    return sum(1 for record in negatives if refused_by_qid.get(record.qid, False)) / len(negatives)


def false_abstention_rate(
    records: Sequence[EvalRecord], refused_by_qid: dict[str, bool]
) -> float | None:
    """Share of *answerable* questions the system wrongly refused."""
    positives = [record for record in records if record.answerable]
    if not positives:
        return None
    return sum(1 for record in positives if refused_by_qid.get(record.qid, False)) / len(positives)
