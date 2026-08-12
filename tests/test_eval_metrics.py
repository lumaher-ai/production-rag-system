"""Recall, and the unanswerable-denominator rule that is easy to get wrong."""

from production_rag.eval.metrics import (
    all_gold_at_k,
    false_abstention_rate,
    mean_recall_at_k,
    recall_at_k,
    refusal_rate,
)

from ._eval import make_gold, make_record

SINGLE = make_record(gold=[make_gold(chunk_index=3)])
MULTI = make_record(
    query_type="multi_hop",
    question="Does the batch timeout outlast the job timeout that would kill it?",
    gold=[make_gold(chunk_index=3), make_gold(chunk_index=31)],
)


def test_a_retrieved_gold_chunk_scores_one():
    assert recall_at_k(SINGLE, [("doc-a.md", 3), ("doc-a.md", 9)], k=5) == 1.0


def test_a_missed_gold_chunk_scores_zero():
    assert recall_at_k(SINGLE, [("doc-a.md", 9)], k=5) == 0.0


def test_a_gold_chunk_below_the_cutoff_does_not_count():
    retrieved = [("doc-a.md", 9), ("doc-a.md", 10), ("doc-a.md", 3)]
    assert recall_at_k(SINGLE, retrieved, k=2) == 0.0
    assert recall_at_k(SINGLE, retrieved, k=3) == 1.0


def test_recall_at_k_gives_partial_credit_for_a_two_gold_question():
    assert recall_at_k(MULTI, [("doc-a.md", 3)], k=5) == 0.5


def test_all_gold_at_k_is_zero_unless_every_primary_chunk_is_retrieved():
    assert all_gold_at_k(MULTI, [("doc-a.md", 3)], k=5) == 0.0
    assert all_gold_at_k(MULTI, [("doc-a.md", 3), ("doc-a.md", 31)], k=5) == 1.0


def test_all_gold_and_recall_agree_for_a_single_gold_question():
    for retrieved in ([("doc-a.md", 3)], [("doc-a.md", 9)]):
        assert recall_at_k(SINGLE, retrieved, k=5) == all_gold_at_k(SINGLE, retrieved, k=5)


def test_a_hit_on_an_overlap_chunk_counts_as_a_hit():
    # The overlap chunk holds the same text at a different index — penalising
    # the retriever for returning it would measure CHUNK_OVERLAP, not retrieval.
    record = make_record(
        gold=[make_gold(chunk_index=3), make_gold(chunk_index=4, role="overlap")]
    )
    assert recall_at_k(record, [("doc-a.md", 4)], k=5) == 1.0


def test_an_overlap_chunk_carrying_a_different_snippet_does_not_satisfy_the_primary():
    record = make_record(
        gold=[
            make_gold(chunk_index=3, snippet="the quick brown fox"),
            make_gold(chunk_index=4, snippet="an entirely different span", role="overlap"),
        ]
    )
    assert recall_at_k(record, [("doc-a.md", 4)], k=5) == 0.0


def test_an_unanswerable_record_has_no_recall_rather_than_a_recall_of_zero():
    negative = make_record(query_type="unanswerable")
    assert recall_at_k(negative, [("doc-a.md", 3)], k=5) is None
    assert all_gold_at_k(negative, [], k=5) is None


def test_unanswerable_items_are_excluded_from_the_recall_denominator():
    records = [SINGLE, make_record(query_type="unanswerable")]
    retrieved = {SINGLE.qid: [("doc-a.md", 3)]}
    # Averaging a 0.0 in for the negative would report 0.5 and make every
    # retrieval number in the report look ~20% worse than it is.
    assert mean_recall_at_k(records, retrieved, k=5) == 1.0


def test_recall_over_an_all_unanswerable_slice_is_undefined_not_zero():
    assert mean_recall_at_k([make_record(query_type="unanswerable")], {}, k=5) is None


def test_a_question_with_no_retrieval_recorded_scores_zero_rather_than_being_skipped():
    assert mean_recall_at_k([SINGLE], {}, k=5) == 0.0


def test_refusal_rate_counts_only_the_unanswerable_slice():
    negative = make_record(query_type="unanswerable")
    records = [SINGLE, negative]
    assert refusal_rate(records, {negative.qid: True, SINGLE.qid: True}) == 1.0
    assert refusal_rate(records, {negative.qid: False}) == 0.0


def test_false_abstention_rate_counts_only_the_answerable_slice():
    negative = make_record(query_type="unanswerable")
    records = [SINGLE, negative]
    assert false_abstention_rate(records, {SINGLE.qid: True, negative.qid: True}) == 1.0


def test_refusal_rate_is_undefined_without_any_negatives():
    assert refusal_rate([SINGLE], {}) is None
