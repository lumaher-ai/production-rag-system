"""The human half: rendering a sheet, reading verdicts, and merging them."""

from uuid import uuid4

import pytest

from production_rag.eval.audit import (
    AuditError,
    acceptance_summary,
    apply_verdicts,
    parse_sheet,
    render_sheet,
    select_for_audit,
    validate_verdicts,
)
from production_rag.eval.corpus import CorpusIndex
from production_rag.models.document import DocumentChunk

from ._eval import make_gold, make_record

DOCUMENT_ID = uuid4()
CHUNK_TEXT = "Iterative scan re-scans until the limit is satisfied."


def build_index() -> CorpusIndex:
    chunks = [
        DocumentChunk(
            document_id=DOCUMENT_ID,
            owner_id=uuid4(),
            document_title="doc-a",
            chunk_index=index,
            content=f"{CHUNK_TEXT} (chunk {index})",
            token_count=200,
            embedding=[0.0],
        )
        for index in range(3)
    ]
    return CorpusIndex(chunks, {DOCUMENT_ID: "doc-a.md"})


def a_verdict(qid: str, decision: str, **extra: str) -> dict[str, str]:
    return {
        "qid": qid,
        "decision": decision,
        "reason": extra.get("reason", ""),
        "question": extra.get("question", ""),
        "answer": extra.get("answer", ""),
    }


# ─── Selection ───


def test_the_review_sheet_is_stratified_across_the_four_query_types():
    templates = {
        "paraphrase": "Paraphrase question number {} about foxes?",
        "exact_term": "What does the setting ef_search_{} control here?",
        "multi_hop": "Multi hop question number {} spanning two chunks?",
        "unanswerable": "Unanswerable question number {} about pricing?",
    }
    records = [
        make_record(query_type, template.format(i))
        for query_type, template in templates.items()
        for i in range(30)
    ]
    selected = select_for_audit(records, seed=1)
    counts: dict[str, int] = {}
    for record in selected:
        counts[record.query_type] = counts.get(record.query_type, 0) + 1
    assert counts == {"paraphrase": 12, "exact_term": 10, "multi_hop": 15, "unanswerable": 13}
    assert len(selected) == 50


def test_every_flagged_item_appears_in_the_review_sheet():
    flagged = [
        make_record(
            "paraphrase",
            f"Flagged question number {i} about the fox?",
            warnings=["snippet_length"],
        )
        for i in range(12)
    ]
    clean = [
        make_record("paraphrase", f"Clean question number {i} about the dog?")
        for i in range(40)
    ]
    selected = select_for_audit(clean + flagged, seed=1, quotas={"paraphrase": 12})
    assert {record.qid for record in selected} == {record.qid for record in flagged}


def test_selection_is_reproducible_for_a_given_seed():
    records = [
        make_record("paraphrase", f"Question number {i} about the brown fox?")
        for i in range(40)
    ]
    first = select_for_audit(records, seed=5, quotas={"paraphrase": 10})
    second = select_for_audit(records, seed=5, quotas={"paraphrase": 10})
    assert [r.qid for r in first] == [r.qid for r in second]


# ─── Rendering ───


def test_the_review_sheet_contains_the_full_gold_chunk_text():
    record = make_record(gold=[make_gold(chunk_index=1)])
    sheet = render_sheet([record], build_index(), "run1")
    assert "(chunk 1)" in sheet
    assert record.question in sheet
    assert "```verdict" in sheet
    assert f"qid: {record.qid}" in sheet


def test_the_sheet_warns_when_a_gold_chunk_no_longer_matches_what_was_generated_against():
    # content_sha256 is computed from the snippet by the builder, so it will not
    # match the chunk text in the index — exactly the drift the tripwire catches.
    sheet = render_sheet([make_record(gold=[make_gold(chunk_index=1)])], build_index(), "run1")
    assert "has CHANGED since generation" in sheet


def test_the_sheet_warns_when_a_gold_chunk_has_vanished_from_the_corpus():
    sheet = render_sheet([make_record(gold=[make_gold(chunk_index=99)])], build_index(), "run1")
    assert "no longer exists in the corpus" in sheet


# ─── Parsing ───


def test_verdicts_round_trip_from_the_rendered_sheet():
    record = make_record()
    sheet = render_sheet([record], build_index(), "run1")
    filled = sheet.replace("decision:", "decision: accept")
    verdicts = parse_sheet(filled)
    assert len(verdicts) == 1
    assert verdicts[0]["qid"] == record.qid
    assert verdicts[0]["decision"] == "accept"


def test_a_multi_line_replacement_answer_is_joined_rather_than_truncated():
    block = "```verdict\nqid: pa_x\ndecision: edit\nanswer: first line\n  second line\n```"
    assert parse_sheet(block)[0]["answer"] == "first line second line"


def test_an_unknown_verdict_key_fails_the_merge_loudly():
    block = "```verdict\nqid: pa_x\ndecision: accept\nverdct: typo\n```"
    with pytest.raises(AuditError, match="Unknown key"):
        parse_sheet(block)


def test_a_verdict_block_with_no_decision_fails_validation():
    with pytest.raises(AuditError, match="expected one of"):
        validate_verdicts([a_verdict("pa_x", "")], {"pa_x"})


def test_a_reject_with_no_reason_fails_validation():
    with pytest.raises(AuditError, match="needs a reason"):
        validate_verdicts([a_verdict("pa_x", "reject")], {"pa_x"})


def test_an_edit_with_neither_a_question_nor_an_answer_fails_validation():
    with pytest.raises(AuditError, match="needs a question or an answer"):
        validate_verdicts([a_verdict("pa_x", "edit")], {"pa_x"})


def test_a_verdict_for_an_unknown_record_fails_validation():
    with pytest.raises(AuditError, match="no such record"):
        validate_verdicts([a_verdict("pa_gone", "accept")], {"pa_x"})


def test_validation_reports_every_problem_at_once_rather_than_the_first():
    with pytest.raises(AuditError) as exc:
        validate_verdicts(
            [a_verdict("pa_x", ""), a_verdict("pa_y", "reject")], {"pa_x", "pa_y"}
        )
    assert "pa_x" in str(exc.value) and "pa_y" in str(exc.value)


# ─── Merging ───


def test_an_accepted_item_is_kept_and_marked():
    record = make_record()
    curated, _ = apply_verdicts([record], [a_verdict(record.qid, "accept")], reviewer="lu")
    assert curated[0].audit.status == "accepted"
    assert curated[0].audit.reviewer == "lu"


def test_a_rejected_item_never_reaches_the_curated_file():
    record = make_record()
    curated, _ = apply_verdicts([record], [a_verdict(record.qid, "reject", reason="ambiguous")])
    assert curated == []


def test_unreviewed_items_are_kept_because_the_audit_samples_a_third_of_the_file():
    reviewed = make_record(question="A reviewed question about the brown fox?")
    untouched = make_record(question="An unreviewed question about the lazy dog?")
    curated, _ = apply_verdicts(
        [reviewed, untouched], [a_verdict(reviewed.qid, "accept")]
    )
    assert {record.qid for record in curated} == {reviewed.qid, untouched.qid}
    assert next(r for r in curated if r.qid == untouched.qid).audit.status == "pending"


def test_an_edit_replaces_the_question_and_keeps_the_original():
    record = make_record()
    curated, _ = apply_verdicts(
        [record],
        [a_verdict(record.qid, "edit", question="What colour is the jumping fox?")],
    )
    assert curated[0].question == "What colour is the jumping fox?"
    assert curated[0].audit.original_question == record.question
    assert curated[0].audit.status == "edited"


def test_an_edited_question_gets_a_new_qid():
    record = make_record()
    curated, _ = apply_verdicts(
        [record], [a_verdict(record.qid, "edit", question="A wholly different question here?")]
    )
    assert curated[0].qid != record.qid


def test_an_edited_answer_keeps_the_qid_because_the_question_is_unchanged():
    record = make_record()
    curated, _ = apply_verdicts(
        [record], [a_verdict(record.qid, "edit", answer="It is a brown fox.")]
    )
    assert curated[0].qid == record.qid
    assert curated[0].audit.original_answer == record.answer


def test_a_verdict_whose_record_no_longer_exists_warns_instead_of_applying_elsewhere():
    record = make_record()
    curated, warnings = apply_verdicts(
        [record], [a_verdict(record.qid, "accept"), a_verdict("pa_regenerated", "reject")]
    )
    assert len(curated) == 1
    assert any("pa_regenerated" in warning for warning in warnings)


def test_applying_an_audit_twice_produces_an_identical_curated_file():
    records = [
        make_record(question="A first question about the brown fox jumping?"),
        make_record(question="A second question about the lazy sleeping dog?"),
    ]
    verdicts = [a_verdict(records[0].qid, "accept")]
    first, _ = apply_verdicts(records, verdicts)
    second, _ = apply_verdicts(records, verdicts)
    assert [r.model_dump(exclude={"audit"}) for r in first] == [
        r.model_dump(exclude={"audit"}) for r in second
    ]


def test_the_summary_reports_an_acceptance_rate_and_declines_to_claim_agreement():
    records = [
        make_record(question=f"Question number {i} about the brown fox jumping?")
        for i in range(4)
    ]
    verdicts = [
        a_verdict(records[0].qid, "accept"),
        a_verdict(records[1].qid, "accept"),
        a_verdict(records[2].qid, "edit", question="Edited question about the fox?"),
        a_verdict(records[3].qid, "reject", reason="answerable elsewhere"),
    ]
    curated, _ = apply_verdicts(records, verdicts)
    summary = acceptance_summary(curated, verdicts)
    assert summary["reviewed"] == 4
    assert summary["acceptance_rate"] == 0.5
    assert summary["usable_rate"] == 0.75
    # One rater cannot measure inter-rater agreement, and reporting a kappa of
    # 1.0 against oneself would be a lie dressed as a measurement.
    assert summary["inter_rater_agreement"] == "not measured (single reviewer)"
