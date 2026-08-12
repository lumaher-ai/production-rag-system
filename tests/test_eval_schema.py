"""The record shape, and the identity decision baked into it."""

import json

import pytest

from production_rag.eval.schema import (
    SCHEMA_VERSION,
    RejectedRecord,
    append_jsonl,
    make_qid,
    read_jsonl,
    read_unit_ids,
    write_jsonl,
)

from ._eval import make_gold, make_record


def test_a_record_round_trips_through_jsonl_unchanged(tmp_path):
    original = make_record()
    path = tmp_path / "d.jsonl"
    write_jsonl(path, [original])
    assert list(read_jsonl(path)) == [original]


def test_a_record_with_an_unknown_schema_version_is_rejected_loudly(tmp_path):
    path = tmp_path / "d.jsonl"
    payload = make_record().model_dump(mode="json")
    payload["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        list(read_jsonl(path))


def test_a_malformed_line_names_its_line_number_rather_than_being_skipped(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text('{"ok": 1}\nnot json at all\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r":1"):
        list(read_jsonl(path))


def test_a_record_never_stores_a_chunk_uuid():
    # Pins the whole identity decision: chunk ids are uuid4 and are regenerated
    # on every re-ingest, so storing one would make the dataset expire silently.
    serialized = json.dumps(make_record().model_dump(mode="json"))
    assert "chunk_id" not in serialized


def test_the_qid_is_stable_across_calls():
    keys = [("doc-a.md", 3)]
    assert make_qid("paraphrase", "What is NFKC?", keys) == make_qid(
        "paraphrase", "What is NFKC?", keys
    )


def test_the_qid_ignores_case_and_surrounding_whitespace():
    keys = [("doc-a.md", 3)]
    assert make_qid("paraphrase", "  What is NFKC?  ", keys) == make_qid(
        "paraphrase", "what is nfkc?", keys
    )


def test_the_qid_changes_when_the_question_changes():
    keys = [("doc-a.md", 3)]
    assert make_qid("paraphrase", "What is NFKC?", keys) != make_qid(
        "paraphrase", "What is NFKD?", keys
    )


def test_the_qid_changes_when_the_gold_chunk_changes():
    assert make_qid("paraphrase", "q?", [("a.md", 1)]) != make_qid(
        "paraphrase", "q?", [("a.md", 2)]
    )


def test_the_qid_is_independent_of_gold_chunk_order():
    forward = make_qid("multi_hop", "q?", [("a.md", 1), ("a.md", 9)])
    backward = make_qid("multi_hop", "q?", [("a.md", 9), ("a.md", 1)])
    assert forward == backward


def test_the_qid_carries_a_prefix_naming_its_stratum():
    assert make_qid("unanswerable", "q?", []).startswith("un_")
    assert make_qid("multi_hop", "q?", []).startswith("mh_")


def test_an_unanswerable_record_serializes_an_empty_gold_list_not_null():
    payload = make_record(query_type="unanswerable").model_dump(mode="json")
    assert payload["gold"] == []
    assert payload["answerable"] is False
    assert payload["answer"] is None


def test_a_multi_hop_record_reports_both_of_its_primary_gold_keys():
    record = make_record(
        query_type="multi_hop",
        gold=[make_gold(chunk_index=1), make_gold(chunk_index=9)],
    )
    assert record.primary_gold_keys == [("doc-a.md", 1), ("doc-a.md", 9)]


def test_overlap_entries_are_excluded_from_the_primary_keys_but_not_from_all_keys():
    record = make_record(
        gold=[make_gold(chunk_index=1), make_gold(chunk_index=2, role="overlap")]
    )
    assert record.primary_gold_keys == [("doc-a.md", 1)]
    assert record.all_gold_keys == [("doc-a.md", 1), ("doc-a.md", 2)]


def test_appending_leaves_a_file_readable_after_every_record(tmp_path):
    path = tmp_path / "d.jsonl"
    append_jsonl(path, make_record(question="First question about the fox?"))
    assert len(list(read_jsonl(path))) == 1
    append_jsonl(path, make_record(question="Second question about the dog?"))
    assert len(list(read_jsonl(path))) == 2


def test_reading_a_missing_file_yields_nothing_rather_than_raising(tmp_path):
    assert list(read_jsonl(tmp_path / "absent.jsonl")) == []


def test_unit_ids_are_collected_from_both_the_silver_and_rejected_files(tmp_path):
    silver = tmp_path / "silver.jsonl"
    rejected = tmp_path / "rejected.jsonl"
    record = make_record()
    record.generation.unit_id = "u_kept"
    append_jsonl(silver, record)
    append_jsonl(rejected, RejectedRecord(drop_reason="snippet_not_verbatim", unit_id="u_gone"))
    # Missing the rejected file would retry every gated-out unit on every run.
    assert read_unit_ids(silver, rejected) == {"u_kept", "u_gone"}
