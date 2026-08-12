"""Reading JSON back from a model that was only asked nicely to produce it."""

import pytest

from production_rag.eval.parsing import (
    MalformedGenerationError,
    extract_items,
    extract_json_object,
)


def test_bare_json_is_parsed():
    assert extract_json_object('{"items": []}') == {"items": []}


def test_fenced_json_is_parsed_when_the_model_wraps_it_in_prose():
    reply = (
        'Sure! Here you go:\n\n```json\n{"items": [{"question": "q?"}]}\n```\n\nHope so.'
    )
    assert extract_items(reply) == [{"question": "q?"}]


def test_unfenced_json_after_prose_is_parsed():
    assert extract_items('Here it is: {"items": [{"question": "q?"}]}') == [{"question": "q?"}]


def test_json_containing_braces_inside_a_snippet_is_parsed():
    # A regex cannot balance braces, and this corpus is full of them.
    reply = '{"items": [{"snippet": "default={} and cfg = {\\"a\\": 1}", "question": "q?"}]}'
    items = extract_items(reply)
    assert items[0]["snippet"] == 'default={} and cfg = {"a": 1}'


def test_a_trailing_comma_does_not_lose_the_whole_batch():
    assert extract_items('{"items": [{"question": "q?"},]}') == [{"question": "q?"}]


def test_a_bare_list_is_accepted_rather_than_costing_a_repair_round_trip():
    assert extract_items('[{"question": "q?"}]') == [{"question": "q?"}]


def test_a_single_unwrapped_item_is_accepted():
    assert extract_items('{"question": "q?", "answer": "a"}') == [
        {"question": "q?", "answer": "a"}
    ]


def test_non_dict_entries_in_items_are_discarded_rather_than_crashing_the_run():
    assert extract_items('{"items": [{"question": "q?"}, "oops", null]}') == [{"question": "q?"}]


def test_an_empty_response_raises_rather_than_returning_empty():
    with pytest.raises(MalformedGenerationError):
        extract_json_object("   ")


def test_an_unparseable_response_raises_rather_than_returning_empty():
    with pytest.raises(MalformedGenerationError):
        extract_json_object("I'm sorry, I can't help with that.")


def test_items_of_the_wrong_type_raise_rather_than_being_silently_ignored():
    with pytest.raises(MalformedGenerationError):
        extract_items('{"items": "not a list"}')
