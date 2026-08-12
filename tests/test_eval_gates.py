"""The gates decide what a human ever sees, so they get the most tests."""

from production_rag.eval.gates import (
    contains_exact_term,
    deduplicate,
    find_overlap_chunks,
    has_valid_length,
    is_self_referential,
    jaccard,
    leaks_answer,
    recover_snippet,
    shingles,
    verify_snippet,
)

CHUNK = (
    "Iterative scan re-scans until the limit is satisfied. `ef_search` widens each\n"
    "pass, because pgvector's default of 40 is tuned for unfiltered search.\n\n"
    "SET LOCAL scopes both to the surrounding transaction."
)


def test_a_snippet_copied_verbatim_is_accepted():
    assert verify_snippet("Iterative scan re-scans until the limit is satisfied.", CHUNK)


def test_a_snippet_not_present_in_its_chunk_is_rejected():
    assert not verify_snippet("Iterative scan doubles the candidate list.", CHUNK)


def test_a_paraphrased_snippet_is_rejected_because_a_citation_is_exact_or_nothing():
    # Every content word appears in the chunk; the sentence does not.
    assert not verify_snippet("The scan re-scans the limit until satisfied.", CHUNK)


def test_a_snippet_whose_whitespace_was_reflowed_is_recovered_rather_than_rejected():
    reflowed = "`ef_search` widens each pass, because pgvector's default of 40 is tuned"
    recovered = recover_snippet(reflowed, CHUNK)
    assert recovered is not None
    # The chunk's own text is returned, newline and all — not the model's reflow.
    assert recovered in CHUNK
    assert "\n" in recovered


def test_a_snippet_containing_a_ligature_survives_normalization():
    chunk = "The workflow is defined by the office."
    assert verify_snippet("deﬁned by the oﬃce", chunk)


def test_a_snippet_separated_by_a_non_breaking_space_is_accepted():
    assert verify_snippet("Iterative scan re-scans", CHUNK)


def test_an_empty_snippet_is_rejected():
    assert not verify_snippet("   ", CHUNK)


def test_a_snippet_present_in_a_neighbouring_chunk_is_reported_as_overlapping():
    snippet = "SET LOCAL scopes both"
    neighbours = [(4, "…prelude. SET LOCAL scopes both to the transaction."), (6, "unrelated")]
    assert find_overlap_chunks(snippet, neighbours) == [4]


def test_a_snippet_absent_from_both_neighbours_reports_no_overlap():
    assert find_overlap_chunks("SET LOCAL scopes both", [(4, "unrelated"), (6, "also no")]) == []


def test_a_question_that_refers_to_the_document_itself_is_rejected():
    assert is_self_referential("According to the document, what does ef_search do?")
    assert is_self_referential("What does this passage say about iterative scan?")
    assert is_self_referential("¿Qué dice el texto anterior sobre el escaneo?")


def test_a_standalone_question_is_not_self_referential():
    assert not is_self_referential("What does hnsw.ef_search control in pgvector?")


def test_a_question_shorter_than_the_minimum_is_rejected():
    assert not has_valid_length("Why?")
    assert has_valid_length("What does ef_search control?")
    assert not has_valid_length("x" * 301)


def test_a_question_quoting_eight_words_of_its_own_answer_is_rejected():
    snippet = "Iterative scan re-scans until the limit is satisfied by pgvector"
    question = "Is it true that iterative scan re-scans until the limit is satisfied by pgvector?"
    assert leaks_answer(question, snippet)


def test_a_question_sharing_only_a_short_phrase_does_not_count_as_leakage():
    assert not leaks_answer("What does iterative scan do?", CHUNK)


def test_an_exact_term_absent_from_its_chunk_is_detected():
    assert contains_exact_term("ef_search", CHUNK)
    assert not contains_exact_term("hnsw.m", CHUNK)


def test_two_questions_with_the_same_content_words_are_deduplicated():
    pairs = [
        ("pa_aaaaaaaaaaaa", "What does ef_search control in the vector index?"),
        ("pa_bbbbbbbbbbbb", "What does ef_search control in a vector index?"),
        ("pa_cccccccccccc", "Which transaction scope does SET LOCAL apply?"),
    ]
    survivors, duplicates = deduplicate(pairs)
    assert len(survivors) == 2
    assert duplicates == {"pa_bbbbbbbbbbbb": "pa_aaaaaaaaaaaa"}


def test_dedup_keeps_the_same_survivor_regardless_of_input_order():
    pairs = [
        ("pa_bbbbbbbbbbbb", "What does ef_search control in a vector index?"),
        ("pa_aaaaaaaaaaaa", "What does ef_search control in the vector index?"),
    ]
    forward, _ = deduplicate(pairs)
    backward, _ = deduplicate(list(reversed(pairs)))
    # Order-dependence here would make the whole silver file stop being a
    # function of its inputs, which is what resumability rests on.
    assert forward == backward == ["pa_aaaaaaaaaaaa"]


def test_distinct_questions_all_survive_deduplication():
    pairs = [
        ("pa_aaaaaaaaaaaa", "What does ef_search control in the vector index?"),
        ("pa_bbbbbbbbbbbb", "Which transaction scope does SET LOCAL apply to?"),
        ("pa_cccccccccccc", "How many pages does the batch processor accept?"),
    ]
    survivors, duplicates = deduplicate(pairs)
    assert len(survivors) == 3
    assert duplicates == {}


def test_jaccard_of_an_empty_shingle_set_is_zero_rather_than_a_division_error():
    assert jaccard(set(), shingles("What does ef_search control?")) == 0.0
