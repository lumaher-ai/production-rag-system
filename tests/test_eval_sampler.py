"""The sampler decides what the dataset is about, so its biases are its output."""

from production_rag.eval.sampling import MIN_MULTI_HOP_GAP, eligible, sample_chunks, sample_pairs

from ._eval import make_ref


def corpus(long_doc: int = 100, short_doc: int = 10) -> list:
    return [
        *(make_ref("long.md", index) for index in range(long_doc)),
        *(make_ref("short.md", index) for index in range(short_doc)),
    ]


def test_the_sampler_returns_the_same_chunks_for_the_same_seed():
    first = sample_chunks(corpus(), target=20, seed=7)
    second = sample_chunks(corpus(), target=20, seed=7)
    assert first == second


def test_a_different_seed_selects_a_different_sample():
    assert sample_chunks(corpus(), target=20, seed=7) != sample_chunks(corpus(), target=20, seed=8)


def test_one_long_document_does_not_dominate_the_sample():
    chosen = sample_chunks(corpus(long_doc=100, short_doc=10), target=20, seed=7)
    from_short = sum(1 for ref in chosen if ref.document_key == "short.md")
    # Proportional sampling would give the 10-chunk document about 2 of 20.
    # Round-robin gives it half until it runs out.
    assert from_short == 10


def test_chunks_below_the_token_floor_are_skipped():
    refs = [make_ref("a.md", 0, token_count=5), make_ref("a.md", 1, token_count=200)]
    assert [ref.chunk_index for ref in eligible(refs)] == [1]


def test_chunks_above_the_token_ceiling_are_skipped():
    refs = [make_ref("a.md", 0, token_count=9000), make_ref("a.md", 1, token_count=200)]
    assert [ref.chunk_index for ref in eligible(refs)] == [1]


def test_the_sampler_returns_fewer_than_requested_rather_than_repeating_a_chunk():
    chosen = sample_chunks(corpus(long_doc=3, short_doc=2), target=50, seed=7)
    assert len(chosen) == 5
    assert len({(ref.document_key, ref.chunk_index) for ref in chosen}) == 5


def test_an_empty_corpus_samples_nothing_rather_than_raising():
    assert sample_chunks([], target=10, seed=7) == []
    assert sample_pairs([], target=10, seed=7) == []


def test_multi_hop_pairs_are_never_adjacent_within_a_document():
    pairs = sample_pairs(corpus(), target=25, seed=7)
    assert pairs
    for left, right in pairs:
        assert abs(left.chunk_index - right.chunk_index) >= MIN_MULTI_HOP_GAP


def test_multi_hop_pairs_never_reuse_a_chunk():
    pairs = sample_pairs(corpus(), target=25, seed=7)
    used = [(ref.document_key, ref.chunk_index) for pair in pairs for ref in pair]
    assert len(used) == len(set(used))


def test_multi_hop_pairs_come_from_a_single_document_each():
    for left, right in sample_pairs(corpus(), target=25, seed=7):
        assert left.document_key == right.document_key


def test_multi_hop_pairing_is_deterministic_for_a_given_seed():
    assert sample_pairs(corpus(), target=15, seed=3) == sample_pairs(corpus(), target=15, seed=3)


def test_a_document_too_short_to_hold_a_gap_yields_no_pairs():
    refs = [make_ref("tiny.md", index) for index in range(3)]
    assert sample_pairs(refs, target=5, seed=7, min_gap=5) == []
