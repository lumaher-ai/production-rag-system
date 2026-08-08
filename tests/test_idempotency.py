from uuid import uuid4

from production_rag.ingestion.idempotency import (
    canonicalize_filters,
    content_hash,
    query_idempotency_key,
)

_BASE = {
    "user_id": uuid4(),
    "question": "What is Python?",
    "filters": None,
    "top_k": 5,
    "normalizer_version": "nfkc-ws-v1",
    "chunker_version": "recursive-char-v1",
    "embedding_model": "text-embedding-3-small",
}


def test_content_hash_is_deterministic_and_sensitive() -> None:
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("hello!")
    assert len(content_hash("hello")) == 64  # sha256 hex


def test_canonicalize_filters_is_key_order_independent() -> None:
    assert canonicalize_filters({"a": 1, "b": 2}) == canonicalize_filters({"b": 2, "a": 1})
    assert canonicalize_filters(None) == canonicalize_filters({})
    assert canonicalize_filters({"a": 1}) != canonicalize_filters({"a": 2})


def test_query_key_is_deterministic() -> None:
    assert query_idempotency_key(**_BASE) == query_idempotency_key(**_BASE)


def test_query_key_changes_with_every_component() -> None:
    base_key = query_idempotency_key(**_BASE)
    variations = [
        {"user_id": uuid4()},
        {"question": "What is Rust?"},
        {"filters": {"doc": "x"}},
        {"top_k": 10},
        # Each of these is a silent input to the vectors an answer was built
        # from — a key that ignored any of them would serve an answer computed
        # under configuration that no longer applies.
        {"normalizer_version": "nfkc-ws-v2"},
        {"chunker_version": "recursive-char-v2"},
        {"embedding_model": "text-embedding-3-large"},
    ]
    for override in variations:
        assert query_idempotency_key(**{**_BASE, **override}) != base_key, override
