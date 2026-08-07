"""Deterministic hashing helpers for idempotent ingestion and query caching.

This is the single place idempotency keys are built. Every key threads through
``chunker_version`` and ``embedding_model`` so a stale chunking or embedding
config can never serve cached data computed under a different one — the key
functions simply have no way to omit them.
"""

import hashlib
import json
from typing import Any
from uuid import UUID


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def content_hash(content: str) -> str:
    """Stable sha256 hex of document content (identity for idempotent ingestion)."""
    return _sha256(content)


def canonicalize_filters(filters: dict[str, Any] | None) -> str:
    """Serialize a metadata/context filter to a key-order-independent string.

    ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` must map to the same string so
    equivalent queries share a cache entry.
    """
    return json.dumps(filters or {}, sort_keys=True, separators=(",", ":"))


def query_idempotency_key(
    *,
    user_id: UUID,
    question: str,
    filters: dict[str, Any] | None,
    top_k: int,
    chunker_version: str,
    embedding_model: str,
) -> str:
    """Deterministic cache key for a query.

    Key = hash(user_id + question + filters + top_k + chunker_version + embedding_model).
    The last two are mandatory positional-by-name inputs so no caller can build a
    key that ignores the chunking/embedding config the answer depends on.
    """
    payload = "|".join(
        [
            str(user_id),
            question,
            canonicalize_filters(filters),
            str(top_k),
            chunker_version,
            embedding_model,
        ]
    )
    return _sha256(payload)
