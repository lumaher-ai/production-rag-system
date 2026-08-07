"""Deterministic hashing helpers for idempotent ingestion and query caching.

This is the single place idempotency keys are built. Every key threads through
``normalizer_version``, ``chunker_version`` and ``embedding_model`` so a stale
normalization, chunking, or embedding config can never serve cached data
computed under a different one — the key functions simply have no way to omit
them.

Those three are the complete set of *silent inputs* to a stored vector: change
any one and the vectors mean something different, while the text that produced
them looks identical. Anything added to that pipeline belongs here too.
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
    normalizer_version: str,
    chunker_version: str,
    embedding_model: str,
) -> str:
    """Deterministic cache key for a query.

    Key = hash(user_id + question + filters + top_k + normalizer_version +
    chunker_version + embedding_model). The last three are mandatory
    keyword-only inputs so no caller can build a key that ignores the
    normalization/chunking/embedding config the answer depends on.

    ``normalizer_version`` matters here in a way it does not on the ingest path.
    A document's identity is its *normalized* content hash, so a rules change
    alters that hash and trips the gate by itself. A query is never stored and
    never hashed into a row — the question is normalized, embedded, and
    discarded. So this version string is the only thing that can invalidate a
    cached answer when the normalization rules move.
    """
    payload = "|".join(
        [
            str(user_id),
            question,
            canonicalize_filters(filters),
            str(top_k),
            normalizer_version,
            chunker_version,
            embedding_model,
        ]
    )
    return _sha256(payload)
