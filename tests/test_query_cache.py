from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.dependencies import get_embedding_service, get_llm_client
from production_rag.llm.client import LLMClient, LLMResponse
from production_rag.llm.embedding_service import EmbeddingService
from production_rag.main import app
from production_rag.models import User
from production_rag.repositories.query_cache_repository import QueryCacheRepository
from production_rag.services.auth_service import hash_password
from tests._jobs import drain_jobs

pytestmark = pytest.mark.asyncio(loop_scope="module")


def _mock_embedding_service() -> EmbeddingService:
    mock = AsyncMock(spec=EmbeddingService)
    mock.embed_text.side_effect = lambda text: [0.1] * 1536
    mock.embed_batch.side_effect = lambda texts: [[0.1] * 1536 for _ in texts]
    mock.model = "text-embedding-3-small"
    return mock


def _mock_llm_client() -> LLMClient:
    mock = AsyncMock(spec=LLMClient)
    mock.chat.return_value = LLMResponse(
        content="The answer is 42.",
        model="gpt-4.1-nano",
        input_tokens=500,
        output_tokens=20,
        total_tokens=520,
        cost_usd=0.0001,
        latency_ms=300.0,
        provider="openai",
    )
    return mock


async def _auth_token(client: AsyncClient, email: str) -> str:
    await client.post(
        "/auth/signup",
        json={"name": "Asker", "email": email, "password": "securepass123"},
    )
    login = await client.post(
        "/auth/login",
        json={"email": email, "password": "securepass123"},
    )
    return login.json()["access_token"]


async def _make_user(session: AsyncSession, email: str) -> UUID:
    user = User(name="T", email=email, hashed_password=hash_password("pw"), role="user")
    session.add(user)
    await session.flush()
    return user.id


# ─── Repository-level: TTL, conditional write, invalidation ───


async def test_cache_get_respects_ttl(pg_session: AsyncSession) -> None:
    uid = await _make_user(pg_session, "ttl@example.com")
    repo = QueryCacheRepository(pg_session)

    past = datetime.now(UTC) - timedelta(seconds=1)
    await repo.put("k-expired", uid, {"answer": "x"}, expires_at=past)
    assert await repo.get("k-expired") is None  # expired → miss

    future = datetime.now(UTC) + timedelta(hours=1)
    await repo.put("k-live", uid, {"answer": "y"}, expires_at=future)
    live = await repo.get("k-live")
    assert live is not None and live.response_json["answer"] == "y"


async def test_cache_put_is_conditional(pg_session: AsyncSession) -> None:
    uid = await _make_user(pg_session, "cond@example.com")
    repo = QueryCacheRepository(pg_session)
    future = datetime.now(UTC) + timedelta(hours=1)

    await repo.put("dup", uid, {"answer": "first"}, expires_at=future)
    await repo.put("dup", uid, {"answer": "second"}, expires_at=future)  # no error, ignored

    got = await repo.get("dup")
    assert got is not None and got.response_json["answer"] == "first"


async def test_cache_delete_by_user(pg_session: AsyncSession) -> None:
    uid = await _make_user(pg_session, "del@example.com")
    repo = QueryCacheRepository(pg_session)
    future = datetime.now(UTC) + timedelta(hours=1)
    await repo.put("d-a", uid, {"answer": "a"}, expires_at=future)
    await repo.put("d-b", uid, {"answer": "b"}, expires_at=future)

    await repo.delete_by_user(uid)

    assert await repo.get("d-a") is None
    assert await repo.get("d-b") is None


# ─── Endpoint-level: cache hit / miss / invalidation ───


async def test_identical_query_served_from_cache(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    mock_emb = _mock_embedding_service()
    mock_llm = _mock_llm_client()
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb
    app.dependency_overrides[get_llm_client] = lambda: mock_llm
    token = await _auth_token(pg_async_client, "cachehit@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    await pg_async_client.post(
        "/documents/upload",
        files={"file": ("kb.txt", b"Python is great. " * 100, "text/plain")},
        headers=headers,
    )
    await drain_jobs(pg_session, job_queue, mock_emb)
    query = {"question": "What is Python?", "top_k": 3}
    r1 = await pg_async_client.post("/documents/query", json=query, headers=headers)
    r2 = await pg_async_client.post("/documents/query", json=query, headers=headers)

    assert r1.json()["cached"] is False
    assert r2.json()["cached"] is True
    assert r1.json()["answer"] == r2.json()["answer"]
    # The cached second query embedded nothing and made no LLM call.
    assert mock_emb.embed_text.call_count == 1
    assert mock_llm.chat.call_count == 1

    app.dependency_overrides.pop(get_embedding_service, None)
    app.dependency_overrides.pop(get_llm_client, None)


async def test_different_top_k_misses_cache(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    mock_emb = _mock_embedding_service()
    mock_llm = _mock_llm_client()
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb
    app.dependency_overrides[get_llm_client] = lambda: mock_llm
    token = await _auth_token(pg_async_client, "topk@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    await pg_async_client.post(
        "/documents/upload",
        files={"file": ("kb.txt", b"Rust is fast. " * 100, "text/plain")},
        headers=headers,
    )
    await drain_jobs(pg_session, job_queue, mock_emb)
    r1 = await pg_async_client.post(
        "/documents/query", json={"question": "What is Rust?", "top_k": 3}, headers=headers
    )
    r2 = await pg_async_client.post(
        "/documents/query", json={"question": "What is Rust?", "top_k": 5}, headers=headers
    )

    assert r1.json()["cached"] is False
    assert r2.json()["cached"] is False  # different top_k → different key
    assert mock_llm.chat.call_count == 2

    app.dependency_overrides.pop(get_embedding_service, None)
    app.dependency_overrides.pop(get_llm_client, None)


async def test_ingest_invalidates_cache(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    mock_emb = _mock_embedding_service()
    mock_llm = _mock_llm_client()
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb
    app.dependency_overrides[get_llm_client] = lambda: mock_llm
    token = await _auth_token(pg_async_client, "invalidate@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    await pg_async_client.post(
        "/documents/upload",
        files={"file": ("kb.txt", b"Go is simple. " * 100, "text/plain")},
        headers=headers,
    )
    await drain_jobs(pg_session, job_queue, mock_emb)
    query = {"question": "What is Go?", "top_k": 3}
    r1 = await pg_async_client.post("/documents/query", json=query, headers=headers)  # miss
    r2 = await pg_async_client.post("/documents/query", json=query, headers=headers)  # hit
    assert r1.json()["cached"] is False and r2.json()["cached"] is True

    # New ingestion invalidates the user's cached answers.
    await pg_async_client.post(
        "/documents/upload",
        files={"file": ("more.txt", b"Go has goroutines. " * 100, "text/plain")},
        headers=headers,
    )
    await drain_jobs(pg_session, job_queue, mock_emb)
    r3 = await pg_async_client.post("/documents/query", json=query, headers=headers)  # miss again

    assert r3.json()["cached"] is False
    assert mock_llm.chat.call_count == 2  # first query + recomputation after invalidation

    app.dependency_overrides.pop(get_embedding_service, None)
    app.dependency_overrides.pop(get_llm_client, None)
