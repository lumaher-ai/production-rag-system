from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient

from production_rag.dependencies import get_embedding_service, get_llm_client
from production_rag.llm.client import LLMClient, LLMResponse
from production_rag.llm.embedding_service import EmbeddingService
from production_rag.main import app
from tests._jobs import drain_jobs

pytestmark = pytest.mark.asyncio(loop_scope="module")


def _mock_embedding_service() -> EmbeddingService:
    mock = AsyncMock(spec=EmbeddingService)
    mock.embed_text.side_effect = lambda text: [0.1] * 1536
    mock.embed_batch.side_effect = lambda texts: [[0.1] * 1536 for _ in texts]
    mock.model = "text-embedding-3-small"  # part of the idempotency key
    return mock


def _mock_llm_client() -> LLMClient:
    mock = AsyncMock(spec=LLMClient)
    mock.chat.return_value = LLMResponse(
        content="Based on the context, the answer is 42.",
        model="gpt-4o-mini",
        input_tokens=500,
        output_tokens=20,
        total_tokens=520,
        cost_usd=0.0001,
        latency_ms=300.0,
        provider="openai",
    )
    return mock


async def test_upload_document_creates_document_via_worker(
    pg_async_client: AsyncClient, pg_session, job_queue
) -> None:
    """Upload is accepted (202), then the worker produces the document."""
    mock_emb = _mock_embedding_service()
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb

    # Signup + login
    await pg_async_client.post(
        "/auth/signup",
        json={"name": "Doc User", "email": "doc@example.com", "password": "securepass123"},
    )
    login = await pg_async_client.post(
        "/auth/login",
        json={"email": "doc@example.com", "password": "securepass123"},
    )
    token = login.json()["access_token"]

    response = await pg_async_client.post(
        "/documents/upload",
        data={"title": "Test Document"},
        files={
            "file": (
                "test.txt",
                b"This is a test document with enough content to be chunked. " * 50,
                "text/plain",
            )
        },
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "pending"

    await drain_jobs(pg_session, job_queue, mock_emb)

    listed = await pg_async_client.get("/documents", headers={"Authorization": f"Bearer {token}"})
    document = next(d for d in listed.json() if d["title"] == "Test Document")
    assert document["chunk_count"] > 0

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_query_returns_answer_with_sources(
    pg_async_client: AsyncClient, pg_session, job_queue
) -> None:
    mock_emb = _mock_embedding_service()
    mock_llm = _mock_llm_client()
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb
    app.dependency_overrides[get_llm_client] = lambda: mock_llm

    # Signup + login
    await pg_async_client.post(
        "/auth/signup",
        json={"name": "Query User", "email": "query@example.com", "password": "securepass123"},
    )
    login = await pg_async_client.post(
        "/auth/login",
        json={"email": "query@example.com", "password": "securepass123"},
    )
    token = login.json()["access_token"]

    # Upload document first
    await pg_async_client.post(
        "/documents/upload",
        data={"title": "Knowledge Base"},
        files={
            "file": (
                "kb.txt",
                b"Python is a programming language. " * 100,
                "text/plain",
            )
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    # Ingestion is async now — run the worker before the document is queryable.
    await drain_jobs(pg_session, job_queue, mock_emb)

    # Query
    response = await pg_async_client.post(
        "/documents/query",
        json={"question": "What is Python?", "top_k": 3},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["sources"]) > 0
    assert data["sources"][0]["document_title"] == "Knowledge Base"
    # Similarity score is exposed, numeric, and in the valid cosine range.
    top_score = data["sources"][0]["similarity_score"]
    assert isinstance(top_score, float)
    assert -1.0 <= top_score <= 1.0
    # Results are ordered best-match-first: scores are non-increasing by rank.
    scores = [s["similarity_score"] for s in data["sources"]]
    assert scores == sorted(scores, reverse=True)
    assert data["model"] == "gpt-4o-mini"

    app.dependency_overrides.pop(get_embedding_service, None)
    app.dependency_overrides.pop(get_llm_client, None)


async def test_query_requires_auth(pg_async_client: AsyncClient) -> None:
    response = await pg_async_client.post(
        "/documents/query",
        json={"question": "test"},
    )
    assert response.status_code == 401


async def test_upload_requires_auth(pg_async_client: AsyncClient) -> None:
    response = await pg_async_client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"test content here with enough length", "text/plain")},
    )
    assert response.status_code == 401
