"""DELETE /documents/{document_id} — and the references it has to take with it.

The interesting assertions here are not "the row is gone". They are the four
things that point at a document and would otherwise outlive it: its chunks
(which is what retrieval actually reads), the ingestion job that produced it,
the owner's cached answers, and another tenant's ability to name the id at all.
"""

from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.dependencies import get_embedding_service, get_llm_client
from production_rag.llm.client import LLMClient, LLMResponse
from production_rag.main import app
from production_rag.models.document import DocumentChunk
from production_rag.models.ingestion_job import IngestionJob
from tests._jobs import drain_jobs, mock_embedding_service

pytestmark = pytest.mark.asyncio(loop_scope="module")


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
        json={"name": "Owner", "email": email, "password": "securepass123"},
    )
    login = await client.post(
        "/auth/login", json={"email": email, "password": "securepass123"}
    )
    return login.json()["access_token"]


async def _ingest(
    client: AsyncClient, session: AsyncSession, queue, headers: dict[str, str], name: str
) -> str:
    """Upload a file, run the worker, return the resulting document id."""
    emb = mock_embedding_service()
    app.dependency_overrides[get_embedding_service] = lambda: emb
    await client.post(
        "/documents/upload",
        files={"file": (name, b"The refund policy is 30 days. " * 100, "text/plain")},
        headers=headers,
    )
    await drain_jobs(session, queue, emb)
    documents = (await client.get("/documents", headers=headers)).json()
    return next(doc["id"] for doc in documents if doc["source"].endswith(name))


async def _chunk_count(session: AsyncSession, document_id: str) -> int:
    result = await session.execute(
        select(func.count())
        .select_from(DocumentChunk)
        .where(DocumentChunk.document_id == UUID(document_id))
    )
    return int(result.scalar_one())


async def test_delete_removes_the_document_and_its_chunks(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    """The chunks are the deletion — an orphaned chunk is still retrievable."""
    token = await _auth_token(pg_async_client, "delete-basic@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    document_id = await _ingest(pg_async_client, pg_session, job_queue, headers, "kb.txt")

    before = await _chunk_count(pg_session, document_id)
    assert before > 0

    response = await pg_async_client.delete(f"/documents/{document_id}", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == document_id
    # Counted from the DELETE, so it is evidence rather than a copy of the
    # document's stored chunk_count.
    assert body["chunks_deleted"] == before
    assert body["cache_invalidated"] is True

    assert await _chunk_count(pg_session, document_id) == 0
    remaining = (await pg_async_client.get("/documents", headers=headers)).json()
    assert all(doc["id"] != document_id for doc in remaining)

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_delete_detaches_the_ingestion_job_without_erasing_it(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    """SET NULL, not CASCADE: the record that this source was ingested survives."""
    token = await _auth_token(pg_async_client, "delete-job@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    document_id = await _ingest(pg_async_client, pg_session, job_queue, headers, "job.txt")

    job = (
        await pg_session.execute(
            select(IngestionJob).where(IngestionJob.document_id == UUID(document_id))
        )
    ).scalar_one()
    job_id, job_source = job.id, job.source

    await pg_async_client.delete(f"/documents/{document_id}", headers=headers)

    await pg_session.refresh(job)
    assert job.document_id is None  # the dangling pointer is gone…
    assert job.id == job_id and job.source == job_source  # …but the history is not
    status_response = await pg_async_client.get(f"/documents/jobs/{job_id}", headers=headers)
    assert status_response.status_code == 200
    assert status_response.json()["document_id"] is None

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_delete_invalidates_cached_answers(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    """A cached answer has its sources baked in and never touches the index.

    Without invalidation the deleted document would keep being quoted, verbatim,
    with no vector search involved to notice its chunks are gone.
    """
    mock_llm = _mock_llm_client()
    app.dependency_overrides[get_llm_client] = lambda: mock_llm
    token = await _auth_token(pg_async_client, "delete-cache@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    document_id = await _ingest(pg_async_client, pg_session, job_queue, headers, "cache.txt")

    query = {"question": "What is the refund policy?", "top_k": 3}
    miss = await pg_async_client.post("/documents/query", json=query, headers=headers)
    hit = await pg_async_client.post("/documents/query", json=query, headers=headers)
    assert miss.json()["cached"] is False and hit.json()["cached"] is True
    assert hit.json()["sources"]

    await pg_async_client.delete(f"/documents/{document_id}", headers=headers)

    after = await pg_async_client.post("/documents/query", json=query, headers=headers)
    assert after.json()["cached"] is False
    assert after.json()["sources"] == []
    assert after.json()["answer"].startswith("No documents found")

    app.dependency_overrides.pop(get_embedding_service, None)
    app.dependency_overrides.pop(get_llm_client, None)


async def test_delete_another_users_document_is_404(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    """404, not 403 — a 403 would confirm the id exists."""
    owner_headers = {
        "Authorization": f"Bearer {await _auth_token(pg_async_client, 'delete-owner@example.com')}"
    }
    intruder_headers = {
        "Authorization": f"Bearer {await _auth_token(pg_async_client, 'delete-other@example.com')}"
    }
    document_id = await _ingest(
        pg_async_client, pg_session, job_queue, owner_headers, "private.txt"
    )

    response = await pg_async_client.delete(
        f"/documents/{document_id}", headers=intruder_headers
    )

    assert response.status_code == 404
    # And the owner's copy is untouched by the attempt.
    assert await _chunk_count(pg_session, document_id) > 0

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_delete_unknown_document_is_404(pg_async_client: AsyncClient) -> None:
    token = await _auth_token(pg_async_client, "delete-missing@example.com")
    response = await pg_async_client.delete(
        f"/documents/{uuid4()}", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 404


async def test_delete_requires_authentication(pg_async_client: AsyncClient) -> None:
    response = await pg_async_client.delete(f"/documents/{uuid4()}")
    assert response.status_code == 401
