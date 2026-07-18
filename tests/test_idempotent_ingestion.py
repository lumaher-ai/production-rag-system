from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.dependencies import get_embedding_service
from production_rag.llm.embedding_service import EmbeddingService
from production_rag.main import app
from production_rag.models.document import Document, DocumentChunk

pytestmark = pytest.mark.asyncio(loop_scope="module")


def _mock_embedding_service() -> EmbeddingService:
    mock = AsyncMock(spec=EmbeddingService)
    mock.embed_text.side_effect = lambda text: [0.1] * 1536
    mock.embed_batch.side_effect = lambda texts: [[0.1] * 1536 for _ in texts]
    mock.model = "text-embedding-3-small"
    return mock


async def _auth_token(client: AsyncClient, email: str) -> str:
    await client.post(
        "/auth/signup",
        json={"name": "Ingestor", "email": email, "password": "securepass123"},
    )
    login = await client.post(
        "/auth/login",
        json={"email": email, "password": "securepass123"},
    )
    return login.json()["access_token"]


async def test_reingest_same_content_skips_embedding(pg_async_client: AsyncClient) -> None:
    mock_emb = _mock_embedding_service()
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb
    token = await _auth_token(pg_async_client, "idem@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"title": "Doc", "content": "repeatable content to be chunked. " * 50}

    r1 = await pg_async_client.post("/documents", json=payload, headers=headers)
    r2 = await pg_async_client.post("/documents", json=payload, headers=headers)

    assert r1.status_code == 201 and r2.status_code == 201
    # Second ingest returns the SAME document and did not re-embed.
    assert r1.json()["id"] == r2.json()["id"]
    assert r1.json()["chunk_count"] == r2.json()["chunk_count"]
    assert mock_emb.embed_batch.call_count == 1

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_different_content_creates_new_document(pg_async_client: AsyncClient) -> None:
    mock_emb = _mock_embedding_service()
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb
    token = await _auth_token(pg_async_client, "distinct@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    r1 = await pg_async_client.post(
        "/documents", json={"title": "A", "content": "alpha content here " * 30}, headers=headers
    )
    r2 = await pg_async_client.post(
        "/documents", json={"title": "B", "content": "beta content here " * 30}, headers=headers
    )

    assert r1.json()["id"] != r2.json()["id"]
    assert mock_emb.embed_batch.call_count == 2

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_chunks_carry_owner_id(
    pg_async_client: AsyncClient, pg_session: AsyncSession
) -> None:
    mock_emb = _mock_embedding_service()
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb
    token = await _auth_token(pg_async_client, "owner@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    resp = await pg_async_client.post(
        "/documents",
        json={"title": "Owned", "content": "owned content to chunk " * 40},
        headers=headers,
    )
    doc_id = UUID(resp.json()["id"])

    document = await pg_session.get(Document, doc_id)
    assert document is not None
    owner_ids = (
        await pg_session.execute(
            select(DocumentChunk.owner_id).where(DocumentChunk.document_id == doc_id)
        )
    ).scalars().all()
    assert owner_ids  # chunks exist
    assert all(oid == document.user_id for oid in owner_ids)  # denormalized owner matches

    app.dependency_overrides.pop(get_embedding_service, None)
