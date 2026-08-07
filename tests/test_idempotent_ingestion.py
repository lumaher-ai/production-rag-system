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


async def _upload(
    client: AsyncClient, token: str, filename: str, content: bytes
) -> dict:
    resp = await client.post(
        "/documents/upload",
        files={"file": (filename, content, "text/plain")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def test_reupload_same_source_unchanged_skips_embedding(
    pg_async_client: AsyncClient,
) -> None:
    mock_emb = _mock_embedding_service()
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb
    token = await _auth_token(pg_async_client, "idem@example.com")
    content = b"repeatable content to be chunked. " * 50

    r1 = await _upload(pg_async_client, token, "notes.txt", content)
    r2 = await _upload(pg_async_client, token, "notes.txt", content)

    # Same (owner, source) + identical bytes → the same document, no re-embed.
    assert r1["id"] == r2["id"]
    assert r1["chunk_count"] == r2["chunk_count"]
    assert mock_emb.embed_batch.call_count == 1

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_reupload_edited_source_replaces_document(
    pg_async_client: AsyncClient, pg_session: AsyncSession
) -> None:
    mock_emb = _mock_embedding_service()
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb
    token = await _auth_token(pg_async_client, "replace@example.com")

    original = await _upload(
        pg_async_client, token, "contract.txt", b"original clause body here. " * 40
    )
    edited = await _upload(
        pg_async_client,
        token,
        "contract.txt",
        b"an entirely rewritten and much longer edited clause body here. " * 80,
    )

    # Same source, different content → the SAME document, replaced in place.
    assert edited["id"] == original["id"]
    # The edit re-embedded (unchanged short-circuit did NOT fire).
    assert mock_emb.embed_batch.call_count == 2

    doc_id = UUID(edited["id"])
    # Exactly one document row for this source (no duplicate was created).
    docs = (
        await pg_session.execute(select(Document.id).where(Document.id == doc_id))
    ).scalars().all()
    assert docs == [doc_id]

    # Old chunks are gone: the persisted chunk count matches the edited response
    # and no chunk carries the original body.
    chunk_rows = (
        await pg_session.execute(
            select(DocumentChunk.content).where(DocumentChunk.document_id == doc_id)
        )
    ).scalars().all()
    assert len(chunk_rows) == edited["chunk_count"]
    assert all("original clause" not in c for c in chunk_rows)
    assert any("rewritten" in c for c in chunk_rows)

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_different_source_creates_new_document(
    pg_async_client: AsyncClient,
) -> None:
    mock_emb = _mock_embedding_service()
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb
    token = await _auth_token(pg_async_client, "distinct@example.com")

    r1 = await _upload(pg_async_client, token, "a.txt", b"alpha content here " * 30)
    r2 = await _upload(pg_async_client, token, "b.txt", b"beta content here " * 30)

    # Distinct sources → distinct documents, each embedded once.
    assert r1["id"] != r2["id"]
    assert mock_emb.embed_batch.call_count == 2

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_chunks_carry_owner_id(
    pg_async_client: AsyncClient, pg_session: AsyncSession
) -> None:
    mock_emb = _mock_embedding_service()
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb
    token = await _auth_token(pg_async_client, "owner@example.com")

    resp = await _upload(
        pg_async_client, token, "owned.txt", b"owned content to chunk " * 40
    )
    doc_id = UUID(resp["id"])

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
