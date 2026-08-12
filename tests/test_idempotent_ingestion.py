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
from production_rag.services import ingestion_service
from tests._jobs import drain_jobs

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
    client: AsyncClient,
    token: str,
    filename: str,
    content: bytes,
    session: AsyncSession = None,
    queue=None,
    embeddings=None,
) -> dict:
    """Upload and run the job, returning the resulting document.

    Ingestion is asynchronous: the endpoint returns 202 and a worker finishes
    the job. These tests are about ingestion *outcomes* (idempotency, replace,
    ownership), so the helper drives both halves and hands back the document.
    """
    resp = await client.post(
        "/documents/upload",
        files={"file": (filename, content, "text/plain")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 202, resp.text
    source = resp.json()["source"]

    await drain_jobs(session, queue, embeddings)

    listed = await client.get("/documents", headers={"Authorization": f"Bearer {token}"})
    return next(d for d in listed.json() if d["source"] == source)


async def test_reupload_same_source_unchanged_skips_embedding(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    mock_emb = _mock_embedding_service()
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb
    token = await _auth_token(pg_async_client, "idem@example.com")
    content = b"repeatable content to be chunked. " * 50

    r1 = await _upload(
        pg_async_client, token, "notes.txt", content, pg_session, job_queue, mock_emb
    )
    r2 = await _upload(
        pg_async_client, token, "notes.txt", content, pg_session, job_queue, mock_emb
    )

    # Same (owner, source) + identical bytes → the same document, no re-embed.
    assert r1["id"] == r2["id"]
    assert r1["chunk_count"] == r2["chunk_count"]
    assert mock_emb.embed_batch.call_count == 1

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_unchanged_reingest_does_not_re_extract_metadata(
    pg_async_client: AsyncClient,
    pg_session: AsyncSession,
    job_queue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-op path must not pay for work it will not use.

    This ordering is invisible from the outside: extracting *before* the
    idempotency gate produces byte-identical metadata and passes every other
    test in this file, so a regression here breaks nothing and reports nothing.
    It is pinned because the cost is not fixed. Extraction is cheap today; an
    extractor that ever called a model would bill a request on every re-ingest
    that decided to do nothing, and the code would still look correct.
    """
    calls: list[str] = []
    real_extract = ingestion_service.extract_metadata

    def counting_extract(text: str, **kwargs: object) -> dict:
        calls.append(text[:32])
        return real_extract(text, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ingestion_service, "extract_metadata", counting_extract)

    mock_emb = _mock_embedding_service()
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb
    token = await _auth_token(pg_async_client, "extract-once@example.com")
    content = b"stable body uploaded twice, unchanged. " * 50

    await _upload(pg_async_client, token, "stable.txt", content, pg_session, job_queue, mock_emb)
    assert len(calls) == 1, "the first ingest must extract"

    await _upload(pg_async_client, token, "stable.txt", content, pg_session, job_queue, mock_emb)
    assert len(calls) == 1, "an unchanged re-ingest must not extract again"

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_reupload_edited_source_replaces_document(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    mock_emb = _mock_embedding_service()
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb
    token = await _auth_token(pg_async_client, "replace@example.com")

    original = await _upload(
        pg_async_client,
        token,
        "contract.txt",
        b"original clause body here. " * 40,
        pg_session,
        job_queue,
        mock_emb,
    )
    edited = await _upload(
        pg_async_client,
        token,
        "contract.txt",
        b"an entirely rewritten and much longer edited clause body here. " * 80,
        pg_session,
        job_queue,
        mock_emb,
    )

    # Same source, different content → the SAME document, replaced in place.
    assert edited["id"] == original["id"]
    # The edit re-embedded (unchanged short-circuit did NOT fire).
    assert mock_emb.embed_batch.call_count == 2

    doc_id = UUID(edited["id"])
    # Exactly one document row for this source (no duplicate was created).
    docs = (
        (await pg_session.execute(select(Document.id).where(Document.id == doc_id))).scalars().all()
    )
    assert docs == [doc_id]

    # Old chunks are gone: the persisted chunk count matches the edited response
    # and no chunk carries the original body.
    chunk_rows = (
        (
            await pg_session.execute(
                select(DocumentChunk.content).where(DocumentChunk.document_id == doc_id)
            )
        )
        .scalars()
        .all()
    )
    assert len(chunk_rows) == edited["chunk_count"]
    assert all("original clause" not in c for c in chunk_rows)
    assert any("rewritten" in c for c in chunk_rows)

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_different_source_creates_new_document(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    mock_emb = _mock_embedding_service()
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb
    token = await _auth_token(pg_async_client, "distinct@example.com")

    r1 = await _upload(
        pg_async_client,
        token,
        "a.txt",
        b"alpha content here " * 30,
        pg_session,
        job_queue,
        mock_emb,
    )
    r2 = await _upload(
        pg_async_client,
        token,
        "b.txt",
        b"beta content here " * 30,
        pg_session,
        job_queue,
        mock_emb,
    )

    # Distinct sources → distinct documents, each embedded once.
    assert r1["id"] != r2["id"]
    assert mock_emb.embed_batch.call_count == 2

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_chunks_carry_owner_id(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    mock_emb = _mock_embedding_service()
    app.dependency_overrides[get_embedding_service] = lambda: mock_emb
    token = await _auth_token(pg_async_client, "owner@example.com")

    resp = await _upload(
        pg_async_client,
        token,
        "owned.txt",
        b"owned content to chunk " * 40,
        pg_session,
        job_queue,
        mock_emb,
    )
    doc_id = UUID(resp["id"])

    document = await pg_session.get(Document, doc_id)
    assert document is not None
    owner_ids = (
        (
            await pg_session.execute(
                select(DocumentChunk.owner_id).where(DocumentChunk.document_id == doc_id)
            )
        )
        .scalars()
        .all()
    )
    assert owner_ids  # chunks exist
    assert all(oid == document.user_id for oid in owner_ids)  # denormalized owner matches

    app.dependency_overrides.pop(get_embedding_service, None)
