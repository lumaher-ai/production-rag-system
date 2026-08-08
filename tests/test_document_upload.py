from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.dependencies import get_embedding_service
from production_rag.llm.embedding_service import EmbeddingService
from production_rag.main import app
from production_rag.models.document import DocumentChunk
from tests._file_builders import make_docx, make_pdf
from tests._jobs import drain_jobs

pytestmark = pytest.mark.asyncio(loop_scope="module")


def _mock_embedding_service() -> EmbeddingService:
    mock = AsyncMock(spec=EmbeddingService)
    mock.embed_text.side_effect = lambda text: [0.1] * 1536
    mock.embed_batch.side_effect = lambda texts: [[0.1] * 1536 for _ in texts]
    mock.model = "text-embedding-3-small"  # part of the idempotency key
    return mock


async def _document_id_for(session: AsyncSession, source: str) -> UUID:
    """Documents are created by the worker, so the id comes from the source URI."""
    from production_rag.models.document import Document

    result = await session.execute(select(Document.id).where(Document.source == source))
    return result.scalar_one()


async def _auth_token(client: AsyncClient, email: str) -> str:
    await client.post(
        "/auth/signup",
        json={"name": "Uploader", "email": email, "password": "securepass123"},
    )
    login = await client.post(
        "/auth/login",
        json={"email": email, "password": "securepass123"},
    )
    return login.json()["access_token"]


async def test_upload_txt_is_accepted_and_ingested(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    app.dependency_overrides[get_embedding_service] = _mock_embedding_service
    token = await _auth_token(pg_async_client, "txt@example.com")

    response = await pg_async_client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"plain text content " * 50, "text/plain")},
        headers={"Authorization": f"Bearer {token}"},
    )

    # 202: the work is queued, not done. source is a URI, not a bare filename.
    assert response.status_code == 202
    accepted = response.json()
    assert accepted["source"].startswith("upload://")
    assert accepted["source"].endswith("/notes.txt")

    await drain_jobs(pg_session, job_queue)

    listed = await pg_async_client.get(
        "/documents", headers={"Authorization": f"Bearer {token}"}
    )
    document = next(d for d in listed.json() if d["source"] == accepted["source"])
    assert document["title"] == "notes"  # filename stem
    assert document["chunk_count"] > 0

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_upload_title_override(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    app.dependency_overrides[get_embedding_service] = _mock_embedding_service
    token = await _auth_token(pg_async_client, "title@example.com")

    response = await pg_async_client.post(
        "/documents/upload",
        data={"title": "Custom Title"},
        files={"file": ("ignored.txt", b"some content here " * 20, "text/plain")},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 202
    await drain_jobs(pg_session, job_queue)

    listed = await pg_async_client.get(
        "/documents", headers={"Authorization": f"Bearer {token}"}
    )
    assert any(d["title"] == "Custom Title" for d in listed.json())

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_upload_pdf_persists_page_provenance(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    app.dependency_overrides[get_embedding_service] = _mock_embedding_service
    token = await _auth_token(pg_async_client, "pdf@example.com")

    pdf = make_pdf(["First page content here.", "Second page content here."])
    response = await pg_async_client.post(
        "/documents/upload",
        files={"file": ("report.pdf", pdf, "application/pdf")},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 202
    await drain_jobs(pg_session, job_queue)
    doc_id = await _document_id_for(pg_session, response.json()["source"])

    # Chunks share the request's session (conftest override) — page must be set.
    result = await pg_session.execute(
        select(DocumentChunk.page).where(DocumentChunk.document_id == doc_id)
    )
    pages = {row[0] for row in result.all()}
    assert pages == {1, 2}

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_upload_docx_persists_section_provenance(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    app.dependency_overrides[get_embedding_service] = _mock_embedding_service
    token = await _auth_token(pg_async_client, "docx@example.com")

    docx_bytes = make_docx([("Introduction", "Intro body " * 30)])
    response = await pg_async_client.post(
        "/documents/upload",
        files={
            "file": (
                "paper.docx",
                docx_bytes,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 202
    await drain_jobs(pg_session, job_queue)
    doc_id = await _document_id_for(pg_session, response.json()["source"])

    result = await pg_session.execute(
        select(DocumentChunk.section).where(DocumentChunk.document_id == doc_id)
    )
    sections = {row[0] for row in result.all()}
    assert "Introduction" in sections

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_upload_unsupported_type_returns_415(
    pg_async_client: AsyncClient, job_queue
) -> None:
    app.dependency_overrides[get_embedding_service] = _mock_embedding_service
    token = await _auth_token(pg_async_client, "unsupported@example.com")

    response = await pg_async_client.post(
        "/documents/upload",
        files={"file": ("archive.xyz", b"binary data", "application/octet-stream")},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 415

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_upload_oversized_returns_413(
    pg_async_client: AsyncClient, job_queue
) -> None:
    from production_rag.config import get_settings

    app.dependency_overrides[get_embedding_service] = _mock_embedding_service

    tiny_settings = get_settings().model_copy(update={"max_upload_bytes": 10})
    app.dependency_overrides[get_settings] = lambda: tiny_settings
    token = await _auth_token(pg_async_client, "big@example.com")

    response = await pg_async_client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"content larger than ten bytes", "text/plain")},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 413

    app.dependency_overrides.pop(get_embedding_service, None)
    app.dependency_overrides.pop(get_settings, None)


async def test_upload_requires_auth(pg_async_client: AsyncClient) -> None:
    response = await pg_async_client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"content", "text/plain")},
    )
    assert response.status_code == 401
