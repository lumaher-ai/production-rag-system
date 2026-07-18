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

pytestmark = pytest.mark.asyncio(loop_scope="module")


def _mock_embedding_service() -> EmbeddingService:
    mock = AsyncMock(spec=EmbeddingService)
    mock.embed_text.side_effect = lambda text: [0.1] * 1536
    mock.embed_batch.side_effect = lambda texts: [[0.1] * 1536 for _ in texts]
    return mock


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


async def test_upload_txt_returns_201(pg_async_client: AsyncClient) -> None:
    app.dependency_overrides[get_embedding_service] = _mock_embedding_service
    token = await _auth_token(pg_async_client, "txt@example.com")

    response = await pg_async_client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"plain text content " * 50, "text/plain")},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "notes"  # filename stem
    assert data["chunk_count"] > 0

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_upload_title_override(pg_async_client: AsyncClient) -> None:
    app.dependency_overrides[get_embedding_service] = _mock_embedding_service
    token = await _auth_token(pg_async_client, "title@example.com")

    response = await pg_async_client.post(
        "/documents/upload",
        data={"title": "Custom Title"},
        files={"file": ("ignored.txt", b"some content here " * 20, "text/plain")},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 201
    assert response.json()["title"] == "Custom Title"

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_upload_pdf_persists_page_provenance(
    pg_async_client: AsyncClient, pg_session: AsyncSession
) -> None:
    app.dependency_overrides[get_embedding_service] = _mock_embedding_service
    token = await _auth_token(pg_async_client, "pdf@example.com")

    pdf = make_pdf(["First page content here.", "Second page content here."])
    response = await pg_async_client.post(
        "/documents/upload",
        files={"file": ("report.pdf", pdf, "application/pdf")},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 201
    doc_id = response.json()["id"]

    # Chunks share the request's session (conftest override) — page must be set.
    result = await pg_session.execute(
        select(DocumentChunk.page).where(DocumentChunk.document_id == UUID(doc_id))
    )
    pages = {row[0] for row in result.all()}
    assert pages == {1, 2}

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_upload_docx_persists_section_provenance(
    pg_async_client: AsyncClient, pg_session: AsyncSession
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

    assert response.status_code == 201
    doc_id = response.json()["id"]

    result = await pg_session.execute(
        select(DocumentChunk.section).where(DocumentChunk.document_id == UUID(doc_id))
    )
    sections = {row[0] for row in result.all()}
    assert "Introduction" in sections

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_upload_unsupported_type_returns_415(pg_async_client: AsyncClient) -> None:
    app.dependency_overrides[get_embedding_service] = _mock_embedding_service
    token = await _auth_token(pg_async_client, "unsupported@example.com")

    response = await pg_async_client.post(
        "/documents/upload",
        files={"file": ("archive.xyz", b"binary data", "application/octet-stream")},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 415

    app.dependency_overrides.pop(get_embedding_service, None)


async def test_upload_oversized_returns_413(pg_async_client: AsyncClient) -> None:
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
