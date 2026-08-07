"""End-to-end ingest-by-URI: fetch an external document and store it.

Exercises the full path — parse URI, fetch bytes, dispatch a loader, chunk,
embed, persist — against a real local HTTP origin, so connectors are proven to
compose with the existing loader and idempotency machinery rather than mocked
into agreement with it.
"""

import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.config import get_settings
from production_rag.dependencies import get_embedding_service
from production_rag.llm.embedding_service import EmbeddingService
from production_rag.main import app
from production_rag.models.document import Document, DocumentChunk
from tests._file_builders import make_pdf

pytestmark = pytest.mark.asyncio(loop_scope="module")

MARKDOWN = b"# Quarterly Report\n\nRevenue rose sharply this quarter. " * 20


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/report.md":
            self._respond(200, MARKDOWN, "text/markdown")
        elif self.path == "/paper.pdf":
            self._respond(200, make_pdf(["Page one text.", "Page two text."]), "application/pdf")
        elif self.path == "/notes.xyz":
            self._respond(200, b"binary", "application/octet-stream")
        elif self.path == "/empty.txt":
            self._respond(200, b"   \n  ", "text/plain")
        else:
            self._respond(404, b"", "text/plain")

    def _respond(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        """Silence per-request logging to stderr."""


@pytest.fixture(scope="module")
def origin() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def _mock_embedding_service() -> EmbeddingService:
    mock = AsyncMock(spec=EmbeddingService)
    mock.embed_text.side_effect = lambda text: [0.1] * 1536
    mock.embed_batch.side_effect = lambda texts: [[0.1] * 1536 for _ in texts]
    mock.model = "text-embedding-3-small"  # part of the idempotency key
    return mock


@pytest.fixture
def local_fetch_allowed() -> Iterator[None]:
    """Allow fetching the loopback test origin, and mock embeddings."""
    settings = get_settings().model_copy(
        update={"allow_private_network_sources": True, "allow_insecure_http_sources": True}
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedding_service] = _mock_embedding_service
    yield
    app.dependency_overrides.pop(get_settings, None)
    app.dependency_overrides.pop(get_embedding_service, None)


async def _auth_token(client: AsyncClient, email: str) -> str:
    await client.post(
        "/auth/signup",
        json={"name": "Ingester", "email": email, "password": "securepass123"},
    )
    login = await client.post(
        "/auth/login",
        json={"email": email, "password": "securepass123"},
    )
    return login.json()["access_token"]


async def test_ingest_url_stores_document_with_uri_source(
    pg_async_client: AsyncClient, origin: str, local_fetch_allowed: None
) -> None:
    token = await _auth_token(pg_async_client, "url@example.com")

    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": f"{origin}/report.md"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["chunk_count"] > 0
    assert data["title"] == "report"  # filename stem from the URL
    # The fetched URL *is* the identity — stored verbatim, not reduced to a name.
    assert data["source"] == f"{origin}/report.md"


async def test_ingest_pdf_url_keeps_page_provenance(
    pg_async_client: AsyncClient, pg_session: AsyncSession, origin: str, local_fetch_allowed: None
) -> None:
    """Connectors add sources, not formats: the PDF loader runs unchanged."""
    token = await _auth_token(pg_async_client, "pdfurl@example.com")

    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": f"{origin}/paper.pdf"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 201
    doc_id = response.json()["id"]

    result = await pg_session.execute(
        select(DocumentChunk.page).where(DocumentChunk.document_id == UUID(doc_id))
    )
    assert {row[0] for row in result.all()} == {1, 2}


async def test_ingest_title_override(
    pg_async_client: AsyncClient, origin: str, local_fetch_allowed: None
) -> None:
    token = await _auth_token(pg_async_client, "ingesttitle@example.com")

    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": f"{origin}/report.md", "title": "Q3 Numbers"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 201
    assert response.json()["title"] == "Q3 Numbers"


async def test_reingesting_same_uri_replaces_rather_than_duplicates(
    pg_async_client: AsyncClient, pg_session: AsyncSession, origin: str, local_fetch_allowed: None
) -> None:
    """The point of making source a URI: it is a stable identity."""
    token = await _auth_token(pg_async_client, "repeat@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    body = {"uri": f"{origin}/report.md"}

    first = await pg_async_client.post("/documents/ingest", json=body, headers=headers)
    second = await pg_async_client.post("/documents/ingest", json=body, headers=headers)

    assert first.status_code == second.status_code == 201
    # Same document id back, and only one row for this source.
    assert first.json()["id"] == second.json()["id"]

    result = await pg_session.execute(
        select(Document).where(Document.source == f"{origin}/report.md")
    )
    assert len(result.scalars().all()) == 1


async def test_upload_and_url_of_same_name_stay_distinct(
    pg_async_client: AsyncClient, origin: str, local_fetch_allowed: None
) -> None:
    """The scheme is load-bearing: same filename, different origin, two documents."""
    token = await _auth_token(pg_async_client, "distinct@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    uploaded = await pg_async_client.post(
        "/documents/upload",
        files={"file": ("report.md", MARKDOWN, "text/markdown")},
        headers=headers,
    )
    fetched = await pg_async_client.post(
        "/documents/ingest", json={"uri": f"{origin}/report.md"}, headers=headers
    )

    assert uploaded.status_code == fetched.status_code == 201
    assert uploaded.json()["id"] != fetched.json()["id"]
    assert uploaded.json()["source"].startswith("upload://")
    assert fetched.json()["source"] == f"{origin}/report.md"


# ─── Error surfaces ───


async def test_ingest_rejects_bare_filename_with_422(
    pg_async_client: AsyncClient, local_fetch_allowed: None
) -> None:
    token = await _auth_token(pg_async_client, "bare@example.com")

    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": "reporte.pdf"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 422


async def test_ingest_rejects_unshipped_scheme_with_422(
    pg_async_client: AsyncClient, local_fetch_allowed: None
) -> None:
    token = await _auth_token(pg_async_client, "s3@example.com")

    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": "s3://bucket/report.pdf"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 422


async def test_ingest_blocks_private_address_with_502(pg_async_client: AsyncClient) -> None:
    """The SSRF guard at its default setting, through the real endpoint."""
    app.dependency_overrides[get_embedding_service] = _mock_embedding_service
    settings = get_settings().model_copy(update={"allow_insecure_http_sources": True})
    app.dependency_overrides[get_settings] = lambda: settings
    token = await _auth_token(pg_async_client, "ssrf@example.com")

    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": "http://169.254.169.254/latest/meta-data/"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 502
    assert "non-public" in response.json()["detail"]

    app.dependency_overrides.pop(get_settings, None)
    app.dependency_overrides.pop(get_embedding_service, None)


async def test_ingest_unsupported_format_returns_415(
    pg_async_client: AsyncClient, origin: str, local_fetch_allowed: None
) -> None:
    token = await _auth_token(pg_async_client, "badformat@example.com")

    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": f"{origin}/notes.xyz"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 415


async def test_ingest_empty_document_returns_422(
    pg_async_client: AsyncClient, origin: str, local_fetch_allowed: None
) -> None:
    token = await _auth_token(pg_async_client, "emptydoc@example.com")

    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": f"{origin}/empty.txt"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 422


async def test_ingest_upstream_404_returns_502(
    pg_async_client: AsyncClient, origin: str, local_fetch_allowed: None
) -> None:
    token = await _auth_token(pg_async_client, "upstream@example.com")

    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": f"{origin}/gone.pdf"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 502


async def test_ingest_requires_auth(pg_async_client: AsyncClient, origin: str) -> None:
    response = await pg_async_client.post(
        "/documents/ingest", json={"uri": f"{origin}/report.md"}
    )
    assert response.status_code == 401
