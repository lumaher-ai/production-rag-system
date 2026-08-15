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

from production_rag.config import Settings, get_settings
from production_rag.dependencies import get_embedding_service
from production_rag.llm.embedding_service import EmbeddingService
from production_rag.main import app
from production_rag.models.document import Document, DocumentChunk
from production_rag.models.enums import JobStatus
from tests._file_builders import make_pdf
from tests._jobs import drain_expecting_failure, drain_jobs

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
def local_fetch_allowed(job_queue) -> Iterator[Settings]:
    """Allow fetching the loopback test origin, and mock embeddings.

    Yields the relaxed Settings because the worker half runs outside FastAPI —
    `drain_jobs` needs the same permissive config the route was given, and a
    dependency override does not reach it.
    """
    settings = get_settings().model_copy(
        update={"allow_private_network_sources": True, "allow_insecure_http_sources": True}
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedding_service] = _mock_embedding_service
    yield settings
    app.dependency_overrides.pop(get_settings, None)
    app.dependency_overrides.pop(get_embedding_service, None)


async def _own_document(client: AsyncClient, token: str, source: str) -> dict:
    """Find a document by source among *this caller's* documents.

    Identity is (user_id, source), so a bare source lookup can match several
    users' rows. Going through the authenticated list keeps it owner-scoped.
    """
    listed = await client.get("/documents", headers={"Authorization": f"Bearer {token}"})
    return next(d for d in listed.json() if d["source"] == source)


async def _document_id_for(session: AsyncSession, source: str) -> UUID:
    """Look up a worker-created document id by source URI (single-owner tests)."""
    result = await session.execute(select(Document.id).where(Document.source == source))
    return result.scalars().first()


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
    pg_async_client: AsyncClient,
    pg_session: AsyncSession,
    origin: str,
    job_queue,
    local_fetch_allowed: Settings,
) -> None:
    token = await _auth_token(pg_async_client, "url@example.com")

    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": f"{origin}/report.md"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 202
    # The fetched URL *is* the identity — stored verbatim, not reduced to a name.
    assert response.json()["source"] == f"{origin}/report.md"

    await drain_jobs(pg_session, job_queue, settings=local_fetch_allowed)

    document = await _own_document(pg_async_client, token, f"{origin}/report.md")
    assert document["chunk_count"] > 0
    assert document["title"] == "report"  # filename stem from the URL


async def test_ingest_pdf_url_keeps_page_provenance(
    pg_async_client: AsyncClient,
    pg_session: AsyncSession,
    origin: str,
    job_queue,
    local_fetch_allowed: Settings,
) -> None:
    """Connectors add sources, not formats: the PDF loader runs unchanged."""
    token = await _auth_token(pg_async_client, "pdfurl@example.com")

    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": f"{origin}/paper.pdf"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 202
    await drain_jobs(pg_session, job_queue, settings=local_fetch_allowed)
    doc_id = await _document_id_for(pg_session, f"{origin}/paper.pdf")

    result = await pg_session.execute(
        select(DocumentChunk.page).where(DocumentChunk.document_id == doc_id)
    )
    assert {row[0] for row in result.all()} == {1, 2}


async def test_ingest_title_override(
    pg_async_client: AsyncClient,
    pg_session: AsyncSession,
    origin: str,
    job_queue,
    local_fetch_allowed: Settings,
) -> None:
    token = await _auth_token(pg_async_client, "ingesttitle@example.com")

    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": f"{origin}/report.md", "title": "Q3 Numbers"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 202
    await drain_jobs(pg_session, job_queue, settings=local_fetch_allowed)

    document = await _own_document(pg_async_client, token, f"{origin}/report.md")
    assert document["title"] == "Q3 Numbers"


async def test_reingesting_same_uri_replaces_rather_than_duplicates(
    pg_async_client: AsyncClient,
    pg_session: AsyncSession,
    origin: str,
    job_queue,
    local_fetch_allowed: Settings,
) -> None:
    """The point of making source a URI: it is a stable identity."""
    token = await _auth_token(pg_async_client, "repeat@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    body = {"uri": f"{origin}/report.md"}

    first = await pg_async_client.post("/documents/ingest", json=body, headers=headers)
    await drain_jobs(pg_session, job_queue, settings=local_fetch_allowed)
    second = await pg_async_client.post("/documents/ingest", json=body, headers=headers)
    await drain_jobs(pg_session, job_queue, settings=local_fetch_allowed)

    assert first.status_code == second.status_code == 202
    # Two jobs, but only ever one document for this source *for this owner*.
    listed = await pg_async_client.get("/documents", headers=headers)
    matching = [d for d in listed.json() if d["source"] == f"{origin}/report.md"]
    assert len(matching) == 1


async def test_upload_and_url_of_same_name_stay_distinct(
    pg_async_client: AsyncClient,
    pg_session: AsyncSession,
    origin: str,
    job_queue,
    local_fetch_allowed: Settings,
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

    assert uploaded.status_code == fetched.status_code == 202
    assert uploaded.json()["source"].startswith("upload://")
    assert fetched.json()["source"] == f"{origin}/report.md"

    await drain_jobs(pg_session, job_queue, settings=local_fetch_allowed)

    # Two distinct documents, because the scheme distinguishes them.
    upload_doc = await _own_document(pg_async_client, token, uploaded.json()["source"])
    url_doc = await _own_document(pg_async_client, token, fetched.json()["source"])
    assert upload_doc["id"] != url_doc["id"]


# ─── Error surfaces ───


async def test_ingest_rejects_bare_filename_with_422(
    pg_async_client: AsyncClient, local_fetch_allowed: Settings
) -> None:
    token = await _auth_token(pg_async_client, "bare@example.com")

    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": "reporte.pdf"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 422


async def test_ingest_rejects_unshipped_scheme_with_422(
    pg_async_client: AsyncClient, local_fetch_allowed: Settings
) -> None:
    token = await _auth_token(pg_async_client, "s3@example.com")

    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": "s3://bucket/report.pdf"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 422


async def test_ingest_blocks_private_address(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    """The SSRF guard still fires — now in the worker, where the fetch happens."""
    app.dependency_overrides[get_embedding_service] = _mock_embedding_service
    settings = get_settings().model_copy(update={"allow_insecure_http_sources": True})
    app.dependency_overrides[get_settings] = lambda: settings
    token = await _auth_token(pg_async_client, "ssrf@example.com")

    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": "http://169.254.169.254/latest/meta-data/"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 202

    job = await drain_expecting_failure(pg_session, job_queue, settings)
    assert job.status == JobStatus.FAILED.value
    assert "non-public" in job.error
    assert job.document_id is None  # nothing was ingested

    app.dependency_overrides.pop(get_settings, None)
    app.dependency_overrides.pop(get_embedding_service, None)


async def test_ingest_unsupported_format_fails_the_job(
    pg_async_client: AsyncClient,
    pg_session: AsyncSession,
    origin: str,
    job_queue,
    local_fetch_allowed: Settings,
) -> None:
    """The format is unknowable until the fetch, so this fails on the job.

    Unlike an upload, where the filename is in the request and a 415 is returned
    synchronously, a URL's content type is only known once it has been
    retrieved — so the caller gets 202 and then a failed job with the reason.
    """
    token = await _auth_token(pg_async_client, "badformat@example.com")

    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": f"{origin}/notes.xyz"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 202

    job = await drain_expecting_failure(pg_session, job_queue, local_fetch_allowed)
    assert job.status == JobStatus.FAILED.value
    assert "Unsupported file type" in job.error


async def test_ingest_empty_document_fails_the_job(
    pg_async_client: AsyncClient,
    pg_session: AsyncSession,
    origin: str,
    job_queue,
    local_fetch_allowed: Settings,
) -> None:
    token = await _auth_token(pg_async_client, "emptydoc@example.com")

    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": f"{origin}/empty.txt"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 202

    job = await drain_expecting_failure(pg_session, job_queue, local_fetch_allowed)
    assert job.status == JobStatus.FAILED.value
    assert "No extractable text" in job.error


async def test_ingest_upstream_404_fails_the_job(
    pg_async_client: AsyncClient,
    pg_session: AsyncSession,
    origin: str,
    job_queue,
    local_fetch_allowed: Settings,
) -> None:
    token = await _auth_token(pg_async_client, "upstream@example.com")

    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": f"{origin}/gone.pdf"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 202

    job = await drain_expecting_failure(pg_session, job_queue, local_fetch_allowed)
    assert job.status == JobStatus.FAILED.value
    assert "404" in job.error


async def test_ingest_requires_auth(pg_async_client: AsyncClient, origin: str) -> None:
    response = await pg_async_client.post("/documents/ingest", json={"uri": f"{origin}/report.md"})
    assert response.status_code == 401
