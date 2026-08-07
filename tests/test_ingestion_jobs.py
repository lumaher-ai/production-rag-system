"""Asynchronous ingestion: the endpoint hands off, the worker finishes.

The behaviour that matters here is **resume**. Committing per batch is what
turns a mid-job failure from "re-embed everything" into "continue from chunk
N", and the resume path is the one that silently degrades into duplicate or
missing chunks if the cursor and the chunk indices ever disagree. Several tests
below exist specifically to pin that down.
"""

from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.config import get_settings
from production_rag.ingestion.loaders import ExtractedSegment
from production_rag.llm.embedding_service import EmbeddingService
from production_rag.models import User
from production_rag.models.document import Document, DocumentChunk
from production_rag.models.enums import JobStatus
from production_rag.repositories.document_repository import DocumentRepository
from production_rag.repositories.ingestion_job_repository import (
    IngestionJobRepository,
)
from production_rag.repositories.query_cache_repository import QueryCacheRepository
from production_rag.services.auth_service import hash_password
from production_rag.services.document_service import build_chunks
from production_rag.services.ingestion_service import IngestionService

pytestmark = pytest.mark.asyncio(loop_scope="module")

# Long enough to split into many chunks at CHUNK_SIZE=1000, so batching and
# resume have something to actually do.
LONG_TEXT = "Vector search relies on approximate nearest neighbour indexes. " * 400


def _mock_embeddings(fail_after: int | None = None) -> EmbeddingService:
    """Embedding service that can be made to fail once, mid-document.

    ``fail_after`` counts *chunks embedded across all calls*, so the failure
    lands inside a batch boundary rather than neatly between documents — which
    is the case resume has to survive.
    """
    mock = AsyncMock(spec=EmbeddingService)
    state = {"count": 0}

    async def embed_batch(texts: list[str]) -> list[list[float]]:
        state["count"] += len(texts)
        if fail_after is not None and state["count"] > fail_after:
            raise RuntimeError("embedding provider exploded")
        return [[0.1] * 1536 for _ in texts]

    mock.embed_batch.side_effect = embed_batch
    mock.embed_text.side_effect = lambda text: [0.1] * 1536
    mock.model = "text-embedding-3-small"
    return mock


def _service(
    session: AsyncSession, embeddings: EmbeddingService, batch_size: int = 10
) -> IngestionService:
    return IngestionService(
        document_repository=DocumentRepository(session),
        embedding_service=embeddings,
        query_cache_repository=QueryCacheRepository(session),
        job_repository=IngestionJobRepository(session),
        batch_size=batch_size,
    )


async def _make_user(session: AsyncSession, email: str) -> User:
    user = User(
        name="Jobs",
        email=email,
        hashed_password=hash_password("pw"),
        role="user",
    )
    session.add(user)
    await session.commit()
    return user


async def _count_chunks(session: AsyncSession, document_id: UUID) -> int:
    result = await session.execute(
        select(func.count())
        .select_from(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
    )
    return int(result.scalar_one())


async def _auth_token(client: AsyncClient, email: str) -> str:
    await client.post(
        "/auth/signup",
        json={"name": "Jobs", "email": email, "password": "securepass123"},
    )
    login = await client.post(
        "/auth/login", json={"email": email, "password": "securepass123"}
    )
    return login.json()["access_token"]


# ─── The endpoint hands off instead of doing the work ───


async def test_upload_returns_202_with_a_job_id(
    pg_async_client: AsyncClient, job_queue, pg_session: AsyncSession
) -> None:
    token = await _auth_token(pg_async_client, "async-upload@example.com")

    response = await pg_async_client.post(
        "/documents/upload",
        files={"file": ("notes.txt", LONG_TEXT.encode(), "text/plain")},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == JobStatus.PENDING.value
    assert body["source"].startswith("upload://")
    # The work was handed to the queue, not done inline.
    assert [str(j) for j in job_queue.enqueued] == [body["job_id"]]

    # No document exists yet — that is the whole point of 202.
    documents = await pg_session.execute(
        select(Document).where(Document.source == body["source"])
    )
    assert documents.scalar_one_or_none() is None


async def test_ingest_by_uri_returns_202_and_stages_no_payload(
    pg_async_client: AsyncClient, job_queue, pg_session: AsyncSession
) -> None:
    """A fetchable source needs no staged bytes — the worker re-fetches it."""
    token = await _auth_token(pg_async_client, "async-uri@example.com")

    response = await pg_async_client.post(
        "/documents/ingest",
        json={"uri": "https://example.com/report.pdf"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 202
    jobs = IngestionJobRepository(pg_session)
    job = await jobs.get(UUID(response.json()["job_id"]))
    assert job.payload is None
    assert job.source == "https://example.com/report.pdf"


async def test_upload_still_rejects_bad_input_synchronously(
    pg_async_client: AsyncClient, job_queue
) -> None:
    """Cheap checks stay in the request: a 415 now beats a job that fails later."""
    token = await _auth_token(pg_async_client, "async-badtype@example.com")

    response = await pg_async_client.post(
        "/documents/upload",
        files={"file": ("archive.xyz", b"binary", "application/octet-stream")},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 415
    assert job_queue.enqueued == []  # nothing queued for work that cannot succeed


# ─── The worker completes the job ───


async def test_worker_run_completes_job_and_creates_document(
    pg_session: AsyncSession,
) -> None:
    user = await _make_user(pg_session, "worker-ok@example.com")
    jobs = IngestionJobRepository(pg_session)
    job = await jobs.create(
        user_id=user.id,
        source=f"upload://{user.id}/notes.txt",
        title="Notes",
        payload=LONG_TEXT.encode(),
        filename="notes.txt",
        content_type="text/plain",
    )

    service = _service(pg_session, _mock_embeddings())
    document = await service.run_job(job, get_settings())

    assert document is not None
    assert job.status == JobStatus.SUCCEEDED.value
    assert job.document_id == document.id
    assert job.total_chunks and job.total_chunks > 1
    assert job.processed_chunks == job.total_chunks
    assert await _count_chunks(pg_session, document.id) == job.total_chunks
    # The staged payload is released once the work is durable.
    assert job.payload is None


async def test_progress_is_observable_while_running(pg_session: AsyncSession) -> None:
    """processed_chunks advances per batch, not only at the end."""
    user = await _make_user(pg_session, "worker-progress@example.com")
    jobs = IngestionJobRepository(pg_session)
    job = await jobs.create(
        user_id=user.id,
        source=f"upload://{user.id}/p.txt",
        payload=LONG_TEXT.encode(),
        filename="p.txt",
        content_type="text/plain",
    )

    # Spy on the exact repository instance the service will use, so the
    # assertion is about real checkpoints rather than a parallel object.
    seen: list[int] = []
    original = jobs.advance

    async def spy(j, processed):  # noqa: ANN001
        seen.append(processed)
        return await original(j, processed)

    jobs.advance = spy  # type: ignore[method-assign]
    service = IngestionService(
        document_repository=DocumentRepository(pg_session),
        embedding_service=_mock_embeddings(),
        query_cache_repository=QueryCacheRepository(pg_session),
        job_repository=jobs,
        batch_size=10,
    )
    await service.run_job(job, get_settings())

    # Multiple checkpoints, strictly increasing, ending at the total.
    assert len(seen) > 1, f"expected several batch checkpoints, got {seen}"
    assert seen == sorted(seen)
    assert seen[-1] == job.total_chunks


# ─── Resume: the reason this design exists ───


async def test_failure_preserves_completed_work(pg_session: AsyncSession) -> None:
    """A mid-job failure keeps its committed chunks and its cursor."""
    user = await _make_user(pg_session, "worker-fail@example.com")
    jobs = IngestionJobRepository(pg_session)
    job = await jobs.create(
        user_id=user.id,
        source=f"upload://{user.id}/f.txt",
        payload=LONG_TEXT.encode(),
        filename="f.txt",
        content_type="text/plain",
    )

    # Fail partway through, after at least two batches have committed.
    service = _service(pg_session, _mock_embeddings(fail_after=25), batch_size=10)
    with pytest.raises(RuntimeError, match="exploded"):
        await service.run_job(job, get_settings())

    await pg_session.rollback()  # as the worker does before recording failure
    await jobs.mark_failed(job, "embedding provider exploded")

    assert job.status == JobStatus.FAILED.value
    # The cursor survives — this is the work a retry will not repeat.
    assert job.processed_chunks == 20
    assert job.payload is not None  # kept, so the retry needs no re-upload
    assert job.document_id is None


async def test_retry_resumes_and_produces_exactly_one_clean_document(
    pg_session: AsyncSession,
) -> None:
    """The core guarantee: resume yields no duplicates and no gaps.

    Compared against the chunk count a clean run would produce, so the assertion
    is about correctness of the final document rather than internal bookkeeping.
    """
    user = await _make_user(pg_session, "worker-resume@example.com")
    jobs = IngestionJobRepository(pg_session)
    job = await jobs.create(
        user_id=user.id,
        source=f"upload://{user.id}/r.txt",
        payload=LONG_TEXT.encode(),
        filename="r.txt",
        content_type="text/plain",
    )

    _, expected_chunks = build_chunks([ExtractedSegment(text=LONG_TEXT)])
    expected = len(expected_chunks)

    # Attempt 1 dies partway.
    failing = _service(pg_session, _mock_embeddings(fail_after=25), batch_size=10)
    with pytest.raises(RuntimeError):
        await failing.run_job(job, get_settings())
    await pg_session.rollback()
    await jobs.mark_failed(job, "boom")
    partial = job.processed_chunks
    assert 0 < partial < expected

    # Attempt 2 succeeds.
    healthy = _service(pg_session, _mock_embeddings(), batch_size=10)
    document = await healthy.run_job(job, get_settings())

    assert document is not None
    assert job.status == JobStatus.SUCCEEDED.value
    assert job.processed_chunks == expected
    # No duplicates, no gap: exactly the clean-run count.
    assert await _count_chunks(pg_session, document.id) == expected
    # And chunk_index is a complete 0..n-1 sequence across the resume boundary.
    rows = await pg_session.execute(
        select(DocumentChunk.chunk_index)
        .where(DocumentChunk.document_id == document.id)
        .order_by(DocumentChunk.chunk_index)
    )
    assert [r[0] for r in rows.all()] == list(range(expected))


async def test_resume_re_embeds_only_the_remaining_chunks(
    pg_session: AsyncSession,
) -> None:
    """Resume must not re-pay for work already committed."""
    user = await _make_user(pg_session, "worker-cost@example.com")
    jobs = IngestionJobRepository(pg_session)
    job = await jobs.create(
        user_id=user.id,
        source=f"upload://{user.id}/c.txt",
        payload=LONG_TEXT.encode(),
        filename="c.txt",
        content_type="text/plain",
    )

    failing = _service(pg_session, _mock_embeddings(fail_after=25), batch_size=10)
    with pytest.raises(RuntimeError):
        await failing.run_job(job, get_settings())
    await pg_session.rollback()
    await jobs.mark_failed(job, "boom")
    already_done = job.processed_chunks

    second = _mock_embeddings()
    await _service(pg_session, second, batch_size=10).run_job(job, get_settings())

    embedded_on_retry = sum(
        len(call.args[0]) for call in second.embed_batch.call_args_list
    )
    assert embedded_on_retry == job.total_chunks - already_done


async def test_chunker_change_restarts_instead_of_splicing(
    pg_session: AsyncSession,
) -> None:
    """A cursor from a different chunker is meaningless — start over.

    Resuming across a chunker change would interleave two different chunk lists
    into one document, which no later check would catch.
    """
    user = await _make_user(pg_session, "worker-chunker@example.com")
    jobs = IngestionJobRepository(pg_session)
    job = await jobs.create(
        user_id=user.id,
        source=f"upload://{user.id}/v.txt",
        payload=LONG_TEXT.encode(),
        filename="v.txt",
        content_type="text/plain",
    )
    # Simulate progress recorded under a chunker that no longer exists.
    job.processed_chunks = 15
    job.chunker_version = "recursive-char-v0-ancient"
    await pg_session.commit()

    service = _service(pg_session, _mock_embeddings(), batch_size=10)
    document = await service.run_job(job, get_settings())

    assert document is not None
    assert job.chunker_version != "recursive-char-v0-ancient"
    # Rebuilt in full, not continued from the stale cursor.
    assert await _count_chunks(pg_session, document.id) == job.total_chunks


async def test_unchanged_source_completes_without_embedding(
    pg_session: AsyncSession,
) -> None:
    """Idempotency still short-circuits — a re-queued job costs nothing."""
    user = await _make_user(pg_session, "worker-idem@example.com")
    jobs = IngestionJobRepository(pg_session)
    source = f"upload://{user.id}/same.txt"

    first_job = await jobs.create(
        user_id=user.id,
        source=source,
        payload=LONG_TEXT.encode(),
        filename="same.txt",
        content_type="text/plain",
    )
    document = await _service(pg_session, _mock_embeddings()).run_job(
        first_job, get_settings()
    )
    assert document is not None

    second_embeddings = _mock_embeddings()
    second_job = await jobs.create(
        user_id=user.id,
        source=source,
        payload=LONG_TEXT.encode(),
        filename="same.txt",
        content_type="text/plain",
    )
    again = await _service(pg_session, second_embeddings).run_job(
        second_job, get_settings()
    )

    assert again is not None and again.id == document.id
    assert second_job.status == JobStatus.SUCCEEDED.value
    second_embeddings.embed_batch.assert_not_called()


# ─── Status endpoint ───


async def test_status_endpoint_reports_progress_and_document(
    pg_async_client: AsyncClient, pg_session: AsyncSession, job_queue
) -> None:
    token = await _auth_token(pg_async_client, "status@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    accepted = await pg_async_client.post(
        "/documents/upload",
        files={"file": ("s.txt", LONG_TEXT.encode(), "text/plain")},
        headers=headers,
    )
    job_id = accepted.json()["job_id"]

    pending = await pg_async_client.get(f"/documents/jobs/{job_id}", headers=headers)
    assert pending.status_code == 200
    assert pending.json()["status"] == JobStatus.PENDING.value
    assert pending.json()["processed_chunks"] == 0
    assert pending.json()["document_id"] is None

    # Run the worker's half against the same row.
    jobs = IngestionJobRepository(pg_session)
    job = await jobs.get(UUID(job_id))
    await _service(pg_session, _mock_embeddings()).run_job(job, get_settings())

    done = await pg_async_client.get(f"/documents/jobs/{job_id}", headers=headers)
    body = done.json()
    assert body["status"] == JobStatus.SUCCEEDED.value
    assert body["document_id"] is not None
    assert body["processed_chunks"] == body["total_chunks"] > 0


async def test_status_of_another_users_job_is_404(
    pg_async_client: AsyncClient, job_queue
) -> None:
    """Not 403 — confirming the id exists would leak another user's job."""
    owner_token = await _auth_token(pg_async_client, "owner-job@example.com")
    accepted = await pg_async_client.post(
        "/documents/upload",
        files={"file": ("o.txt", LONG_TEXT.encode(), "text/plain")},
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    job_id = accepted.json()["job_id"]

    intruder_token = await _auth_token(pg_async_client, "intruder-job@example.com")
    response = await pg_async_client.get(
        f"/documents/jobs/{job_id}",
        headers={"Authorization": f"Bearer {intruder_token}"},
    )

    assert response.status_code == 404


async def test_unknown_job_is_404(pg_async_client: AsyncClient, job_queue) -> None:
    token = await _auth_token(pg_async_client, "unknown-job@example.com")
    response = await pg_async_client.get(
        f"/documents/jobs/{uuid4()}", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 404


async def test_jobs_list_is_scoped_to_owner(
    pg_async_client: AsyncClient, job_queue
) -> None:
    token = await _auth_token(pg_async_client, "listjobs@example.com")
    headers = {"Authorization": f"Bearer {token}"}
    await pg_async_client.post(
        "/documents/upload",
        files={"file": ("l.txt", LONG_TEXT.encode(), "text/plain")},
        headers=headers,
    )

    response = await pg_async_client.get("/documents/jobs", headers=headers)

    assert response.status_code == 200
    assert len(response.json()) >= 1
    assert all(job["status"] in {s.value for s in JobStatus} for job in response.json())


async def test_jobs_endpoints_require_auth(pg_async_client: AsyncClient) -> None:
    assert (await pg_async_client.get("/documents/jobs")).status_code == 401
    assert (await pg_async_client.get(f"/documents/jobs/{uuid4()}")).status_code == 401

