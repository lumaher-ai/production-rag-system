"""Test helper: run the worker's half of ingestion, in-process.

Ingestion is asynchronous now — an endpoint returns 202 and a worker finishes
the job. Tests that assert on the *resulting document* therefore need to drive
the worker too. This runs the real ``IngestionService`` against the enqueued
job ids, so those tests exercise the same code path production does rather than
a synchronous shortcut kept alive for their benefit.
"""

from unittest.mock import AsyncMock
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.config import Settings, get_settings
from production_rag.llm.embedding_service import EmbeddingService
from production_rag.models.ingestion_job import IngestionJob
from production_rag.repositories.document_repository import DocumentRepository
from production_rag.repositories.ingestion_job_repository import (
    IngestionJobRepository,
)
from production_rag.repositories.query_cache_repository import QueryCacheRepository
from production_rag.services.ingestion_service import IngestionService

EMBEDDING_DIMS = 1536


def mock_embedding_service(model: str = "text-embedding-3-small") -> EmbeddingService:
    """A deterministic stand-in — the model name is part of the idempotency key."""
    mock = AsyncMock(spec=EmbeddingService)
    mock.embed_text.side_effect = lambda text: [0.1] * EMBEDDING_DIMS
    mock.embed_batch.side_effect = lambda texts: [[0.1] * EMBEDDING_DIMS for _ in texts]
    mock.model = model
    return mock


async def drain_jobs(
    session: AsyncSession,
    queue,
    embeddings: EmbeddingService | None = None,
    settings: Settings | None = None,
) -> list[UUID]:
    """Run every job the recording queue has collected, then clear it.

    Returns the ids processed. Failures propagate rather than being recorded on
    the job row: a test that drains a job expects it to succeed, and swallowing
    the exception here would turn a real break into a confusing assertion
    failure further down. Use ``drain_expecting_failure`` when the failure is
    the thing under test.
    """
    jobs = IngestionJobRepository(session)
    service = IngestionService(
        document_repository=DocumentRepository(session),
        embedding_service=embeddings or mock_embedding_service(),
        query_cache_repository=QueryCacheRepository(session),
        job_repository=jobs,
    )

    processed = list(queue.enqueued)
    queue.enqueued.clear()
    for job_id in processed:
        job = await jobs.get(job_id)
        await service.run_job(job, settings or get_settings())
    return processed


async def drain_expecting_failure(
    session: AsyncSession,
    queue,
    settings: Settings | None = None,
) -> IngestionJob:
    """Run one queued job that is expected to fail, and return its row.

    Mirrors what ``worker.ingest_document`` does on an exception — roll back the
    partial transaction, then record the failure — so tests assert against the
    same state a real failed job leaves behind.
    """
    jobs = IngestionJobRepository(session)
    service = IngestionService(
        document_repository=DocumentRepository(session),
        embedding_service=mock_embedding_service(),
        query_cache_repository=QueryCacheRepository(session),
        job_repository=jobs,
    )

    job_id = queue.enqueued.pop(0)
    job = await jobs.get(job_id)
    try:
        await service.run_job(job, settings or get_settings())
    except Exception as exc:
        await session.rollback()
        await jobs.mark_failed(job, f"{type(exc).__name__}: {exc}")
        return job
    raise AssertionError(f"job {job_id} was expected to fail but succeeded")
