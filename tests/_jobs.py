"""Test helper: run the worker's half of ingestion, in-process.

Ingestion is asynchronous now — an endpoint returns 202 and a worker finishes
the job. Tests that assert on the *resulting document* therefore need to drive
the worker too. This runs the real ``IngestionService`` against the enqueued
job ids, so those tests exercise the same code path production does rather than
a synchronous shortcut kept alive for their benefit.
"""

import hashlib
import random
from unittest.mock import AsyncMock
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.config import Settings, get_settings
from production_rag.ingestion.failures import classify, diagnostics
from production_rag.llm.embedding_service import EmbeddingService
from production_rag.models.ingestion_job import IngestionJob
from production_rag.repositories.document_repository import DocumentRepository
from production_rag.repositories.failed_ingestion_repository import (
    FailedIngestionRepository,
)
from production_rag.repositories.ingestion_job_repository import (
    IngestionJobRepository,
)
from production_rag.repositories.query_cache_repository import QueryCacheRepository
from production_rag.services.ingestion_service import IngestionService

EMBEDDING_DIMS = 1536


def mock_embedding_service(model: str = "text-embedding-3-small") -> EmbeddingService:
    """A deterministic stand-in — the model name is part of the idempotency key.

    Every text maps to the *same* vector, which is fine for ingestion tests (they
    assert on rows written, not on ranking) and useless for retrieval tests: with
    all chunks equidistant, any ordering assertion passes vacuously. Those want
    ``hashed_embedding_service`` below.
    """
    mock = AsyncMock(spec=EmbeddingService)
    mock.embed_text.side_effect = lambda text: [0.1] * EMBEDDING_DIMS
    mock.embed_batch.side_effect = lambda texts: [[0.1] * EMBEDDING_DIMS for _ in texts]
    mock.model = model
    return mock


def embed_deterministically(text: str) -> list[float]:
    """A stable pseudo-random unit vector derived from ``text``.

    Distinct texts get distinct directions and identical texts get identical
    ones, so a query embedded from a chunk's own text is that chunk's nearest
    neighbour at distance ~0. That is what makes rank and recall assertions mean
    something without calling a real embedding API.

    Seeded from a content hash rather than the global RNG so the vectors do not
    depend on test ordering.
    """
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    vector = [rng.gauss(0.0, 1.0) for _ in range(EMBEDDING_DIMS)]
    norm = sum(value * value for value in vector) ** 0.5
    return [value / norm for value in vector]


def hashed_embedding_service(model: str = "text-embedding-3-small") -> EmbeddingService:
    """An embedding stand-in whose vectors actually distinguish texts."""
    mock = AsyncMock(spec=EmbeddingService)
    mock.embed_text.side_effect = embed_deterministically
    mock.embed_batch.side_effect = lambda texts: [embed_deterministically(t) for t in texts]
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
    embeddings: EmbeddingService | None = None,
) -> IngestionJob:
    """Run one queued job that is expected to fail, and return its row.

    Mirrors what ``worker.ingest_document`` does on an exception — roll back the
    partial transaction, classify the failure, write it to both the job row and
    the dead-letter table — so tests assert against the same state a real failed
    job leaves behind, including whether the failure was judged terminal.
    """
    settings = settings or get_settings()
    jobs = IngestionJobRepository(session)
    failures = FailedIngestionRepository(session)
    service = IngestionService(
        document_repository=DocumentRepository(session),
        embedding_service=embeddings or mock_embedding_service(),
        query_cache_repository=QueryCacheRepository(session),
        job_repository=jobs,
    )

    job_id = queue.enqueued.pop(0)
    job = await jobs.get(job_id)
    try:
        await service.run_job(job, settings)
    except Exception as exc:
        await session.rollback()
        classification = classify(exc)
        terminal = not classification.retryable or job.attempts >= settings.ingestion_max_attempts
        message = f"{type(exc).__name__}: {exc}"
        await jobs.mark_failed(job, message, reason=classification.reason.value)
        await failures.record(
            job,
            classification,
            message,
            is_terminal=terminal,
            diagnostics=diagnostics(exc),
        )
        return job
    raise AssertionError(f"job {job_id} was expected to fail but succeeded")
