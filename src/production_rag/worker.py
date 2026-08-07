"""arq worker: runs ingestion outside the request/response cycle.

    uv run arq production_rag.worker.WorkerSettings

Each task opens its own database session. It is deliberately *not* the API's
session — the work outlives any request, and the whole point is that a caller
can poll progress while it runs, which an open request transaction would hide.
"""

# ruff: noqa: E402
# Mirrors main.py: .env must be in os.environ before litellm and friends read
# provider keys at import time.
from dotenv import load_dotenv

load_dotenv()

from typing import Any
from uuid import UUID

from arq.connections import RedisSettings

from production_rag.config import get_settings
from production_rag.database import close_db, get_session, init_db
from production_rag.llm.embedding_service import EmbeddingService
from production_rag.logging_config import configure_logging, get_logger
from production_rag.queue import INGEST_TASK
from production_rag.repositories.document_repository import DocumentRepository
from production_rag.repositories.ingestion_job_repository import (
    IngestionJobRepository,
)
from production_rag.repositories.query_cache_repository import QueryCacheRepository
from production_rag.services.ingestion_service import IngestionService

configure_logging()
logger = get_logger(__name__)


async def ingest_document(ctx: dict[str, Any], job_id: str) -> str:
    """Ingest one document. Idempotent and resumable.

    Failures are recorded on the job row *and* re-raised: the row is what a user
    polling for status sees, and the exception is what tells arq to retry. On
    the final attempt the row is the only record left, which is why the message
    is stored rather than only logged.
    """
    settings = get_settings()
    uuid = UUID(job_id)

    async with get_session() as session:
        jobs = IngestionJobRepository(session)
        job = await jobs.get(uuid)

        service = IngestionService(
            document_repository=DocumentRepository(session),
            embedding_service=EmbeddingService(model=settings.embedding_model),
            query_cache_repository=QueryCacheRepository(session),
            job_repository=jobs,
            batch_size=settings.ingestion_batch_size,
        )

        try:
            document = await service.run_job(job, settings)
        except Exception as exc:
            # Roll back whatever the failed batch left open before writing the
            # failure, or the status update would be lost with it.
            await session.rollback()
            await jobs.mark_failed(job, f"{type(exc).__name__}: {exc}")
            logger.exception(
                "ingestion_job_failed",
                job_id=job_id,
                attempt=job.attempts,
                processed_chunks=job.processed_chunks,
            )
            raise

        logger.info(
            "ingestion_job_succeeded",
            job_id=job_id,
            document_id=str(document.id) if document else None,
            chunks=job.processed_chunks,
        )
        return str(document.id) if document else ""


async def startup(ctx: dict[str, Any]) -> None:
    await init_db()
    logger.info("worker_started")


async def shutdown(ctx: dict[str, Any]) -> None:
    await close_db()
    logger.info("worker_stopped")


# arq resolves the task by function name; keep it aligned with what the API
# enqueues so a rename cannot silently orphan messages.
assert ingest_document.__name__ == INGEST_TASK


_settings = get_settings()


class WorkerSettings:
    """arq reads these as plain class attributes at worker startup."""

    functions = [ingest_document]
    on_startup = startup
    on_shutdown = shutdown

    redis_settings = RedisSettings.from_dsn(_settings.redis_url)
    # A large document legitimately takes minutes; the timeout has to exceed the
    # slowest realistic job or the worker would kill its own work mid-batch.
    job_timeout = _settings.ingestion_job_timeout_seconds
    max_tries = _settings.ingestion_max_attempts
    # Retries resume from the last committed batch rather than restarting, so
    # backoff is about waiting out transient upstream failures, not redoing work.
    retry_jobs = True
