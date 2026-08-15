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

from arq import cron
from arq.connections import RedisSettings

from production_rag.config import get_settings
from production_rag.database import close_db, get_session, init_db
from production_rag.ingestion.failures import classify, diagnostics
from production_rag.llm.embedding_service import EmbeddingService
from production_rag.logging_config import configure_logging, get_logger
from production_rag.queue import INGEST_TASK, ArqJobQueue
from production_rag.repositories.document_repository import DocumentRepository
from production_rag.repositories.failed_ingestion_repository import (
    FailedIngestionRepository,
)
from production_rag.repositories.ingestion_job_repository import (
    IngestionJobRepository,
)
from production_rag.repositories.query_cache_repository import QueryCacheRepository
from production_rag.services.ingestion_service import IngestionService

configure_logging()
logger = get_logger(__name__)


async def ingest_document(ctx: dict[str, Any], job_id: str) -> str:
    """Ingest one document. Idempotent and resumable.

    Failures are recorded in three places, because they answer three different
    questions: the **job row** is what a user polling for status sees, the
    **dead-letter table** is what an operator counts and re-drives, and the
    **log** is what a developer reads.

    A failure is only re-raised — which is what tells arq to retry — when
    retrying could plausibly change the answer. A scanned PDF rejected by the
    quality gate is still a scanned PDF on attempt three, so those return
    normally and the job stops burning attempts (and, once OCR is in the path,
    money) on a verdict that will not move.
    """
    settings = get_settings()
    uuid = UUID(job_id)

    async with get_session() as session:
        jobs = IngestionJobRepository(session)
        failures = FailedIngestionRepository(session)
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
            classification = classify(exc)
            terminal = (
                not classification.retryable or job.attempts >= settings.ingestion_max_attempts
            )
            message = f"{type(exc).__name__}: {exc}"

            await jobs.mark_failed(job, message, reason=classification.reason.value)
            await failures.record(
                job,
                classification,
                message,
                is_terminal=terminal,
                diagnostics=diagnostics(exc),
            )
            logger.exception(
                "ingestion_job_failed",
                job_id=job_id,
                attempt=job.attempts,
                processed_chunks=job.processed_chunks,
                reason=classification.reason.value,
                stage=classification.stage.value,
                terminal=terminal,
            )
            if terminal:
                # Deliberately not re-raised: arq retries whatever raises, and
                # there is nothing here worth retrying.
                return ""
            raise

        logger.info(
            "ingestion_job_succeeded",
            job_id=job_id,
            document_id=str(document.id) if document else None,
            chunks=job.processed_chunks,
        )
        return str(document.id) if document else ""


async def recover_stale_jobs(ctx: dict[str, Any]) -> int:
    """Re-queue jobs whose worker died without recording a failure.

    arq retries a task that *raises*. It does not rescue one whose worker was
    killed outright — SIGKILL, an OOM, a pod eviction — because nothing is left
    to report the failure. Verified: after `kill -9` mid-job, the row stays
    'running' and no retry ever comes. Without this sweep, "resumable" would be
    true of the design and false of the system.

    Staleness is measured by heartbeat, not start time, so a legitimately slow
    job is left alone. Re-queued jobs resume from their committed cursor, so a
    recovered job repeats no completed work.
    """
    settings = get_settings()
    async with get_session() as session:
        jobs = IngestionJobRepository(session)
        stale = await jobs.find_stale_running(settings.ingestion_stale_after_seconds)
        if not stale:
            return 0

        queue = ArqJobQueue(ctx["redis"])
        for job in stale:
            logger.warning(
                "ingestion_job_recovered",
                job_id=str(job.id),
                processed_chunks=job.processed_chunks,
                total_chunks=job.total_chunks,
                attempts=job.attempts,
            )
            await queue.enqueue_ingestion(job.id)
        return len(stale)


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

    functions = [ingest_document, recover_stale_jobs]
    # Sweeps for jobs orphaned by a killed worker. Runs on every worker, but the
    # sweep is idempotent — a re-queued job resumes from its cursor, and a job
    # already picked up has a fresh heartbeat and so is no longer stale.
    cron_jobs = [cron(recover_stale_jobs, minute=set(range(0, 60, 2)), run_at_startup=True)]
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
