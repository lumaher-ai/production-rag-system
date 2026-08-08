"""The boundary between the API and the worker.

A thin wrapper over arq rather than direct calls, for two reasons: routes depend
on an interface instead of a Redis connection, and tests swap in a recorder so
the suite needs no live Redis to assert that a job was enqueued.
"""

from typing import Protocol
from uuid import UUID

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings

from production_rag.logging_config import get_logger

logger = get_logger(__name__)

# The worker-side task name. A string rather than a direct import so the API
# process never imports the worker module (and with it the whole ingestion
# stack) just to enqueue.
INGEST_TASK = "ingest_document"


class JobQueue(Protocol):
    async def enqueue_ingestion(self, job_id: UUID) -> None:
        """Hand a persisted ingestion job to a worker."""
        ...


class ArqJobQueue:
    """Production queue, backed by Redis.

    Only the job id crosses the wire — everything the worker needs is already in
    the ``ingestion_jobs`` row. That keeps the message tiny and makes the
    database the single source of truth for job state, so a message lost or
    replayed cannot contradict what the row says.
    """

    def __init__(self, pool: ArqRedis) -> None:
        self._pool = pool

    async def enqueue_ingestion(self, job_id: UUID) -> None:
        # Deliberately no arq ``_job_id``. Setting it to a value derived from the
        # ingestion job id makes arq treat a *re-enqueue as a duplicate and drop
        # it silently* — which breaks the retry of an orphaned job, the one case
        # the retry path exists for. Deduplication is not needed here anyway:
        # every request mints a fresh job row with its own UUID, and running the
        # same job twice is harmless because ingestion is idempotent (content
        # hash) and resumable (the progress cursor).
        await self._pool.enqueue_job(INGEST_TASK, str(job_id))
        logger.info("ingestion_job_enqueued", job_id=str(job_id))


async def create_queue_pool(redis_url: str) -> ArqRedis:
    return await create_pool(RedisSettings.from_dsn(redis_url))
