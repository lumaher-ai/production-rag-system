from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.exceptions import NotFoundError
from production_rag.models.enums import JobStatus
from production_rag.models.ingestion_job import IngestionJob


class IngestionJobNotFoundError(NotFoundError):
    pass


class IngestionJobRepository:
    """Persistence for ingestion jobs.

    Every state transition commits immediately rather than riding on a caller's
    transaction. That is deliberate: the whole point of the job row is to be
    observable *while* work is in flight, so a status poll must see progress
    that an open transaction would otherwise hide.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        user_id: UUID,
        source: str,
        title: str | None = None,
        payload: bytes | None = None,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> IngestionJob:
        """Create and **commit** a pending job.

        Committing here rather than leaving it to the request is required for
        correctness, not convenience: the caller enqueues immediately afterwards,
        and a worker can pick the message up before this request would otherwise
        have committed — finding no row and failing.
        """
        job = IngestionJob(
            user_id=user_id,
            source=source,
            title=title,
            payload=payload,
            filename=filename,
            content_type=content_type,
            status=JobStatus.PENDING.value,
        )
        self._session.add(job)
        await self._session.commit()
        await self._session.refresh(job)
        return job

    async def get(self, job_id: UUID) -> IngestionJob:
        """Fetch by id without an owner check — worker use only."""
        result = await self._session.execute(
            select(IngestionJob).where(IngestionJob.id == job_id)
        )
        job = result.scalar_one_or_none()
        if job is None:
            raise IngestionJobNotFoundError(f"Ingestion job {job_id} not found")
        return job

    async def get_for_owner(self, job_id: UUID, user_id: UUID) -> IngestionJob:
        """Fetch scoped to its owner.

        A job belonging to someone else raises not-found rather than forbidden:
        confirming that an id exists would leak that another user has it.
        """
        result = await self._session.execute(
            select(IngestionJob).where(
                IngestionJob.id == job_id,
                IngestionJob.user_id == user_id,
            )
        )
        job = result.scalar_one_or_none()
        if job is None:
            raise IngestionJobNotFoundError(f"Ingestion job {job_id} not found")
        return job

    async def list_for_owner(self, user_id: UUID, limit: int = 20) -> list[IngestionJob]:
        result = await self._session.execute(
            select(IngestionJob)
            .where(IngestionJob.user_id == user_id)
            .order_by(IngestionJob.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def find_stale_running(
        self, older_than_seconds: int, limit: int = 50
    ) -> list[IngestionJob]:
        """Jobs stuck in 'running' whose worker has stopped reporting progress.

        A worker killed by SIGKILL, an OOM, or a pod eviction never reaches its
        failure handler, so its job stays 'running' with nobody working on it and
        nothing to retry it. A heartbeat older than one batch's realistic
        duration is the signal that the owning worker is gone.

        The cutoff must exceed the time one batch can take, or a slow-but-healthy
        job would be handed to a second worker and both would write the same
        chunks.
        """
        cutoff = datetime.now(UTC) - timedelta(seconds=older_than_seconds)
        result = await self._session.execute(
            select(IngestionJob)
            .where(
                IngestionJob.status == JobStatus.RUNNING.value,
                # NULL predates the heartbeat column; treat it as stale so old
                # rows are not stranded forever.
                or_(
                    IngestionJob.heartbeat_at.is_(None),
                    IngestionJob.heartbeat_at < cutoff,
                ),
            )
            .order_by(IngestionJob.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def mark_running(self, job: IngestionJob) -> IngestionJob:
        """Claim the job for this attempt.

        ``attempts`` increments here rather than on failure so that a worker
        killed mid-job (which never reaches a failure handler) is still counted.
        """
        now = datetime.now(UTC)
        job.status = JobStatus.RUNNING.value
        job.attempts += 1
        job.error = None
        job.failure_reason = None
        job.started_at = job.started_at or now
        job.heartbeat_at = now
        await self._session.commit()
        return job

    async def set_plan(
        self,
        job: IngestionJob,
        total_chunks: int,
        normalizer_version: str,
        chunker_version: str,
    ) -> IngestionJob:
        """Record the chunk count and the config it was derived under.

        Both versions are stored because either one moving invalidates the
        resume cursor — the chunk list a retry derives would not line up with
        the chunks already written.
        """
        job.total_chunks = total_chunks
        job.normalizer_version = normalizer_version
        job.chunker_version = chunker_version
        await self._session.commit()
        return job

    async def reset_progress(self, job: IngestionJob) -> IngestionJob:
        """Rewind the resume cursor to zero (chunker changed between attempts)."""
        job.processed_chunks = 0
        await self._session.commit()
        return job

    async def advance(self, job: IngestionJob, processed_chunks: int) -> IngestionJob:
        """Commit the resume cursor after a batch is durably written.

        Called once per batch, inside the same transaction as that batch's chunk
        rows, so the cursor can never claim more progress than was persisted.
        Doubles as the liveness heartbeat: a job making batch progress is
        demonstrably alive, so the sweeper leaves it alone.
        """
        job.processed_chunks = processed_chunks
        job.heartbeat_at = datetime.now(UTC)
        await self._session.commit()
        return job

    async def mark_succeeded(self, job: IngestionJob, document_id: UUID) -> IngestionJob:
        """Finish the job and release its staged payload.

        Dropping ``payload`` here is what keeps 100 MiB uploads from
        accumulating; it is safe only once the document is fully written.
        """
        job.status = JobStatus.SUCCEEDED.value
        job.document_id = document_id
        job.payload = None
        job.error = None
        job.failure_reason = None
        job.finished_at = datetime.now(UTC)
        await self._session.commit()
        return job

    async def mark_failed(
        self, job: IngestionJob, error: str, reason: str | None = None
    ) -> IngestionJob:
        """Record a failure, keeping the payload so a retry needs no re-upload.

        ``processed_chunks`` is left intact — it is the resume cursor, and
        clearing it would discard exactly the work this design exists to save.

        ``reason`` is the countable form of the same failure (a FailureReason
        value). Optional so a caller that has not classified the exception can
        still record it, rather than being forced to guess a code.
        """
        job.status = JobStatus.FAILED.value
        job.error = error[:2000]
        job.failure_reason = reason
        job.finished_at = datetime.now(UTC)
        await self._session.commit()
        return job

    async def requeue(self, job: IngestionJob) -> IngestionJob:
        """Return a failed job to 'pending' so it can be enqueued again.

        ``attempts`` resets to zero because this is a *human* deciding to try
        again, usually after fixing whatever caused the failure — carrying the
        old count over would give the retry one attempt instead of the full
        budget, and it would fail terminally for a reason that no longer holds.

        ``processed_chunks`` does not reset: it is the resume cursor, and the
        chunks it counts are still in the database.
        """
        job.status = JobStatus.PENDING.value
        job.attempts = 0
        job.error = None
        job.failure_reason = None
        job.finished_at = None
        job.heartbeat_at = None
        await self._session.commit()
        return job
