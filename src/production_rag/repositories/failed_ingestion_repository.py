from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.exceptions import NotFoundError
from production_rag.ingestion.failures import FailureClassification
from production_rag.models.failed_ingestion import FailedIngestion
from production_rag.models.ingestion_job import IngestionJob


class FailedIngestionNotFoundError(NotFoundError):
    pass


class FailedIngestionRepository:
    """Persistence for the dead-letter records.

    ``record`` commits immediately, for the same reason every
    ``IngestionJobRepository`` transition does: it is written on the failure
    path, right after the caller rolled back whatever the failed work left open.
    A failure record that rides on someone else's transaction is a failure record
    that disappears exactly when the thing it describes goes wrong.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self,
        job: IngestionJob,
        classification: FailureClassification,
        error: str,
        *,
        is_terminal: bool,
        diagnostics: dict[str, Any] | None = None,
    ) -> FailedIngestion:
        """Append one failed attempt.

        The job's identifying fields are copied rather than referenced so the
        row still says what it was about after the job is gone.
        """
        failure = FailedIngestion(
            job_id=job.id,
            user_id=job.user_id,
            source=job.source,
            filename=job.filename,
            content_type=job.content_type,
            reason=classification.reason.value,
            stage=classification.stage.value,
            error=error[:2000],
            attempt=job.attempts,
            is_terminal=is_terminal,
            diagnostics=diagnostics or {},
        )
        self._session.add(failure)
        await self._session.commit()
        await self._session.refresh(failure)
        return failure

    async def list_for_owner(
        self,
        user_id: UUID,
        *,
        terminal_only: bool = True,
        limit: int = 20,
    ) -> list[FailedIngestion]:
        """This user's failures, newest first.

        ``terminal_only`` defaults to True because the operator question is
        "what needs me?", and a failure that is about to be retried does not.
        Passing False gives the full attempt history, which is what you want
        when diagnosing *why* something took three tries.
        """
        stmt = select(FailedIngestion).where(FailedIngestion.user_id == user_id)
        if terminal_only:
            stmt = stmt.where(FailedIngestion.is_terminal.is_(True))
        result = await self._session.execute(
            stmt.order_by(FailedIngestion.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())

    async def get_for_owner(self, failure_id: UUID, user_id: UUID) -> FailedIngestion:
        """Fetch scoped to its owner.

        Someone else's failure id raises not-found rather than forbidden:
        confirming an id exists would leak that another user has it.
        """
        result = await self._session.execute(
            select(FailedIngestion).where(
                FailedIngestion.id == failure_id,
                FailedIngestion.user_id == user_id,
            )
        )
        failure = result.scalar_one_or_none()
        if failure is None:
            raise FailedIngestionNotFoundError(f"Failed ingestion {failure_id} not found")
        return failure
