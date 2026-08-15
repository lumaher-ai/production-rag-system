import enum


class UserRole(enum.StrEnum):
    USER = "user"
    ADMIN = "admin"


class JobStatus(enum.StrEnum):
    """Lifecycle of an ingestion job.

    ``pending`` and ``running`` are non-terminal; ``succeeded`` and ``failed``
    are terminal. A job can re-enter ``running`` from ``failed`` when the queue
    retries it — the resume cursor (``processed_chunks``) is what makes that
    safe.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED)
