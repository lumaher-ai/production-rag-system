from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from production_rag.database import Base
from production_rag.models.document import MetadataJSON


class FailedIngestion(Base):
    """One record per failed ingestion attempt — the dead-letter surface.

    ``ingestion_jobs`` already knows a job failed, but it knows it in the way a
    status light knows: one free-text ``error`` column, overwritten by each
    attempt. That loses two things this table keeps.

    **History.** Attempt three's message replaces attempt two's, so the sequence
    "timeout, timeout, then a parse error" is indistinguishable from a single
    parse error. Here every attempt is a row.

    **Countability.** ``error`` embeds filenames and page numbers, so "what is
    our ingestion failure rate, by cause" is not a query you can write against
    it. ``reason`` is a small closed vocabulary, and that is the whole point:
    it is the column a GROUP BY lands on.

    Rows are append-only. Nothing updates them, and the operator surface reads
    ``is_terminal`` to separate "this will be retried" from "this needs a human".
    """

    __tablename__ = "failed_ingestions"

    # The shape the operator endpoint queries: this user's terminal failures,
    # newest first. Composite rather than three separate indexes because no
    # query here filters on is_terminal without also filtering on the owner.
    __table_args__ = (
        Index(
            "ix_failed_ingestions_owner_terminal",
            "user_id",
            "is_terminal",
            "created_at",
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)

    # SET NULL rather than CASCADE, matching ingestion_jobs.document_id: deleting
    # the job should not erase the record that ingestion was attempted and
    # failed. The denormalized columns below are what keep the row readable once
    # this goes null.
    job_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("ingestion_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # CASCADE, unlike job_id: a deleted user's failures are their data, and the
    # tenant boundary outranks the audit trail.
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ─── What was being ingested, copied rather than joined ───
    # Denormalized so the row survives its job and still says what it was about.
    # A dead-letter record that can only be read by joining to a row that may be
    # gone is a dead-letter record that stops working exactly when it matters.
    source: Mapped[str] = mapped_column(String(1024), nullable=False)
    filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # ─── Why it failed ───
    # A FailureReason value. Stored as a string, not a PG enum, matching
    # ingestion_jobs.status: adding a reason should be a deploy, not a migration.
    reason: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(16), nullable=False)
    # The human-readable message. Truncated at the same 2000 chars as
    # ingestion_jobs.error — a stack-trace-length string here is a log line that
    # wandered into a database.
    error: Mapped[str] = mapped_column(Text, nullable=False)

    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    # True when no further attempt is coming: either the retry budget is spent,
    # or the failure is one that retrying cannot change (a scanned PDF is still
    # a scanned PDF on attempt three). This is the flag that makes the table a
    # dead-letter queue rather than a log.
    is_terminal: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )

    # Measurements from the failure, when it had any — today the quality gate's
    # chars-per-page numbers. JSONB rather than columns for the same reason
    # document metadata is: which measurements matter is not settled, and a
    # column per measurement is an ALTER TABLE per measurement forever.
    diagnostics: Mapped[dict[str, Any]] = mapped_column(
        MetadataJSON, nullable=False, default=dict, server_default="{}"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
