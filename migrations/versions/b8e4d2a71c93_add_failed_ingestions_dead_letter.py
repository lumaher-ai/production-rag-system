"""add a dead-letter table and a countable failure reason on jobs

Revision ID: b8e4d2a71c93
Revises: f2c7a91d4e60
Create Date: 2026-08-08

Ingestion failures were recorded in exactly one place: a free-text ``error``
column on ``ingestion_jobs``, overwritten by every attempt. That loses two
things.

**History.** Attempt three's message replaces attempt two's. A document that
timed out twice and then hit a parse error is indistinguishable from one that
hit a parse error immediately, which is precisely the distinction that tells a
flaky origin apart from a broken file.

**Countability.** ``error`` holds strings like
``LowTextYieldError: This document yielded 12 characters per page across 340
pages...``. It embeds a filename and a page count, so "what is our ingestion
failure rate, by cause" is not a query anyone can write against it. Decision A6
asks for exactly that number as its proof.

So this adds ``failed_ingestions`` — one append-only row per failed attempt,
carrying a small closed-vocabulary ``reason``, the pipeline ``stage``, the
attempt number, and an ``is_terminal`` flag that separates "will be retried"
from "needs a human" — plus ``ingestion_jobs.failure_reason``, the same code on
the job row so a status poller can branch on the cause instead of pattern
matching a message.

**Why job_id is ON DELETE SET NULL, not CASCADE.** Same reasoning as
``ingestion_jobs.document_id``: deleting the job should not erase the record
that ingestion was attempted and failed. The identifying columns (``source``,
``filename``, ``content_type``) are denormalized onto the row for that reason —
a dead-letter record readable only by joining to a row that may be gone is a
record that stops working exactly when it is needed. ``user_id`` is CASCADE,
because a deleted tenant's failures are their data.

**Why the index is composite.** ``(user_id, is_terminal, created_at)`` is the
shape the operator endpoint queries and the only shape it queries; no path here
filters on ``is_terminal`` without also scoping to an owner.

**No backfill.** Failures that happened before this table existed were never
recorded in a recoverable form — ``error`` holds only the last attempt of each,
with no stage, no attempt number, and no way to tell whether it was terminal.
Synthesizing rows from it would put guesses in a table whose entire value is
that its numbers are trustworthy.

Downgrade discards the failure history. That is acceptable: it is diagnostic
data, not a source of truth — the documents it describes were never ingested,
and the jobs that failed are still there in ``ingestion_jobs``.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "b8e4d2a71c93"
down_revision: str | Sequence[str] | None = "f2c7a91d4e60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "failed_ingestions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=True),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("source", sa.String(length=1024), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=True),
        sa.Column("content_type", sa.String(length=255), nullable=True),
        sa.Column("reason", sa.String(length=32), nullable=False),
        sa.Column("stage", sa.String(length=16), nullable=False),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column(
            "is_terminal", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column(
            "diagnostics",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["ingestion_jobs.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_failed_ingestions_job_id", "failed_ingestions", ["job_id"])
    op.create_index("ix_failed_ingestions_user_id", "failed_ingestions", ["user_id"])
    op.create_index("ix_failed_ingestions_reason", "failed_ingestions", ["reason"])
    op.create_index("ix_failed_ingestions_is_terminal", "failed_ingestions", ["is_terminal"])
    op.create_index(
        "ix_failed_ingestions_owner_terminal",
        "failed_ingestions",
        ["user_id", "is_terminal", "created_at"],
    )

    # Nullable with no default: a job that has not failed has no reason, and
    # 'none'-style sentinels would have to be filtered out of every count.
    op.add_column(
        "ingestion_jobs", sa.Column("failure_reason", sa.String(length=32), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema. Discards failure history — see the note above."""
    op.drop_column("ingestion_jobs", "failure_reason")
    op.drop_index("ix_failed_ingestions_owner_terminal", table_name="failed_ingestions")
    op.drop_index("ix_failed_ingestions_is_terminal", table_name="failed_ingestions")
    op.drop_index("ix_failed_ingestions_reason", table_name="failed_ingestions")
    op.drop_index("ix_failed_ingestions_user_id", table_name="failed_ingestions")
    op.drop_index("ix_failed_ingestions_job_id", table_name="failed_ingestions")
    op.drop_table("failed_ingestions")
