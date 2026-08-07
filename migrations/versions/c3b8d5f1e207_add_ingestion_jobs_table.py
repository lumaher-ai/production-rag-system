"""add ingestion_jobs table for asynchronous document ingestion

Revision ID: c3b8d5f1e207
Revises: a1f7c2e94b31
Create Date: 2026-08-07

Ingestion used to run inside the HTTP request: parse, embed every chunk, write,
all in one transaction. A large document held the connection and the transaction
open for a minute or more, and a failure part-way discarded every embedding
already paid for.

This table makes the work durable and observable. A request records a job and
returns; a worker drives it and updates the row as it goes.

Two columns carry most of the design:

  - ``payload`` stages uploaded bytes, because the worker is a separate process
    and cannot read the request's UploadFile. It is NULL for https:// and
    gdrive:// sources, which the worker re-fetches from ``source`` instead, and
    it is cleared once the job succeeds so large blobs do not accumulate.

  - ``processed_chunks`` is both the progress counter and the **resume cursor**.
    Chunking is deterministic for a given (content, chunker_version), so a retry
    re-derives the same chunk list and skips this many — which is why resuming
    needs no persisted chunk text. ``chunker_version`` records the chunker the
    completed work was produced under; if it moves between attempts the cursor
    is meaningless and the job restarts from zero.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3b8d5f1e207"
down_revision: str | Sequence[str] | None = "a1f7c2e94b31"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("source", sa.String(length=1024), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("payload", sa.LargeBinary(), nullable=True),
        sa.Column("filename", sa.String(length=512), nullable=True),
        sa.Column("content_type", sa.String(length=255), nullable=True),
        sa.Column("total_chunks", sa.Integer(), nullable=True),
        sa.Column("processed_chunks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("chunker_version", sa.String(length=64), nullable=True),
        sa.Column("document_id", sa.Uuid(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        # Liveness signal. A worker killed outright never records a failure, so
        # a stale heartbeat is the only way to distinguish a dead job from a
        # slow one and hand it back to the queue.
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        # SET NULL, not CASCADE: deleting a document should not erase the record
        # that it was once ingested.
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="SET NULL"),
    )
    # The status endpoints always filter by owner; the worker and any operator
    # view filter by status.
    op.create_index("ix_ingestion_jobs_user_id", "ingestion_jobs", ["user_id"])
    op.create_index("ix_ingestion_jobs_status", "ingestion_jobs", ["status"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_ingestion_jobs_status", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_user_id", table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")
