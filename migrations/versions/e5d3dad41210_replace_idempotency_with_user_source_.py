"""replace idempotency with user_source uniqueness

Revision ID: e5d3dad41210
Revises: 2174b955a8c3
Create Date: 2026-07-21 11:01:00.583748

Makes (user_id, source) the ingestion identity so re-uploading an edited file
replaces the existing document instead of adding a second copy.

  - Drop UNIQUE(user_id, content_hash, chunker_version, embedding_model). Identity
    is no longer content-based: two differently-named files with identical bytes
    are now two documents, and an edited re-upload of the same source is an
    in-place replace (delete old chunks → write new chunks → update the row).
  - source : NOT NULL — every document now has an origin (the uploaded filename).
  - Add UNIQUE(user_id, source) : one document per source per user.

content_hash / chunker_version / embedding_model columns are kept; content_hash
now serves the "unchanged re-upload → no-op" check within a source.

NOTE: source was previously nullable. This migration assumes no NULL/duplicate
source rows exist (clean-slate policy — wipe documents/document_chunks before
running). No backfill is performed.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e5d3dad41210"
down_revision: str | Sequence[str] | None = "2174b955a8c3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint("uq_documents_idempotency", "documents", type_="unique")
    op.alter_column(
        "documents",
        "source",
        existing_type=sa.String(length=1024),
        nullable=False,
    )
    op.create_unique_constraint(
        "uq_documents_user_source",
        "documents",
        ["user_id", "source"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("uq_documents_user_source", "documents", type_="unique")
    op.alter_column(
        "documents",
        "source",
        existing_type=sa.String(length=1024),
        nullable=True,
    )
    op.create_unique_constraint(
        "uq_documents_idempotency",
        "documents",
        ["user_id", "content_hash", "chunker_version", "embedding_model"],
    )
