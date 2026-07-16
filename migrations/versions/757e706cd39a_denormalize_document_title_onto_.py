"""denormalize document_title onto document_chunks

Revision ID: 757e706cd39a
Revises: 1188038e4c5b
Create Date: 2026-07-15 23:30:05.230244

Adds a denormalized ``document_title`` column to ``document_chunks``.

Retrieval returns chunks ranked by similarity; building the response previously
required a separate ``SELECT`` on ``documents`` for every chunk to fetch its
title (the classic N+1 pattern). Copying the title onto each chunk at ingestion
time removes those per-chunk lookups.

The column is added nullable, backfilled from the parent document, then made
NOT NULL so new rows must always carry a title. Backfill is a single set-based
UPDATE (not a row-by-row loop), so it stays cheap on existing data.

Trade-off: a document rename would leave these copies stale. Titles are only
set at ingestion today (no update path exists), so this is safe now; a future
rename endpoint must also update ``document_chunks.document_title``.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "757e706cd39a"
down_revision: str | Sequence[str] | None = "1188038e4c5b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1. Add nullable so the column can be created on a table with existing rows.
    op.add_column(
        "document_chunks",
        sa.Column("document_title", sa.String(length=255), nullable=True),
    )
    # 2. Backfill each chunk's title from its parent document (single UPDATE).
    op.execute(
        """
        UPDATE document_chunks AS c
        SET document_title = d.title
        FROM documents AS d
        WHERE c.document_id = d.id;
        """
    )
    # 3. Enforce NOT NULL now that every row has a value.
    op.alter_column("document_chunks", "document_title", nullable=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("document_chunks", "document_title")
