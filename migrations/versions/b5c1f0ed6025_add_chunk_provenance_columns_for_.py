"""add chunk provenance columns for citation mapping

Revision ID: b5c1f0ed6025
Revises: 757e706cd39a
Create Date: 2026-07-15 23:36:44.979459

Adds provenance columns to ``document_chunks`` so an answer can later be linked
back to the exact span it came from (citation mapping):

  - char_start / char_end : the chunk's character offsets within the parent
                            document's content. Populated at ingestion going
                            forward (via the splitter's start_index).
  - page                  : source page number — requires a structure-aware
                            loader (e.g. PDF), so NULL until Phase 2.
  - section               : source section/heading — likewise Phase 2.

All four are nullable: page/section are unknown for plain-text ingestion, and
existing chunks predate offset tracking, so their values remain NULL. No
backfill is attempted — recomputing offsets would couple this migration to the
splitter configuration.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b5c1f0ed6025"
down_revision: str | Sequence[str] | None = "757e706cd39a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("document_chunks", sa.Column("char_start", sa.Integer(), nullable=True))
    op.add_column("document_chunks", sa.Column("char_end", sa.Integer(), nullable=True))
    op.add_column("document_chunks", sa.Column("page", sa.Integer(), nullable=True))
    op.add_column("document_chunks", sa.Column("section", sa.String(length=255), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("document_chunks", "section")
    op.drop_column("document_chunks", "page")
    op.drop_column("document_chunks", "char_end")
    op.drop_column("document_chunks", "char_start")
