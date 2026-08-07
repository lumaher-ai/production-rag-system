"""add normalizer_version to documents and ingestion_jobs

Revision ID: d4a91c6b8f52
Revises: c3b8d5f1e207
Create Date: 2026-08-07

Text is now normalized (Unicode NFKC + whitespace) before chunking and embedding.
Normalization rules are a *silent input* to every stored vector: change them and
the vectors mean something different while the source text looks identical.

``normalizer_version`` records which rules produced a row, joining
``chunker_version`` and ``embedding_model`` in the idempotency gate. Without it,
a rules change would leave the gate reporting "unchanged" while serving vectors
computed under rules that no longer apply — mixing incompatible embeddings in one
index with nothing raising an error.

**Existing rows default to 'none', not to the current version.** This is the
whole point of the migration and the easy thing to get wrong. Those documents
were ingested before normalization existed, so their chunks were embedded from
un-normalized text. Stamping them with the current version would mark them as
up to date and permanently hide them from the gate — exactly the failure this
column exists to prevent. 'none' is chosen to never match a real version string,
so they fail the gate and are re-processed.

``ingestion_jobs.normalizer_version`` is nullable, matching ``chunker_version``:
a job records it once its plan is computed, and a mid-flight change invalidates
the resume cursor (resuming would splice two normalizations into one document).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4a91c6b8f52"
down_revision: str | Sequence[str] | None = "c3b8d5f1e207"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Sentinel for content processed before normalization existed. Must never equal
# a real NORMALIZER_VERSION, or pre-normalization rows would pass the gate.
PRE_NORMALIZATION = "none"


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "documents",
        sa.Column(
            "normalizer_version",
            sa.String(length=64),
            nullable=False,
            server_default=PRE_NORMALIZATION,
        ),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column("normalizer_version", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("ingestion_jobs", "normalizer_version")
    op.drop_column("documents", "normalizer_version")
