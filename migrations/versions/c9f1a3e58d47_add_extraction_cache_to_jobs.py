"""cache a remote extraction on the job row

Revision ID: c9f1a3e58d47
Revises: b8e4d2a71c93
Create Date: 2026-08-08

Ingestion gained a remote extractor (Document AI), and that breaks an
assumption the resume design was built on.

Resume works because ``processed_chunks`` is a cursor into a chunk list a retry
can re-derive exactly: chunking is deterministic for a given (content,
normalizer, chunker), so the retry skips N chunks and lands where the previous
attempt stopped. Local parsing preserves that — the same bytes give the same
segments every time. A call to a versioned, model-backed service does not
guarantee it, and a re-extraction that drifted even slightly would splice two
different parses into one document, with nothing downstream able to detect it.

Caching the extraction on the job row makes the guarantee local again: a
resumed job re-uses the exact segments the first attempt worked from, so the
cursor still points where it did.

The second reason is money. ``ingestion_max_attempts`` is 3 and Layout Parser
bills roughly $10 per 1,000 pages, so without this a 400-page scan that fails
while embedding is billed for OCR three times to produce the same text.

Stored as JSONB rather than a table because it is a cache with the lifetime of
one job — written once, read at most a few times, and dropped in
``mark_succeeded`` alongside ``payload``. It is nullable and only ever populated
on the Document AI path: local parsing is free and deterministic, so caching it
would cost a write and buy nothing.

Downgrade drops the column. Safe: it holds no source of truth, only a copy of
something re-derivable from bytes that are still staged.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "c9f1a3e58d47"
down_revision: str | Sequence[str] | None = "b8e4d2a71c93"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "ingestion_jobs",
        sa.Column(
            "extracted_segments",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Downgrade schema. Discards a cache, not data — see the note above."""
    op.drop_column("ingestion_jobs", "extracted_segments")
