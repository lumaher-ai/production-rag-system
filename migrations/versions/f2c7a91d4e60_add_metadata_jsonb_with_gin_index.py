"""add extractive metadata as a JSONB column with a GIN index

Revision ID: f2c7a91d4e60
Revises: d4a91c6b8f52
Create Date: 2026-08-07

Adds ``metadata`` to ``documents`` and ``document_chunks``: language, document
date, document type and MIME, extracted at ingestion by
``production_rag.ingestion.metadata``.

**One JSONB column, not a column per field.** Which metadata a retrieval system
needs is not knowable in advance — this system starts with four fields and will
plausibly want author, retention class, and a sensitivity label. Typed columns
would make each of those an ALTER TABLE on a table that is expensive to rewrite,
coordinated with a deploy. JSONB makes them writes. What is given up is per-key
NOT NULL and type enforcement; validation lives in ``ingestion.metadata``
instead, which is where the extraction rules already are.

The index is ``gin (metadata jsonb_path_ops)``, not the default ``jsonb_ops``.
``jsonb_path_ops`` supports only containment (``@>``) and is markedly smaller and
faster for it. Retrieval filters exclusively by containment, so the key-existence
operators the default op-class would add are index bloat paid for on every write.

Retrieval must use ``metadata @> '{"language":"es"}'::jsonb`` to hit this index.
The intuitive ``metadata->>'language' = 'es'`` does **not** use it and degrades to
a scan; making that form indexable needs one expression index *per key*, which is
the per-field migration this column exists to avoid.

**Existing rows are not backfilled.** They get ``{}``. Backfilling would mean
re-running the extractor, which reads *normalized* text and lives in application
code a migration must not import — and the result would be metadata produced by
whatever extractor version happened to be deployed at migration time, recorded
nowhere. The consequence is correct rather than silent: ``@>`` never matches an
empty document, so an un-extracted row is excluded from filtered queries instead
of leaking into them. Unknown is not "matches".

Those rows repopulate on their next ingest or re-index, and finding them needs no
migration — which is the whole argument for this column::

    SELECT id, source FROM documents
    WHERE metadata->>'extractor_version' IS DISTINCT FROM 'extractive-v1';
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f2c7a91d4e60"
down_revision: str | Sequence[str] | None = "d4a91c6b8f52"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = ("documents", "document_chunks")


def upgrade() -> None:
    """Upgrade schema."""
    for table in TABLES:
        op.add_column(
            table,
            sa.Column(
                "metadata",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
        )
        # Raw SQL because Alembic's op.create_index cannot express an operator
        # class, and jsonb_path_ops is the point of this index rather than a
        # detail of it. IF NOT EXISTS so it converges with the identical index
        # declared on the model, which the test fixtures create directly.
        op.execute(
            f"""
            CREATE INDEX IF NOT EXISTS ix_{table}_metadata_gin
            ON {table}
            USING gin (metadata jsonb_path_ops);
            """
        )


def downgrade() -> None:
    """Downgrade schema.

    Dropping the column discards the extracted metadata, which is fine: it is
    derived, not authored, and re-ingesting reproduces it exactly. The indexes go
    with the columns.
    """
    for table in TABLES:
        op.execute(f"DROP INDEX IF EXISTS ix_{table}_metadata_gin;")
        op.drop_column(table, "metadata")
