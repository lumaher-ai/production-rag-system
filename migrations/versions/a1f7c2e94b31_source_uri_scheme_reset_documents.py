"""adopt source URIs and reset documents ingested under bare filenames

Revision ID: a1f7c2e94b31
Revises: e5d3dad41210
Create Date: 2026-08-07

``documents.source`` is half the ``(user_id, source)`` ingestion identity, and it
now holds a real URI (``upload://<user_id>/report.pdf``, ``https://…``,
``gdrive://<file_id>``) rather than a bare filename.

Rows written before this change hold a bare ``report.pdf``. They cannot be
matched by the new upload path, so re-uploading an unchanged file would create a
*second* document instead of replacing the original — the identity would appear
to work while silently duplicating.

Backfilling them is possible (``upload://<user_id>/<quoted source>``) but this
system has no production data, so the honest, verifiable choice is a clean reset:
delete the documents rather than migrate rows that nobody depends on.

Deleted here:
  - ``documents`` — ``document_chunks`` follow via ON DELETE CASCADE.
  - ``query_cache`` — cached answers cite chunks that no longer exist, so
    serving them after this migration would attribute an answer to a deleted
    source.

``downgrade()`` cannot restore deleted rows; it is a no-op, and the docstring is
the record of that.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1f7c2e94b31"
down_revision: str | Sequence[str] | None = "e5d3dad41210"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Chunks cascade from documents; the cache is cleared explicitly because it
    # has no FK to either.
    op.execute("DELETE FROM query_cache;")
    op.execute("DELETE FROM documents;")


def downgrade() -> None:
    """Downgrade schema.

    Intentionally empty: the upgrade deletes data, which cannot be recreated.
    Documents must be re-ingested (their sources will then be bare filenames
    again, matching the pre-URI code this downgrade returns you to).
    """
