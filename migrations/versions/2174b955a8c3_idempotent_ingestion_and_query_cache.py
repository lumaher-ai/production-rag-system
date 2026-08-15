"""idempotent ingestion and query cache

Revision ID: 2174b955a8c3
Revises: b5c1f0ed6025
Create Date: 2026-07-18 00:00:00.000000

Adds content-hash idempotency to ingestion and a deterministic query-result cache.

documents:
  - content_hash / chunker_version / embedding_model : the idempotency identity.
    A repeat ingest of the same content by the same user under the same chunker +
    embedding model is skipped (no re-embedding). A UNIQUE(user_id, content_hash,
    chunker_version, embedding_model) makes concurrent duplicate ingests collapse
    to one row. content_hash is added nullable, backfilled via pgcrypto
    (encode(digest(content,'sha256'),'hex'), byte-for-byte equal to Python's
    hashlib.sha256(content.encode()).hexdigest()), then set NOT NULL. The two
    version columns use a server_default of the current config to backfill
    existing rows, dropped afterwards so the app must supply them going forward.
  - source : origin pointer (uploaded filename / URI); nullable — unknown for
    pre-existing rows and the JSON route stores "inline".

document_chunks:
  - owner_id : denormalized owner/tenant (= parent document's user_id). Added
    nullable, backfilled from the parent (single set-based UPDATE, mirroring the
    document_title denormalization), then NOT NULL. Lets retrieval enforce tenant
    isolation without joining documents.

query_cache: new table keyed by a deterministic idempotency key; a hit skips
embedding, retrieval, and the LLM call. Rows expire at expires_at (TTL) and are
deleted for a user on new ingestion.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2174b955a8c3"
down_revision: str | Sequence[str] | None = "b5c1f0ed6025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Current config used to backfill existing documents (kept in sync with
# EmbeddingService / DocumentService constants at the time of this migration).
_CHUNKER_VERSION = "recursive-char-v1"
_EMBEDDING_MODEL = "text-embedding-3-small"


def upgrade() -> None:
    """Upgrade schema."""
    # --- documents: idempotency identity ---
    # Version columns carry a server_default so existing rows backfill, then we
    # drop the default so the application must supply the value on new rows.
    op.add_column(
        "documents",
        sa.Column(
            "chunker_version",
            sa.String(length=64),
            nullable=False,
            server_default=_CHUNKER_VERSION,
        ),
    )
    op.add_column(
        "documents",
        sa.Column(
            "embedding_model",
            sa.String(length=128),
            nullable=False,
            server_default=_EMBEDDING_MODEL,
        ),
    )
    op.alter_column("documents", "chunker_version", server_default=None)
    op.alter_column("documents", "embedding_model", server_default=None)

    op.add_column("documents", sa.Column("source", sa.String(length=1024), nullable=True))

    # content_hash: add nullable, backfill via pgcrypto, then enforce NOT NULL.
    op.add_column("documents", sa.Column("content_hash", sa.String(length=64), nullable=True))
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
    op.execute("UPDATE documents SET content_hash = encode(digest(content, 'sha256'), 'hex');")
    op.alter_column("documents", "content_hash", nullable=False)

    op.create_unique_constraint(
        "uq_documents_idempotency",
        "documents",
        ["user_id", "content_hash", "chunker_version", "embedding_model"],
    )

    # --- document_chunks: denormalized owner/tenant ---
    op.add_column("document_chunks", sa.Column("owner_id", sa.Uuid(), nullable=True))
    op.execute(
        """
        UPDATE document_chunks AS c
        SET owner_id = d.user_id
        FROM documents AS d
        WHERE c.document_id = d.id;
        """
    )
    op.alter_column("document_chunks", "owner_id", nullable=False)
    op.create_foreign_key(
        "fk_document_chunks_owner_id_users",
        "document_chunks",
        "users",
        ["owner_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "ix_document_chunks_owner_id",
        "document_chunks",
        ["owner_id"],
    )

    # --- query_cache: deterministic query-result cache ---
    op.create_table(
        "query_cache",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("response_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_query_cache_idempotency_key",
        "query_cache",
        ["idempotency_key"],
        unique=True,
    )
    op.create_index("ix_query_cache_user_id", "query_cache", ["user_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_query_cache_user_id", table_name="query_cache")
    op.drop_index("ix_query_cache_idempotency_key", table_name="query_cache")
    op.drop_table("query_cache")

    op.drop_index("ix_document_chunks_owner_id", table_name="document_chunks")
    op.drop_constraint("fk_document_chunks_owner_id_users", "document_chunks", type_="foreignkey")
    op.drop_column("document_chunks", "owner_id")

    op.drop_constraint("uq_documents_idempotency", "documents", type_="unique")
    op.drop_column("documents", "content_hash")
    op.drop_column("documents", "source")
    op.drop_column("documents", "embedding_model")
    op.drop_column("documents", "chunker_version")
