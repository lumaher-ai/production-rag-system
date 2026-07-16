"""add hnsw index on document_chunks embedding

Revision ID: 1188038e4c5b
Revises: 0557189f61a3
Create Date: 2026-07-15 23:03:39.470247

Adds an HNSW approximate-nearest-neighbour index on ``document_chunks.embedding``.

Before this migration every similarity query was an exact, sequential scan over
all chunks (O(n) per query). HNSW turns retrieval into an approximate graph walk
that stays fast as the corpus grows.

Operator class ``vector_cosine_ops`` is chosen to match the retrieval query,
which orders by cosine distance (``embedding <=> query`` via
``DocumentChunk.embedding.cosine_distance(...)``). The operator class MUST match
the distance used at query time or the planner will ignore the index.

``m`` and ``ef_construction`` are pgvector's defaults, set explicitly so the
build-time recall/latency trade-off is documented rather than implicit:
  - m = 16              : edges per node (higher = better recall, more memory)
  - ef_construction = 64: candidate list size while building (higher = better
                          recall, slower build)
Query-time recall is tuned separately via ``hnsw.ef_search`` at the session level.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1188038e4c5b"
down_revision: str | Sequence[str] | None = "0557189f61a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "ix_document_chunks_embedding_hnsw"


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {INDEX_NAME}
        ON document_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(f"DROP INDEX IF EXISTS {INDEX_NAME};")
