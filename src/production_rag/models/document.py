from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from production_rag.database import Base

# Extractive metadata (language, document date, doc type, MIME) lives in ONE JSONB
# column per table rather than a typed column per field. Which fields matter is
# genuinely not known yet — governance labels, retention class, and author are all
# plausible next — and a typed column means an ALTER TABLE and a code change per
# field, forever. JSONB + a GIN index buys the opposite: new keys are writes, and
# filtering on them is indexed the day they first appear.
#
# The cost is real and accepted: no per-key NOT NULL, no per-key type enforcement,
# no foreign keys. Validation lives in ``ingestion.metadata`` instead of the schema.
#
# JSONB (not JSON) is required — ``@>`` containment and GIN indexing are JSONB-only.
# The variant keeps the SQLite engine the fast unit tests run on able to create
# these tables; the containment query itself is Postgres-only, as is pgvector.
MetadataJSON = JSONB().with_variant(JSON(), "sqlite")


class Document(Base):
    __tablename__ = "documents"

    # Ingestion identity: one document per (owner, source). Re-uploading the same
    # source replaces it in place — unchanged content is a no-op, edited content
    # rewrites its chunks. Enforced in the DB so concurrent uploads of the same
    # source collapse to one row (IntegrityError → return existing).
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "source",
            name="uq_documents_user_source",
        ),
        # ``jsonb_path_ops`` rather than the default ``jsonb_ops``: it indexes
        # only containment (``@>``), which is the sole operator retrieval uses,
        # and in exchange is substantially smaller and faster. Giving it up would
        # buy key-existence (``?``) queries this system does not make.
        Index(
            "ix_documents_metadata_gin",
            "metadata",
            postgresql_using="gin",
            postgresql_ops={"metadata": "jsonb_path_ops"},
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # sha256 hex of `content` — used to detect whether a re-uploaded source is
    # unchanged (no-op) or edited (rewrite chunks).
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # Config the embeddings were computed under; a change to any of these forces
    # a re-embed of an otherwise-unchanged source rather than serving stale
    # vectors. Together they are the complete set of *silent inputs* to a stored
    # vector — change one and the vectors mean something different while the
    # source text looks identical.
    #
    # ``normalizer_version`` is 'none' for rows ingested before normalization
    # existed. That value is deliberately chosen never to match a real version,
    # so those rows fail the gate and are re-processed rather than being
    # mistaken for current.
    normalizer_version: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default="none"
    )
    chunker_version: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    # Origin pointer (uploaded filename / URI). Part of the (user_id, source)
    # identity, so every document must have one.
    source: Mapped[str] = mapped_column(String(1024), nullable=False)
    # Extractive metadata produced at ingestion by ``ingestion.metadata``:
    # language, document_date, doc_type, mime_type, extractor_version. Deliberately
    # NOT one of the version columns above — it is derived from content and never
    # reaches the embedding model, so a change to the extraction rules must not
    # force a re-embed. The extractor version travels *inside* the document, which
    # is what makes "which rows predate the current extractor?" answerable with a
    # query instead of a migration.
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",  # the attribute cannot be named `metadata`: Base.metadata owns it
        MetadataJSON,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    __table_args__ = (
        Index(
            "ix_document_chunks_metadata_gin",
            "metadata",
            postgresql_using="gin",
            postgresql_ops={"metadata": "jsonb_path_ops"},
        ),
        # Declared here as well as in migration 1188038e4c5b, which created it.
        # The test fixtures build their schema with ``Base.metadata.create_all``
        # rather than by running Alembic, so an index that exists only in a
        # migration does not exist under test — and without the HNSW index every
        # test query is an exact sequential scan, which is precisely the condition
        # under which the filtered-ANN under-return bug (D3) cannot be reproduced.
        # Both definitions use the same name, and the migration uses
        # CREATE INDEX IF NOT EXISTS, so they converge rather than conflict.
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    # Source pointer to the parent document (a chunk's origin).
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Denormalized owner/tenant (= parent document's user_id). This app has no
    # separate tenant concept — user_id IS the tenant boundary. Carrying it here
    # lets retrieval enforce isolation and scope by owner without joining documents.
    owner_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Denormalized copy of the parent document's title. Retrieval returns chunks
    # ranked by similarity; carrying the title here lets query()/search_documents
    # build sources without an extra per-chunk lookup of the parent (avoids N+1).
    document_title: Mapped[str] = mapped_column(String(255), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding = mapped_column(Vector(1536), nullable=False)
    # Provenance for future citation mapping (linking an answer back to the exact
    # source span). char_start/char_end are the chunk's offsets in the parent
    # document's content and are populated at ingestion. page/section require
    # structure-aware loaders (PDF pages, Markdown headers) and stay NULL until
    # those land in Phase 2 — hence all four are nullable.
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Denormalized copy of the parent document's metadata — same reasoning as
    # owner_id and document_title above. Retrieval filters on it inside the ANN
    # query, and that query is the hot path: joining documents to reach the
    # filter would put a join between the HNSW scan and its LIMIT.
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        MetadataJSON,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
