from datetime import datetime, timezone
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from production_rag.database import Base


class Document(Base):
    __tablename__ = "documents"

    # Idempotency scope: the same content re-ingested by the same user under the
    # same chunker + embedding model must not be re-embedded. This composite is
    # the ingestion idempotency key, enforced in the DB so concurrent duplicate
    # ingests collapse to one row (IntegrityError → return existing).
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "content_hash",
            "chunker_version",
            "embedding_model",
            name="uq_documents_idempotency",
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
    # sha256 hex of `content` — the identity used to skip re-embedding.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # Config the embeddings were computed under; part of the idempotency key so a
    # config change forces re-embedding rather than serving stale vectors.
    chunker_version: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    # Origin pointer (uploaded filename / URI; "inline" for the JSON route).
    source: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
