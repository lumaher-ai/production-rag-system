from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from production_rag.database import Base


class QueryCache(Base):
    """Deterministic query-result cache.

    Keyed by a hash of (user_id, question, filters, top_k, chunker_version,
    embedding_model). A hit returns the stored answer without re-embedding,
    retrieval, or an LLM call. Entries are invalidated on new ingestion for the
    owner and expire at ``expires_at`` (TTL).
    """

    __tablename__ = "query_cache"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # Owner scope — indexed so invalidate-on-ingest can delete a user's rows cheaply.
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Serialized QueryResponse (JSONB on Postgres, TEXT on SQLite test engine).
    response_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
