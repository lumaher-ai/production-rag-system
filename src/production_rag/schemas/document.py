from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    id: UUID
    title: str
    chunk_count: int
    created_at: datetime
    source: str = Field(
        ...,
        description="Canonical source URI — the document's identity together with "
        "its owner. e.g. upload://<user_id>/report.pdf, https://…, gdrive://<file_id>. "
        "Re-ingesting the same source replaces this document rather than duplicating it.",
    )

    model_config = {"from_attributes": True}


class IngestFromUriRequest(BaseModel):
    uri: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Source URI to pull from: https://…, http://… (only when "
        "explicitly enabled), or gdrive://<file_id>. upload:// is not fetchable — "
        "use POST /documents/upload for direct file uploads.",
        examples=["https://example.com/report.pdf", "gdrive://1A2b3C4d5E6f7G8h9I"],
    )
    title: str | None = Field(
        default=None,
        max_length=255,
        description="Overrides the title derived from the fetched filename.",
    )


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Metadata/context filter. Part of the cache key; retrieval "
        "filtering on it is not applied yet (reserved for forward-compat).",
    )


class ChunkSource(BaseModel):
    chunk_id: UUID
    document_title: str
    content_preview: str
    similarity_rank: int
    similarity_score: float = Field(
        ...,
        description="Cosine similarity to the query in [-1, 1]; 1.0 = identical. Higher is better.",
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[ChunkSource]
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool = Field(
        default=False,
        description="True when served from the deterministic query cache "
        "(no embedding, retrieval, or LLM call was made).",
    )
