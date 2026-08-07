from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    id: UUID
    title: str
    chunk_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


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
