from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentUploadRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=10, max_length=500_000)


class DocumentResponse(BaseModel):
    id: UUID
    title: str
    chunk_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class ChunkSource(BaseModel):
    chunk_id: UUID
    document_title: str
    content_preview: str
    similarity_rank: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[ChunkSource]
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
