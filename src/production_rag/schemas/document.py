from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from production_rag.ingestion.metadata import validate_metadata_filter


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
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        # The ORM attribute cannot be called `metadata` (Declarative owns that
        # name), but the column and the API field can and should be.
        validation_alias="doc_metadata",
        description="Metadata extracted at ingestion: language, document_date, "
        "doc_type, mime_type, extractor_version. Keys that could not be determined "
        "are absent rather than null. These are the keys accepted by the query "
        "filter. Empty for documents ingested before extraction existed.",
    )

    model_config = {"from_attributes": True, "populate_by_name": True}


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
        description="Restrict the search to chunks whose extracted metadata contains "
        "every key/value given. Applied inside the vector query, so top_k counts "
        "matching chunks. Flat map of scalars only — nested objects and arrays are "
        "rejected; ranges and OR are not supported. Keys come from the document's "
        "metadata (language, doc_type, document_date, mime_type).",
        examples=[{"language": "es", "doc_type": "contract"}],
    )

    @field_validator("filters")
    @classmethod
    def _check_filters(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """Reject a malformed filter at the edge, as a 422 rather than a 500.

        Delegates to the same validator retrieval uses, so a filter accepted by
        the API is one the query layer can actually run.
        """
        return validate_metadata_filter(value)


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
