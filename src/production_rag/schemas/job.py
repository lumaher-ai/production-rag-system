from datetime import datetime
from uuid import UUID

from pydantic import AliasChoices, BaseModel, Field

from production_rag.models.enums import JobStatus

# The ORM column is `id`; the API calls it `job_id` so a caller holding several
# kinds of id can tell them apart. AliasChoices lets both populate the field, so
# model_validate works directly on the ORM row.
_JOB_ID = Field(validation_alias=AliasChoices("id", "job_id"))


class JobAcceptedResponse(BaseModel):
    """Returned by the ingestion endpoints — the work is queued, not done.

    Paired with HTTP 202: the caller holds a ``job_id`` and polls for the
    outcome. There is no document id yet, because no document exists yet.
    """

    job_id: UUID = _JOB_ID
    status: JobStatus
    source: str = Field(description="Canonical source URI the job will ingest.")

    model_config = {"from_attributes": True, "populate_by_name": True}


class JobStatusResponse(BaseModel):
    job_id: UUID = _JOB_ID
    status: JobStatus
    source: str
    processed_chunks: int = Field(
        description="Chunks embedded and durably committed so far. A retry "
        "resumes from this point rather than restarting."
    )
    total_chunks: int | None = Field(
        default=None,
        description="Known only once the document has been parsed and chunked; "
        "null while the job is still pending or fetching.",
    )
    document_id: UUID | None = Field(
        default=None,
        description="Set once the job succeeds — the document is queryable from that point.",
    )
    error: str | None = Field(default=None, description="Failure message, when status is 'failed'.")
    failure_reason: str | None = Field(
        default=None,
        description="The same failure as a countable code — low_text_yield, "
        "fetch_failed, unsupported_type, ocr_not_configured, ... Branch on this "
        "rather than on `error`, whose wording is free to change.",
    )
    attempts: int
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def progress(self) -> float | None:
        if not self.total_chunks:
            return None
        return round(self.processed_chunks / self.total_chunks, 4)

    model_config = {"from_attributes": True, "populate_by_name": True}
