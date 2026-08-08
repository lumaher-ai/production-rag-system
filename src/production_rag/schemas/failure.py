from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import AliasChoices, BaseModel, Field

# The ORM column is `id`; the API calls it `failure_id` so a caller holding a
# job id, a document id and a failure id can tell them apart. Mirrors the
# `job_id` treatment in schemas/job.py.
_FAILURE_ID = Field(validation_alias=AliasChoices("id", "failure_id"))


class FailureResponse(BaseModel):
    """One failed ingestion attempt, as an operator needs to read it."""

    failure_id: UUID = _FAILURE_ID
    job_id: UUID | None = Field(
        default=None,
        description="The job that failed. Null once that job has been deleted — "
        "the record of the attempt outlives it.",
    )
    source: str
    filename: str | None = None
    reason: str = Field(
        description="Countable failure cause: low_text_yield, fetch_failed, "
        "unsupported_type, ocr_not_configured, parse_error, internal, ..."
    )
    stage: str = Field(description="Where it failed: fetch, parse, quality_gate, ocr, embed, ...")
    error: str = Field(description="The human-readable message.")
    attempt: int
    is_terminal: bool = Field(
        description="True when no further attempt is coming — either the retry "
        "budget is spent or retrying cannot change the outcome. These are the "
        "ones that need a human."
    )
    diagnostics: dict[str, Any] = Field(
        default_factory=dict,
        description="Measurements from the failure, when it had any — e.g. the "
        "chars-per-page numbers behind a low_text_yield rejection.",
    )
    created_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}
