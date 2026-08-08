from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from production_rag.config import Settings, get_settings
from production_rag.dependencies import (
    get_current_user,
    get_document_service,
    get_ingestion_job_repository,
    get_job_queue,
)
from production_rag.exceptions import FileTooLargeError, UnsupportedFileTypeError
from production_rag.ingestion.loaders import is_ingestible, unsupported_type_message
from production_rag.ingestion.sources import build_upload_uri, parse_source_uri
from production_rag.models import User
from production_rag.queue import JobQueue
from production_rag.repositories.ingestion_job_repository import IngestionJobRepository
from production_rag.schemas.document import (
    DocumentResponse,
    IngestFromUriRequest,
    QueryRequest,
    QueryResponse,
)
from production_rag.schemas.job import JobAcceptedResponse, JobStatusResponse
from production_rag.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_file(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    current_user: User = Depends(get_current_user),
    jobs: IngestionJobRepository = Depends(get_ingestion_job_repository),
    queue: JobQueue = Depends(get_job_queue),
    settings: Settings = Depends(get_settings),
) -> JobAcceptedResponse:
    """Queue an uploaded file for ingestion.

    Returns **202 with a job id**, not the finished document. Parsing and
    embedding a large document takes minutes; doing it here would hold the
    connection and a database transaction open for the duration, and lose every
    completed embedding if it failed part-way. Poll
    ``GET /documents/jobs/{job_id}`` for progress.

    Cheap checks still run synchronously — an oversized or unreadable file is
    rejected immediately rather than becoming a job that fails later.
    """
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise FileTooLargeError(
            f"File exceeds the maximum upload size of {settings.max_upload_bytes} bytes."
        )
    # Deliberately depends on deployment state, not just on the filename: a
    # spreadsheet is ingestible where Document AI is configured and a 415 where
    # it is not, and saying so now beats a 202 followed by a job that fails a
    # second later with the same message.
    if not is_ingestible(
        file.filename, file.content_type, ocr_available=settings.ocr_available
    ):
        raise UnsupportedFileTypeError(
            unsupported_type_message(file.filename, ocr_available=settings.ocr_available)
        )

    stem = Path(file.filename).stem if file.filename else "document"
    job = await jobs.create(
        user_id=current_user.id,
        # A real URI, so an uploaded report.pdf and one pulled from a URL of the
        # same name remain distinct documents.
        source=build_upload_uri(current_user.id, file.filename),
        title=(title or stem or "document")[:255],
        # The worker is a different process and cannot read this UploadFile, so
        # the bytes are staged on the job row and cleared once it succeeds.
        payload=content,
        filename=file.filename,
        content_type=file.content_type,
    )
    await queue.enqueue_ingestion(job.id)
    return JobAcceptedResponse.model_validate(job)


@router.post(
    "/ingest",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_from_uri(
    data: IngestFromUriRequest,
    current_user: User = Depends(get_current_user),
    jobs: IngestionJobRepository = Depends(get_ingestion_job_repository),
    queue: JobQueue = Depends(get_job_queue),
) -> JobAcceptedResponse:
    """Queue an external source (https:// or gdrive://) for ingestion.

    No bytes are staged: the URI can be fetched again, so the worker re-fetches
    it. That keeps a 100 MiB blob out of the database for every source that has
    an address.

    The URI is validated here so a malformed one is a 422 now, rather than a job
    that fails asynchronously.
    """
    parsed = parse_source_uri(data.uri)

    job = await jobs.create(
        user_id=current_user.id,
        source=parsed.uri,
        title=data.title[:255] if data.title else None,
    )
    await queue.enqueue_ingestion(job.id)
    return JobAcceptedResponse.model_validate(job)


@router.get("/jobs", response_model=list[JobStatusResponse])
async def list_jobs(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    jobs: IngestionJobRepository = Depends(get_ingestion_job_repository),
) -> list[JobStatusResponse]:
    records = await jobs.list_for_owner(current_user.id, limit=min(limit, 100))
    return [JobStatusResponse.model_validate(job) for job in records]


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    jobs: IngestionJobRepository = Depends(get_ingestion_job_repository),
) -> JobStatusResponse:
    """Progress and outcome of one ingestion job.

    Scoped to its owner: someone else's job id is a 404, not a 403, so the
    endpoint does not confirm that an id exists.
    """
    job = await jobs.get_for_owner(job_id, current_user.id)
    return JobStatusResponse.model_validate(job)


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    data: QueryRequest,
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> QueryResponse:
    return await service.query(
        question=data.question,
        user_id=current_user.id,
        top_k=data.top_k,
        filters=data.filters,
    )


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> list[DocumentResponse]:
    documents = await service.list_user_documents(user_id=current_user.id)
    return [DocumentResponse.model_validate(doc) for doc in documents]
