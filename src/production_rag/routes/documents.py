from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from production_rag.config import Settings, get_settings
from production_rag.dependencies import get_current_user, get_document_service
from production_rag.exceptions import FileTooLargeError
from production_rag.ingestion.loaders import load_file
from production_rag.models import User
from production_rag.schemas.document import (
    DocumentResponse,
    QueryRequest,
    QueryResponse,
)
from production_rag.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_file(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    current_user: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
    settings: Settings = Depends(get_settings),
) -> DocumentResponse:
    """Ingest an uploaded file (PDF/DOCX/HTML/TXT/Markdown) into the RAG store."""
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise FileTooLargeError(
            f"File exceeds the maximum upload size of {settings.max_upload_bytes} bytes."
        )

    # Raises UnsupportedFileTypeError (415) / ValidationError (422) on bad input.
    segments = load_file(content, file.filename, file.content_type)

    # Title defaults to the filename stem; an explicit form field overrides it.
    stem = Path(file.filename).stem if file.filename else "document"
    doc_title = (title or stem or "document")[:255]

    # source is the document's identity for (owner, source) replace-on-re-upload.
    # Fall back to the title if the client omitted a filename.
    source = (file.filename or doc_title)[:1024]

    document = await service.ingest_segments(
        title=doc_title,
        segments=segments,
        user_id=current_user.id,
        source=source,
    )
    return DocumentResponse.model_validate(document)


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
