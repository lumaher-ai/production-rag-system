"""Resumable ingestion: parse → chunk → embed in batches → persist.

This is the half of ingestion that used to live inside the HTTP request. It runs
in a worker now, and its defining property is that **it can be interrupted**.
Work is committed in batches, and the job row records how far it got, so a
retry continues instead of starting over.

The trade-off that buys: ingestion is no longer atomic. A document under active
ingestion is partially visible to retrieval. For a new document that is
harmless — every chunk written is real and searchable. For a *replace*, the old
chunks are dropped on the first attempt, so that document has reduced content
until the job finishes. The clean fix is an ``is_ready`` flag on ``documents``
filtered at retrieval, deferred because it forces a join into the hot ANN query
and interacts with the open filtered-ANN issue (decision D3).

``run_job`` and ``ingest_now`` share one persistence path. The only difference
is whether a job row is being updated, which also decides transaction
boundaries: with a job, each batch commits (progress must be observable while
in flight); without one, the caller owns the transaction.
"""

from pathlib import Path
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from production_rag.config import Settings
from production_rag.exceptions import ValidationError
from production_rag.ingestion.connectors import fetch_source
from production_rag.ingestion.idempotency import content_hash
from production_rag.ingestion.loaders import ExtractedSegment, load_file
from production_rag.ingestion.sources import parse_source_uri
from production_rag.llm.client import count_tokens
from production_rag.llm.embedding_service import EmbeddingService
from production_rag.logging_config import get_logger
from production_rag.models.document import Document
from production_rag.models.ingestion_job import IngestionJob
from production_rag.repositories.document_repository import DocumentRepository
from production_rag.repositories.ingestion_job_repository import IngestionJobRepository
from production_rag.repositories.query_cache_repository import QueryCacheRepository
from production_rag.services.document_service import (
    CHUNKER_VERSION,
    PreparedChunk,
    build_chunks,
)

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 100


class IngestionService:
    def __init__(
        self,
        document_repository: DocumentRepository,
        embedding_service: EmbeddingService,
        query_cache_repository: QueryCacheRepository,
        job_repository: IngestionJobRepository | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._documents = document_repository
        self._embeddings = embedding_service
        self._query_cache = query_cache_repository
        self._jobs = job_repository
        self._batch_size = max(1, batch_size)

    # ─── Entry points ───

    async def run_job(self, job: IngestionJob, settings: Settings) -> Document | None:
        """Drive a job to completion. Safe to call again after a failure.

        Returns the document, or ``None`` when the source was unchanged and no
        work was needed. Raises on failure so the queue can retry — recording
        the failure on the job row is the caller's (worker's) responsibility, so
        that a worker killed outright is still distinguishable from one that
        caught an error.
        """
        if self._jobs is None:
            raise RuntimeError("run_job requires a job repository")

        await self._jobs.mark_running(job)

        # Resume is only valid if the chunker has not moved since the completed
        # work was produced; otherwise the cursor points into a different chunk
        # list and would splice two chunkings together.
        if job.processed_chunks > 0 and job.chunker_version != CHUNKER_VERSION:
            logger.warning(
                "ingestion_job_chunker_changed",
                job_id=str(job.id),
                was=job.chunker_version,
                now=CHUNKER_VERSION,
                discarded_chunks=job.processed_chunks,
            )
            await self._jobs.reset_progress(job)

        segments, filename = await self._materialize(job, settings)

        return await self._persist(
            # The caller's title wins; otherwise derive it from the filename the
            # fetch actually resolved, which for a URI source is only known now.
            title=job.title or _fallback_title(filename, job.source),
            segments=segments,
            user_id=job.user_id,
            source=job.source,
            job=job,
        )

    async def ingest_now(
        self,
        title: str,
        segments: list[ExtractedSegment],
        user_id: UUID,
        source: str,
    ) -> Document | None:
        """Ingest synchronously, without a job row.

        Same persistence path as ``run_job`` — not a second implementation. Used
        for seeding and for direct in-process ingestion where the caller already
        owns the transaction.
        """
        return await self._persist(
            title=title,
            segments=segments,
            user_id=user_id,
            source=source,
            job=None,
        )

    # ─── Stages ───

    async def _materialize(
        self, job: IngestionJob, settings: Settings
    ) -> tuple[list[ExtractedSegment], str | None]:
        """Get the document's bytes and parse them into segments.

        Uploads carry their bytes on the job row (the worker cannot read the
        request's UploadFile). Connector sources carry only a URI and are
        re-fetched here, which keeps a 100 MiB blob out of the database for
        every source that can be fetched again.

        Returns the filename alongside the segments: for a URI source it is
        discovered by the fetch (from Content-Disposition or the URL path) and
        is not knowable when the job is created.
        """
        if job.payload is not None:
            content = job.payload
            filename = job.filename
            content_type = job.content_type
        else:
            parsed = parse_source_uri(job.source)
            fetched = await fetch_source(parsed, settings)
            content = fetched.content
            filename = fetched.filename
            content_type = fetched.content_type

        return load_file(content, filename, content_type), filename

    async def _persist(
        self,
        title: str,
        segments: list[ExtractedSegment],
        user_id: UUID,
        source: str,
        job: IngestionJob | None,
    ) -> Document | None:
        full_content, chunks = build_chunks(segments)
        if not chunks:
            raise ValidationError("No extractable text found in the document.")

        chash = content_hash(full_content)
        embedding_model = self._embeddings.model
        resuming = job is not None and job.processed_chunks > 0

        existing = await self._documents.find_document_by_source(
            user_id=user_id, source=source
        )

        # Unchanged content short-circuits before any embedding spend. Skipped
        # when resuming: the hash already matches (the document row was written
        # on the first attempt) but its chunks are only partly there.
        if (
            not resuming
            and existing is not None
            and existing.content_hash == chash
            and existing.chunker_version == CHUNKER_VERSION
            and existing.embedding_model == embedding_model
        ):
            logger.info(
                "document_ingest_skipped",
                document_id=str(existing.id),
                title=title,
                reason="unchanged_source",
            )
            if job is not None and self._jobs is not None:
                await self._jobs.set_plan(job, len(chunks), CHUNKER_VERSION)
                await self._jobs.advance(job, len(chunks))
                await self._jobs.mark_succeeded(job, existing.id)
            return existing

        logger.info(
            "document_chunked",
            title=title,
            total_chars=len(full_content),
            segment_count=len(segments),
            chunk_count=len(chunks),
        )

        document = await self._upsert_document(
            existing=existing,
            resuming=resuming,
            title=title,
            full_content=full_content,
            user_id=user_id,
            source=source,
            chash=chash,
            embedding_model=embedding_model,
            chunk_count=len(chunks),
        )
        if document is None:  # lost a concurrent race; the winner owns this source
            return await self._documents.find_document_by_source(
                user_id=user_id, source=source
            )

        if job is not None and self._jobs is not None:
            await self._jobs.set_plan(job, len(chunks), CHUNKER_VERSION)

        await self._embed_and_store(
            document=document,
            chunks=chunks,
            user_id=user_id,
            start_index=job.processed_chunks if job is not None else 0,
            job=job,
        )

        # New content invalidates this user's cached answers.
        await self._query_cache.delete_by_user(user_id)

        if job is not None and self._jobs is not None:
            await self._jobs.mark_succeeded(job, document.id)

        logger.info(
            "document_ingested",
            document_id=str(document.id),
            title=title,
            chunks_created=len(chunks),
        )
        return document

    async def _upsert_document(
        self,
        existing: Document | None,
        resuming: bool,
        title: str,
        full_content: str,
        user_id: UUID,
        source: str,
        chash: str,
        embedding_model: str,
        chunk_count: int,
    ) -> Document | None:
        """Create or replace the document row. ``None`` means a race was lost."""
        if existing is not None:
            if not resuming:
                # Only on a fresh attempt: a resumed job's partial chunks are the
                # work we are trying to preserve, so they must not be deleted.
                await self._documents.delete_chunks_by_document_id(existing.id)
            document = await self._documents.update_document_content(
                document=existing,
                title=title,
                content=full_content,
                chunk_count=chunk_count,
                content_hash=chash,
                chunker_version=CHUNKER_VERSION,
                embedding_model=embedding_model,
            )
            logger.info("document_replaced", document_id=str(document.id), title=title)
            return document

        # A concurrent ingest of the same source may have raced us to the unique
        # (user_id, source) constraint — treat that as a hit.
        try:
            async with self._documents.savepoint():
                return await self._documents.create_document(
                    title=title,
                    content=full_content,
                    user_id=user_id,
                    chunk_count=chunk_count,
                    content_hash=chash,
                    chunker_version=CHUNKER_VERSION,
                    embedding_model=embedding_model,
                    source=source,
                )
        except IntegrityError:
            logger.info("document_ingest_raced", source=source)
            return None

    async def _embed_and_store(
        self,
        document: Document,
        chunks: list[PreparedChunk],
        user_id: UUID,
        start_index: int,
        job: IngestionJob | None,
    ) -> None:
        """Embed and write chunks in batches, checkpointing after each.

        Batching is what makes the job resumable, and it also bounds each
        embedding request — the previous code sent every chunk of a document in
        one call with no cap, which fails outright on a large enough document
        (decision C4).

        With a job, ``advance`` commits the batch's chunk rows and the cursor
        together, so the cursor can never claim progress that was not persisted.
        Without one, the caller's transaction stays in charge.
        """
        total = len(chunks)
        for offset in range(start_index, total, self._batch_size):
            batch = chunks[offset : offset + self._batch_size]
            embeddings = await self._embeddings.embed_batch([c.text for c in batch])

            for position, (chunk, embedding) in enumerate(
                zip(batch, embeddings, strict=True)
            ):
                await self._documents.create_chunk(
                    document_id=document.id,
                    owner_id=user_id,
                    document_title=document.title,
                    # Absolute index, so a resumed batch lands where it belongs.
                    chunk_index=offset + position,
                    content=chunk.text,
                    token_count=count_tokens(chunk.text),
                    embedding=embedding,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    page=chunk.page,
                    section=chunk.section,
                )

            processed = offset + len(batch)
            if job is not None and self._jobs is not None:
                await self._jobs.advance(job, processed)
                logger.info(
                    "ingestion_batch_committed",
                    job_id=str(job.id),
                    processed_chunks=processed,
                    total_chunks=total,
                )


def _fallback_title(filename: str | None, source: str) -> str:
    """Title for a job whose caller supplied none.

    The filename stem, so ``report.md`` titles a document "report" whether it
    arrived by upload or by URL — the extension is a transport detail, not part
    of what the document is called.
    """
    if filename:
        stem = Path(filename).stem
        if stem:
            return stem[:255]
    stem = Path(source.rsplit("/", 1)[-1]).stem
    return (stem or "document")[:255]
