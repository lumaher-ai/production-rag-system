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
from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from production_rag.config import Settings
from production_rag.exceptions import ValidationError
from production_rag.ingestion.connectors import fetch_source
from production_rag.ingestion.idempotency import content_hash
from production_rag.ingestion.loaders import ExtractedSegment, load_file, resolve_mime
from production_rag.ingestion.metadata import METADATA_VERSION, extract_metadata
from production_rag.ingestion.normalize import NORMALIZER_VERSION
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

        # Resume is only valid if neither the normalizer nor the chunker has
        # moved since the completed work was produced. Either change makes the
        # cursor point into a different chunk list, and resuming would splice two
        # incompatible processings into one document — with nothing downstream
        # able to detect it.
        if job.processed_chunks > 0 and (
            job.chunker_version != CHUNKER_VERSION
            or job.normalizer_version != NORMALIZER_VERSION
        ):
            logger.warning(
                "ingestion_job_config_changed",
                job_id=str(job.id),
                was_chunker=job.chunker_version,
                now_chunker=CHUNKER_VERSION,
                was_normalizer=job.normalizer_version,
                now_normalizer=NORMALIZER_VERSION,
                discarded_chunks=job.processed_chunks,
            )
            await self._jobs.reset_progress(job)

        segments, filename, content_type = await self._materialize(job, settings)

        return await self._persist(
            # The caller's title wins; otherwise derive it from the filename the
            # fetch actually resolved, which for a URI source is only known now.
            title=job.title or _fallback_title(filename, job.source),
            segments=segments,
            user_id=job.user_id,
            source=job.source,
            job=job,
            mime_type=resolve_mime(filename, content_type),
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
    ) -> tuple[list[ExtractedSegment], str | None, str | None]:
        """Get the document's bytes and parse them into segments.

        Uploads carry their bytes on the job row (the worker cannot read the
        request's UploadFile). Connector sources carry only a URI and are
        re-fetched here, which keeps a 100 MiB blob out of the database for
        every source that can be fetched again.

        Returns the filename and content type alongside the segments: for a URI
        source both are discovered by the fetch (from Content-Disposition, the
        response headers, or the URL path) and are not knowable when the job is
        created. They outlive the loader dispatch because the resolved MIME is
        stored as document metadata, not just used to pick a parser.

        Three sources of bytes, in preference order:

          1. **staged payload** — a job still holding its uploaded bytes;
          2. **a fetchable URI** (``https://``, ``gdrive://``) — re-fetched, so
             the loader runs again and **page/section provenance is preserved**;
          3. **the stored document body** — the re-index fallback.

        Branch 3 exists because a re-index of an ``upload://`` document reaches
        neither of the first two: its payload was cleared when the original job
        succeeded, and ``upload://`` is deliberately not fetchable (it names
        bytes that only ever arrived over multipart). **Its cost is real and is
        logged per document**: ``Document.content`` is the *joined* segment text,
        so re-chunking it recovers no ``page``/``section`` values — those columns
        come back NULL. Documents with structure worth keeping should be
        re-uploaded rather than re-indexed through this path.
        """
        if job.payload is not None:
            content = job.payload
            filename = job.filename
            content_type = job.content_type
        else:
            parsed = parse_source_uri(job.source)
            if parsed.is_fetchable:
                fetched = await fetch_source(parsed, settings)
                content = fetched.content
                filename = fetched.filename
                content_type = fetched.content_type
            else:
                return await self._segments_from_stored_body(job)

        return load_file(content, filename, content_type), filename, content_type

    async def _segments_from_stored_body(
        self, job: IngestionJob
    ) -> tuple[list[ExtractedSegment], str | None, str | None]:
        """Rebuild segments from the persisted document body (re-index only).

        The body survives as ``Document.content``, so a document can be
        re-processed under new normalization/chunking rules without asking the
        user to re-upload. What does not survive is structure: the stored body is
        segments already joined, so this returns a single section-less,
        page-less segment and the rebuilt chunks lose that provenance. The
        content type is likewise gone — the job's payload carried it and was
        cleared — so ``metadata.mime_type`` is re-derived from the source URI's
        extension where one survives, and otherwise omitted.
        """
        existing = await self._documents.find_document_by_source(
            user_id=job.user_id, source=job.source
        )
        if existing is None:
            raise ValidationError(
                f"Cannot re-index '{job.source}': its bytes are not staged, the "
                f"source is not fetchable, and no stored document body exists."
            )

        had_structure = await self._documents.document_has_provenance(existing.id)
        logger.warning(
            "ingestion_reindex_from_stored_body",
            document_id=str(existing.id),
            source=job.source,
            provenance_lost=had_structure,
            detail=(
                "page/section provenance cannot be recovered from the stored body; "
                "re-upload the original file to restore it"
                if had_structure
                else "document had no page/section provenance to lose"
            ),
        )
        return [ExtractedSegment(text=existing.content)], existing.title, None

    async def _persist(
        self,
        title: str,
        segments: list[ExtractedSegment],
        user_id: UUID,
        source: str,
        job: IngestionJob | None,
        mime_type: str | None = None,
    ) -> Document | None:
        full_content, chunks = build_chunks(segments)
        if not chunks:
            raise ValidationError("No extractable text found in the document.")

        # Metadata is NOT extracted here. Both places that need it sit past the
        # idempotency gate below, and the gate's whole purpose is that an
        # unchanged, up-to-date document does no work — so extraction happens at
        # each use site instead. The two are mutually exclusive (the skip branch
        # returns), so this still runs at most once per call, and not at all on
        # the common no-op re-ingest.
        #
        # Today that saves a few milliseconds. It also keeps the cheap path cheap
        # if extraction ever stops being cheap: an extractor that called a model
        # would otherwise bill a request every time a re-ingest decided to do
        # nothing, and nothing about the code would look wrong.
        chash = content_hash(full_content)
        embedding_model = self._embeddings.model
        resuming = job is not None and job.processed_chunks > 0

        existing = await self._documents.find_document_by_source(
            user_id=user_id, source=source
        )

        # Unchanged content short-circuits before any embedding spend. Skipped
        # when resuming: the hash already matches (the document row was written
        # on the first attempt) but its chunks are only partly there.
        #
        # METADATA_VERSION is deliberately NOT one of these conditions. The other
        # three are silent inputs to a stored *vector* — change one and the
        # embeddings mean something different while the text looks identical, so
        # the only safe response is to re-embed. Extracted metadata never reaches
        # the embedding model, so gating on it would turn adding a keyword to a
        # marker list into a full-corpus re-embed. It is handled below instead, at
        # the cost it actually warrants: two UPDATEs.
        if (
            not resuming
            and existing is not None
            and existing.content_hash == chash
            and existing.normalizer_version == NORMALIZER_VERSION
            and existing.chunker_version == CHUNKER_VERSION
            and existing.embedding_model == embedding_model
        ):
            # Content is unchanged but the extractor may have moved since this row
            # was written. Left alone, it would keep answering filters under rules
            # that no longer apply — invisible, because nothing about a stale
            # metadata value looks like an error.
            if existing.doc_metadata.get("extractor_version") != METADATA_VERSION:
                await self._documents.refresh_metadata(
                    existing, extract_metadata(full_content, mime_type=mime_type)
                )
                # Cached answers were computed against the old metadata, so the
                # filters that produced them may no longer select the same chunks.
                await self._query_cache.delete_by_user(user_id)
                logger.info(
                    "document_metadata_refreshed",
                    document_id=str(existing.id),
                    extractor_version=METADATA_VERSION,
                )

            logger.info(
                "document_ingest_skipped",
                document_id=str(existing.id),
                title=title,
                reason="unchanged_source",
            )
            if job is not None and self._jobs is not None:
                await self._jobs.set_plan(job, len(chunks), NORMALIZER_VERSION, CHUNKER_VERSION)
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

        # Extracted from the *normalized* body, so the extractor reads the same
        # string the embedding model does rather than the loader's ligatures and
        # stray whitespace.
        metadata = extract_metadata(full_content, mime_type=mime_type)

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
            metadata=metadata,
        )
        if document is None:  # lost a concurrent race; the winner owns this source
            return await self._documents.find_document_by_source(
                user_id=user_id, source=source
            )

        if job is not None and self._jobs is not None:
            await self._jobs.set_plan(job, len(chunks), NORMALIZER_VERSION, CHUNKER_VERSION)

        await self._embed_and_store(
            document=document,
            chunks=chunks,
            user_id=user_id,
            start_index=job.processed_chunks if job is not None else 0,
            job=job,
            metadata=metadata,
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
        metadata: dict[str, Any],
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
                normalizer_version=NORMALIZER_VERSION,
                chunker_version=CHUNKER_VERSION,
                embedding_model=embedding_model,
                doc_metadata=metadata,
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
                    normalizer_version=NORMALIZER_VERSION,
                    chunker_version=CHUNKER_VERSION,
                    embedding_model=embedding_model,
                    source=source,
                    doc_metadata=metadata,
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
        metadata: dict[str, Any],
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
                    # Denormalized from the document so the ANN query can filter
                    # without joining. Document-level, so every chunk gets the
                    # same dict — it is not derived per chunk, which keeps
                    # build_chunks pure and a resumed job's chunks consistent
                    # with the ones written before the interruption.
                    doc_metadata=metadata,
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
