import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from sqlalchemy import CursorResult, delete, func, or_, select, text, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased, defer

from production_rag.exceptions import NotFoundError
from production_rag.logging_config import get_logger
from production_rag.models.document import Document, DocumentChunk

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSessionTransaction

logger = get_logger(__name__)

# Query-time recall knobs for the HNSW index, applied per statement. Defaults
# here match ``Settings``; callers pass the configured values through.
DEFAULT_EF_SEARCH = 100
DEFAULT_ITERATIVE_SCAN = "relaxed_order"


class DocumentNotFoundError(NotFoundError):
    pass


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def savepoint(self) -> "AsyncSessionTransaction":
        """A nested transaction (SAVEPOINT) so a unique-violation on one write can
        be caught and recovered from without poisoning the outer transaction."""
        return self._session.begin_nested()

    async def create_document(
        self,
        title: str,
        content: str,
        user_id: UUID,
        chunk_count: int,
        content_hash: str,
        normalizer_version: str,
        chunker_version: str,
        embedding_model: str,
        source: str,
        doc_metadata: dict[str, Any] | None = None,
    ) -> Document:
        document = Document(
            title=title,
            content=content,
            user_id=user_id,
            chunk_count=chunk_count,
            content_hash=content_hash,
            normalizer_version=normalizer_version,
            chunker_version=chunker_version,
            embedding_model=embedding_model,
            source=source,
            doc_metadata=doc_metadata or {},
        )
        self._session.add(document)
        await self._session.flush()
        return document

    async def find_document_by_source(
        self,
        user_id: UUID,
        source: str,
    ) -> Document | None:
        """Return this user's existing document for ``source``, if any.

        The identity mirrors the ``uq_documents_user_source`` constraint: at most
        one row per (owner, source). Used to decide create vs. replace on upload.
        """
        result = await self._session.execute(
            select(Document).where(
                Document.user_id == user_id,
                Document.source == source,
            )
        )
        return result.scalar_one_or_none()

    async def update_document_content(
        self,
        document: Document,
        title: str,
        content: str,
        chunk_count: int,
        content_hash: str,
        normalizer_version: str,
        chunker_version: str,
        embedding_model: str,
        doc_metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Rewrite an existing document's body/metadata in place (source is kept).

        Callers replace the document's chunks separately via
        ``delete_chunks_by_document_id`` + ``create_chunk``.
        """
        document.title = title
        document.content = content
        document.chunk_count = chunk_count
        document.content_hash = content_hash
        document.normalizer_version = normalizer_version
        document.chunker_version = chunker_version
        document.embedding_model = embedding_model
        document.doc_metadata = doc_metadata or {}
        await self._session.flush()
        return document

    async def refresh_metadata(
        self,
        document: Document,
        doc_metadata: dict[str, Any],
    ) -> None:
        """Rewrite a document's metadata and its chunks', touching nothing else.

        The cheap half of a re-ingest, for the case where the content is provably
        unchanged but the extractor that produced its metadata has moved. Two
        UPDATEs and no embedding calls — which is exactly why the extractor
        version is kept out of the idempotency gate: putting it there would turn
        a keyword-list tweak into a full corpus re-embed.
        """
        document.doc_metadata = doc_metadata
        await self._session.execute(
            update(DocumentChunk)
            .where(DocumentChunk.document_id == document.id)
            .values(doc_metadata=doc_metadata)
        )
        await self._session.flush()

    async def delete_chunks_by_document_id(self, document_id: UUID) -> int:
        """Delete all chunks belonging to a document (used before re-chunking).

        Returns how many rows went, which is what lets a delete report what it
        actually removed rather than echoing ``Document.chunk_count`` — those two
        disagree for any document whose ingestion was interrupted part-way.
        """
        # ``execute`` is typed as returning ``Result``, which has no rowcount;
        # a DML statement always yields a ``CursorResult``, which does.
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
            ),
        )
        await self._session.flush()
        return result.rowcount or 0

    async def create_chunk(
        self,
        document_id: UUID,
        owner_id: UUID,
        document_title: str,
        chunk_index: int,
        content: str,
        token_count: int,
        embedding: list[float],
        char_start: int | None = None,
        char_end: int | None = None,
        page: int | None = None,
        section: str | None = None,
        doc_metadata: dict[str, Any] | None = None,
    ) -> DocumentChunk:
        chunk = DocumentChunk(
            document_id=document_id,
            owner_id=owner_id,
            document_title=document_title,
            chunk_index=chunk_index,
            content=content,
            token_count=token_count,
            embedding=embedding,
            char_start=char_start,
            char_end=char_end,
            page=page,
            section=section,
            doc_metadata=doc_metadata or {},
        )
        self._session.add(chunk)
        await self._session.flush()
        return chunk

    async def get_document_by_id(self, document_id: UUID) -> Document:
        result = await self._session.execute(select(Document).where(Document.id == document_id))
        document = result.scalar_one_or_none()
        if document is None:
            raise DocumentNotFoundError(f"Document {document_id} not found")
        return document

    async def get_document_for_owner(self, document_id: UUID, user_id: UUID) -> Document:
        """Fetch a document, but only if this user owns it.

        Someone else's document id raises ``DocumentNotFoundError`` — a 404, not
        a 403, matching ``jobs.get_for_owner``: a 403 would confirm that the id
        exists, which is exactly what a caller probing for other tenants' ids
        wants to learn.
        """
        result = await self._session.execute(
            select(Document).where(
                Document.id == document_id,
                Document.user_id == user_id,
            )
        )
        document = result.scalar_one_or_none()
        if document is None:
            raise DocumentNotFoundError(f"Document {document_id} not found")
        return document

    async def delete_document(self, document: Document) -> int:
        """Delete a document and its chunks, returning the chunk count removed.

        Chunks are deleted explicitly rather than left to the ``ON DELETE
        CASCADE`` on ``document_chunks.document_id``. The cascade is real and
        would fire on Postgres, but it is the *database's* behaviour, not this
        code's: SQLite (which the fast unit tests run on) does not enforce
        foreign keys unless asked, so relying on it would leave orphan chunks
        under test — and an orphan chunk still carries ``owner_id`` and is still
        returned by retrieval, so the document would keep answering questions
        after being deleted. Doing it here makes the outcome the same on both
        engines and observable in a test.

        Both statements share the caller's transaction, so a failure between
        them rolls the whole delete back rather than stranding a document with
        no chunks.
        """
        chunks_deleted = await self.delete_chunks_by_document_id(document.id)
        await self._session.delete(document)
        await self._session.flush()
        return chunks_deleted

    async def _apply_ann_settings(self, ef_search: int, iterative_scan: str) -> None:
        """Set the HNSW query-time GUCs for the current transaction.

        ``iterative_scan`` is the fix for pgvector's filtered-ANN under-return: an
        HNSW scan walks the graph and applies ``WHERE`` to what it finds, so a
        query whose filters exclude most of the graph's first ``ef_search`` hits
        returns **fewer rows than LIMIT, with no error**. Recall degrades silently
        as tenants and metadata predicates multiply. Iterative scan re-scans until
        the limit is satisfied. ``ef_search`` widens each pass (pgvector's default
        of 40 is tuned for unfiltered search).

        ``SET LOCAL`` scopes both to the surrounding transaction, so a pooled
        connection never carries them into an unrelated request.

        Unsupported settings (pgvector < 0.8 has no ``iterative_scan``) degrade
        rather than fail: the query still returns correct rows, just possibly
        fewer than ``top_k``. Each attempt runs in its own savepoint because a
        failed statement otherwise aborts the caller's transaction.
        """
        # SET takes a literal, never a bind parameter — ``SET LOCAL x = $1`` is a
        # syntax error, and because these statements are allowed to fail softly it
        # would have failed *silently*, leaving the settings unapplied. Both values
        # come from Settings rather than from a request, and both are constrained
        # here so the interpolation cannot carry anything but what it claims to.
        statements: list[str] = []
        if ef_search > 0:
            statements.append(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
        if iterative_scan:
            if iterative_scan not in ("relaxed_order", "strict_order", "off"):
                raise ValueError(f"Invalid hnsw.iterative_scan value: {iterative_scan!r}")
            statements.append(f"SET LOCAL hnsw.iterative_scan = '{iterative_scan}'")

        for sql in statements:
            try:
                async with self._session.begin_nested():
                    await self._session.execute(text(sql))
            except DBAPIError as exc:
                logger.warning(
                    "hnsw_setting_unsupported",
                    statement=sql,
                    detail=str(exc.orig),
                    consequence="filtered search may return fewer than top_k rows",
                )

    async def search_similar_chunks(
        self,
        query_embedding: list[float],
        user_id: UUID,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        ef_search: int = DEFAULT_EF_SEARCH,
        iterative_scan: str = DEFAULT_ITERATIVE_SCAN,
    ) -> list[tuple[DocumentChunk, float]]:
        """Find the most similar chunks to a query embedding using cosine distance.

        Returns each chunk paired with its cosine *similarity* score in ``[-1, 1]``
        (``1.0`` = identical direction). pgvector's ``<=>`` operator yields cosine
        *distance*; similarity is ``1 - distance``, so higher is a better match.
        The score is selected alongside the row rather than recomputed, so callers
        can rank, threshold, or surface it without re-querying.

        ``filters`` is a flat metadata map, applied as JSONB **containment**
        (``metadata @> :filters``) against the denormalized per-chunk metadata.
        Containment is what the ``jsonb_path_ops`` GIN index accelerates; the
        equivalent-looking ``metadata->>'language' = 'es'`` does not use that index
        and degrades to a scan, and making it indexable would need one expression
        index per key — the per-field migration the JSONB column exists to avoid.

        The filter is a *pre-filter in SQL*, but pgvector applies it after walking
        the HNSW graph, which is why ``_apply_ann_settings`` runs first. Callers
        must validate ``filters`` (see ``ingestion.metadata.validate_metadata_filter``)
        — it is trusted here to be a flat map of scalars.
        """
        await self._apply_ann_settings(ef_search, iterative_scan)

        distance = DocumentChunk.embedding.cosine_distance(query_embedding)
        # Owner scope comes from the denormalized owner_id — no join to documents
        # needed (title and metadata are carried on the chunk too).
        stmt = select(DocumentChunk, distance.label("distance")).where(
            DocumentChunk.owner_id == user_id
        )
        if filters:
            # Bound as a parameter, so a filter value can never be SQL. The JSONB
            # cast is explicit because the driver sends the dict as text.
            stmt = stmt.where(
                DocumentChunk.doc_metadata.op("@>", is_comparison=True)(
                    text("CAST(:metadata_filter AS jsonb)").bindparams(
                        metadata_filter=json.dumps(filters, sort_keys=True)
                    )
                )
            )
        stmt = stmt.order_by(distance).limit(top_k)

        # ``relaxed_order`` lets pgvector return rows in approximate distance
        # order. ScoredChunk.score is what eval harnesses rank on, so exact order
        # is restored outside the scan rather than trusted from it.
        subquery = stmt.subquery()
        chunk = aliased(DocumentChunk, subquery)
        result = await self._session.execute(
            select(chunk, subquery.c.distance).order_by(subquery.c.distance)
        )
        return [(row[0], 1.0 - row.distance) for row in result.all()]

    async def list_chunks_for_owner(
        self,
        user_id: UUID,
        limit: int = 2000,
        offset: int = 0,
    ) -> list[DocumentChunk]:
        """Every chunk this user owns, in stable ``(document_id, chunk_index)`` order.

        Ordered by that pair rather than by ``created_at`` because the pair *is*
        a chunk's durable identity: ``DocumentChunk.id`` is a fresh ``uuid4`` on
        every ingestion pass, so anything keyed on it points at nothing the
        morning after a re-index. A reader that took rows in insertion order
        would also silently reorder itself whenever one document is re-ingested,
        which would change the sample a seeded eval run picks without changing
        the seed.

        ``embedding`` is deferred: this returns text for callers that read chunk
        content, and shipping 1536 floats per row to a consumer that never looks
        at them is pure transfer cost. Callers needing the vector should touch
        ``chunk.embedding``, which will lazily load it, or use
        ``search_similar_chunks``.
        """
        result = await self._session.execute(
            select(DocumentChunk)
            .options(defer(DocumentChunk.embedding))
            .where(DocumentChunk.owner_id == user_id)
            .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_chunks_by_keys(
        self,
        user_id: UUID,
        keys: Sequence[tuple[UUID, int]],
    ) -> dict[tuple[UUID, int], DocumentChunk]:
        """Resolve stable ``(document_id, chunk_index)`` keys back to rows.

        This is what makes that pair a real identity rather than a convention:
        the eval dataset stores those keys and nothing else, so every consumer —
        snippet verification, the audit sheet, the metrics harness — comes back
        through here.

        Missing keys are simply absent from the result rather than raising. A
        caller that finds a key unresolved is looking at a dataset whose corpus
        has moved underneath it, and letting it see *which* keys died is more
        useful than aborting on the first one.

        Issued as one ``document_id IN (...) AND chunk_index IN (...)`` query
        with the exact pairs filtered in Python. A row-value ``IN`` over tuples
        would be tighter, but the fast unit-test path runs on SQLite and this
        form behaves identically on both engines; the over-fetch is bounded by
        the number of keys asked for.
        """
        if not keys:
            return {}

        wanted = set(keys)
        result = await self._session.execute(
            select(DocumentChunk)
            .options(defer(DocumentChunk.embedding))
            .where(
                DocumentChunk.owner_id == user_id,
                DocumentChunk.document_id.in_({document_id for document_id, _ in wanted}),
                DocumentChunk.chunk_index.in_({index for _, index in wanted}),
            )
        )
        found: dict[tuple[UUID, int], DocumentChunk] = {}
        for chunk in result.scalars().all():
            key = (chunk.document_id, chunk.chunk_index)
            if key in wanted:
                found[key] = chunk
        return found

    async def count_chunks_for_owner(self, user_id: UUID) -> int:
        """How many chunks this user owns.

        The denominator a random-chunk baseline needs in order to state its own
        expected score (``k / N``) *before* it is run — a measured number that
        matches a prediction made in advance is evidence; one that merely looks
        low is not.
        """
        result = await self._session.execute(
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.owner_id == user_id)
        )
        return int(result.scalar_one())

    async def document_has_provenance(self, document_id: UUID) -> bool:
        """Whether any of this document's chunks carry page or section values.

        Used before a re-index falls back to the stored body, so the warning can
        say whether provenance is actually being lost or whether there was none
        to lose — the difference between a real caveat and noise.
        """
        result = await self._session.execute(
            select(DocumentChunk.id)
            .where(
                DocumentChunk.document_id == document_id,
                or_(
                    DocumentChunk.page.is_not(None),
                    DocumentChunk.section.is_not(None),
                ),
            )
            .limit(1)
        )
        return result.scalar_one_or_none() is not None

    async def find_stale(
        self,
        normalizer_version: str,
        chunker_version: str,
        embedding_model: str,
        limit: int = 500,
    ) -> list[Document]:
        """Documents whose stored vectors were produced under superseded config.

        These three columns are the silent inputs to every stored vector, so a
        row differing on any of them holds embeddings that no longer match how
        the system would process the same text today. They stay queryable and
        retrievable — nothing is broken — but they are inconsistent with newer
        documents in a way that only this comparison can reveal.

        This is what the version columns buy that a content hash cannot: the
        question "which documents predate the fix?" is an indexed lookup here,
        rather than re-normalizing the whole corpus and diffing hashes.
        """
        result = await self._session.execute(
            select(Document)
            .where(
                or_(
                    Document.normalizer_version != normalizer_version,
                    Document.chunker_version != chunker_version,
                    Document.embedding_model != embedding_model,
                )
            )
            .order_by(Document.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_documents_by_user(
        self,
        user_id: UUID,
        limit: int = 20,
    ) -> list[Document]:
        result = await self._session.execute(
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
