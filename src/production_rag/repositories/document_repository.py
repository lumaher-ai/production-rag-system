from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.exceptions import NotFoundError
from production_rag.models.document import Document, DocumentChunk


class DocumentNotFoundError(NotFoundError):
    pass


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_document(
        self,
        title: str,
        content: str,
        user_id: UUID,
        chunk_count: int,
        content_hash: str,
        chunker_version: str,
        embedding_model: str,
        source: str | None = None,
    ) -> Document:
        document = Document(
            title=title,
            content=content,
            user_id=user_id,
            chunk_count=chunk_count,
            content_hash=content_hash,
            chunker_version=chunker_version,
            embedding_model=embedding_model,
            source=source,
        )
        self._session.add(document)
        await self._session.flush()
        return document

    async def get_document_by_idempotency(
        self,
        user_id: UUID,
        content_hash: str,
        chunker_version: str,
        embedding_model: str,
    ) -> Document | None:
        """Return the document already ingested for this exact identity, if any.

        The identity mirrors the ``uq_documents_idempotency`` constraint so a
        repeat ingest can short-circuit before any embedding work happens.
        """
        result = await self._session.execute(
            select(Document).where(
                Document.user_id == user_id,
                Document.content_hash == content_hash,
                Document.chunker_version == chunker_version,
                Document.embedding_model == embedding_model,
            )
        )
        return result.scalar_one_or_none()

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

    async def search_similar_chunks(
        self,
        query_embedding: list[float],
        user_id: UUID,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """Find the most similar chunks to a query embedding using cosine distance.

        Returns each chunk paired with its cosine *similarity* score in ``[-1, 1]``
        (``1.0`` = identical direction). pgvector's ``<=>`` operator yields cosine
        *distance*; similarity is ``1 - distance``, so higher is a better match.
        The score is selected alongside the row rather than recomputed, so callers
        can rank, threshold, or surface it without re-querying.
        """
        distance = DocumentChunk.embedding.cosine_distance(query_embedding)
        result = await self._session.execute(
            # Owner scope comes from the denormalized owner_id — no join to
            # documents needed (title is already carried on the chunk too).
            select(DocumentChunk, distance.label("distance"))
            .where(DocumentChunk.owner_id == user_id)
            .order_by(distance)
            .limit(top_k)
        )
        return [(row[0], 1.0 - row.distance) for row in result.all()]

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
