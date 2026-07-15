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
    ) -> Document:
        document = Document(
            title=title,
            content=content,
            user_id=user_id,
            chunk_count=chunk_count,
        )
        self._session.add(document)
        await self._session.flush()
        return document

    async def create_chunk(
        self,
        document_id: UUID,
        chunk_index: int,
        content: str,
        token_count: int,
        embedding: list[float],
    ) -> DocumentChunk:
        chunk = DocumentChunk(
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            token_count=token_count,
            embedding=embedding,
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
    ) -> list[DocumentChunk]:
        """Find the most similar chunks to a query embedding using cosine distance."""
        result = await self._session.execute(
            select(DocumentChunk)
            .join(Document, DocumentChunk.document_id == Document.id)
            .where(Document.user_id == user_id)
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
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
