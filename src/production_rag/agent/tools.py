from uuid import UUID

from langchain_core.tools import BaseTool, tool

from production_rag.llm.embedding_service import EmbeddingService
from production_rag.repositories.document_repository import DocumentRepository

_DOC_TRUNCATE_CHARS = 5000


def build_db_tools(
    document_repository: DocumentRepository,
    embedding_service: EmbeddingService,
    user_id: UUID,
) -> list[BaseTool]:
    """Tools that talk to Postgres / the document store."""

    @tool
    async def search_documents(query: str, top_k: int = 5) -> str:
        """Search the user's uploaded documents for information relevant to a query.

        Use this when the user asks about content in their documents.

        Args:
            query: The search query to find relevant document chunks.
            top_k: Number of results to return (default: 5).
        """
        query_embedding = await embedding_service.embed_text(query)
        chunks = await document_repository.search_similar_chunks(
            query_embedding=query_embedding,
            user_id=user_id,
            top_k=top_k,
        )

        if not chunks:
            return "No relevant documents found."

        results = []
        for i, chunk in enumerate(chunks):
            doc = await document_repository.get_document_by_id(chunk.document_id)
            results.append(f"[Result {i + 1} from '{doc.title}']\n{chunk.content}")

        return "\n\n---\n\n".join(results)

    @tool
    async def list_documents() -> str:
        """List all documents the user has uploaded.

        Use this when the user asks what documents are available.
        """
        documents = await document_repository.list_documents_by_user(user_id=user_id)

        if not documents:
            return "No documents uploaded yet."

        lines = [f"- {doc.title} (ID: {doc.id}, {doc.chunk_count} chunks)" for doc in documents]
        return "Available documents:\n" + "\n".join(lines)

    @tool
    async def get_document_content(document_id: str) -> str:
        """Get the full content of a specific document by its ID.

        Use this when the user asks about a specific document.

        Args:
            document_id: The UUID of the document to retrieve.
        """
        try:
            doc = await document_repository.get_document_by_id(UUID(document_id))
        except Exception:
            return f"Document with ID '{document_id}' not found."

        content = doc.content
        if len(content) > _DOC_TRUNCATE_CHARS:
            content = (
                content[:_DOC_TRUNCATE_CHARS]
                + f"\n\n[...truncated, {len(doc.content)} total chars]"
            )

        return f"Title: {doc.title}\n\n{content}"

    return [search_documents, list_documents, get_document_content]


def build_production_rag_tools(
    document_repository: DocumentRepository,
    embedding_service: EmbeddingService,
    user_id: UUID,
) -> list[BaseTool]:
    """Assemble every tool the agent should have access to.

    Each sub-module owns its own tools; this function is just the assembler
    that combines them into the single list bound to the agent.
    """
    db_tools = build_db_tools(document_repository, embedding_service, user_id)
    return db_tools
