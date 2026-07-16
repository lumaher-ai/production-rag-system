from uuid import UUID

from langchain_text_splitters import RecursiveCharacterTextSplitter

from production_rag.llm.client import LLMClient, count_tokens
from production_rag.llm.embedding_service import EmbeddingService
from production_rag.logging_config import get_logger
from production_rag.models.document import Document
from production_rag.repositories.document_repository import DocumentRepository
from production_rag.schemas.document import ChunkSource, QueryResponse

logger = get_logger(__name__)

# Chunking config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Prompt

RAG_SYSTEM_PROMPT = """You are an assistant that answers questions based on the provided context.

RULES:
- Answer ONLY based on the context provided below.
- If the context doesn't contain enough information to answer, say: 
"I don't have enough information to answer that question based on the available documents."
- Do NOT make up information that isn't in the context.
- Cite which parts of the context you're using in your answer.
- Be concise and direct.

CONTEXT:
{context}"""


class DocumentService:
    def __init__(
        self,
        repository: DocumentRepository,
        embedding_service: EmbeddingService,
        llm_client: LLMClient,
    ) -> None:
        self._repository = repository
        self._embedding_service = embedding_service
        self._llm = llm_client
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],  # Recursive character splitting strategy
        )

    async def query(
        self,
        question: str,
        user_id: UUID,
        top_k: int = 5,
        model: str = "gpt-4.1-nano",
    ) -> QueryResponse:
        """Full RAG pipeline: embed query → search → build prompt → generate answer."""
        # Step 1: Embed the question (Retrieval)
        query_embedding = await self._embedding_service.embed_text(question)

        # Step 2: Find similar chunks (Retrieval) — each paired with its similarity score
        scored_chunks = await self._repository.search_similar_chunks(
            query_embedding=query_embedding,
            user_id=user_id,
            top_k=top_k,
        )

        if not scored_chunks:
            return QueryResponse(
                answer="No documents found. Please upload documents first.",
                sources=[],
                model=model,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
            )

        # Step 3: Build context from chunks (Augmented)
        context_parts = []
        for i, (chunk, _score) in enumerate(scored_chunks):
            context_parts.append(f"[Source {i + 1}: {chunk.document_title}]\n{chunk.content}")
        context = "\n\n---\n\n".join(context_parts)

        # Step 4: Call LLM with context (Generation)
        system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
        llm_response = await self._llm.chat(
            messages=[{"role": "user", "content": question}],
            system=system_prompt,
            model=model,
        )

        # Step 5: Build response with sources (Generation)
        sources = []
        for i, (chunk, score) in enumerate(scored_chunks):
            sources.append(
                ChunkSource(
                    chunk_id=chunk.id,
                    document_title=chunk.document_title,
                    content_preview=chunk.content[:200] + "..."
                    if len(chunk.content) > 200
                    else chunk.content,
                    similarity_rank=i + 1,
                    similarity_score=round(score, 4),
                )
            )

        logger.info(
            "rag_query_completed",
            question_length=len(question),
            chunks_used=len(scored_chunks),
            chunk_sources=sources,
            answer=llm_response.content,
            model=model,
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            cost_usd=llm_response.cost_usd,
        )

        return QueryResponse(
            answer=llm_response.content,
            sources=sources,
            model=llm_response.model,
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            cost_usd=llm_response.cost_usd,
        )

    async def ingest_document(
        self,
        title: str,
        content: str,
        user_id: UUID,
    ) -> Document:
        """Process a document: chunk it, embed chunks, and store everything."""
        # Step 1: Split into chunks
        chunk_texts = self._splitter.split_text(content)
        logger.info(
            "document_chunked",
            title=title,
            total_chars=len(content),
            chunk_count=len(chunk_texts),
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        # Step 2: Embed all chunks in a single batch call
        embeddings = await self._embedding_service.embed_batch(chunk_texts)

        # Step 3: Create document record
        document = await self._repository.create_document(
            title=title,
            content=content,
            user_id=user_id,
            chunk_count=len(chunk_texts),
        )

        # Step 4: Create chunk records with embeddings
        chunks = []
        for i, (text, embedding) in enumerate(zip(chunk_texts, embeddings, strict=True)):
            chunk = await self._repository.create_chunk(
                document_id=document.id,
                document_title=document.title,
                chunk_index=i,
                content=text,
                token_count=count_tokens(text),
                embedding=embedding,
            )
            chunks.append(chunk)

        logger.info(
            "document_ingested",
            document_id=str(document.id),
            title=title,
            chunks_created=len(chunks),
            total_tokens=sum(c.token_count for c in chunks),
        )

        return document

    async def list_user_documents(self, user_id: UUID) -> list[Document]:
        return await self._repository.list_documents_by_user(user_id)
