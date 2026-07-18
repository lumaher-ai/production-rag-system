from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.exc import IntegrityError

from production_rag.ingestion.idempotency import content_hash, query_idempotency_key
from production_rag.ingestion.loaders import ExtractedSegment
from production_rag.llm.client import LLMClient, count_tokens
from production_rag.llm.embedding_service import EmbeddingService
from production_rag.logging_config import get_logger
from production_rag.models.document import Document
from production_rag.repositories.document_repository import DocumentRepository
from production_rag.repositories.query_cache_repository import QueryCacheRepository
from production_rag.schemas.document import ChunkSource, QueryResponse

logger = get_logger(__name__)

# Chunking config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# Version of the chunking behaviour above. Part of every idempotency key — BUMP
# this whenever CHUNK_SIZE / CHUNK_OVERLAP / separators / splitter change, so
# content chunked under the old config is re-embedded instead of served stale.
CHUNKER_VERSION = "recursive-char-v1"

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
        query_cache_repository: QueryCacheRepository,
        query_cache_ttl_seconds: int = 3600,
    ) -> None:
        self._repository = repository
        self._embedding_service = embedding_service
        self._llm = llm_client
        self._query_cache = query_cache_repository
        self._query_cache_ttl_seconds = query_cache_ttl_seconds
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],  # Recursive character splitting strategy
            add_start_index=True,  # record each chunk's offset in the source (for citations)
        )

    async def query(
        self,
        question: str,
        user_id: UUID,
        top_k: int = 5,
        model: str = "gpt-4.1-nano",
        filters: dict[str, Any] | None = None,
    ) -> QueryResponse:
        """Full RAG pipeline: embed query → search → build prompt → generate answer.

        Guarded by a deterministic cache: an identical (user, question, filters,
        top_k, chunker version, embedding model) returns the stored answer without
        embedding, retrieval, or an LLM call.
        """
        # Step 0: Cache lookup on the deterministic idempotency key.
        cache_key = query_idempotency_key(
            user_id=user_id,
            question=question,
            filters=filters,
            top_k=top_k,
            chunker_version=CHUNKER_VERSION,
            embedding_model=self._embedding_service.model,
        )
        cached = await self._query_cache.get(cache_key)
        if cached is not None:
            logger.info("query_cache_hit", user_id=str(user_id), cache_key=cache_key)
            response = QueryResponse.model_validate(cached.response_json)
            response.cached = True
            return response

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

        response = QueryResponse(
            answer=llm_response.content,
            sources=sources,
            model=llm_response.model,
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            cost_usd=llm_response.cost_usd,
        )

        # Persist to the deterministic cache (conditional write; TTL from config).
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self._query_cache_ttl_seconds)
        await self._query_cache.put(
            idempotency_key=cache_key,
            user_id=user_id,
            response_json=response.model_dump(mode="json"),
            expires_at=expires_at,
        )
        return response

    async def ingest_document(
        self,
        title: str,
        content: str,
        user_id: UUID,
        source: str | None = "inline",
    ) -> Document:
        """Ingest a raw text document (JSON route). Delegates to the segment core."""
        return await self.ingest_segments(
            title=title,
            segments=[ExtractedSegment(text=content)],
            user_id=user_id,
            source=source,
        )

    async def ingest_segments(
        self,
        title: str,
        segments: list[ExtractedSegment],
        user_id: UUID,
        source: str | None = None,
    ) -> Document:
        """Chunk, embed, and store a document from structured segments.

        Idempotent: if this exact content was already ingested by this user under
        the current chunker + embedding model, the existing document is returned
        and **no embedding happens**. Each segment is chunked independently so its
        ``page``/``section`` provenance carries onto every chunk; char offsets are
        made global against the joined ``full_content`` (the stored document body).
        """
        # Join with a fixed separator; SEP_LEN keeps the running offset aligned
        # with the stored full_content so char_start/char_end point into it.
        separator = "\n\n"
        sep_len = len(separator)
        full_content = separator.join(seg.text for seg in segments)

        # Step 0: Idempotency short-circuit — skip re-embedding unchanged content.
        chash = content_hash(full_content)
        embedding_model = self._embedding_service.model
        existing = await self._repository.get_document_by_idempotency(
            user_id=user_id,
            content_hash=chash,
            chunker_version=CHUNKER_VERSION,
            embedding_model=embedding_model,
        )
        if existing is not None:
            logger.info(
                "document_ingest_skipped",
                document_id=str(existing.id),
                title=title,
                reason="content_hash_match",
            )
            return existing

        # Step 1: Chunk each segment, carrying its provenance and a global offset.
        chunk_texts: list[str] = []
        chunk_meta: list[dict] = []  # {char_start, char_end, page, section}
        base_offset = 0
        for seg in segments:
            split_docs = self._splitter.create_documents([seg.text])
            for doc in split_docs:
                text = doc.page_content
                # start_index is -1 if the splitter couldn't locate the chunk; treat
                # that as "unknown" (None) rather than storing a bogus offset.
                local_start = doc.metadata.get("start_index")
                if local_start is not None and local_start < 0:
                    local_start = None
                char_start = base_offset + local_start if local_start is not None else None
                char_end = char_start + len(text) if char_start is not None else None
                chunk_texts.append(text)
                chunk_meta.append(
                    {
                        "char_start": char_start,
                        "char_end": char_end,
                        "page": seg.page,
                        "section": seg.section,
                    }
                )
            base_offset += len(seg.text) + sep_len

        logger.info(
            "document_chunked",
            title=title,
            total_chars=len(full_content),
            segment_count=len(segments),
            chunk_count=len(chunk_texts),
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        # Step 2: Embed all chunks in a single batch call
        embeddings = await self._embedding_service.embed_batch(chunk_texts)

        # Step 3: Create document record. A concurrent identical ingest may have
        # raced us to the unique idempotency constraint — treat that as a hit.
        try:
            async with self._repository.savepoint():
                document = await self._repository.create_document(
                    title=title,
                    content=full_content,
                    user_id=user_id,
                    chunk_count=len(chunk_texts),
                    content_hash=chash,
                    chunker_version=CHUNKER_VERSION,
                    embedding_model=embedding_model,
                    source=source,
                )
        except IntegrityError:
            winner = await self._repository.get_document_by_idempotency(
                user_id=user_id,
                content_hash=chash,
                chunker_version=CHUNKER_VERSION,
                embedding_model=embedding_model,
            )
            if winner is None:
                raise
            logger.info("document_ingest_raced", document_id=str(winner.id), title=title)
            return winner

        # Step 4: Create chunk records with embeddings and provenance
        chunks = []
        for i, (text, meta, embedding) in enumerate(
            zip(chunk_texts, chunk_meta, embeddings, strict=True)
        ):
            chunk = await self._repository.create_chunk(
                document_id=document.id,
                owner_id=user_id,
                document_title=document.title,
                chunk_index=i,
                content=text,
                token_count=count_tokens(text),
                embedding=embedding,
                char_start=meta["char_start"],
                char_end=meta["char_end"],
                page=meta["page"],
                section=meta["section"],
            )
            chunks.append(chunk)

        # New content invalidates this user's cached answers (they may now be stale).
        await self._query_cache.delete_by_user(user_id)

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
