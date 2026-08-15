from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from langchain_text_splitters import RecursiveCharacterTextSplitter

from production_rag.ingestion.idempotency import query_idempotency_key
from production_rag.ingestion.loaders import ExtractedSegment
from production_rag.ingestion.metadata import validate_metadata_filter
from production_rag.ingestion.normalize import (
    NORMALIZER_VERSION,
    normalize_segments,
    normalize_text,
)
from production_rag.llm.client import LLMClient
from production_rag.llm.embedding_service import EmbeddingService
from production_rag.logging_config import get_logger
from production_rag.models.document import Document, DocumentChunk
from production_rag.repositories.document_repository import (
    DEFAULT_EF_SEARCH,
    DEFAULT_ITERATIVE_SCAN,
    DocumentRepository,
)
from production_rag.repositories.ingestion_job_repository import IngestionJobRepository
from production_rag.repositories.query_cache_repository import QueryCacheRepository
from production_rag.schemas.document import ChunkSource, QueryResponse

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PreparedChunk:
    """One chunk of a document, with its provenance, before embedding.

    Produced by ``build_chunks`` and consumed by the ingestion pipeline. Kept
    separate from the ORM row so chunking can be derived, compared, and counted
    without touching the database — which is what lets a resumed job re-derive
    the same list and skip the part already written.
    """

    text: str
    char_start: int | None
    char_end: int | None
    page: int | None
    section: str | None


@dataclass(frozen=True, slots=True)
class DeletedDocument:
    """What a delete actually removed, captured before the rows are gone.

    ``chunks_deleted`` is counted from the DELETE, not read from
    ``Document.chunk_count``: an ingestion interrupted part-way leaves those two
    disagreeing, and the honest number is the one the statement returned.
    """

    id: UUID
    title: str
    source: str
    chunks_deleted: int


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A retrieved chunk paired with its relevance score.

    ``score`` is cosine *similarity* in ``[-1, 1]`` (``1.0`` = identical
    direction), i.e. ``1 - cosine_distance`` — **higher is a better match**. It
    is a similarity, NOT a distance; ranking is by *descending* score. This is
    the field eval harnesses rank on for Recall@k / nDCG, so its orientation
    matters: reading it as a distance would silently invert those metrics.
    """

    chunk: DocumentChunk
    score: float


# Chunking config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# Version of the chunking behaviour above. Part of every idempotency key — BUMP
# this whenever CHUNK_SIZE / CHUNK_OVERLAP / separators / splitter change, so
# content chunked under the old config is re-embedded instead of served stale.
# It is also what makes a resumed ingestion job safe: a job records the version
# its completed chunks were produced under, and restarts from zero if it moved.
CHUNKER_VERSION = "recursive-char-v1"

# Separator used to join segments into the stored document body. Chunk offsets
# are computed against the joined text, so this must not change without a
# CHUNKER_VERSION bump.
SEGMENT_SEPARATOR = "\n\n"


def build_splitter() -> RecursiveCharacterTextSplitter:
    """The chunker, as a single source of truth.

    Built from module-level constants so the request path, the worker, and any
    eval harness are provably chunking identically — a resumed job that chunked
    differently from its first attempt would splice two chunkings together.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # Recursive character splitting strategy
        add_start_index=True,  # record each chunk's offset in the source (for citations)
    )


def build_chunks(segments: list[ExtractedSegment]) -> tuple[str, list[PreparedChunk]]:
    """Normalize, then split into chunks. Pure, deterministic, side-effect free.

    Returns the normalized document body and its chunks. Each segment is
    normalized and chunked independently so its ``page``/``section`` provenance
    carries onto every chunk; char offsets are made global against the joined
    body, which is what gets stored as ``Document.content``.

    **Normalization is inside this function, not a step callers perform first.**
    That is the same principle ``ingestion.idempotency`` applies to version
    inputs: a caller must have no way to skip it. Chunking un-normalized text
    would embed ligatures and stray whitespace as tokens, and — worse — would
    produce chunks whose offsets point into a differently-shaped string than the
    one stored as ``Document.content``.

    **Determinism is a correctness requirement, not a nicety.** A worker
    resuming a partly-done job re-runs this function and skips the first
    ``processed_chunks`` entries; if the output differed between attempts, the
    document would end up with a mixture of two chunkings.
    """
    segments = normalize_segments(segments)
    sep_len = len(SEGMENT_SEPARATOR)
    full_content = SEGMENT_SEPARATOR.join(seg.text for seg in segments)

    splitter = build_splitter()
    chunks: list[PreparedChunk] = []
    base_offset = 0
    for seg in segments:
        for doc in splitter.create_documents([seg.text]):
            text = doc.page_content
            # start_index is -1 if the splitter couldn't locate the chunk; treat
            # that as "unknown" (None) rather than storing a bogus offset.
            local_start = doc.metadata.get("start_index")
            if local_start is not None and local_start < 0:
                local_start = None
            char_start = base_offset + local_start if local_start is not None else None
            chunks.append(
                PreparedChunk(
                    text=text,
                    char_start=char_start,
                    char_end=char_start + len(text) if char_start is not None else None,
                    page=seg.page,
                    section=seg.section,
                )
            )
        base_offset += len(seg.text) + sep_len

    return full_content, chunks


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
        job_repository: IngestionJobRepository | None = None,
        query_cache_ttl_seconds: int = 3600,
        hnsw_ef_search: int = DEFAULT_EF_SEARCH,
        hnsw_iterative_scan: str = DEFAULT_ITERATIVE_SCAN,
    ) -> None:
        self._repository = repository
        self._embedding_service = embedding_service
        self._llm = llm_client
        self._query_cache = query_cache_repository
        # Only ``delete_document`` needs it, so retrieval-only callers (the agent
        # tools, eval harnesses) are not made to construct one. Deleting without
        # it raises rather than silently skipping the job detach — a delete that
        # quietly left dangling references would be the exact bug this argument
        # exists to prevent.
        self._jobs = job_repository
        self._query_cache_ttl_seconds = query_cache_ttl_seconds
        self._hnsw_ef_search = hnsw_ef_search
        self._hnsw_iterative_scan = hnsw_iterative_scan

    async def retrieve(
        self,
        question: str,
        user_id: UUID,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> list[ScoredChunk]:
        """Retrieval only: embed the question and return the top-k most similar
        chunks with their similarity scores. **No LLM, no synthesis.**

        This is the stage evals measure (Recall@k / nDCG) — cheap and fast, and
        the exact code path ``query()`` runs, so measuring it here doesn't lie
        about production. Retrieval itself is never cached (the answer cache
        lives in ``query()``), so calling this always exercises real embedding +
        vector search. ``use_cache`` is accepted so callers express intent
        uniformly; an eval harness passes ``use_cache=False`` to guarantee it is
        measuring retrieval and never a warm answer cache.

        ``filters`` restricts the search to chunks whose extracted metadata
        contains every given key/value — ``{"language": "es", "doc_type":
        "contract"}`` searches only Spanish contracts. It is applied *inside* the
        vector query, not by discarding results afterwards, so ``top_k`` counts
        matching chunks rather than being silently eaten by non-matching ones.
        """
        # Normalize the question under the SAME rules the corpus was normalized
        # with. Skipping this leaves an asymmetry that never raises an error and
        # only shows up as quietly worse recall: the corpus has been NFKC-folded,
        # so a question containing a ligature or a non-breaking space would be
        # embedded against text that no longer contains those characters.
        question = normalize_text(question)

        # Validated here rather than only at the HTTP edge: retrieve() is called
        # directly by the agent tools and by eval harnesses, which do not pass
        # through the request schema.
        filters = validate_metadata_filter(filters)

        # Embed the question, then rank chunks by cosine similarity (higher is
        # better). search_similar_chunks already returns (chunk, similarity).
        query_embedding = await self._embedding_service.embed_text(question)
        scored = await self._repository.search_similar_chunks(
            query_embedding=query_embedding,
            user_id=user_id,
            top_k=top_k,
            filters=filters,
            ef_search=self._hnsw_ef_search,
            iterative_scan=self._hnsw_iterative_scan,
        )
        return [ScoredChunk(chunk=chunk, score=score) for chunk, score in scored]

    async def query(
        self,
        question: str,
        user_id: UUID,
        top_k: int = 5,
        model: str = "gpt-4.1-nano",
        filters: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> QueryResponse:
        """Full RAG pipeline: ``retrieve()`` then synthesize an answer.

        Retrieval is delegated to ``retrieve()`` (the same path evals measure);
        this method adds only prompt-building + generation on top. Guarded by a
        deterministic answer cache (skipped when ``use_cache=False``): an
        identical (user, question, filters, top_k, chunker version, embedding
        model) returns the stored answer without embedding, retrieval, or an
        LLM call.
        """
        # Step 0: Answer-cache lookup on the deterministic idempotency key.
        cache_key = query_idempotency_key(
            user_id=user_id,
            question=question,
            filters=filters,
            top_k=top_k,
            normalizer_version=NORMALIZER_VERSION,
            chunker_version=CHUNKER_VERSION,
            embedding_model=self._embedding_service.model,
        )
        if use_cache:
            cached = await self._query_cache.get(cache_key)
            if cached is not None:
                logger.info("query_cache_hit", user_id=str(user_id), cache_key=cache_key)
                response = QueryResponse.model_validate(cached.response_json)
                response.cached = True
                return response

        # Step 1: Retrieve (embed + vector search) — the eval-measurable stage.
        scored_chunks = await self.retrieve(
            question=question,
            user_id=user_id,
            top_k=top_k,
            filters=filters,
            use_cache=use_cache,
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

        # Step 2: Build context from chunks (Augmented)
        context_parts = []
        for i, sc in enumerate(scored_chunks):
            context_parts.append(f"[Source {i + 1}: {sc.chunk.document_title}]\n{sc.chunk.content}")
        context = "\n\n---\n\n".join(context_parts)

        # Step 3: Call LLM with context (Generation)
        system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
        llm_response = await self._llm.chat(
            messages=[{"role": "user", "content": question}],
            system=system_prompt,
            model=model,
        )

        # Step 4: Build response with sources (Generation)
        sources = []
        for i, sc in enumerate(scored_chunks):
            chunk = sc.chunk
            sources.append(
                ChunkSource(
                    chunk_id=chunk.id,
                    document_title=chunk.document_title,
                    content_preview=chunk.content[:200] + "..."
                    if len(chunk.content) > 200
                    else chunk.content,
                    similarity_rank=i + 1,
                    similarity_score=round(sc.score, 4),
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

        # Persist to the deterministic answer cache (conditional write; TTL from
        # config). Skipped when caching is disabled so evals never populate it.
        if use_cache:
            expires_at = datetime.now(UTC) + timedelta(seconds=self._query_cache_ttl_seconds)
            await self._query_cache.put(
                idempotency_key=cache_key,
                user_id=user_id,
                response_json=response.model_dump(mode="json"),
                expires_at=expires_at,
            )
        return response

    async def list_user_documents(self, user_id: UUID) -> list[Document]:
        return await self._repository.list_documents_by_user(user_id)

    async def delete_document(self, document_id: UUID, user_id: UUID) -> DeletedDocument:
        """Remove a document and everything that points at it.

        Four things reference a document, and a delete that skips any of them
        leaves the system answering questions from a document the user believes
        is gone:

        1. **its chunks** — deleted. These are what retrieval actually reads, so
           they are the deletion. A chunk carries its own ``owner_id``,
           ``document_title`` and metadata (denormalized for the ANN query), so
           an orphaned chunk is fully self-sufficient and would keep being
           retrieved and cited forever.
        2. **ingestion jobs** — their ``document_id`` is nulled, not deleted.
           The job row is the record that this source was ingested, when, and at
           what cost; erasing it would destroy the audit trail to tidy up a
           pointer. This matches the ``ON DELETE SET NULL`` on the column.
        3. **cached answers** — the owner's query cache is dropped. An entry
           holds a fully-rendered answer with its source previews baked in, so a
           cached hit would keep quoting the deleted document with no vector
           search involved and nothing to notice the rows are gone. Same
           invalidation ingestion performs, for the same reason.
        4. **dead-letter failures** — untouched, deliberately. They reference a
           *job*, never a document, and denormalize their own source string
           precisely so they survive it.

        Ownership is checked by fetching the row scoped to ``user_id``, so
        another tenant's id is a 404.

        **Known limit:** this does not cancel an ingestion job that is running
        right now for the same document. A worker mid-flight will fail on its
        next batch (its foreign key no longer resolves) and land in the
        dead-letter table — recoverable and visible, but noisier than a real
        cancellation. Cancellation needs a signal the worker polls, which does
        not exist yet.
        """
        if self._jobs is None:
            raise RuntimeError(
                "DocumentService was constructed without a job repository, so it "
                "cannot detach ingestion jobs from a deleted document."
            )

        document = await self._repository.get_document_for_owner(document_id, user_id)
        # Read what the response needs before the row is deleted.
        summary_title, summary_source = document.title, document.source

        # Jobs first: the pointer has to go before the row it points at.
        jobs_detached = await self._jobs.detach_document(document_id)
        chunks_deleted = await self._repository.delete_document(document)
        await self._query_cache.delete_by_user(user_id)

        logger.info(
            "document_deleted",
            document_id=str(document_id),
            user_id=str(user_id),
            title=summary_title,
            source=summary_source,
            chunks_deleted=chunks_deleted,
            jobs_detached=jobs_detached,
        )
        return DeletedDocument(
            id=document_id,
            title=summary_title,
            source=summary_source,
            chunks_deleted=chunks_deleted,
        )
