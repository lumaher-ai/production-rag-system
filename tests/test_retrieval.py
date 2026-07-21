from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from production_rag.ingestion.loaders import ExtractedSegment
from production_rag.llm.client import LLMClient
from production_rag.llm.embedding_service import EmbeddingService
from production_rag.models import User
from production_rag.repositories.document_repository import DocumentRepository
from production_rag.repositories.query_cache_repository import QueryCacheRepository
from production_rag.services.auth_service import hash_password
from production_rag.services.document_service import DocumentService, ScoredChunk

pytestmark = pytest.mark.asyncio(loop_scope="module")


def _mock_embedding_service() -> EmbeddingService:
    mock = AsyncMock(spec=EmbeddingService)
    mock.embed_text.side_effect = lambda text: [0.1] * 1536
    mock.embed_batch.side_effect = lambda texts: [[0.1] * 1536 for _ in texts]
    mock.model = "text-embedding-3-small"
    return mock


def _service(session: AsyncSession, embedding: EmbeddingService) -> DocumentService:
    # A real repository over the test session; the LLM is a mock that retrieval
    # must never touch (asserted below).
    return DocumentService(
        repository=DocumentRepository(session),
        embedding_service=embedding,
        llm_client=AsyncMock(spec=LLMClient),
        query_cache_repository=QueryCacheRepository(session),
    )


async def test_retrieve_returns_scored_chunks_without_llm(pg_session: AsyncSession) -> None:
    """retrieve() is pure retrieval: chunks + scores, no server, no LLM call."""
    emb = _mock_embedding_service()
    service = _service(pg_session, emb)

    user = User(
        name="R",
        email="retrieve@example.com",
        hashed_password=hash_password("pw"),
        role="user",
    )
    pg_session.add(user)
    await pg_session.flush()

    await service.ingest_segments(
        title="KB",
        segments=[ExtractedSegment(text="pgvector powers similarity search. " * 40)],
        user_id=user.id,
        source="kb.txt",
    )

    results = await service.retrieve(
        question="How does similarity search work?",
        user_id=user.id,
        top_k=3,
        use_cache=False,
    )

    # Returns scored chunks (not bare chunks) — the eval-measurable stage.
    assert results, "expected at least one retrieved chunk"
    assert len(results) <= 3
    assert all(isinstance(r, ScoredChunk) for r in results)
    # Every result carries a numeric score and its underlying chunk content.
    assert all(isinstance(r.score, float) for r in results)
    assert all(r.chunk.content for r in results)
    # score is a similarity (higher = better): results come back ranked descending.
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    # Retrieval synthesises nothing: the LLM was never called.
    service._llm.chat.assert_not_called()
