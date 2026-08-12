"""Builders and fakes for the eval-dataset tests.

Matches the ``tests/_jobs.py`` convention: underscore-prefixed so pytest does
not collect it, plain functions rather than fixtures so a test can construct
exactly the shape it needs inline.
"""

from dataclasses import dataclass, field
from typing import Any

from production_rag.eval.sampling import ChunkRef
from production_rag.eval.schema import (
    Corpus,
    EvalRecord,
    Gates,
    Generation,
    GoldChunk,
    content_sha256,
    make_qid,
)


def make_ref(
    document_key: str = "doc-a.md",
    chunk_index: int = 0,
    token_count: int = 200,
    document_title: str = "doc-a",
) -> ChunkRef:
    return ChunkRef(
        document_key=document_key,
        chunk_index=chunk_index,
        token_count=token_count,
        document_title=document_title,
    )


def make_gold(
    document_key: str = "doc-a.md",
    chunk_index: int = 0,
    snippet: str = "the quick brown fox jumps over the lazy dog",
    role: str = "primary",
    content: str | None = None,
) -> GoldChunk:
    return GoldChunk(
        document_key=document_key,
        chunk_index=chunk_index,
        content_sha256=content_sha256(content if content is not None else snippet),
        snippet=snippet,
        role=role,  # type: ignore[arg-type]
    )


def make_record(
    query_type: str = "paraphrase",
    question: str = "What colour is the fox that jumps over the dog?",
    answer: str | None = "It is brown.",
    gold: list[GoldChunk] | None = None,
    answerable: bool | None = None,
    warnings: list[str] | None = None,
    **overrides: Any,
) -> EvalRecord:
    if gold is None:
        gold = [] if query_type == "unanswerable" else [make_gold()]
    if answerable is None:
        answerable = query_type != "unanswerable"
    keys = [(item.document_key, item.chunk_index) for item in gold if item.role == "primary"]
    return EvalRecord(
        qid=make_qid(query_type, question, keys),
        query_type=query_type,  # type: ignore[arg-type]
        answerable=answerable,
        question=question,
        answer=answer if answerable else None,
        gold=gold,
        generation=Generation(
            prompt_version="test-v1",
            requested_model="gpt-4o-mini",
            served_model="gpt-4o-mini",
            run_id="test",
            unit_id="u_test",
        ),
        corpus=Corpus(
            normalizer_version="nfkc-ws-v1",
            chunker_version="recursive-char-v1",
            embedding_model="text-embedding-3-small",
        ),
        gates=Gates(warnings=warnings or [], snippet_verified=True),
        **overrides,
    )


@dataclass
class FakeLLMResponse:
    content: str
    model: str = "gpt-4o-mini"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    provider: str = "openai"


@dataclass
class FakeLLMClient:
    """An ``LLMClient``-shaped fake that returns queued replies.

    Queued rather than computed so a test can pin the exact malformed reply it
    wants to exercise. Records every call so a test can assert on what was sent
    — which is how "this chunk was never sent to the model" is checked.
    """

    replies: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    default: str = '{"items": []}'
    model: str = "gpt-4o-mini"

    async def chat(self, **kwargs: Any) -> FakeLLMResponse:
        self.calls.append(kwargs)
        content = self.replies.pop(0) if self.replies else self.default
        return FakeLLMResponse(content=content, model=self.model)
