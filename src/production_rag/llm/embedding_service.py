import time

import litellm

from production_rag.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self._model = model

    async def embed_text(self, text: str) -> list[float]:
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        start_time = time.perf_counter()

        response = await litellm.aembedding(
            model=self._model,
            input=texts,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000
        total_tokens = response.usage.total_tokens if response.usage else 0

        try:
            cost = litellm.completion_cost(
                model=self._model,
                prompt="",
                completion="",
                prompt_tokens=total_tokens,
                completion_tokens=0,
            )
        except Exception:
            cost = (total_tokens / 1_000_000) * 0.02  # Fallback manual

        logger.info(
            "embedding_completed",
            model=self._model,
            text_count=len(texts),
            total_tokens=total_tokens,
            cost_usd=round(cost, 6),
            latency_ms=round(latency_ms, 1),
        )

        sorted_data = sorted(response.data, key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]
