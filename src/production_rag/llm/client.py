import time
from dataclasses import dataclass

import litellm
import tiktoken
from litellm import acompletion, completion_cost
from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
)
from litellm.types.utils import ModelResponse

from production_rag.config import get_settings
from production_rag.logging_config import get_logger

logger = get_logger(__name__)

# Configure LiteLLM behavior
litellm.success_callback = []
litellm.failure_callback = []
litellm.drop_params = True


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    provider: str


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens for a string using tiktoken (OpenAI models only)."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # For non-OpenAI models, approximate: 1 token ≈ 4 chars
        return len(text) // 4


class LLMClient:
    """Production LLM client powered by LiteLLM -> Handles retries, fallbacks, and cost tracking"""

    def __init__(self) -> None:
        self._settings = get_settings()

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Send a chat completion request through LiteLLM.

        Works with any provider: OpenAI, Anthropic, Bedrock, etc.
        LiteLLM normalizes the interface — all models use OpenAI message format.

        ``response_format`` requests structured output, e.g. a
        ``{"type": "json_schema", ...}`` block. **It is a request, not a
        guarantee, and callers must parse defensively.** ``litellm.drop_params``
        is True module-wide and this method always passes ``fallbacks``, so a
        rate limit on the primary provider silently drops the parameter and
        serves the call from a model that never saw it. ``LLMResponse.model``
        reports which model actually answered; a caller that cares should check
        it rather than assume.
        """
        model = model or self._settings.default_model
        start_time = time.perf_counter()

        # Build the messages array with system prompt
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        # Build kwargs — only include params that are set
        kwargs = {
            "model": model,
            "messages": all_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = await acompletion(
                **kwargs,
                num_retries=3,
                fallbacks=[self._settings.fallback_model],
            )
        except (RateLimitError, ServiceUnavailableError, APIConnectionError) as e:
            logger.error(
                "llm_call_failed_after_retries",
                model=model,
                error_type=type(e).__name__,
                error=str(e),
            )
            raise

        latency_ms = (time.perf_counter() - start_time) * 1000

        # acompletion returns ModelResponse | CustomStreamWrapper; we never
        # set stream=True, so narrow the union (and fail fast otherwise).
        assert isinstance(response, ModelResponse)

        # Extract response data
        choice = response.choices[0]
        content = choice.message.content or ""
        usage = getattr(response, "usage", None)

        # Calculate cost using LiteLLM's built-in pricing
        try:
            cost = completion_cost(completion_response=response)
        except Exception:
            cost = 0.0  # If pricing data is unavailable, don't crash

        # Detect which provider actually served the response
        actual_model = response.model or model
        provider = self._detect_provider(actual_model)

        result = LLMResponse(
            content=content,
            model=actual_model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            cost_usd=round(cost, 6),
            latency_ms=round(latency_ms, 1),
            provider=provider,
        )

        logger.info(
            "llm_call_completed",
            model=result.model,
            provider=result.provider,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
        )

        return result

    async def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        """Send a completion with tools and return the raw response.

        Returns the raw LiteLLM response (OpenAI format) for the agent loop
        to inspect tool_calls directly. This avoids double-wrapping.
        """
        model = model or self._settings.default_model

        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        response = await acompletion(
            model=model,
            messages=all_messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            num_retries=3,
            fallbacks=[self._settings.fallback_model],
        )

        # acompletion returns ModelResponse | CustomStreamWrapper; we never
        # set stream=True, so narrow the union (and fail fast otherwise).
        assert isinstance(response, ModelResponse)

        return response

    @staticmethod
    def _detect_provider(model: str) -> str:
        """Detect the provider from the model name."""
        if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
            return "openai"
        elif model.startswith("claude"):
            return "anthropic"
        elif model.startswith("bedrock/"):
            return "aws_bedrock"
        elif model.startswith("gemini"):
            return "google"
        else:
            return "unknown"
