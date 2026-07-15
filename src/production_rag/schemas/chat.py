from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(default="gpt-4o-mini", description="LLM model to use")
    system: str | None = Field(
        default="You are a helpful assistant.",
        description="System prompt",
    )


class ChatResponse(BaseModel):
    response: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
