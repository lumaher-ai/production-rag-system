from fastapi import APIRouter, Depends

from production_rag.dependencies import get_current_user, get_llm_client
from production_rag.llm.client import LLMClient
from production_rag.models import User
from production_rag.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    data: ChatRequest,
    current_user: User = Depends(get_current_user),
    llm: LLMClient = Depends(get_llm_client),
) -> ChatResponse:
    result = await llm.chat(
        messages=[{"role": "user", "content": data.message}],
        model=data.model,
        system=data.system,
    )

    return ChatResponse(
        response=result.content,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
    )
