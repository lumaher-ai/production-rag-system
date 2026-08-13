from uuid import uuid4

from fastapi import APIRouter, Depends
from langgraph.checkpoint.base import BaseCheckpointSaver

from production_rag.agent.agent_loop import (
    AgentConfig,
    AgentExecutionError,
    AgentLoop,
)
from production_rag.agent.tools import build_production_rag_tools
from production_rag.dependencies import (
    get_checkpointer,
    get_current_user,
    get_document_repository,
    get_embedding_service,
)
from production_rag.exceptions import PaddingtonError
from production_rag.llm.embedding_service import EmbeddingService
from production_rag.logging_config import get_logger
from production_rag.models import User
from production_rag.repositories.document_repository import DocumentRepository
from production_rag.schemas.agent import AgentRunRequest, AgentRunResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(
    data: AgentRunRequest,
    current_user: User = Depends(get_current_user),
    doc_repo: DocumentRepository = Depends(get_document_repository),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    checkpointer: BaseCheckpointSaver | None = Depends(get_checkpointer),
) -> AgentRunResponse:
    client_thread_id = data.thread_id or str(uuid4())
    scoped_thread_id = f"{current_user.id}:{client_thread_id}"

    tools = build_production_rag_tools(
        document_repository=doc_repo,
        embedding_service=embedding_service,
        user_id=current_user.id,
    )

    config = AgentConfig(
        model=data.model,
        max_iterations=data.max_iterations,
    )
    agent = AgentLoop(tools=tools, config=config, checkpointer=checkpointer)

    try:
        result = await agent.run(user_message=data.message, thread_id=scoped_thread_id)
    except PaddingtonError:
        # Domain errors (recursion limit, budget, etc.) already carry the right
        # status code — let the registered handler shape the response.
        raise
    except Exception as exc:
        # An unexpected failure inside a tool or the graph should surface as a
        # structured error, not a naked 500 with a raw stack trace.
        logger.error(
            "agent_run_failed",
            error_type=type(exc).__name__,
            thread_id=scoped_thread_id,
            exc_info=exc,
        )
        raise AgentExecutionError(
            "The agent failed to complete the request due to an internal tool error."
        ) from exc

    return AgentRunResponse(
        answer=result.answer,
        iterations=result.iterations,
        tools_used=result.tools_used,
        total_input_tokens=result.total_input_tokens,
        total_output_tokens=result.total_output_tokens,
        total_cost_usd=result.total_cost_usd,
        thread_id=client_thread_id,
    )
