from dataclasses import dataclass
from typing import Any

import litellm
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    dynamic_prompt,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from production_rag.config import get_settings
from production_rag.exceptions import PaddingtonError
from production_rag.logging_config import get_logger

logger = get_logger(__name__)

# Default prompt used when no per-phase prompt is supplied. Built from the shared
# browser rules plus a generic preamble; phase nodes override this per invocation.
_DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant with access to tools.
Use them when needed to answer the user's question accurately.
If you can answer without tools, do so directly.
Be concise and cite your sources when using search results"""


class AgentBudgetExceededError(PaddingtonError):
    status_code = 429


class AgentRecursionLimitError(PaddingtonError):
    status_code = 422


class AgentExecutionError(PaddingtonError):
    status_code = 502


@dataclass
class AgentResult:
    answer: str
    iterations: int
    tools_used: list[str]
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float


@dataclass
class AgentConfig:
    model: str = "gpt-4o-mini"
    max_iterations: int = 3
    max_cost_usd: float = 0.50
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT


@dataclass
class AgentInvocationContext:
    """Per-invocation runtime context (NOT persisted in the checkpoint).

    Tells the budget middleware how many messages already existed at the
    start of this turn, so it can compute cost/tokens only from new messages.

    ``system_prompt`` carries the per-phase prompt for this invocation; when None
    the agent falls back to the default. Because the graph is compiled once, this
    is how each phase node supplies its own goal/"ends when" prompt without a
    rebuild — see ``_phase_system_prompt``.
    """

    baseline_message_count: int
    system_prompt: str | None = None


def _resolve_system_prompt(context: "AgentInvocationContext | None") -> str:
    """Return the per-phase prompt when the invocation supplied one, else the default."""
    if context is not None and context.system_prompt:
        return context.system_prompt
    return _DEFAULT_SYSTEM_PROMPT


@dynamic_prompt
def _phase_system_prompt(request: ModelRequest) -> str:
    """Resolve the system prompt for this model call from the invocation context."""
    return _resolve_system_prompt(request.runtime.context)


_INTERRUPTED_TOOL_RESULT = (
    "The previous tool call did not complete because the run was interrupted. "
    "Retry the action if it is still needed."
)


def _dangling_tool_messages(messages: list[BaseMessage]) -> list[ToolMessage]:
    """Return synthetic ToolMessages for any tool_call left without a result.

    An interruption mid-tool-execution (a crashing tool, a process kill, a
    recursion/budget abort) can leave an AIMessage whose tool_calls were never
    answered by a ToolMessage. Replaying that history makes the OpenAI/Anthropic
    APIs reject the request ("tool_call_ids did not have response messages"). We
    synthesize a placeholder result per dangling id so the history is valid again.
    """
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    repairs: list[ToolMessage] = []
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        for call in m.tool_calls or []:
            call_id = call.get("id")
            if call_id and call_id not in answered:
                repairs.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        name=call.get("name"),
                        content=_INTERRUPTED_TOOL_RESULT,
                    )
                )
                answered.add(call_id)
    return repairs


class AgentLoop:
    """ReAct agent powered by LangGraph + ChatLiteLLM."""

    def __init__(
        self,
        tools: list[BaseTool],
        config: AgentConfig | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self._config = config or AgentConfig()
        self._checkpointer = checkpointer
        self._graph = self._build_graph(tools)

    @property
    def model(self) -> str:
        """The LLM model id this loop runs on (used to build sibling LLM calls)."""
        return self._config.model

    def _build_graph(self, tools: list[BaseTool]) -> CompiledStateGraph[Any, Any, Any, Any]:
        settings = get_settings()
        model = ChatLiteLLM(
            model=self._config.model,
            temperature=0.0,
            max_retries=3,
            model_kwargs={"fallbacks": [settings.fallback_model]},
        )
        # The system prompt is supplied per invocation via the dynamic-prompt
        # middleware (reading AgentInvocationContext), not baked in here — that is
        # how each phase node gets its own goal/"ends when" prompt on one graph.
        return create_agent(
            model,
            tools=tools,
            middleware=[
                _phase_system_prompt,
                BudgetMiddleware(self._config.max_cost_usd, self._config.model),
                # Trim stale page snapshots/screenshots from each request (nearest the
                # model call) so a phase's context stays bounded across ReAct turns.
                SnapshotPruneMiddleware(),
            ],
            context_schema=AgentInvocationContext,
            checkpointer=self._checkpointer,
        )

    async def run(
        self,
        user_message: str,
        thread_id: str,
        system_prompt: str | None = None,
    ) -> AgentResult:
        invoke_config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._config.max_iterations * 2,
        }

        baseline = 0
        dangling: list[ToolMessage] = []
        if self._checkpointer is not None:
            snapshot = await self._graph.aget_state({"configurable": {"thread_id": thread_id}})
            if snapshot and snapshot.values:
                messages = snapshot.values.get("messages", [])
                baseline = len(messages)

                # Repair any tool_call left unanswered by a prior interrupted run.
                # Without this, the persisted history is invalid and every future
                # request on this thread is rejected by the LLM API. We inject the
                # synthetic results into this turn's input so the add_messages
                # reducer appends them right after the dangling tool_call, before
                # the new user message.
                dangling = _dangling_tool_messages(messages)
                if dangling:
                    logger.warning(
                        "repaired_dangling_tool_calls",
                        thread_id=thread_id,
                        count=len(dangling),
                    )

        try:
            final = await self._graph.ainvoke(
                {"messages": [*dangling, HumanMessage(content=user_message)]},
                config=invoke_config,
                context=AgentInvocationContext(
                    baseline_message_count=baseline,
                    system_prompt=system_prompt or self._config.system_prompt,
                ),
            )
        except GraphRecursionError as exc:
            logger.warning(
                "agent_recursion_limit_reached",
                max_iterations=self._config.max_iterations,
                recursion_limit=invoke_config["recursion_limit"],
                thread_id=thread_id,
                # The compact tool trace is the whole point on this path: a loop shows up
                # as the same call repeated (e.g. click(el_142) toggling a seat on/off).
                tool_trace=await self._recent_tool_trace(thread_id),
            )
            raise AgentRecursionLimitError(
                f"Agent stopped after reaching its step limit "
                f"(max_iterations={self._config.max_iterations}) without finishing. "
                f"Retry with a higher max_iterations for multi-step tasks like browsing."
            ) from exc

        new_messages = final["messages"][baseline:]
        total_cost, total_input, total_output = _accumulate_usage(new_messages, self._config.model)

        iterations = sum(1 for m in new_messages if isinstance(m, AIMessage))
        tools_used = [
            tc["name"]
            for m in new_messages
            if isinstance(m, AIMessage)
            for tc in (m.tool_calls or [])
        ]
        answer = next(
            (
                m.content
                for m in reversed(new_messages)
                if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content
            ),
            "",
        )

        if total_cost > self._config.max_cost_usd:
            raise AgentBudgetExceededError(
                f"Agent exceeded budget: ${total_cost:.4f} > ${self._config.max_cost_usd:.2f}"
            )

        logger.info(
            "agent_completed",
            iterations=iterations,
            tools_used=tools_used,
            total_cost=round(total_cost, 6),
        )

        return AgentResult(
            answer=answer,
            iterations=iterations,
            tools_used=tools_used,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_cost_usd=round(total_cost, 6),
        )

    async def _recent_tool_trace(self, thread_id: str, limit: int = 16) -> list[str]:
        """Best-effort compact trace of the last tool calls (name + key arg) for debugging.

        Read from the persisted checkpoint after a failure (e.g. a recursion limit) so the
        log shows *what the agent actually did*. Each entry is ``name(detail)`` where detail
        is the click ref / navigate url when present — a loop then reads as the same call
        repeated. Returns ``[]`` if state can't be read (no checkpointer, unknown thread).
        """
        if self._checkpointer is None:
            return []
        try:
            snapshot = await self._graph.aget_state({"configurable": {"thread_id": thread_id}})
        except Exception:  # noqa: BLE001 — tracing must never mask the original error
            return []
        if not snapshot or not snapshot.values:
            return []
        trace: list[str] = []
        for m in snapshot.values.get("messages", []):
            if not isinstance(m, AIMessage):
                continue
            for tc in m.tool_calls or []:
                args = tc.get("args") or {}
                detail = args.get("ref") or args.get("url") or ""
                name = tc.get("name")
                trace.append(f"{name}({detail})" if detail else str(name))
        return trace[-limit:]


def _accumulate_usage(messages: list, model: str) -> tuple[float, int, int]:
    """Sum cost + tokens across every AIMessage in the given slice."""
    total_cost = 0.0
    total_input = 0
    total_output = 0
    for msg in messages:
        if not isinstance(msg, AIMessage) or not msg.usage_metadata:
            continue
        input_tokens = msg.usage_metadata.get("input_tokens", 0)
        output_tokens = msg.usage_metadata.get("output_tokens", 0)
        total_input += input_tokens
        total_output += output_tokens
        try:
            in_cost, out_cost = litellm.cost_per_token(
                model=model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
            total_cost += in_cost + out_cost
        except Exception as e:
            logger.warning("cost_calculation_failed", model=model, error=str(e))
    return total_cost, total_input, total_output


class BudgetMiddleware(AgentMiddleware):
    """Halt the agent when *this turn's* LLM cost exceeds the configured budget.

    Reads `runtime.context.baseline_message_count` to ignore messages from
    previous turns when computing cost. If we're over budget and the model
    just requested tools, strip the tool_calls so the loop terminates;
    `AgentLoop.run` then raises `AgentBudgetExceededError`.
    """

    def __init__(self, max_cost_usd: float, model: str) -> None:
        super().__init__()
        self._max_cost_usd = max_cost_usd
        self._model = model

    def after_model(
        self, state: AgentState, runtime: Runtime[AgentInvocationContext]
    ) -> dict[str, Any] | None:
        baseline = runtime.context.baseline_message_count if runtime.context else 0
        new_messages = state["messages"][baseline:]
        total_cost, _, _ = _accumulate_usage(new_messages, self._model)
        if total_cost <= self._max_cost_usd:
            return None

        logger.warning(
            "agent_budget_exceeded",
            cost=round(total_cost, 6),
            budget=self._max_cost_usd,
        )
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            stripped = AIMessage(
                id=last.id,
                content=last.content or "Stopped: budget exceeded before completing the answer.",
                response_metadata=last.response_metadata,
                usage_metadata=last.usage_metadata,
            )
            return {"messages": [stripped]}
        return None


# --- Snapshot pruning ------------------------------------------------------------
# Inside one phase run, the ReAct loop calls get_snapshot / take_screenshot on nearly
# every turn, so the message history fills with stale ~8k-char page dumps and base64
# screenshots. Only the most recent of each reflects the live page (refs are valid only
# "until the next snapshot"), so the rest are pure token waste — and their refs are
# actively misleading. We collapse the stale ones in the *outgoing request only*.

_SNAPSHOT_TOOL_NAME = "get_snapshot"
_SNAPSHOT_PLACEHOLDER = (
    "[stale page snapshot pruned to save context — call get_snapshot to re-read this "
    "page; earlier refs are no longer valid]"
)
_SCREENSHOT_PLACEHOLDER = "[earlier screenshot omitted to save context]"


def _is_screenshot_message(message: BaseMessage) -> bool:
    """True for a HumanMessage carrying an image block (a take_screenshot result).

    take_screenshot appends the PNG as a HumanMessage whose content is a list with an
    ``image_url`` block (browser/tools.py); that image is the token-heavy part.
    """
    if not isinstance(message, HumanMessage) or not isinstance(message.content, list):
        return False
    return any(
        isinstance(part, dict) and part.get("type") in {"image_url", "image"}
        for part in message.content
    )


def _prune_snapshots(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Collapse every get_snapshot ToolMessage and screenshot image except the most
    recent of each to a short placeholder, returning a NEW list.

    Non-destructive: originals are never mutated (they may be shared with the persisted
    checkpoint) — collapsed messages are fresh ``model_copy`` copies. A stale snapshot
    ToolMessage's *content* is replaced, never the message removed: every tool_call needs
    a matching ToolMessage or the LLM API rejects the history (see
    ``_dangling_tool_messages``). When there is nothing to prune the input list is
    returned by identity so the caller can skip the ``override``.
    """
    # Map tool_call_id -> tool name so we can tell which ToolMessages are snapshots.
    call_names: dict[str, str | None] = {}
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                call_id = tc.get("id")
                if call_id:
                    call_names[call_id] = tc.get("name")

    def _is_snapshot(m: BaseMessage) -> bool:
        return isinstance(m, ToolMessage) and (
            call_names.get(m.tool_call_id) == _SNAPSHOT_TOOL_NAME or m.name == _SNAPSHOT_TOOL_NAME
        )

    snapshot_idxs = [i for i, m in enumerate(messages) if _is_snapshot(m)]
    screenshot_idxs = [i for i, m in enumerate(messages) if _is_screenshot_message(m)]

    # Keep the most recent of each type; collapse everything earlier.
    stale_snapshots = set(snapshot_idxs[:-1])
    stale_screenshots = set(screenshot_idxs[:-1])
    if not stale_snapshots and not stale_screenshots:
        return messages

    pruned: list[AnyMessage] = []
    for i, m in enumerate(messages):
        if i in stale_snapshots:
            pruned.append(m.model_copy(update={"content": _SNAPSHOT_PLACEHOLDER}))
        elif i in stale_screenshots:
            pruned.append(m.model_copy(update={"content": _SCREENSHOT_PLACEHOLDER}))
        else:
            pruned.append(m)
    return pruned


class SnapshotPruneMiddleware(AgentMiddleware):
    """Keep only the most-recent snapshot and screenshot in the model request.

    Snapshots/screenshots accumulate across a phase's ReAct turns, but only the latest
    reflects the live page. We prune the outgoing ``ModelRequest`` only — the persisted
    checkpoint keeps the full transcript, so this is reversible and doesn't perturb budget
    accounting (which reads persisted ``usage_metadata``, not the request). Async because
    the loop runs under ``ainvoke``.
    """

    async def awrap_model_call(self, request: ModelRequest, handler):
        pruned = _prune_snapshots(request.messages)
        if pruned is not request.messages:
            request = request.override(messages=pruned)
        return await handler(request)
