"""Recovering JSON from a language model, in escalating desperation.

This module is not redundant with ``response_format``, and the reason is
specific rather than defensive. ``litellm.drop_params = True`` is set globally
in ``llm/client.py``, and ``LLMClient.chat`` passes
``fallbacks=[settings.fallback_model]`` on every call. So when the primary
provider rate-limits, LiteLLM retries against Claude — and silently *drops* the
``response_format`` parameter on the way, because Claude does not accept it.
There is no exception and no warning. The caller that was promised a strict
schema receives Claude's habitual prose-wrapped, fence-delimited JSON instead.

Structured output is therefore a property of the primary model, not of the call.
Parse accordingly.
"""

import json
import re
from typing import Any


class MalformedGenerationError(ValueError):
    """The model's reply could not be read as JSON by any available means."""


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _first_balanced_object(text: str) -> str | None:
    """The first brace-balanced ``{...}`` span, honouring strings and escapes.

    Scanned rather than matched with a regex because a regex cannot balance
    braces, and the snippets this corpus generates are full of them — JSON
    literals, dict defaults, f-strings. A greedy ``\\{.*\\}`` would swallow
    trailing prose; a lazy one would stop at the first nested close.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for position in range(start, len(text)):
        char = text[position]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : position + 1]
    return None


def extract_json_object(content: str) -> dict[str, Any]:
    """Read a JSON object out of an LLM reply. Four passes, cheapest first.

    1. ``json.loads`` on the whole string — what ``response_format`` guarantees
       when it survives, and the only pass that costs nothing.
    2. The contents of the first ``` fence, for a model that narrates.
    3. The first brace-balanced span, for a model that narrates without fencing.
    4. Pass 3 with trailing commas stripped.

    Raises ``MalformedGenerationError`` when all four fail, so the caller can
    quarantine one unit and keep going rather than losing the run.
    """
    if not content or not content.strip():
        raise MalformedGenerationError("empty response")

    attempts: list[str] = [content.strip()]

    fenced = _FENCE.search(content)
    if fenced:
        attempts.append(fenced.group(1).strip())

    balanced = _first_balanced_object(content)
    if balanced:
        attempts.append(balanced)
        attempts.append(_TRAILING_COMMA.sub(r"\1", balanced))

    for candidate in attempts:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        # A model told to return {"items": [...]} occasionally returns the bare
        # list. Accepting it costs one line and saves a paid repair round-trip.
        if isinstance(parsed, list):
            return {"items": parsed}

    raise MalformedGenerationError(
        f"no JSON object found in a {len(content)}-character response"
    )


def extract_items(content: str) -> list[dict[str, Any]]:
    """The ``items`` array, with the shapes models actually return normalised."""
    payload = extract_json_object(content)
    items = payload.get("items")
    if items is None:
        # A single item returned unwrapped — recoverable, and cheaper to accept
        # than to re-ask for.
        return [payload] if "question" in payload else []
    if not isinstance(items, list):
        raise MalformedGenerationError(f"'items' is {type(items).__name__}, not a list")
    return [item for item in items if isinstance(item, dict)]
