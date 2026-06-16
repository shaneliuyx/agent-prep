"""
src/llm_client.py — the ONE oMLX client every pattern shares.

Copied verbatim in spirit from lab-04-react-from-scratch/src/react.py §1:
  - local oMLX OpenAI-compatible endpoint (http://127.0.0.1:8000/v1)
  - workhorse model gemma-4-26B-A4B-it-heretic-4bit via MODEL_SONNET
  - api_key placeholder "not-needed" (SDK rejects empty; oMLX ignores auth)

NO cloud / VibeProxy profiles. Zero cloud spend. If you point OMLX_URL elsewhere
the whole zoo follows — the model is selected by the `model:` field, one endpoint.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, cast

from openai import OpenAI

CLIENT = OpenAI(
    base_url=os.getenv("OMLX_URL", "http://127.0.0.1:8000/v1"),
    api_key=os.getenv("OMLX_API_KEY", "not-needed"),
    timeout=120,
    max_retries=0,
)
MODEL = os.getenv("MODEL_SONNET", "gemma-4-26B-A4B-it-heretic-4bit")


@dataclass
class LLMResponse:
    """Structured return so callers don't navigate nested SDK attributes."""

    content: str
    tool_calls: list
    prompt_tokens: int
    completion_tokens: int


def call_llm(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> LLMResponse:
    """Single chat-completions round-trip against oMLX.

    `tools=None` sends no tool schema (used by CodeAct / ReWOO planner+solver,
    which want plain text out, not structured tool calls). Real usage numbers
    come straight from response.usage — never estimated.
    """
    resp = CLIENT.chat.completions.create(
        model=MODEL,
        messages=cast(Any, messages),
        tools=cast(Any, tools) if tools else None,
        tool_choice="auto" if tools else cast(Any, None),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    usage = resp.usage  # may be None on some local backends
    return LLMResponse(
        content=choice.message.content or "",
        tool_calls=choice.message.tool_calls or [],
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )
