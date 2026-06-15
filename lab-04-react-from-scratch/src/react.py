"""
src/react.py — ReAct agent loop, no framework.

Architecture:
  - context_for_llm()   : assemble the full prompt from parts
  - call_llm()          : single LLM call; returns (content, tool_calls, usage)
  - parse_tool_call()   : extract tool name + args from the model's response
  - run_tool()          : dispatch to registered tool functions
  - agent_run()         : the outer loop; returns final answer string

Observability:
  - Every iteration logs a row to SQLite via log_event() (see src/obs.py Phase 4).
  - Pass obs=False to agent_run() to disable logging (useful in unit tests).

Usage:
  from src.react import agent_run, register_tool
  register_tool("my_tool", my_fn, MY_SCHEMA)
  answer = agent_run("What is 12 * 34?")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. Client setup
#    MODEL_SONNET defaults to gemma-4-26B-A4B-it-heretic-4bit — the measured
#    workhorse (the only fleet model scoring 1.00 on tool+json+reason+instr;
#    see §1.4). oMLX serves all models on ONE endpoint (:8000/v1), routed by
#    the `model:` field, so retiering swaps MODEL only — the URL is constant.
#    For a larger reasoning attempt use the `hard_loop` role (Qwen3.5-27B).
#    The fallback strings let the file run without a .env if needed (e.g., in
#    CI against a stub server).
#    oMLX requires no auth, but the OpenAI SDK rejects an empty api_key at
#    construction time, so we pass a non-empty placeholder ("not-needed").
# ---------------------------------------------------------------------------
_client = OpenAI(
    base_url=os.getenv("OMLX_URL", "http://127.0.0.1:8000/v1"),
    api_key=os.getenv("OMLX_API_KEY", "not-needed"),
)
MODEL = os.getenv("MODEL_SONNET", "gemma-4-26B-A4B-it-heretic-4bit")

# ---------------------------------------------------------------------------
# 2. Loop constants
#    MAX_ITER is the "retry budget." If the loop runs this many times without
#    returning a final answer, something is wrong — stop and surface it.
#    CONTEXT_TOKEN_LIMIT is a soft guard: if the scratchpad grows past this
#    threshold, truncate the oldest tool results to prevent a 400 error from
#    the inference server.
# ---------------------------------------------------------------------------
MAX_ITER: int = int(os.getenv("REACT_MAX_ITER", "12"))
CONTEXT_TOKEN_LIMIT: int = int(os.getenv("REACT_CTX_LIMIT", "28000"))

# ---------------------------------------------------------------------------
# 3. Tool registry
#    Tools are stored in a module-level dict. register_tool() adds entries.
#    Each entry has three keys:
#      fn      : the Python callable
#      schema  : the OpenAI-format tool definition (type/function/parameters)
#      budget  : max calls per agent_run(); enforced in run_tool()
# ---------------------------------------------------------------------------
_TOOLS: dict[str, dict[str, Any]] = {}


def register_tool(
    name: str,
    fn: Callable,
    schema: dict,
    max_calls: int = 5,
) -> None:
    """Register a tool so the loop can dispatch to it."""
    _TOOLS[name] = {"fn": fn, "schema": schema, "budget": max_calls}


def tool_schemas() -> list[dict]:
    """Return all registered tool schemas in the format OpenAI expects."""
    return [t["schema"] for t in _TOOLS.values()]


# ---------------------------------------------------------------------------
# 4. Scratchpad
#    The scratchpad is the event log. It grows by appending (tool_call, result)
#    pairs; it is never mutated in place. This is the "event sourcing" half of
#    the loop — if you replay the scratchpad you can reconstruct the full
#    reasoning trace.
# ---------------------------------------------------------------------------
@dataclass
class ScratchpadEntry:
    role: str           # "assistant" | "tool"
    content: str        # the raw text the model emitted, or the tool result
    tool_call_id: str = ""
    name: str = ""      # tool name, only for role=="tool"


@dataclass
class Scratchpad:
    entries: list[ScratchpadEntry] = field(default_factory=list)

    def append_assistant(self, content: str, tool_calls: list | None = None) -> None:
        """Record what the assistant said (or requested)."""
        self.entries.append(
            ScratchpadEntry(role="assistant", content=content or "")
        )
        # Store the raw tool_call objects on the entry for context_for_llm()
        self.entries[-1]._tool_calls = tool_calls or []

    def append_tool_result(self, tool_call_id: str, name: str, result: str) -> None:
        """Record the result of executing one tool call."""
        self.entries.append(
            ScratchpadEntry(
                role="tool",
                content=result,
                tool_call_id=tool_call_id,
                name=name,
            )
        )

    def estimated_tokens(self) -> int:
        """Rough token count: 1 token ≈ 4 characters (good enough for the guard)."""
        total_chars = sum(len(e.content) for e in self.entries)
        return total_chars // 4


# ---------------------------------------------------------------------------
# 5. Context assembly
#    Every iteration rebuilds the full message list from scratch. This is
#    intentionally stateless: if you have the system prompt, the user message,
#    and the scratchpad, you can reconstruct the exact context the model saw.
#    This is the "idempotent re-computation" property — important for debugging.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a capable assistant with access to tools.

When you need information or need to perform an action, call the appropriate tool.
When you have gathered enough information to answer the user, respond in plain text
with no tool call — that ends the loop.

Rules:
- Call tools one at a time unless the task clearly requires parallel calls.
- If a tool returns an error, acknowledge it and decide whether to retry with
  different arguments, try a different tool, or explain to the user why you cannot proceed.
- Never call the same tool with identical arguments more than twice in a row.
- If you are uncertain and no tool will help, say so clearly.
"""


def context_for_llm(user_msg: str, scratchpad: Scratchpad) -> list[dict]:
    """
    Assemble the full message list for the LLM call.

    Structure:
      [system] → [user] → [assistant + tool_calls?] → [tool result] → ... repeat
    """
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    # Replay the scratchpad. Each assistant turn may have raw tool_call objects;
    # each tool turn is the result of one of those calls.
    for entry in scratchpad.entries:
        if entry.role == "assistant":
            msg: dict = {"role": "assistant", "content": entry.content}
            raw_calls = getattr(entry, "_tool_calls", [])
            if raw_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in raw_calls
                ]
            messages.append(msg)
        else:
            messages.append({
                "role": "tool",
                "tool_call_id": entry.tool_call_id,
                "name": entry.name,
                "content": entry.content,
            })

    return messages


# ---------------------------------------------------------------------------
# 6. LLM call
#    Thin wrapper over the OpenAI client. Returns a named tuple so the caller
#    does not need to navigate nested attributes from the raw response.
# ---------------------------------------------------------------------------
@dataclass
class LLMResponse:
    content: str                    # text content (may be empty if tool_calls present)
    tool_calls: list                # list of ToolCall objects, may be empty
    prompt_tokens: int
    completion_tokens: int


def call_llm(messages: list[dict]) -> LLMResponse:
    """Send messages to the model; return a structured response."""
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tool_schemas() or None,       # None disables tool_choice parsing
        tool_choice="auto",
        temperature=0.0,                    # determinism; important for bad-case tests
        max_tokens=1024,
    )
    choice = resp.choices[0]
    return LLMResponse(
        content=choice.message.content or "",
        tool_calls=choice.message.tool_calls or [],
        prompt_tokens=resp.usage.prompt_tokens,
        completion_tokens=resp.usage.completion_tokens,
    )


# ---------------------------------------------------------------------------
# 7. Tool dispatch
#    run_tool() resolves the name, enforces the per-tool call budget, calls
#    the function, and truncates the result if it is too large to fit in context.
#    It never raises — errors become string results that the model can read.
# ---------------------------------------------------------------------------
RESULT_TRUNCATION_CHARS = 4000   # ~1000 tokens; prevents 50KB blobs from blowing context


def run_tool(name: str, args: dict, call_counts: dict[str, int]) -> str:
    """
    Execute a registered tool. Returns a string in all cases — errors included.

    call_counts tracks how many times each tool has been called this run;
    enforced against each tool's budget.
    """
    # Guard: unknown tool
    if name not in _TOOLS:
        return (
            f"ERROR: tool '{name}' is not registered. "
            f"Available tools: {list(_TOOLS.keys())}"
        )

    entry = _TOOLS[name]

    # Guard: per-tool call budget (circuit breaker)
    call_counts[name] = call_counts.get(name, 0) + 1
    if call_counts[name] > entry["budget"]:
        return (
            f"ERROR: tool '{name}' has exceeded its call budget "
            f"({entry['budget']} calls). Use a different approach."
        )

    # Execute; catch all exceptions so the loop continues
    try:
        result = entry["fn"](**args)
    except TypeError as e:
        # Most common: missing required argument, or wrong type
        return f"ERROR: bad arguments for tool '{name}': {e}"
    except Exception as e:
        return f"ERROR: tool '{name}' raised {type(e).__name__}: {e}"

    result_str = str(result)

    # Guard: truncate oversized results (the "50KB blob" failure mode)
    if len(result_str) > RESULT_TRUNCATION_CHARS:
        result_str = (
            result_str[:RESULT_TRUNCATION_CHARS]
            + f"\n[TRUNCATED: result was {len(result_str)} chars; "
            f"only the first {RESULT_TRUNCATION_CHARS} shown]"
        )

    return result_str


def _evict_if_over_limit(scratchpad: "Scratchpad") -> "ScratchpadEntry | None":
    """If the scratchpad exceeds CONTEXT_TOKEN_LIMIT, FIFO-evict the OLDEST tool
    result and return it; else return None.

    Tool results are the largest, most-replaceable entries, so they are evicted
    first — assistant reasoning is preserved. One call drops at most one entry
    (agent_run calls it once per iteration, so a very large scratchpad trims
    down over successive turns). Extracted from the loop so the eviction policy
    can be unit-tested directly (Phase 5, Scenario 10)."""
    if scratchpad.estimated_tokens() <= CONTEXT_TOKEN_LIMIT:
        return None
    tool_entries = [e for e in scratchpad.entries if e.role == "tool"]
    if not tool_entries:
        return None
    oldest = tool_entries[0]
    scratchpad.entries.remove(oldest)
    return oldest


# ---------------------------------------------------------------------------
# 8. The outer loop
#    agent_run() is the entry point. It drives iterations until one of three
#    exit conditions: (a) the model returns text with no tool calls, (b) the
#    iteration count hits MAX_ITER, or (c) a context-size guard fires.
# ---------------------------------------------------------------------------
def agent_run(
    user_msg: str,
    obs: bool = True,
    run_id: str | None = None,
) -> str:
    """
    Run the ReAct loop for a single user message.

    Args:
        user_msg: The question or instruction from the user.
        obs:      If True, log every iteration to SQLite (requires src/obs.py).
        run_id:   Optional trace ID; auto-generated if not provided.

    Returns:
        The agent's final text answer.
    """
    if obs:
        try:
            from src.obs import log_event
        except ImportError:
            obs = False   # graceful degradation if obs module not yet created

    run_id = run_id or f"run_{int(time.time()*1000)}"
    scratchpad = Scratchpad()
    call_counts: dict[str, int] = {}
    last_tool_signature: tuple | None = None   # for circular-reasoning detection

    for iteration in range(MAX_ITER):

        # --- context-size guard: FIFO-evict the oldest tool result if over limit ---
        # (extracted to _evict_if_over_limit so the policy is unit-testable)
        _evict_if_over_limit(scratchpad)

        messages = context_for_llm(user_msg, scratchpad)

        # --- LLM call ---
        t0 = time.perf_counter()
        llm_resp = call_llm(messages)
        llm_latency_ms = int((time.perf_counter() - t0) * 1000)

        # --- no tool call → final answer ---
        if not llm_resp.tool_calls:
            if obs:
                log_event(
                    run_id=run_id,
                    iteration=iteration,
                    event_type="final_answer",
                    tool_name=None,
                    prompt_tokens=llm_resp.prompt_tokens,
                    completion_tokens=llm_resp.completion_tokens,
                    tool_latency_ms=0,
                    tool_error=None,
                )
            return llm_resp.content or "(agent produced no output)"

        # --- process each tool call in the response ---
        scratchpad.append_assistant(llm_resp.content, llm_resp.tool_calls)

        for tc in llm_resp.tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}
                tool_result = (
                    f"ERROR: arguments for tool '{tool_name}' were not valid JSON. "
                    f"Raw: {tc.function.arguments!r}"
                )
            else:
                # --- circular reasoning guard ---
                sig = (tool_name, json.dumps(tool_args, sort_keys=True))
                if sig == last_tool_signature:
                    tool_result = (
                        f"ERROR: you just called '{tool_name}' with identical arguments. "
                        "Do not repeat the same call. Try a different approach."
                    )
                else:
                    last_tool_signature = sig
                    t_tool = time.perf_counter()
                    tool_result = run_tool(tool_name, tool_args, call_counts)
                    tool_latency_ms = int((time.perf_counter() - t_tool) * 1000)

            if obs:
                log_event(
                    run_id=run_id,
                    iteration=iteration,
                    event_type="tool_call",
                    tool_name=tool_name,
                    prompt_tokens=llm_resp.prompt_tokens,
                    completion_tokens=llm_resp.completion_tokens,
                    tool_latency_ms=tool_latency_ms if "tool_latency_ms" in dir() else 0,
                    tool_error=tool_result if tool_result.startswith("ERROR") else None,
                )

            scratchpad.append_tool_result(tc.id, tool_name, tool_result)

    # MAX_ITER exhausted — dead-letter handling
    dlq_msg = (
        f"[AGENT STOPPED: reached maximum iterations ({MAX_ITER}) without a final answer. "
        f"Last tool calls: {[tc.function.name for tc in llm_resp.tool_calls]}. "
        "Check the scratchpad for stuck reasoning.]"
    )
    if obs:
        log_event(
            run_id=run_id,
            iteration=MAX_ITER,
            event_type="max_iter_exceeded",
            tool_name=None,
            prompt_tokens=0,
            completion_tokens=0,
            tool_latency_ms=0,
            tool_error=dlq_msg,
        )
    return dlq_msg