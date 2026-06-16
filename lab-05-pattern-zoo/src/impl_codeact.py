"""
src/impl_codeact.py — CodeAct for the pattern zoo.

CodeAct (Wang et al., 2024, "Executable Code Actions Elicit Better LLM Agents"):
the agent's ACTION SPACE is *Python code*, not a structured JSON tool call. Each
turn the model emits a fenced ```python block; the harness executes it in a
restricted namespace where the tools (src/tools.py) are in scope as plain
callables, captures stdout as the OBSERVATION, and loops. The model signals it is
done by printing a line  FINAL: <answer>.

Teaching point vs ReAct: because the action is code, the model can COMPOSE tools,
loop, and do arithmetic in ONE action instead of one LLM round-trip per tool call.
On the canonical task ReAct needs ~5 round-trips (kb_lookup x3, add, multiply,
then answer); CodeAct can do the whole chain in a single code block -> far fewer
LLM calls.

Failure modes guarded (the assignment's "obvious failures"):
  - model emits prose, no code block      -> observation nudges it to emit code
  - model code raises                     -> traceback is fed back as observation
  - bounded retries via MAX_ITER          -> no infinite loop
  - restricted namespace                  -> only the tool callables + a small
                                             allow-list of builtins are exposed;
                                             no import / open / __import__.

Entry point: run(task: str) -> AgentResult

SECURITY NOTE: exec() of model output is NOT a real sandbox. The namespace is
restricted (no __import__, no open, no file/network builtins) which bounds the
non-adversarial teaching agent here, but a determined adversary can still escape
via attribute gadgets. Real isolation (subprocess + rlimits, then containers) is
the lab-04 python_repl story and W11.5 — out of scope for this pattern demo.
"""

from __future__ import annotations

import io
import re
import time
import traceback
from contextlib import redirect_stdout
from typing import Any

from src.llm_client import call_llm
from src.schema import CANONICAL_TASK, AgentResult
from src.tools import TOOLS, tools_doc

PATTERN = "CodeAct"
MAX_ITER = 6

# A code block fence:  ```python ... ```   (also tolerate a bare ``` ... ```).
_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
# The done sentinel the model prints when finished.
_FINAL = re.compile(r"^\s*FINAL:\s*(.+?)\s*$", re.MULTILINE)

# Builtins the executed code is allowed to touch. Deliberately tiny: enough for
# arithmetic / printing / iteration, nothing for I/O, import, or reflection.
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "len": len, "range": range, "enumerate": enumerate, "zip": zip,
    "float": float, "int": int, "str": str, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "print": print, "sorted": sorted, "map": map, "filter": filter,
}

SYSTEM_PROMPT = f"""You are a CodeAct agent. Your ONLY way to act is to emit a
single fenced Python code block. The harness will execute it and return its stdout
as the observation. The following tool functions are already in scope — call them
directly, do NOT import anything:

{tools_doc()}

Protocol:
- Each turn, emit exactly one ```python ... ``` block.
- print(...) any intermediate values you want to observe.
- You may call several tools and do arithmetic IN ONE block — you do not need a
  separate turn per tool.
- When you have the final number, print a line exactly of the form:
      print("FINAL:", <the_number>)
  and the loop will end.
- Do NOT import modules, open files, or use builtins other than basic ones
  (print, int, float, sum, range, ...). Only the tool functions above plus
  arithmetic are available.
"""


def _make_namespace() -> dict[str, Any]:
    """Build the restricted exec namespace: tool callables + safe builtins only."""
    ns: dict[str, Any] = dict(TOOLS)            # kb_lookup / add / multiply
    ns["__builtins__"] = _SAFE_BUILTINS          # block import/open/etc.
    return ns


def _extract_code(text: str) -> str | None:
    """Pull the first fenced code block out of the model's message, or None."""
    m = _CODE_BLOCK.search(text)
    return m.group(1) if m else None


def _execute(code: str, ns: dict[str, Any]) -> str:
    """Exec `code` in `ns`, capturing stdout. Returns stdout, or a traceback
    string prefixed with ERROR: if it raises. Never propagates the exception."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(compile(code, "<codeact>", "exec"), ns)  # noqa: S102 (restricted ns)
    except Exception:
        tb = traceback.format_exc(limit=3)
        captured = buf.getvalue()
        return f"ERROR: code raised:\n{tb}" + (f"\n[stdout before error]\n{captured}" if captured else "")
    out = buf.getvalue().strip()
    return out or "(no output — remember to print() what you want to observe)"


def run(task: str = CANONICAL_TASK) -> AgentResult:
    """Run the CodeAct loop on `task`; return the measured AgentResult."""
    t0 = time.perf_counter()
    ns = _make_namespace()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    llm_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    trace: list[str] = []

    for _ in range(MAX_ITER):
        resp = call_llm(messages, tools=None)   # plain text out; action is code
        llm_calls += 1
        prompt_tokens += resp.prompt_tokens
        completion_tokens += resp.completion_tokens
        assistant_msg = resp.content
        messages.append({"role": "assistant", "content": assistant_msg})

        # A FINAL: sentinel in the assistant text itself ends the loop, even if
        # it came from describing the printed result.
        fin = _FINAL.search(assistant_msg)

        code = _extract_code(assistant_msg)
        if code is None:
            # Failure mode: prose, no code. Nudge and retry (bounded by MAX_ITER).
            if fin:  # model answered in prose with FINAL: — accept it
                trace.append(f"final(prose): {fin.group(1)!r}")
                return _result(PATTERN, fin.group(1).strip(), llm_calls,
                               prompt_tokens, completion_tokens, t0, trace)
            obs = ("ERROR: no ```python``` code block found in your reply. "
                   "Emit exactly one fenced Python block to act.")
            trace.append("no-code -> nudge")
            messages.append({"role": "user", "content": f"Observation:\n{obs}"})
            continue

        # Execute the code; stdout becomes the observation.
        obs = _execute(code, ns)
        trace.append(f"exec -> {obs!r}")

        # Did the executed code print a FINAL: line?
        fin_obs = _FINAL.search(obs)
        if fin_obs:
            return _result(PATTERN, fin_obs.group(1).strip(), llm_calls,
                           prompt_tokens, completion_tokens, t0, trace)

        messages.append({"role": "user", "content": f"Observation:\n{obs}"})

    # MAX_ITER exhausted.
    return _result(
        PATTERN,
        f"[CodeAct stopped: reached MAX_ITER={MAX_ITER} without a FINAL: line]",
        llm_calls, prompt_tokens, completion_tokens, t0, trace,
    )


def _result(pattern, answer, calls, ptok, ctok, t0, trace) -> AgentResult:
    return AgentResult(
        pattern=pattern,
        answer=answer,
        llm_calls=calls,
        prompt_tokens=ptok,
        completion_tokens=ctok,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        trace=tuple(trace),
    )


if __name__ == "__main__":
    print(run())
