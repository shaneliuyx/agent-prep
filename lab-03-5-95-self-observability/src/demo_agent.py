"""W3.5.95 — minimal self-contained decision agent (stands in for the W4 ReAct
loop, which this lab doesn't depend on).

One step: given a task + a tool registry, the LLM picks the single best next tool.
Two seams the chapter is about:
  * WRITE: every decision is appended to OBSERVABILITY (instrumented).
  * READ: if recall is on, the metacognitive-recall block (self-patterns relevant
    to this task) is injected into the decision prompt BEFORE the model chooses.

The paired-trial (tests/) runs the SAME task with recall OFF vs ON and checks
whether the injected self-pattern changes the chosen tool.
"""
from __future__ import annotations

import os
import time
import uuid

from openai import OpenAI

import metacog_recall
import observability as obs

# Tool registry: pairs where a self-pattern can plausibly flip the default choice
# (grep↔rg on big repos, find↔fd on deep trees, web_search↔read_local_notes, …).
TOOLS: dict[str, str] = {
    "grep": "recursive text search; slow / can time out on very large repositories",
    "rg": "ripgrep — fast recursive text search, handles large repositories well",
    "find": "locate files by name; slow on deep directory trees",
    "fd": "fast file finder; handles deep trees well",
    "web_search": "search the public web",
    "read_local_notes": "read the user's own local notes / past decisions",
    "sql_query": "query the database directly",
    "python_repl": "run arbitrary Python for computation",
}

DECIDE_PROMPT = """You are an agent choosing the single best tool for a task.

AVAILABLE TOOLS:
{tools}

{recall}TASK: {task}

Output ONLY the tool name (one token from the list above). No explanation."""


def _agent_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("LLM_BASE_URL", os.getenv("OMLX_BASE_URL", "http://localhost:8000/v1")),
        api_key=os.getenv("LLM_API_KEY", os.getenv("OMLX_API_KEY", "dummy")),
    )


def _tools_block() -> str:
    return "\n".join(f"- {n}: {d}" for n, d in TOOLS.items())


def _normalize_tool(raw: str) -> str:
    """Map the model's free text back to a known tool name (first match wins)."""
    low = raw.lower()
    for name in TOOLS:
        if name in low:
            return name
    return raw.strip().split()[0] if raw.strip() else "<none>"


def decide(conn: obs.sqlite3.Connection, task: str, *, use_recall: bool,
           run_id: str | None = None, step_idx: int = 0,
           model: str | None = None) -> tuple[str, str]:
    """Pick a tool for `task`. Logs the decision to OBSERVABILITY. Returns
    (chosen_tool, recall_block_used). With use_recall, injects relevant
    self-patterns before the model chooses."""
    run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
    model = model or os.getenv("MODEL_AGENT", "Qwen2.5-Coder-14B-Instruct-MLX-4bit")
    recall = metacog_recall.recall_block(conn, task) if use_recall else ""
    recall_section = (recall + "\n\n") if recall else ""
    prompt = DECIDE_PROMPT.format(tools=_tools_block(), recall=recall_section, task=task)

    t0 = time.perf_counter()
    status, raw = "ok", ""
    try:
        resp = _agent_client().chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=12)
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001 — record the failure as an observation too
        status, raw = "error", str(e)
    latency_ms = (time.perf_counter() - t0) * 1000
    chosen = _normalize_tool(raw) if status == "ok" else "<error>"

    obs.log_observation(
        conn, agent_run_id=run_id, step_idx=step_idx, tool_name=chosen,
        args={"task": task, "recall_used": bool(recall)},
        outcome={"raw": raw}, outcome_status=status, latency_ms=latency_ms)
    return chosen, recall
