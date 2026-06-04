"""W3.5.95 — the SAME self-observability seams wired into a REAL agent framework
(smolagents), to show the production integration pattern (chapter's W4-ReAct tie):

  * WRITE seam — instrument the TOOLS (one OBSERVABILITY row per tool call). This
    is framework-agnostic and the canonical seam (chapter Phase 2 "tool wrapper"):
    a CodeAgent executes *code*, not discrete tool_calls, so wrapping the tool is
    cleaner than a step_callback. A step_callback is added too, for step-level
    metadata (errors / timing).
  * READ seam — prepend the metacognitive-recall block to the task before run().

Demonstration only: tools are stubs (the lab is about the memory seams, not real
tool execution). Runs against local oMLX via OpenAIServerModel.
"""
from __future__ import annotations

import functools
import os
import pathlib
import time
import uuid

from smolagents import CodeAgent, OpenAIServerModel, tool

import metacog_recall
import observability as obs

# Shared OBSERVABILITY handle the instrumented tools write to (smolagents tools
# are plain functions; a module handle threads the connection + run/step state).
_OBS: dict = {"conn": None, "run_id": None, "step": 0}


def _instrumented(fn):
    """Wrap a tool so every call appends one OBSERVABILITY row (the WRITE seam)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        status, out = "ok", None
        try:
            out = fn(*args, **kwargs)
            return out
        except Exception as e:  # noqa: BLE001 — record tool failures too
            status, out = "error", str(e)
            raise
        finally:
            if _OBS["conn"] is not None:
                obs.log_observation(
                    _OBS["conn"], agent_run_id=_OBS["run_id"], step_idx=_OBS["step"],
                    tool_name=fn.__name__, args={"args": args, "kwargs": kwargs},
                    outcome={"result": out}, outcome_status=status,
                    latency_ms=(time.perf_counter() - t0) * 1000)
                _OBS["step"] += 1
    return wrapper


# Tool stubs (mirror demo_agent's registry). @tool reads the signature + docstring.
@tool
@_instrumented
def grep(query: str) -> str:
    """Recursive text search (slow on very large repositories).

    Args:
        query: the text pattern to search for.
    """
    return f"grep found 2 matches for {query!r}"


@tool
@_instrumented
def rg(query: str) -> str:
    """Ripgrep — fast recursive text search, handles large repositories well.

    Args:
        query: the text pattern to search for.
    """
    return f"rg found 2 matches for {query!r}"


@tool
@_instrumented
def read_local_notes(topic: str) -> str:
    """Read the user's own local notes / past decisions.

    Args:
        topic: the subject to look up in the local notes.
    """
    return f"local notes on {topic!r}: (cached answer)"


def _step_logger(memory_step, agent):  # step-level metadata seam (errors/timing)
    err = getattr(memory_step, "error", None)
    if err and _OBS["conn"] is not None:
        obs.log_observation(
            _OBS["conn"], agent_run_id=_OBS["run_id"], step_idx=_OBS["step"],
            tool_name="<step_error>", args={"step": getattr(memory_step, "step_number", -1)},
            outcome={"error": str(err)}, outcome_status="error", latency_ms=0.0)
        _OBS["step"] += 1


def build_agent(conn, run_id: str | None = None) -> CodeAgent:
    _OBS.update(conn=conn, run_id=run_id or f"sa-{uuid.uuid4().hex[:8]}", step=0)
    model = OpenAIServerModel(
        model_id=os.getenv("MODEL_AGENT", "Qwen2.5-Coder-14B-Instruct-MLX-4bit"),
        api_base=os.getenv("OMLX_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("OMLX_API_KEY", "dummy"))
    # use_structured_outputs_internally: the agent forces the model to return its
    # action as {"thought":..., "code":...} JSON via response_format, instead of
    # free text scraped between <code></code> tags. This sidesteps smolagents
    # issue #1851 — local/MLX models emit the partial stop sequence "</code"
    # (no closing ">"), which leaks into the parsed Python and crashes ast.parse
    # with a SyntaxError. oMLX supports response_format json_schema, so the JSON
    # path is clean. (BCJ Entry 5.)
    return CodeAgent(tools=[grep, rg, read_local_notes], model=model,
                     step_callbacks=[_step_logger], max_steps=3, verbosity_level=0,
                     use_structured_outputs_internally=True)


def run_with_recall(agent: CodeAgent, conn, task: str, use_recall: bool = True) -> str:
    """READ seam: prepend the recalled self-patterns to the task before the run."""
    block = metacog_recall.recall_block(conn, task) if use_recall else ""
    full_task = (block + "\n\n" + task) if block else task
    return agent.run(full_task)


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
    # check_same_thread=False: smolagents runs the agent's code in a worker thread,
    # so the instrumented tools write log_observation() off the connect() thread.
    conn = obs.connect(
        str(pathlib.Path(__file__).resolve().parent.parent / "data" / "memory.db"),
        check_same_thread=False)
    # Contrarian self-pattern so recall has something to override the prior with;
    # shares terms with the task ("search", "repository") so BM25 recall fires.
    # DELETE-then-INSERT (not bare INSERT) so reruns REPLACE the demo fact instead
    # of stacking identical copies that recall would then echo K times.
    fact = ("When I search this repository, rg segfaults on its symlinked vendor "
            "dirs; plain grep is the search tool that completes here.")
    conn.execute("DELETE FROM learning WHERE pattern_text=?", (fact,))
    conn.execute("INSERT INTO learning(type,pattern_text,confidence,is_self_caused,source_rows,ts) "
                 "VALUES('tool_preference',?,0.8,1,'[]',?)", (fact, time.time())); conn.commit()
    # UNIQUE run_id per invocation: OBSERVABILITY is append-only with PK
    # (run_id, step_idx). A fixed id collides with rows a PREVIOUS run already
    # wrote to the persistent DB → IntegrityError on the second run. (BCJ Entry 8.)
    run_id = f"smolagents-demo-{uuid.uuid4().hex[:8]}"
    agent = build_agent(conn, run_id=run_id)
    task = "Search this repository for the definition of parse_config, then report what you found."
    print(">>> recall block injected:\n" + (metacog_recall.recall_block(conn, task) or "(none)"))
    out = run_with_recall(agent, conn, task)
    print(f"\n>>> agent final answer: {str(out)[:120]}")
    rows = obs.observations_by_run(conn, run_id)
    print(f"\n>>> OBSERVABILITY rows written by the run ({len(rows)}):")
    for r in rows:
        print(f"  step{r['step_idx']} {r['tool_name']} status={r['outcome_status']} {r['latency_ms']:.0f}ms")


if __name__ == "__main__":
    main()
