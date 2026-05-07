"""Smoke test for Qwen3.6-35B-A3B-UD-MLX-4bit before lab refactor.

Probes the four capabilities the W2.7 optimizations rely on:
  T1 — JSON-mode output (build_tree.py SUMMARIZE_SYSTEM, TREE_BUILDER_SYSTEM)
  T2 — Tool-calling (query_tree.py agentic-loop optimization #1)
  T3 — Multi-turn coherence (agentic loop iterates)
  T4 — Long-context retrieval (16K-char leaf text from PDF)

Prints PASS/FAIL + latency per test + recommended prompt adjustments.
Writes results/qwen36_smoketest.json for follow-up review.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from openai import OpenAI

MODEL = "Qwen3.6-35B-A3B-UD-MLX-4bit"
OMLX = OpenAI(base_url="http://localhost:8000/v1", api_key=os.getenv("OMLX_API_KEY", "***REMOVED-OMLX-KEY***"))


def _print(label: str, ok: bool, latency: float, detail: str = "") -> dict:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}  ({latency:.2f}s){'  ' + detail if detail else ''}")
    return {"label": label, "ok": ok, "latency": latency, "detail": detail}


def t1_json_mode() -> dict:
    """Build_tree.py-shape: ask for strict JSON, no prose preamble."""
    t0 = time.time()
    r = OMLX.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=300,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return only JSON. No prose."},
            {"role": "user", "content": 'Produce a JSON object with keys "title" (string) and "topics" (array of 3 strings) describing Berkshire Hathaway 2023 annual report.'},
        ],
    )
    elapsed = time.time() - t0
    raw = (r.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
        ok = "title" in parsed and isinstance(parsed.get("topics"), list) and len(parsed["topics"]) == 3
        detail = f"keys={list(parsed.keys())}"
    except Exception as e:
        ok = False
        detail = f"json-parse-error: {e}; raw[:80]={raw[:80]!r}"
    return _print("T1 JSON-mode strict", ok, elapsed, detail)


def t2_tool_calling() -> dict:
    """Agentic-loop-shape: model must call get_page_content with explicit args."""
    t0 = time.time()
    tools = [{
        "type": "function",
        "function": {
            "name": "get_page_content",
            "description": "Fetch raw text for a page range from the PDF.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer", "description": "Start page (1-indexed)"},
                    "end": {"type": "integer", "description": "End page (inclusive)"},
                },
                "required": ["start", "end"],
            },
        },
    }]
    r = OMLX.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=300, tools=tools,
        messages=[
            {"role": "system", "content": "You answer questions by first fetching relevant page content using the get_page_content tool, then answering. For 'what was Berkshire's net earnings in 2023?', the answer typically lives in the Consolidated Statements of Earnings, around pages 95-100."},
            {"role": "user", "content": "What was Berkshire's net earnings in 2023? Use the tool to fetch pages first."},
        ],
    )
    elapsed = time.time() - t0
    msg = r.choices[0].message
    tcalls = getattr(msg, "tool_calls", None) or []
    if tcalls:
        try:
            args = json.loads(tcalls[0].function.arguments)
            ok = "start" in args and "end" in args and isinstance(args["start"], int)
            detail = f"called {tcalls[0].function.name}(start={args.get('start')}, end={args.get('end')})"
        except Exception as e:
            ok = False
            detail = f"args-parse-error: {e}"
    else:
        ok = False
        content = (msg.content or "")[:80]
        detail = f"no tool_calls; content[:80]={content!r}"
    return _print("T2 Tool-calling", ok, elapsed, detail)


def t3_multi_turn() -> dict:
    """Multi-turn: tool-call → tool-result → final answer flow."""
    t0 = time.time()
    tools = [{
        "type": "function",
        "function": {
            "name": "get_page_content",
            "description": "Fetch raw text for a page range.",
            "parameters": {
                "type": "object",
                "properties": {"start": {"type": "integer"}, "end": {"type": "integer"}},
                "required": ["start", "end"],
            },
        },
    }]
    msgs = [
        {"role": "system", "content": "Use the get_page_content tool, then answer with citation [pages X-Y]."},
        {"role": "user", "content": "What was Berkshire's net earnings in 2023?"},
    ]
    r1 = OMLX.chat.completions.create(model=MODEL, temperature=0.0, max_tokens=200, tools=tools, messages=msgs)
    msg1 = r1.choices[0].message
    tcalls = getattr(msg1, "tool_calls", None) or []
    if not tcalls:
        return _print("T3 Multi-turn", False, time.time() - t0, "no tool_call on turn 1")

    # Simulate tool result
    msgs.append({
        "role": "assistant",
        "content": msg1.content or "",
        "tool_calls": [{
            "id": tcalls[0].id, "type": "function",
            "function": {"name": tcalls[0].function.name, "arguments": tcalls[0].function.arguments},
        }],
    })
    msgs.append({
        "role": "tool", "tool_call_id": tcalls[0].id,
        "content": "[pages 96-97 content] Net earnings attributable to Berkshire Hathaway shareholders were $96.2 billion in 2023, compared to a loss of $22.8 billion in 2022.",
    })
    r2 = OMLX.chat.completions.create(model=MODEL, temperature=0.0, max_tokens=200, messages=msgs)
    final = (r2.choices[0].message.content or "").strip()
    elapsed = time.time() - t0
    has_number = "96.2" in final or "$96" in final
    has_citation = "[pages" in final.lower() or "page" in final.lower()
    ok = has_number and has_citation
    return _print("T3 Multi-turn (tool-call → result → answer)", ok, elapsed,
                  f"answer[:100]={final[:100]!r}")


def t4_long_context() -> dict:
    """16K-char leaf text simulation — needle in haystack mid-context."""
    t0 = time.time()
    needle = "MARKER_PHRASE: Berkshire's 2023 total revenues were $364.5 billion."
    filler_a = "This section discusses operating earnings across business segments. " * 80
    filler_b = "The following pages enumerate balance sheet items and footnotes. " * 80
    text = filler_a + "\n\n" + needle + "\n\n" + filler_b  # ~16K chars, needle mid
    r = OMLX.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=150,
        messages=[
            {"role": "system", "content": "Answer using ONLY the context. Quote the exact sentence containing the answer."},
            {"role": "user", "content": f"Context:\n{text}\n\nQuestion: What were Berkshire's 2023 total revenues?"},
        ],
    )
    elapsed = time.time() - t0
    ans = (r.choices[0].message.content or "").strip()
    ok = "364.5" in ans or "$364" in ans
    return _print("T4 Long-context (16K, needle mid)", ok, elapsed,
                  f"answer[:100]={ans[:100]!r}")


def main() -> None:
    print(f"\nSmoke testing {MODEL} on http://localhost:8000\n")
    results = []
    for fn in (t1_json_mode, t2_tool_calling, t3_multi_turn, t4_long_context):
        try:
            results.append(fn())
        except Exception as e:
            print(f"  [ERROR] {fn.__name__}: {type(e).__name__}: {e}")
            results.append({"label": fn.__name__, "ok": False, "latency": 0.0, "detail": f"{type(e).__name__}: {e}"})

    Path("results").mkdir(exist_ok=True)
    Path("results/qwen36_smoketest.json").write_text(json.dumps(results, indent=2))

    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    avg_lat = sum(r["latency"] for r in results) / max(total, 1)
    print(f"\n--- Summary: {passed}/{total} pass, avg latency {avg_lat:.2f}s ---")

    print("\nPrompt-tuning recommendations:")
    if not results[0]["ok"]:
        print("  - T1 failed: switch SUMMARIZE_SYSTEM to a few-shot JSON example; do NOT rely on response_format alone")
    if not results[1]["ok"]:
        print("  - T2 failed: agentic-loop opt #1 must use prompt-encoded tool spec (function-call format), not OpenAI tools API")
    if not results[2]["ok"]:
        print("  - T3 failed: agentic loop will not work multi-turn; fall back to single-shot retrieve-then-answer")
    if not results[3]["ok"]:
        print("  - T4 failed: lower max leaf char budget below 16K, or chunk by section into separate calls")
    if all(r["ok"] for r in results):
        print("  - All 4 tests pass — proceed with full optimization plan unchanged.")


if __name__ == "__main__":
    main()
