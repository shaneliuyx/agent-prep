"""compare_models.py — head-to-head of three local MLX models on oMLX :8000.

Two MEASURED axes (measured-engineering ethos: every number traces to this run):

  performance  — decode throughput (tok/s) and time-to-first-token (TTFT), from a
                 STREAMED completion. tok/s = completion_tokens / decode_window.
  reasoning    — pass-rate on a small curated task set, graded PROGRAMMATICALLY
                 (no LLM-judge bias): math/logic by exact-match on a tagged final
                 answer; code by running unit-test asserts in a sandboxed subprocess.

Models (served ids on oMLX — note the first drops its `arthurcollet/` org prefix):
  - Codestral-22B-v0.1-mlx-nvfp4
  - gpt-oss-20b-MXFP4-Q8
  - Qwen2.5-Coder-14B-Instruct-MLX-4bit

Run (needs only `openai`; reuses shared/llm.py for the client):
    uv run --with openai python compare_models.py
    # or from any lab venv that already has openai:
    #   python compare_models.py
    # subset / override:
    #   MODELS="gpt-oss-20b-MXFP4-Q8,Qwen2.5-Coder-14B-Instruct-MLX-4bit" python compare_models.py
    #   PERF_TRIALS=5 MAX_TOKENS=384 python compare_models.py

Outputs:
    model-compare-results/raw.json   — per-model, per-task records (verify here)
    model-compare-results/REPORT.md  — ranked summary tables
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, "/Users/yuxinliu/code/agent-prep/shared")
from llm import make_client  # OpenAI-compatible client over oMLX :8000

# ── config ────────────────────────────────────────────────────────────────────
DEFAULT_MODELS = [
    "Codestral-22B-v0.1-mlx-nvfp4",
    "gpt-oss-20b-MXFP4-Q8",
    "Qwen2.5-Coder-14B-Instruct-MLX-4bit",
]
MODELS = [m.strip() for m in os.getenv("MODELS", ",".join(DEFAULT_MODELS)).split(",") if m.strip()]
PERF_TRIALS = int(os.getenv("PERF_TRIALS", "3"))      # median over N trials damps jitter
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))      # perf-prompt generation cap
CODE_TIMEOUT_S = int(os.getenv("CODE_TIMEOUT_S", "10"))
OUT_DIR = Path(os.getenv("OUT_DIR", "model-compare-results"))

# A fixed, model-neutral generation prompt for the throughput benchmark. Long-ish
# output so the decode window dominates and tok/s is meaningful.
PERF_PROMPT = (
    "Explain, in about 200 words, how a hash map achieves average O(1) lookup, "
    "what causes collisions, and how chaining versus open addressing differ."
)


# ── reasoning / coding task set (programmatic graders) ──────────────────────────
@dataclass(frozen=True)
class Task:
    id: str
    category: str                     # "math" | "logic" | "code"
    prompt: str
    expected: str | None = None       # math/logic: exact-match target (normalized)
    entrypoint: str | None = None     # code: function name to test
    tests: str = ""                   # code: assert lines appended after model code


_ANSWER_RULE = "\n\nReason briefly, then end your reply with a line exactly: ANSWER: <value>"

TASKS: list[Task] = [
    # — math (numeric exact-match) —
    Task("math_trains", "math",
         "Two trains start 420 km apart and move toward each other at 60 km/h and "
         "80 km/h. After how many hours do they meet?" + _ANSWER_RULE, expected="3"),
    Task("math_discount", "math",
         "A jacket costs $80. It is discounted 25%, then 10% sales tax is added to the "
         "discounted price. What is the final price in dollars?" + _ANSWER_RULE, expected="66"),
    Task("math_apples", "math",
         "A farmer has 17 apples. He sells 3 baskets of 4 apples each and then buys 9 "
         "more. How many apples does he have?" + _ANSWER_RULE, expected="14"),
    Task("math_avg", "math",
         "The average of 5 numbers is 12. Four of them are 8, 10, 14, 16. "
         "What is the fifth number?" + _ANSWER_RULE, expected="12"),

    # — logic (short-answer exact-match) —
    Task("logic_seq", "logic",
         "What number continues the sequence 2, 6, 12, 20, 30, ...?" + _ANSWER_RULE,
         expected="42"),
    Task("logic_family", "logic",
         "Sarah has the same number of brothers as sisters. Each of her brothers has "
         "twice as many sisters as brothers. How many girls are in the family?" + _ANSWER_RULE,
         expected="4"),
    Task("logic_truth", "logic",
         "A says 'B lies'. B says 'C lies'. C says 'A and B both lie'. Exactly one of "
         "them tells the truth. Who tells the truth? Answer with a single letter A, B, "
         "or C." + _ANSWER_RULE, expected="B"),

    # — code (HumanEval-style: model writes function, we run asserts) —
    Task("code_twosum", "code",
         "Write a Python function `two_sum(nums, target)` that returns a tuple of the "
         "two distinct indices (i, j) with i < j such that nums[i] + nums[j] == target. "
         "Assume exactly one solution exists. Return ONLY a fenced ```python code block.",
         entrypoint="two_sum",
         tests="assert tuple(two_sum([2,7,11,15],9))==(0,1)\n"
               "assert tuple(two_sum([3,2,4],6))==(1,2)\n"
               "assert tuple(two_sum([-1,0,1],0))==(0,2)\n"),
    Task("code_balanced", "code",
         "Write a Python function `is_balanced(s)` that returns True iff the brackets "
         "in s (only the characters ()[]{}) are correctly balanced and nested. Return "
         "ONLY a fenced ```python code block.",
         entrypoint="is_balanced",
         tests="assert is_balanced('()[]{}') is True\n"
               "assert is_balanced('([{}])') is True\n"
               "assert is_balanced('(]') is False\n"
               "assert is_balanced('([)]') is False\n"
               "assert is_balanced('') is True\n"),
    Task("code_rle", "code",
         "Write a Python function `rle(s)` that run-length encodes a string, e.g. "
         "'aaabbc' -> 'a3b2c1'. Single chars still get a count of 1. Return ONLY a "
         "fenced ```python code block.",
         entrypoint="rle",
         tests="assert rle('aaabbc')=='a3b2c1'\n"
               "assert rle('a')=='a1'\n"
               "assert rle('')==''\n"
               "assert rle('xxxx')=='x4'\n"),
    Task("code_primes", "code",
         "Write a Python function `primes_upto(n)` returning a sorted list of all primes "
         "<= n (n >= 0). Return ONLY a fenced ```python code block.",
         entrypoint="primes_upto",
         tests="assert primes_upto(10)==[2,3,5,7]\n"
               "assert primes_upto(1)==[]\n"
               "assert primes_upto(2)==[2]\n"
               "assert primes_upto(20)==[2,3,5,7,11,13,17,19]\n"),
]


# ── perf measurement (streamed) ─────────────────────────────────────────────────
@dataclass
class PerfSample:
    ttft_s: float
    decode_tok_s: float
    total_s: float
    completion_tokens: int
    tokens_estimated: bool


def measure_perf(client, model: str, prompt: str, max_tokens: int) -> PerfSample:
    """One streamed completion. TTFT = first content chunk; decode tok/s over the
    post-first-token window. Prefers server-reported usage; falls back to a chars/4
    estimate (flagged) when the endpoint omits usage on the stream."""
    t0 = time.perf_counter()
    first_t: float | None = None
    text_parts: list[str] = []
    usage_tokens: int | None = None

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if getattr(chunk, "usage", None):  # final usage frame (include_usage)
            usage_tokens = chunk.usage.completion_tokens
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        piece = getattr(delta, "content", None)
        if piece:
            if first_t is None:
                first_t = time.perf_counter()
            text_parts.append(piece)

    end = time.perf_counter()
    first_t = first_t or end
    text = "".join(text_parts)
    estimated = usage_tokens is None
    tokens = usage_tokens if usage_tokens is not None else max(1, len(text) // 4)
    decode_window = max(end - first_t, 1e-6)
    return PerfSample(
        ttft_s=first_t - t0,
        decode_tok_s=tokens / decode_window,
        total_s=end - t0,
        completion_tokens=tokens,
        tokens_estimated=estimated,
    )


def perf_for_model(client, model: str) -> dict:
    samples = [measure_perf(client, model, PERF_PROMPT, MAX_TOKENS) for _ in range(PERF_TRIALS)]
    return {
        "trials": PERF_TRIALS,
        "median_decode_tok_s": round(statistics.median(s.decode_tok_s for s in samples), 1),
        "median_ttft_s": round(statistics.median(s.ttft_s for s in samples), 3),
        "median_total_s": round(statistics.median(s.total_s for s in samples), 2),
        "median_completion_tokens": int(statistics.median(s.completion_tokens for s in samples)),
        "tokens_estimated": any(s.tokens_estimated for s in samples),
    }


# ── reasoning measurement (programmatic grading) ────────────────────────────────
def complete(client, model: str, prompt: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
    )
    return (r.choices[0].message.content or "").strip(), time.perf_counter() - t0


def _norm(s: str) -> str:
    """Normalize an answer token: strip $ , % whitespace, trailing .0, lowercase."""
    s = s.strip().strip(".").replace("$", "").replace(",", "").replace("%", "").strip()
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s.lower()


def grade_exact(reply: str, expected: str) -> bool:
    """Pass iff the tagged `ANSWER: <value>` (last occurrence) matches expected."""
    matches = re.findall(r"ANSWER:\s*(.+)", reply, flags=re.IGNORECASE)
    if not matches:
        return False
    return _norm(matches[-1]) == _norm(expected)


def _extract_code(reply: str) -> str:
    """Pull the first ```python fenced block; fall back to any ``` block, else raw."""
    m = re.search(r"```(?:python)?\s*(.*?)```", reply, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else reply).strip()


def grade_code(reply: str, task: Task) -> bool:
    """Write model code + asserts to a temp file, run in a sandboxed subprocess with a
    hard timeout. Pass iff exit 0. (Local models on your own box — acceptable exec,
    but isolated to a subprocess with no args and a wall-clock kill.)"""
    code = _extract_code(reply)
    if task.entrypoint and task.entrypoint not in code:
        return False
    program = f"{code}\n\n# --- harness asserts ---\n{task.tests}\nprint('OK')\n"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "candidate.py"
        p.write_text(program)
        try:
            proc = subprocess.run(
                [sys.executable, str(p)],
                capture_output=True, text=True, timeout=CODE_TIMEOUT_S, cwd=d,
            )
        except subprocess.TimeoutExpired:
            return False
        return proc.returncode == 0 and proc.stdout.strip().endswith("OK")


def grade(reply: str, task: Task) -> bool:
    if task.category == "code":
        return grade_code(reply, task)
    return grade_exact(reply, task.expected or "")


def reasoning_for_model(client, model: str) -> dict:
    records, latencies = [], []
    by_cat: dict[str, list[bool]] = {}
    for task in TASKS:
        try:
            reply, dt = complete(client, model, task.prompt)
            ok = grade(reply, task)
        except Exception as exc:  # endpoint hiccup → record as fail, keep going
            reply, dt, ok = f"<error: {exc}>", 0.0, False
        latencies.append(dt)
        by_cat.setdefault(task.category, []).append(ok)
        records.append({"id": task.id, "category": task.category,
                        "passed": ok, "latency_s": round(dt, 2),
                        "reply_tail": reply[-240:]})
        print(f"    {model:38s} {task.id:14s} {'PASS' if ok else 'FAIL'}  {dt:5.1f}s")
    total = sum(r["passed"] for r in records)
    return {
        "overall_pass": total,
        "overall_total": len(records),
        "overall_pct": round(100 * total / len(records), 1),
        "by_category": {c: f"{sum(v)}/{len(v)}" for c, v in sorted(by_cat.items())},
        "median_task_latency_s": round(statistics.median(latencies), 2),
        "records": records,
    }


# ── orchestration + report ──────────────────────────────────────────────────────
def render_report(results: dict) -> str:
    perf_rows = sorted(results.items(), key=lambda kv: kv[1]["perf"]["median_decode_tok_s"],
                       reverse=True)
    reas_rows = sorted(results.items(), key=lambda kv: kv[1]["reasoning"]["overall_pct"],
                       reverse=True)
    lines = ["# Local MLX model comparison",
             "",
             f"Endpoint: oMLX `:8000` · perf trials: {PERF_TRIALS} · max_tokens(perf): {MAX_TOKENS}",
             f"Reasoning set: {len(TASKS)} tasks "
             f"({sum(t.category=='math' for t in TASKS)} math, "
             f"{sum(t.category=='logic' for t in TASKS)} logic, "
             f"{sum(t.category=='code' for t in TASKS)} code) · grading: programmatic",
             "",
             "## Performance (higher tok/s = faster decode)",
             "",
             "| Rank | Model | decode tok/s | TTFT (s) | total (s) | tok counted |",
             "|------|-------|-------------:|---------:|----------:|:-----------:|"]
    for i, (m, r) in enumerate(perf_rows, 1):
        p = r["perf"]
        src = "est." if p["tokens_estimated"] else "server"
        lines.append(f"| {i} | `{m}` | {p['median_decode_tok_s']} | {p['median_ttft_s']} "
                     f"| {p['median_total_s']} | {src} |")
    lines += ["",
              "## Reasoning (programmatic pass-rate)",
              "",
              "| Rank | Model | overall | % | math | logic | code | med task (s) |",
              "|------|-------|:-------:|--:|:----:|:-----:|:----:|-------------:|"]
    for i, (m, r) in enumerate(reas_rows, 1):
        q = r["reasoning"]
        c = q["by_category"]
        lines.append(f"| {i} | `{m}` | {q['overall_pass']}/{q['overall_total']} | "
                     f"{q['overall_pct']} | {c.get('math','-')} | {c.get('logic','-')} "
                     f"| {c.get('code','-')} | {q['median_task_latency_s']} |")
    lines += ["",
              "_Numbers come from this run only (model-compare-results/raw.json). "
              "tok/s 'est.' = endpoint omitted stream usage, counted as chars/4._", ""]
    return "\n".join(lines)


def main() -> None:
    client = make_client()
    print(f"Comparing {len(MODELS)} models on oMLX :8000\n  " + "\n  ".join(MODELS) + "\n")
    results: dict[str, dict] = {}
    for model in MODELS:
        print(f"[{model}]")
        print("  · performance (streamed throughput)…")
        perf = perf_for_model(client, model)
        print(f"      decode {perf['median_decode_tok_s']} tok/s · "
              f"TTFT {perf['median_ttft_s']}s")
        print("  · reasoning (programmatic grading)…")
        reasoning = reasoning_for_model(client, model)
        print(f"      {reasoning['overall_pass']}/{reasoning['overall_total']} "
              f"({reasoning['overall_pct']}%)\n")
        results[model] = {"perf": perf, "reasoning": reasoning}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "raw.json").write_text(json.dumps(results, indent=2))
    report = render_report(results)
    (OUT_DIR / "REPORT.md").write_text(report)
    print("\n" + report)
    print(f"wrote {OUT_DIR}/raw.json  and  {OUT_DIR}/REPORT.md")


if __name__ == "__main__":
    main()
