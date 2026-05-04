"""
probe_fleet.py — score the local vMLX fleet on five dimensions, then
recommend a role assignment for src/react.py.

Five probes (run against each model):
  P1  ping_latency       median ms over 5 short completions      (lower = better)
  P2  tool_calling       3 trials calling a tool schema          (pass = tool_calls present)
  P3  json_only          3 trials of "return JSON only"          (pass = json.loads works)
  P4  reasoning          3 multi-step word problems              (pass = correct integer)
  P5  instr_following    3 "exactly N words" constraints         (pass = exact word count)

The harness is deliberately small and frameworkless — it matches the discipline
of Week 4. Run it once, copy the recommendation block into your role-mapping
notes, and use those tiers in src/react.py + Week 5's pattern zoo.

Usage:
  python scripts/probe_fleet.py
  python scripts/probe_fleet.py --json   # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import sys
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse
from typing import Callable, cast

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionToolParam

# ---------------------------------------------------------------------------
# Fleet definition. Edit if you swap models on a port.
# ---------------------------------------------------------------------------
FLEET: list[dict] = [
    # Stable always-hot fleet (fits within 48 GB unified memory).
    {
        "tier": "sonnet",
        "label": "gemma-4-26B-A4B-it-heretic-4bit (dense 26B, 4-bit quant, mid)",
        "model": "gemma-4-26B-A4B-it-heretic-4bit",
        "url": "http://127.0.0.1:8003/v1",
    },
    {
        "tier": "haiku",
        "label": "MLX-Qwen3.5-9B-GLM5.1-Distill-v1-8bit (smallest, fastest)",
        "model": "MLX-Qwen3.5-9B-GLM5.1-Distill-v1-8bit",
        "url": "http://127.0.0.1:8004/v1",
    },
    # Option C: lazy-loaded — keep cold, spin up only when needed.
    # Tier "opus_lazy" is excluded from --only opus,sonnet,haiku by default;
    # use --only opus_lazy or --include-lazy to test in isolation.
    {
        "tier": "opus_lazy",
        "label": "gemma-4-31B-uncensored-heretic-mlx-4bit (lazy; uncensored long-form)",
        "model": "gemma-4-31B-uncensored-heretic-mlx-4bit",
        "url": "http://127.0.0.1:8000/v1",
    },
]
DEFAULT_TIERS = {"sonnet", "haiku"}  # opus_lazy excluded from default run

API_KEY = "not-needed"  # vMLX ignores; SDK requires non-empty
REACH_TIMEOUT_S = 2  # fail fast when port is dead
PROBE_TIMEOUT_S = 60  # generous once we know the server is up


def make_client(url: str, timeout: float = PROBE_TIMEOUT_S) -> OpenAI:
    return OpenAI(base_url=url, api_key=API_KEY, timeout=timeout, max_retries=0)


def is_port_alive(url: str, timeout: float = REACH_TIMEOUT_S) -> bool:
    """TCP-level probe — fails instantly on dead port without httpx overhead."""
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except (OSError, socket.timeout):
            return False


# ---------------------------------------------------------------------------
# Probes — each returns (score: float in [0,1], notes: list[str])
# ---------------------------------------------------------------------------

def probe_ping(c: OpenAI, model: str) -> tuple[float, list[str]]:
    """P1: median latency over 3 cold-ish calls. Score = 1 / (1 + median_s)."""
    latencies = []
    for _ in range(3):
        t0 = time.perf_counter()
        try:
            c.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply: ready"}],
                max_tokens=8,
                temperature=0.0,
            )
            latencies.append(time.perf_counter() - t0)
        except OpenAIError as e:
            return 0.0, [f"failed: {type(e).__name__}: {e}"]
    median_s = statistics.median(latencies)
    score = 1.0 / (1.0 + median_s)  # 0.5s -> 0.67, 2s -> 0.33, 5s -> 0.17
    return score, [f"median {median_s*1000:.0f} ms over {len(latencies)} runs"]


TOOL_SCHEMA: list[ChatCompletionToolParam] = cast(list[ChatCompletionToolParam], [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}])


def probe_tool_calling(c: OpenAI, model: str) -> tuple[float, list[str]]:
    """P2: 3 trials of structured tool call. Pass = tool_calls non-empty."""
    passes = 0
    notes = []
    for city in ["Tokyo", "Beijing", "Berlin"]:
        try:
            r = c.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"What is the weather in {city}?"}],
                tools=TOOL_SCHEMA,
                tool_choice="auto",
                max_tokens=128,
                temperature=0.0,
            )
            tcs = r.choices[0].message.tool_calls or []
            # Tool call may be ChatCompletionMessageFunctionToolCall (.function.name)
            # or a custom variant — use getattr to stay compatible.
            fn = getattr(tcs[0], "function", None) if tcs else None
            name = getattr(fn, "name", None)
            if name == "get_weather":
                passes += 1
            else:
                notes.append(f"{city}: no get_weather tool_call (got name={name!r})")
        except OpenAIError as e:
            notes.append(f"{city}: {type(e).__name__}")
    return passes / 3.0, notes or [f"{passes}/3 calls produced get_weather tool_call"]


def probe_json_only(c: OpenAI, model: str) -> tuple[float, list[str]]:
    """P3: 3 trials of 'return only valid JSON'. Pass = json.loads succeeds."""
    prompts = [
        'Return ONLY this JSON, no markdown, no prose: {"status":"ok","n":1}',
        'Output ONLY the JSON object: {"animal":"cat","legs":4}',
        'Reply with valid JSON only: {"x":3.14,"y":2.71}',
    ]
    passes = 0
    notes = []
    for p in prompts:
        try:
            r = c.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": p}],
                max_tokens=64,
                temperature=0.0,
            )
            txt = (r.choices[0].message.content or "").strip()
            try:
                json.loads(txt)
                passes += 1
            except json.JSONDecodeError:
                notes.append(f"unparseable: {txt[:60]!r}")
        except OpenAIError as e:
            notes.append(f"err: {type(e).__name__}")
    return passes / 3.0, notes or [f"{passes}/3 produced valid JSON"]


def probe_reasoning(c: OpenAI, model: str) -> tuple[float, list[str]]:
    """P4: 3 short word problems. Pass = exact integer answer present in output."""
    cases = [
        ("Alice has 3 apples. Bob gives her 5 more. She eats 2. How many? Answer with a single integer.", "6"),
        ("A train leaves at 9 AM going 60 mph. After 2.5 hours, how many miles? Single integer.", "150"),
        ("If 12 widgets cost $36, how much do 5 cost? Single integer in dollars.", "15"),
    ]
    passes = 0
    notes = []
    for q, want in cases:
        try:
            r = c.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": q}],
                max_tokens=64,
                temperature=0.0,
            )
            txt = (r.choices[0].message.content or "").strip()
            if want in txt.split():
                passes += 1
            elif want in txt:
                passes += 0.5  # right answer but verbose
                notes.append(f"verbose-correct: {txt[:40]!r}")
            else:
                notes.append(f"wrong: want {want}, got {txt[:40]!r}")
        except OpenAIError as e:
            notes.append(f"err: {type(e).__name__}")
    return passes / 3.0, notes or [f"{passes}/3 correct integer answers"]


def probe_instr_following(c: OpenAI, model: str) -> tuple[float, list[str]]:
    """P5: 3 strict word-count constraints. Pass = exact word count match."""
    cases = [(3, "Describe the ocean in EXACTLY 3 words."),
             (5, "Summarize Python in EXACTLY 5 words."),
             (4, "Define agent in EXACTLY 4 words.")]
    passes = 0
    notes = []
    for want, q in cases:
        try:
            r = c.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": q}],
                max_tokens=32,
                temperature=0.0,
            )
            txt = (r.choices[0].message.content or "").strip().rstrip(".")
            n = len(txt.split())
            if n == want:
                passes += 1
            else:
                notes.append(f"wanted {want} words, got {n}: {txt[:40]!r}")
        except OpenAIError as e:
            notes.append(f"err: {type(e).__name__}")
    return passes / 3.0, notes or [f"{passes}/3 exact word counts"]


PROBES: list[tuple[str, Callable]] = [
    ("ping", probe_ping),
    ("tool", probe_tool_calling),
    ("json", probe_json_only),
    ("reason", probe_reasoning),
    ("instr", probe_instr_following),
]


# ---------------------------------------------------------------------------
# Runner + recommendation
# ---------------------------------------------------------------------------

@dataclass
class FleetResult:
    tier: str
    model: str
    url: str
    label: str
    scores: dict[str, float] = field(default_factory=dict)
    notes: dict[str, list[str]] = field(default_factory=dict)
    reachable: bool = True


def run() -> list[FleetResult]:
    results = []
    for entry in FLEET:
        r = FleetResult(tier=entry["tier"], model=entry["model"],
                        url=entry["url"], label=entry["label"])
        if not is_port_alive(entry["url"]):
            r.reachable = False
            r.notes["_reach"] = [f"port closed at {entry['url']}"]
            results.append(r)
            continue
        c = make_client(entry["url"])
        for name, fn in PROBES:
            score, notes = fn(c, entry["model"])
            r.scores[name] = score
            r.notes[name] = notes
        results.append(r)
    return results


def recommend(results: list[FleetResult]) -> dict[str, str]:
    """Pick a tier per scenario based on probe scores."""
    alive = [r for r in results if r.reachable]
    if not alive:
        return {"_error": "no models reachable; start vMLX servers first"}
    best_at = lambda key: max(alive, key=lambda r: r.scores.get(key, 0)).model
    fastest = min(alive, key=lambda r: 1 / (r.scores.get("ping", 1e-9) or 1e-9)).model
    return {
        "main_react_loop":         best_at("tool"),    # tool calling is the loop's hot path
        "tool_arg_synthesis":      best_at("tool"),
        "final_answer_composer":   best_at("instr"),   # follows formatting instructions
        "json_extractor":          best_at("json"),
        "math_or_reasoning_step":  best_at("reason"),
        "cheap_classifier":        fastest,            # used for pre-loop intent triage
        "obs_summarizer_sidecar":  fastest,
    }


def format_table(results: list[FleetResult]) -> str:
    headers = ["tier", "model", "ping", "tool", "json", "reason", "instr"]
    rows = [headers]
    for r in results:
        if not r.reachable:
            rows.append([r.tier, r.model[:38], "DOWN", "-", "-", "-", "-"])
            continue
        rows.append([r.tier, r.model[:38]] + [f"{r.scores.get(k,0):.2f}" for k in ["ping","tool","json","reason","instr"]])
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]
    out = []
    for row in rows:
        out.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--only", help="comma-separated tier filter, e.g. opus,sonnet")
    args = ap.parse_args()

    global FLEET
    if args.only:
        tiers = {t.strip() for t in args.only.split(",")}
    else:
        tiers = DEFAULT_TIERS
    FLEET = [e for e in FLEET if e["tier"] in tiers]
    if not FLEET:
        print(f"No fleet entries match tiers={tiers}", file=sys.stderr)
        sys.exit(2)

    print(f"Probing vMLX fleet ({len(FLEET)} model(s))... cold-call latency on a 35B MoE is ~5s/probe.\n", file=sys.stderr)
    results = run()

    if args.json:
        payload = {
            "fleet": [
                {"tier": r.tier, "model": r.model, "url": r.url,
                 "reachable": r.reachable, "scores": r.scores, "notes": r.notes}
                for r in results
            ],
            "recommendation": recommend(results),
        }
        print(json.dumps(payload, indent=2))
        return

    print("=" * 70)
    print("Per-model scores (1.0 = perfect, 0.0 = failed)")
    print("=" * 70)
    print(format_table(results))
    print()

    recs = recommend(results)
    if "_error" in recs:
        print(f"[!] {recs['_error']}")
        return
    print("=" * 70)
    print("Recommended role assignment for src/react.py + §5 pattern zoo")
    print("=" * 70)
    for role, model in recs.items():
        print(f"  {role:28s} -> {model}")


if __name__ == "__main__":
    main()
