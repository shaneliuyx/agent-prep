"""
probe_fleet.py — score the local oMLX fleet on five dimensions, then
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
import os
import re
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
# All models accessible through oMLX's OpenAI-compatible surface on :8000/v1.
# Route by `model:` field in the request body (bare IDs as listed by GET /v1/models —
# no `models/` or `lmstudio-community/` prefix). Larger on-demand models load on first
# request — first call adds ~10-30s cold-start tax; subsequent calls hit the warm path.
# Override with OMLX_URL if you run oMLX elsewhere.
GATEWAY_URL = os.getenv("OMLX_URL", "http://127.0.0.1:8000/v1")

FLEET: list[dict] = [
    # Stable always-hot fleet (fits within 48 GB unified memory).
    {
        "tier": "sonnet",
        "label": "gemma-4-26B-A4B-it-heretic-4bit (dense 26B, 4-bit quant, mid)",
        "model": "gemma-4-26B-A4B-it-heretic-4bit",
        "url": GATEWAY_URL,
    },
    {
        "tier": "haiku",
        "label": "MLX-Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-4bit (MoE 35B/3B-active; tool-capable, fast decode)",
        "model": "MLX-Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled-4bit",
        "url": GATEWAY_URL,
    },
    {
        # Smallest full model (4 GB), ~260 ms. Clears ALL four dims: tool 1.00
        # STRUCTURED (Qwen3.5 is server-parsed by oMLX — no client parser),
        # json 1.00, instr 1.00, reason 3/3 (given a generous token cap).
        # Replaced Qwen2.5-Coder-7B (text-parsed tools, loose json/instr) 2026-06-15.
        "tier": "fast",
        "label": "Qwen3.5-4B-MLX-4bit (4 GB; fast tier, all 4 dims, structured tools)",
        "model": "Qwen3.5-4B-MLX-4bit",
        "url": GATEWAY_URL,
    },
    # Larger 31B variants — oMLX loads on demand.
    {
        "tier": "opus_lazy",
        "label": "gemma-4-31B-uncensored-heretic-mlx-4bit (uncensored 31B; on-demand)",
        "model": "Gemma4-31",  # served tag/alias for gemma-4-31B-uncensored-heretic-mlx-4bit
        "url": GATEWAY_URL,
    },
    # NOTE: opus_jang (Gemma-4-31B-JANG_4M-CRACK) removed 2026-06-15 — the
    # "4M-CRACK" build 500s on oMLX under both API surfaces (and its dealignai--
    # alias too); it was also redundant with opus_lazy (Gemma 31B uncensored).
    # Re-add a tier here if a loadable 4M-context variant becomes available.
    # Dense 27B distilled from Claude-4.6-Opus reasoning traces.
    {
        "tier": "opus_qwen",
        "label": "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit (dense 27B; opus-distilled)",
        "model": "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
        "url": GATEWAY_URL,
    },
]
DEFAULT_TIERS = {"sonnet", "haiku", "fast"}  # always-hot; lazy/opus tiers load on demand

API_KEY = "not-needed"  # oMLX ignores; SDK requires non-empty
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


# ---------------------------------------------------------------------------
# Client-side tool-call recovery.
#
# Some local models form a correct tool call but the server has no parser for
# that model family, so it falls through as plain TEXT instead of populating
# `message.tool_calls`. Observed leak wrappers (same model, different surfaces):
#   <tool_call>...</tool_call>   Qwen/Hermes native
#   <tools>...</tools>           oMLX OpenAI-compatible surface
#   <function>...</function>     oMLX Anthropic-compatible surface (close tag often missing)
# This recovers {name, arguments} from that text so a text-only model is still
# usable in the loop. Lift this into src/react.py to actually drive such a model.
# ---------------------------------------------------------------------------
_TOOL_TAGS = ("tool_call", "tools", "function", "tool")


def _first_json_obj(s: str) -> dict | None:
    """Return the first balanced {...} that parses as JSON, else None.
    Brace-matched (not regex) so a nested `arguments` object isn't truncated,
    and string-aware so braces inside a JSON string value (e.g. arguments
    serialized as a JSON-encoded string, the OpenAI form) don't miscount."""
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except json.JSONDecodeError:
                        break  # malformed; advance to next candidate brace
        start = s.find("{", start + 1)
    return None


def _normalize_call(obj: object) -> dict | None:
    """Coerce a parsed object into {name, arguments: dict}. None if not a
    dict or missing a name (json.loads may yield a list/scalar)."""
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if name is None:
        return None
    args = obj.get("arguments", obj.get("parameters", obj.get("input", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw": args}
    return {"name": name, "arguments": args if isinstance(args, dict) else {}}


def extract_text_tool_calls(content: str | None) -> list[dict]:
    """Best-effort recovery of tool calls the server left as text.
    Returns [{name, arguments}, ...]; [] when nothing tool-shaped is found."""
    if not content:
        return []
    calls: list[dict] = []
    for tag in _TOOL_TAGS:
        # inner text runs to the close tag OR end-of-string (close tag may be absent)
        for m in re.finditer(rf"<{tag}>\s*(.*?)(?:</{tag}>|\Z)", content,
                             re.DOTALL | re.IGNORECASE):
            call = _normalize_call(_first_json_obj(m.group(1)))
            if call:
                calls.append(call)
    if not calls:  # untagged: a bare {name, arguments} blob
        call = _normalize_call(_first_json_obj(content))
        if call:
            calls.append(call)
    return calls


def probe_tool_calling(c: OpenAI, model: str) -> tuple[float, list[str]]:
    """P2: 3 trials of structured tool call. Pass = a get_weather call is
    obtained, either as structured `tool_calls` or recovered from text."""
    passes = 0
    structured = 0
    text_parsed = 0
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
            msg = r.choices[0].message
            tcs = msg.tool_calls or []
            # Tool call may be ChatCompletionMessageFunctionToolCall (.function.name)
            # or a custom variant — use getattr to stay compatible.
            fn = getattr(tcs[0], "function", None) if tcs else None
            name = getattr(fn, "name", None)
            if name == "get_weather":
                passes += 1
                structured += 1
            elif any(rc["name"] == "get_weather" for rc in extract_text_tool_calls(msg.content)):
                # server didn't parse it; client-side recovery succeeded
                passes += 1
                text_parsed += 1
            else:
                notes.append(f"{city}: no get_weather call (structured name={name!r})")
        except OpenAIError as e:
            notes.append(f"{city}: {type(e).__name__}")
    summary = f"{passes}/3 get_weather calls ({structured} structured, {text_parsed} text-parsed)"
    return passes / 3.0, [summary] + notes


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
                # Generous cap: reasoning is about correctness, not terseness.
                # A 64-token cap unfairly fails verbose-but-correct models that
                # show their work (e.g. Qwen3.5-4B derived 150 then got clipped).
                max_tokens=512,
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
        return {"_error": "no models reachable; start oMLX first"}
    # ping is already scored as 1/(1+median_s), so larger = faster.
    best_at = lambda key: max(alive, key=lambda r: r.scores.get(key, 0)).model

    # "Cheap" roles want the FASTEST model that can still emit a usable answer.
    # Pure argmax(ping) is a trap: a reasoning-distilled model can ping fastest
    # yet be useless for classification. A cheap classifier must BOTH reason
    # (pick the right answer) AND follow output format (emit a clean label), so
    # floor on reason AND instr. Reason alone is insufficient: a reasoning-
    # distilled model clears reason=0.83 but scores instr=0 (think-blocks eat
    # the format), so it would be picked yet produce unparseable labels.
    CHEAP_FLOOR = 0.5
    capable = [r for r in alive
               if r.scores.get("reason", 0) >= CHEAP_FLOOR
               and r.scores.get("instr", 0) >= CHEAP_FLOOR]
    fastest_capable = max(capable or alive,
                          key=lambda r: r.scores.get("ping", 0)).model
    return {
        "main_react_loop":         best_at("tool"),    # tool calling is the loop's hot path
        "tool_arg_synthesis":      best_at("tool"),
        "final_answer_composer":   best_at("instr"),   # follows formatting instructions
        "json_extractor":          best_at("json"),
        "math_or_reasoning_step":  best_at("reason"),
        "cheap_classifier":        fastest_capable,    # fast AND able to answer
        "obs_summarizer_sidecar":  fastest_capable,
    }


def format_table(results: list[FleetResult]) -> str:
    probe_keys = [name for name, _ in PROBES]
    headers = ["tier", "model"] + probe_keys
    rows = [headers]
    for r in results:
        if not r.reachable:
            rows.append([r.tier, r.model[:38], "DOWN"] + ["-"] * (len(probe_keys) - 1))
            continue
        rows.append([r.tier, r.model[:38]] + [f"{r.scores.get(k, 0):.2f}" for k in probe_keys])
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

    lazy_present = any(e["tier"].startswith("opus") for e in FLEET)
    note = ("lazy/cold models in this set wake on first call (~10-30s cold-start tax, then warm)."
            if lazy_present
            else "all selected models are always-hot; probes should be fast.")
    print(f"Probing oMLX fleet ({len(FLEET)} model(s))... {note}\n", file=sys.stderr)
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
