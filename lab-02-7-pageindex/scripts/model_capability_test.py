"""Probe vMLX models on 4 capability dimensions, ONE MODEL AT A TIME.

After each model finishes its 4 probes, unload it via POST /v1/models/{id}/unload
before loading the next. Avoids RAM pressure on machines that can hold only 1
27B-class MLX model at a time.

Probes:
  1. Tool-following     — single tool, well-formed structured tool_calls?
  2. Multi-tool routing — 2 tools, picks the correct one?
  3. Synthesis          — combines 2 facts into 1 coherent sentence?
  4. Refusal            — OOD question, refuses cleanly?

Each probe = 1 LLM call. Score 0/1. Aggregate ranks models.

Usage:
  python scripts/model_capability_test.py                # all 5 models, sequential
  python scripts/model_capability_test.py <model_id>     # single model
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

_LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LAB_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

load_dotenv(_LAB_ROOT / ".env")

BASE_URL = os.getenv("OMLX_BASE_URL", "http://localhost:8080/v1")
API_KEY = os.getenv("OMLX_API_KEY", "nokey")

MODELS = [
    "models/gemma-4-26B-A4B-it-heretic-4bit",
    "models/Qwen3.6-35B-A3B-nvfp4",
    "models/Gemma-4-31B-JANG_4M-CRACK",
    "models/MLX-Qwen3.5-9B-GLM5.1-Distill-v1-8bit",
    "models/gemma-4-31B-uncensored-heretic-mlx-4bit",
]

ADD_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "add_numbers",
        "description": "Add two integers and return the sum.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}
WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
STOCK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "Get current stock price for a ticker symbol.",
        "parameters": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
}


def unload_model(model: str) -> str:
    """POST /v1/models/{id}/unload. Returns short status string."""
    url = f"{BASE_URL}/models/{model}/unload"
    req = urllib.request.Request(url, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return f"unload {model} → HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return f"unload {model} → HTTP {e.code} ({e.reason})"
    except Exception as e:                                  # noqa: BLE001
        return f"unload {model} → {type(e).__name__}: {str(e)[:60]}"


def probe_tool_following(client: OpenAI, model: str) -> dict:
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=200,
            tools=cast(Any, [ADD_TOOL]),
            tool_choice="required",
            messages=[{"role": "user", "content": "Add 37 and 58 using the tool."}],
        )
        lat = time.time() - t0
        msg = r.choices[0].message
        tcs = getattr(msg, "tool_calls", None) or []
        if not tcs:
            return {"score": 0, "lat": lat, "reason": "no tool_calls",
                    "raw": (msg.content or "")[:120]}
        tc = tcs[0]
        name = tc.function.name
        args = json.loads(tc.function.arguments)
        ok = name == "add_numbers" and int(args.get("a")) == 37 and int(args.get("b")) == 58
        return {"score": int(ok), "lat": lat, "name": name, "args": args}
    except Exception as e:                                  # noqa: BLE001
        return {"score": 0, "lat": time.time() - t0,
                "reason": f"{type(e).__name__}: {str(e)[:80]}"}


def probe_multitool_routing(client: OpenAI, model: str) -> dict:
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=200,
            tools=cast(Any, [WEATHER_TOOL, STOCK_TOOL]),
            tool_choice="required",
            messages=[{"role": "user", "content": "What's the weather in Tokyo right now?"}],
        )
        lat = time.time() - t0
        msg = r.choices[0].message
        tcs = getattr(msg, "tool_calls", None) or []
        if not tcs:
            return {"score": 0, "lat": lat, "reason": "no tool_calls"}
        tc = tcs[0]
        name = tc.function.name
        args = json.loads(tc.function.arguments)
        ok = name == "get_weather" and "tokyo" in str(args.get("city", "")).lower()
        return {"score": int(ok), "lat": lat, "name": name, "args": args}
    except Exception as e:                                  # noqa: BLE001
        return {"score": 0, "lat": time.time() - t0,
                "reason": f"{type(e).__name__}: {str(e)[:80]}"}


def probe_synthesis(client: OpenAI, model: str) -> dict:
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=120,
            messages=[{
                "role": "user",
                "content": (
                    "Fact A: Berkshire's 2023 total revenues were $364.5 billion.\n"
                    "Fact B: Berkshire's 2023 net earnings were $96.2 billion.\n"
                    "Combine both facts into ONE sentence. Include both numbers exactly."
                ),
            }],
        )
        lat = time.time() - t0
        ans = (r.choices[0].message.content or "").strip()
        has_rev = "364.5" in ans or "364" in ans
        has_earn = "96.2" in ans or "96" in ans
        # Count sentence-terminators only (period followed by space/EOL),
        # NOT decimals like "364.5" or "96.2".
        import re as _re
        terminators = len(_re.findall(r"[.!?](?:\s|$)", ans))
        one_sentence = terminators <= 1
        ok = has_rev and has_earn and one_sentence
        return {"score": int(ok), "lat": lat, "answer": ans[:200],
                "has_rev": has_rev, "has_earn": has_earn,
                "one_sentence": one_sentence}
    except Exception as e:                                  # noqa: BLE001
        return {"score": 0, "lat": time.time() - t0,
                "reason": f"{type(e).__name__}: {str(e)[:80]}"}


def probe_refusal(client: OpenAI, model: str) -> dict:
    t0 = time.time()
    try:
        r = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=120,
            messages=[{
                "role": "system",
                "content": (
                    "You answer ONLY from the Berkshire Hathaway 2023 annual report. "
                    "If the question is outside that scope, refuse and say "
                    "'This question is outside the document.'"
                ),
            }, {
                "role": "user",
                "content": "What was Apple's revenue in fiscal year 2024?",
            }],
        )
        lat = time.time() - t0
        ans_raw = (r.choices[0].message.content or "").strip()
        ans = ans_raw.lower()
        refused = (
            "outside the document" in ans
            or ("outside" in ans and "document" in ans)
            or "cannot" in ans
            or "don't know" in ans
            or "not in" in ans
            or "i can't" in ans
        )
        gave_number = any(tok in ans for tok in ["391", "383", "billion", "trillion"])
        ok = refused and not gave_number
        return {"score": int(ok), "lat": lat, "answer": ans_raw[:200],
                "refused": refused, "hallucinated": gave_number}
    except Exception as e:                                  # noqa: BLE001
        return {"score": 0, "lat": time.time() - t0,
                "reason": f"{type(e).__name__}: {str(e)[:80]}"}


FETCH_CHUNK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_chunk",
        "description": (
            "Fetch chunk N of a document. Returns one sentence. "
            "Call repeatedly with N=0,1,2,... to read the full document."
        ),
        "parameters": {
            "type": "object",
            "properties": {"n": {"type": "integer", "minimum": 0}},
            "required": ["n"],
        },
    },
}

# Synthetic 8-chunk doc — chunk size matches real RAG observations (~2KB each).
# Cumulative ~16KB across 8 rounds simulates the actual W2.7 agent-loop load
# that triggered Issue #1011 degradation on Qwen3.5-27B-4bit. Probe with
# trivially short chunks (50 bytes) missed the bug entirely.
_CHUNK_SEEDS = [
    ("Chairman's Letter", "Berkshire's Chairman's Letter from Buffett discusses controlled and non-controlled businesses, the not-so-secret weapon of patient long-term capital deployment, and Charlie Munger's influence on company culture."),
    ("Non-controlled businesses", "Buffett discusses non-controlled businesses including Coca-Cola, American Express, Apple, and Bank of America. These investments contribute to look-through earnings beyond GAAP."),
    ("BNSF Railway", "BNSF Railway is Berkshire's railroad subsidiary, described in Item 1 Business Description. Operations span 32,500 route miles across 28 states and Canada."),
    ("Risk Factors", "Item 1A Risk Factors covers cybersecurity, regulatory, climate, supply-chain, insurance underwriting, and operational risks. Threats are continuously assessed."),
    ("Financial Statements", "Item 8 Consolidated Financial Statements report total revenues $364.5 billion, net earnings $96.2 billion, and operating earnings figures broken down by segment."),
    ("Cybersecurity", "Item 1C details cybersecurity governance, board oversight, threat assessment, third-party penetration testing, incident response procedures, and breach disclosure protocols."),
    ("Notes to Statements", "Notes to Consolidated Financial Statements appear after Item 8 audited statements, covering accounting policies, segment reporting, fair value, and contingencies."),
    ("Closing", "The report ends with management certifications under Sarbanes-Oxley sections 302 and 906, exhibits including the 10-K cover page, and required certifications."),
]
# Pad each chunk to ~2KB to match real RAG observation size.
_CHUNKS = [
    f"=== Chunk {i}: {title} ===\n{body}\n\n" + (body + "\n") * 12
    for i, (title, body) in enumerate(_CHUNK_SEEDS)
]


def probe_sustained_tool_loop(client: OpenAI, model: str, rounds: int = 8) -> dict:
    """P5: simulate W2.7 agent-loop. Fire `rounds` sequential tool calls in
    ONE conversation. Track the FIRST round where the model emits empty content
    + no tool_calls (Issue #1011 degradation pattern). Single-call probes 1-4
    miss this entirely.

    Score = (rounds_completed_cleanly / rounds). 1.0 = no degradation."""
    t0 = time.time()
    msgs: list[dict] = [{
        "role": "system",
        "content": (
            "You are reading a document chunk-by-chunk via fetch_chunk(n). "
            "Call fetch_chunk(0), then (1), then (2), ... until you have read "
            "all chunks. After EACH observation, call the NEXT chunk. Do not "
            "stop until told."
        ),
    }, {
        "role": "user",
        "content": f"Read chunks 0 through {rounds - 1}, calling fetch_chunk(n) for each.",
    }]
    completed = 0
    first_fail_round = None
    fail_reason = None
    try:
        for r in range(rounds):
            resp = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=200,
                tools=cast(Any, [FETCH_CHUNK_TOOL]),
                tool_choice="auto",
                messages=cast(Any, msgs),
            )
            msg = resp.choices[0].message
            tcs = getattr(msg, "tool_calls", None) or []
            content = (msg.content or "").strip()
            if not tcs:
                # Degradation pattern: empty content + no tool_calls
                if not content:
                    first_fail_round = r
                    fail_reason = "empty content + no tool_calls"
                    break
                # Model finished early but with content — count as success
                # only if it actually completed enough rounds
                if r >= rounds - 1:
                    completed = r
                else:
                    first_fail_round = r
                    fail_reason = f"finished early at round {r} with content"
                break
            tc = tcs[0]
            try:
                args = json.loads(tc.function.arguments)
                n = int(args.get("n", -1))
            except Exception:                               # noqa: BLE001
                first_fail_round = r
                fail_reason = "malformed args"
                break
            if not 0 <= n < len(_CHUNKS):
                first_fail_round = r
                fail_reason = f"out-of-range n={n}"
                break
            msgs.append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.function.name,
                                             "arguments": tc.function.arguments}}],
            })
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": _CHUNKS[n]})
            completed = r + 1
    except Exception as e:                                  # noqa: BLE001
        first_fail_round = completed
        fail_reason = f"{type(e).__name__}: {str(e)[:60]}"
    lat = time.time() - t0
    score = completed / rounds
    return {"score": round(score, 3), "lat": lat,
            "completed": completed, "rounds": rounds,
            "first_fail_round": first_fail_round, "reason": fail_reason}


def probe_cross_conversation(client: OpenAI, model: str,
                              conversations: int = 8) -> dict:
    """P6: simulate the REAL v1 pattern — fire `conversations` SEPARATE
    chat.completions.create calls, each with its own large system prompt +
    tree-TOC user message + 1 tool call. This is what triggered Issue #1011
    degradation on Qwen3.5-27B-4bit (broke at Q4 in real v1) but P5
    within-conversation could not catch.

    Server-side prefix-cache state may pollute MoE-gate scales across
    separate conversations on flat-quant Qwen MoE."""
    big_system = (
        "You answer questions about a long document. Read the tree, then call "
        "get_page_content(start_page, end_page) for the right range. Page "
        "ranges focused (3-10 pages typical). Cite [pages X-Y] inline."
    ) * 3  # repeat to make it ~1KB
    big_tree = (
        "Tree:\n" +
        "\n".join(f"  [{i}] Section {i} (pages {i*10}-{i*10+9}): topic {i}"
                  for i in range(40))
    )
    page_tool: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "get_page_content",
            "description": "Fetch page text from start_page to end_page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_page": {"type": "integer"},
                    "end_page": {"type": "integer"},
                },
                "required": ["start_page", "end_page"],
            },
        },
    }
    t0 = time.time()
    completed = 0
    first_fail = None
    fail_reason = None
    for i in range(conversations):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=200,
                tools=cast(Any, [page_tool]),
                tool_choice="auto",
                messages=[
                    {"role": "system", "content": big_system},
                    {"role": "user",
                     "content": (
                         f"{big_tree}\n\nQuestion {i}: What is in section "
                         f"{i % 40}? Call get_page_content for that range."
                     )},
                ],
            )
            msg = r.choices[0].message
            tcs = getattr(msg, "tool_calls", None) or []
            content = (msg.content or "").strip()
            if not tcs and not content:
                first_fail = i
                fail_reason = f"q{i}: empty + no tool_calls (Issue #1011 pattern)"
                break
            completed = i + 1
        except Exception as e:                              # noqa: BLE001
            first_fail = i
            fail_reason = f"q{i}: {type(e).__name__}: {str(e)[:60]}"
            break
    lat = time.time() - t0
    score = completed / conversations
    return {"score": round(score, 3), "lat": lat,
            "completed": completed, "rounds": conversations,
            "first_fail_round": first_fail, "reason": fail_reason}


def run_one_model(client: OpenAI, model: str) -> dict:
    print(f"\n=== {model}", flush=True)
    p1 = probe_tool_following(client, model)
    print(f"  P1 single-tool        score={p1['score']} lat={p1['lat']:.2f}s "
          f"{p1.get('reason', '')}", flush=True)
    p2 = probe_multitool_routing(client, model)
    print(f"  P2 multi-tool route   score={p2['score']} lat={p2['lat']:.2f}s "
          f"{p2.get('reason', '')}", flush=True)
    p3 = probe_synthesis(client, model)
    print(f"  P3 synthesis          score={p3['score']} lat={p3['lat']:.2f}s "
          f"rev={p3.get('has_rev')} earn={p3.get('has_earn')} one={p3.get('one_sentence')}",
          flush=True)
    p4 = probe_refusal(client, model)
    print(f"  P4 OOD refusal        score={p4['score']} lat={p4['lat']:.2f}s "
          f"refused={p4.get('refused')} halluc={p4.get('hallucinated')}",
          flush=True)
    p5 = probe_sustained_tool_loop(client, model, rounds=8)
    print(f"  P5 within-conv-load   score={p5['score']:.2f} lat={p5['lat']:.2f}s "
          f"completed={p5['completed']}/{p5['rounds']} "
          f"first_fail={p5.get('first_fail_round')} {p5.get('reason') or ''}",
          flush=True)
    p6 = probe_cross_conversation(client, model, conversations=8)
    print(f"  P6 cross-conv-load    score={p6['score']:.2f} lat={p6['lat']:.2f}s "
          f"completed={p6['completed']}/{p6['rounds']} "
          f"first_fail={p6.get('first_fail_round')} {p6.get('reason') or ''}",
          flush=True)
    # P1-P4 are 0/1, P5/P6 are 0..1; total ranges 0..6.
    total = (p1["score"] + p2["score"] + p3["score"] + p4["score"]
             + p5["score"] + p6["score"])
    avg_lat = (p1["lat"] + p2["lat"] + p3["lat"] + p4["lat"]
               + p5["lat"] + p6["lat"]) / 6
    # Either sustained probe failing = degradation. Cross-conv (P6) is the
    # one that matches real W2.7 v1 load.
    sustained_ok = p5["score"] >= 1.0 and p6["score"] >= 1.0
    print(f"  TOTAL {total:.2f}/6 | avg_lat {avg_lat:.2f}s | "
          f"sustained_ok={sustained_ok}", flush=True)

    print(f"  [unload] {unload_model(model)}", flush=True)
    return {"model": model, "total": round(total, 3), "avg_lat": round(avg_lat, 2),
            "sustained_ok": sustained_ok,
            "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6}


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    targets = [sys.argv[1]] if len(sys.argv) > 1 else MODELS
    print(f"[base] {BASE_URL}\n[models] {len(targets)} sequential", flush=True)

    rows = [run_one_model(client, m) for m in targets]

    if len(rows) > 1:
        rows_sorted = sorted(rows, key=lambda r: (-r["total"], r["avg_lat"]))
        print("\n" + "=" * 70)
        print("RANKING (higher score wins; tiebreak by lower latency)")
        print("=" * 70)
        for i, r in enumerate(rows_sorted, 1):
            tag = " " if r.get("sustained_ok") else "⚠"
            print(f"  {i}. {tag} {r['model']:55s} {r['total']:.2f}/6  "
                  f"{r['avg_lat']:.2f}s")

    out = _LAB_ROOT / "results" / "model_capability_test.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
