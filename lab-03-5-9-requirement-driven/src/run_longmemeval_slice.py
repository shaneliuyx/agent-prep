"""§8.7.3 — LongMemEval cross-validation eval driver.

For each question in ``data/longmemeval_slice_w358.json``, replays the
question's evidence sessions into BOTH EverCore (Bucket-1) and Qdrant
(Bucket-2), retrieves the top-k memories, asks a single-shot reader LLM
to answer, scores via ``judge_sonnet.judge``, and writes per-question
results to ``data/results_w358.jsonl`` (line-by-line for incremental
checkpoint).

Architecture decisions (locked in §8.7.3 design pass, 2026-05-25):
    1. Reader LLM   = ``gemma-4-26B-A4B`` via local oMLX (MODEL_READER; same model for both
                      backends — fair comparison)
    2. EverCore     = one POST per LongMemEval session, distinct
                      ``session_id`` per session; flush each; single 60s
                      async-extraction wait per question (not per session)
    3. Qdrant input = per-session: concat turns into scroll text,
                      ``summarize_scroll`` → ``tm.imprint``; SKIP-gated
                      sessions contribute nothing (tracked in results)
    4. Reader prompt = single-shot "given these memories, answer Q"
                      (no ReAct loop — that's a different experiment)
    5. Per-Q wall cap = 180s; exceeded → marked ``<timeout>``, scored 0

Run from lab root (after ``scripts/build_slice.py``):

    uv run python -m src.run_longmemeval_slice --smoke 1   # 1-Q smoke
    uv run python -m src.run_longmemeval_slice             # full 20-Q
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
import traceback
import urllib.request

from dotenv import load_dotenv

load_dotenv()  # read lab .env (OMLX_BASE_URL/KEY, MODEL_*, EVERCORE_*/cap overrides)

from openai import OpenAI

from src.consolidation import summarize_scroll
from src.judge_sonnet import judge
from src.tiered_memory_qdrant import TieredMemory, TieredMemoryConfig

EVERCORE = "http://localhost:1995"
# Empirical (2026-05-26 timing probe): single EverCore POST ~17s, single FLUSH
# ~85s. With 3 evidence sessions per LongMemEval question, EverCore imprint wall
# alone is ~300s before the 60s async-extraction wait. Cap at 600s leaves room
# for 4-session questions plus retrieve/read overhead.
# All three are env-overridable. EverCore's /memories/flush runs SYNCHRONOUS LLM
# extraction on oMLX; a large LongMemEval session + oMLX contention routinely
# exceeds the old 180s HTTP timeout, so the default is raised to 600s and the
# per-question cap to 1200s. Lower them via env for fast local iteration.
PER_QUESTION_CAP_S = float(os.getenv("PER_QUESTION_CAP_S", "1200"))
EVERCORE_ASYNC_WAIT_S = float(os.getenv("EVERCORE_ASYNC_WAIT_S", "60"))
EVERCORE_HTTP_TIMEOUT_S = float(os.getenv("EVERCORE_HTTP_TIMEOUT_S", "600"))
TOP_K = 5
READER_MODEL = os.getenv("MODEL_READER", os.getenv("MODEL_HAIKU", "gemma-4-26B-A4B-it-heretic-4bit"))

LAB_ROOT = pathlib.Path(__file__).resolve().parent.parent
SLICE_PATH = LAB_ROOT / "data" / "longmemeval_slice_w358.json"
RESULTS_PATH = LAB_ROOT / "data" / "results_w358.jsonl"

READER_PROMPT = """You are answering a question using ONLY the retrieved memories below.

If the memories contain the answer, respond with a single short answer (one short sentence, or a single number/name). If they don't, respond with: I don't know.

QUESTION: {question}

RETRIEVED MEMORIES:
{memories}

ANSWER:"""


# ── EverCore helpers (mirror demo_conversational_imprint.py, kept inline
# so the driver is self-contained and doesn't drag the demo into prod). ──

def _ec_post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{EVERCORE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(req, timeout=EVERCORE_HTTP_TIMEOUT_S).read())


def _ec_imprint_session(user_id: str, session_id: str, turns: list[dict]) -> dict:
    ts = int(time.time() * 1000)
    messages = [
        {"role": t["role"], "timestamp": ts + i, "content": t["content"]}
        for i, t in enumerate(turns)
    ]
    post_resp = _ec_post(
        "/api/v1/memories",
        {"user_id": user_id, "session_id": session_id, "messages": messages},
    )
    flush_resp = _ec_post(
        "/api/v1/memories/flush",
        {"user_id": user_id, "session_id": session_id},
    )
    return {"post_status": post_resp["data"].get("status"),
            "flush_status": flush_resp["data"].get("status")}


def _ec_search(user_id: str, query: str, k: int = TOP_K) -> list[dict]:
    body = {"query": query, "top_k": k, "filters": {"user_id": user_id}}
    return _ec_post("/api/v1/memories/search", body).get("data", {}).get("episodes", [])


# ── Qdrant helpers — use lab's standard TieredMemoryQdrant pipeline. ──

def _qd_tm(user_id: str) -> TieredMemory:
    """Qdrant-variant TieredMemory. Same-name twin of the EverCore variant —
    distinguished only by which module it's imported from (§7.5 pattern)."""
    cfg = TieredMemoryConfig()
    return TieredMemory(user_id=user_id, agent_id="lme-eval", config=cfg)


# ── Backend dispatch (W3.5.9) ────────────────────────────────────────
# Baseline driver had 'qdrant' (W3.5.8 §7.7) and 'evercore' (§7.1, HTTP).
# Phase 3 adds 'mem0'; Phase 4 adds 'atomic_fact' + 'hybrid' (the router).
# EverCore is an HTTP service handled inline in _run_backend, so it is NOT
# built here. qdrant uses the lab's TieredMemory; the W3.5.9 backends are
# duck-typed twins (same imprint(content, metadata) / query_context(query, k)).
OBJECT_BACKENDS = ("qdrant", "mem0", "atomic_fact", "hybrid", "three_tier")
ALL_BACKENDS = ("qdrant", "evercore", "mem0", "atomic_fact", "hybrid", "three_tier")


def _build_backend(backend: str, user_id: str):
    if backend == "qdrant":
        return _qd_tm(user_id)                                  # W3.5.8 2-tier (Qdrant variant)
    if backend == "mem0":
        from src.mem0_backend_adapter import Mem0Adapter        # Phase 3
        return Mem0Adapter(user_id=user_id)
    if backend == "atomic_fact":
        from src.atomic_fact_memory import AtomicFactMemory     # Phase 4
        return AtomicFactMemory(user_id=user_id)
    if backend == "hybrid":
        from src.router_memory import RouterMemory              # Phase 4 — question-type router
        return RouterMemory(user_id=user_id)
    if backend == "three_tier":
        from src.three_tier_memory import ThreeTierMemory       # Phase 7 — L1+L2+L3 (HyperMem)
        return ThreeTierMemory(user_id=user_id)
    raise ValueError(f"unknown object-backend: {backend!r}")


def _session_to_scroll(session: list[dict]) -> str:
    """Concat session turns into a single text blob suitable for
    ``summarize_scroll``. Format mirrors a task-scroll shape: each turn
    becomes one tagged line so the summarizer can locate the salient
    content. Not strictly the lab's `[completed]`/`[journal]` convention,
    but close enough that the summarize prompt produces useful output."""
    lines = []
    for t in session:
        tag = "USER" if t["role"] == "user" else "ASSISTANT"
        lines.append(f"[{tag}] {t['content']}")
    return "\n".join(lines)


def _qd_imprint_session(tm: TieredMemory, qid: str, idx: int,
                        session: list[dict]) -> tuple[bool, str | None]:
    """Returns (imprinted, summary_or_reason)."""
    scroll = _session_to_scroll(session)
    summary = summarize_scroll(scroll)
    if summary is None or summary.strip().upper() == "SKIP":
        return False, "summarize_skip"
    tm.imprint(summary, metadata={"quest_id": f"{qid}-sess{idx}",
                                   "subject": f"LongMemEval session {idx}"})
    return True, summary


# ── Reader LLM — single-shot, same model for both backends ───────────

def _reader_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("OMLX_BASE_URL"),
        api_key=os.getenv("OMLX_API_KEY"),
    )


def _read_answer(question: str, memories: list[dict]) -> str:
    """Format memories + question, ask reader LLM for a short answer."""
    if not memories:
        body = "(no memories retrieved)"
    else:
        lines = []
        for i, m in enumerate(memories[:TOP_K], 1):
            content = (m.get("content") or m.get("summary")
                       or m.get("episode") or "").strip()
            lines.append(f"[{i}] {content[:400]}")
        body = "\n".join(lines)
    prompt = READER_PROMPT.format(question=question, memories=body)
    resp = _reader_client().chat.completions.create(
        model=READER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=120,
    )
    return (resp.choices[0].message.content or "").strip()


# ── Per-question driver ──────────────────────────────────────────────

def _run_backend(backend: str, q: dict) -> dict:
    qid = q["question_id"]
    user_id = f"lme-{qid}-{backend[:2]}"  # isolate per-question per-backend
    t_start = time.perf_counter()
    imprint_walls: list[float] = []
    imprint_meta: list[dict] = []
    tm: TieredMemory | None = None  # hoisted so retrieve branch can see it

    try:
        if backend == "evercore":
            for idx, session in enumerate(q["haystack_sessions"]):
                t0 = time.perf_counter()
                meta = _ec_imprint_session(user_id, f"{qid}-{idx}", session)
                imprint_walls.append(time.perf_counter() - t0)
                imprint_meta.append(meta)
            time.sleep(EVERCORE_ASYNC_WAIT_S)
        else:  # object-backends: qdrant / mem0 / atomic_fact / hybrid / three_tier
            tm = _build_backend(backend, user_id)
            assert tm is not None  # built above — narrows the hoisted Optional
            for idx, session in enumerate(q["haystack_sessions"]):
                t0 = time.perf_counter()
                if backend == "qdrant":
                    # 2-tier write path: summarize the session scroll, then imprint.
                    imprinted, info = _qd_imprint_session(tm, qid, idx, session)
                else:
                    # W3.5.9 backends extract internally (atomic facts / messages /
                    # routed), so imprint the raw session scroll directly.
                    info = tm.imprint(
                        _session_to_scroll(session),
                        metadata={"quest_id": f"{qid}-sess{idx}",
                                  "subject": f"LongMemEval session {idx}"},
                    )
                    imprinted = True
                imprint_walls.append(time.perf_counter() - t0)
                imprint_meta.append({"imprinted": imprinted, "info": str(info)[:80]})

        wall_imprint = sum(imprint_walls)

        # Retrieval
        t0 = time.perf_counter()
        if backend == "evercore":
            hits = _ec_search(user_id, q["question"], k=TOP_K)
        else:
            assert tm is not None  # guaranteed by branch above
            hits = tm.query_context(q["question"], k=TOP_K)
        wall_retrieve = time.perf_counter() - t0

        # Reader
        t0 = time.perf_counter()
        predicted = _read_answer(q["question"], hits)
        wall_read = time.perf_counter() - t0

        # Cap check
        if time.perf_counter() - t_start > PER_QUESTION_CAP_S:
            return {"status": "timeout", "predicted": predicted,
                    "wall_imprint": wall_imprint, "wall_retrieve": wall_retrieve,
                    "wall_read": wall_read, "n_imprinted": len(imprint_meta),
                    "hits": len(hits)}

        return {"status": "ok", "predicted": predicted,
                "wall_imprint": wall_imprint, "wall_retrieve": wall_retrieve,
                "wall_read": wall_read, "n_imprinted": len(imprint_meta),
                "hits": len(hits), "imprint_meta": imprint_meta}

    except Exception as exc:
        return {"status": "error", "error": repr(exc)[:200],
                "trace": traceback.format_exc()[-400:],
                "wall_imprint": sum(imprint_walls)}


def run_one(q: dict, backends: tuple[str, ...] = ("qdrant", "evercore")) -> dict:
    record = {"question_id": q["question_id"], "question_type": q["question_type"],
              "question": q["question"], "gold": str(q["answer"])}
    for backend in backends:
        print(f"  [{backend}] running...")
        result = _run_backend(backend, q)
        if result["status"] == "ok":
            verdict = judge(q["question"], str(q["answer"]), result["predicted"])
            result.update(verdict)
        else:
            result.update({"correct": False, "score": 0.0,
                           "reason": f"<{result['status']}>"})
        record[backend] = result
        print(f"    -> predicted={result.get('predicted','<n/a>')[:100]!r}")
        print(f"    -> correct={result.get('correct')} reason={result.get('reason','')[:120]}")
        print(f"    -> wall: imprint={result.get('wall_imprint',0):.1f}s "
              f"retrieve={result.get('wall_retrieve',0):.2f}s "
              f"read={result.get('wall_read',0):.2f}s")
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0,
                    help="run only first N questions (for wiring validation)")
    ap.add_argument("--skip-evercore", action="store_true",
                    help="skip EverCore backend (run Qdrant only)")
    ap.add_argument("--backend", choices=[*ALL_BACKENDS, "all"], default="all",
                    help="run a single backend, or 'all' for the full comparison "
                         "(qdrant, evercore, mem0, atomic_fact, hybrid)")
    args = ap.parse_args()

    backends = ALL_BACKENDS if args.backend == "all" else (args.backend,)
    if args.skip_evercore:
        backends = tuple(b for b in backends if b != "evercore")
    print(f">>> backends: {backends}")

    qs = json.loads(SLICE_PATH.read_text())
    if args.smoke:
        qs = qs[:args.smoke]
        print(f">>> SMOKE MODE — first {len(qs)} question(s)")
    else:
        print(f">>> FULL SLICE — {len(qs)} questions")

    if args.skip_evercore:
        # Globally short-circuit the evercore branch by monkey-patching the
        # backend loop below. Cleaner than a flag percolating through run_one().
        global _run_backend
        _orig = _run_backend
        def _patched(backend, q):
            if backend == "evercore":
                return {"status": "skipped", "predicted": "<evercore_skipped>",
                        "wall_imprint": 0, "wall_retrieve": 0, "wall_read": 0,
                        "n_imprinted": 0, "hits": 0}
            return _orig(backend, q)
        _run_backend = _patched
        print("    (EverCore backend SKIPPED via --skip-evercore)")

    RESULTS_PATH.unlink(missing_ok=True)
    t_total = time.perf_counter()
    for i, q in enumerate(qs, 1):
        print(f"\n[{i}/{len(qs)}] qid={q['question_id']} type={q['question_type']}")
        record = run_one(q, backends)
        with RESULTS_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")

    elapsed = time.perf_counter() - t_total
    print(f"\n>>> DONE — {len(qs)} questions in {elapsed/60:.1f} min")
    print(f"    results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
