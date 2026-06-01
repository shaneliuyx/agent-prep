"""§8.7.3 — LongMemEval cross-validation eval driver.

For each question in ``data/longmemeval_slice_w358.json``, replays the
question's evidence sessions into BOTH EverCore (Bucket-1) and Qdrant
(Bucket-2), retrieves the top-k memories, asks a single-shot reader LLM
to answer, scores via ``judge_sonnet.judge``, and writes per-question
results to ``data/results_w358.jsonl`` (line-by-line for incremental
checkpoint).

Architecture decisions (locked in §8.7.3 design pass, 2026-05-25):
    1. Reader LLM   = ``gpt-oss-20b`` via local oMLX (same model for both
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

from openai import OpenAI

from src.consolidation import summarize_scroll
from src.judge_sonnet import judge
from src.tiered_memory_qdrant import TieredMemory, TieredMemoryConfig

EVERCORE = "http://localhost:1995"
# Empirical (2026-05-26 timing probe): single EverCore POST ~17s, single FLUSH
# ~85s. With 3 evidence sessions per LongMemEval question, EverCore imprint wall
# alone is ~300s before the 60s async-extraction wait. Cap at 600s leaves room
# for 4-session questions plus retrieve/read overhead.
PER_QUESTION_CAP_S = 600.0
EVERCORE_ASYNC_WAIT_S = 60.0
EVERCORE_HTTP_TIMEOUT_S = 180.0  # FLUSH does synchronous LLM extraction
TOP_K = 5
READER_MODEL = os.getenv("MODEL_HAIKU", "gpt-oss-20b-MXFP4-Q8")

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
        else:  # qdrant
            tm = _qd_tm(user_id)
            for idx, session in enumerate(q["haystack_sessions"]):
                t0 = time.perf_counter()
                imprinted, info = _qd_imprint_session(tm, qid, idx, session)
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


def run_one(q: dict) -> dict:
    record = {"question_id": q["question_id"], "question_type": q["question_type"],
              "question": q["question"], "gold": str(q["answer"])}
    for backend in ("qdrant", "evercore"):
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
    args = ap.parse_args()

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
        record = run_one(q)
        with RESULTS_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")

    elapsed = time.perf_counter() - t_total
    print(f"\n>>> DONE — {len(qs)} questions in {elapsed/60:.1f} min")
    print(f"    results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
