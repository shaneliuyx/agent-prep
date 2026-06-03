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
from src.llm_retry import chat_with_retry, is_cloak  # 503 backoff + cloak detection
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

# Count questions ("how many / how much / how often") need MORE retrieval depth
# (their answer items are heterogeneous and scattered — a single dense query
# can't gather them at k=5) and an enumerate-then-count reader prompt with room
# to list items. Probe-validated: at k=40 over the user-turn store all 3 needles
# surface, and the count prompt turns "I don't know" into a clean enumeration.
COUNT_TOP_K = int(os.getenv("COUNT_TOP_K", "40"))
COUNT_MAX_TOKENS = int(os.getenv("COUNT_MAX_TOKENS", "500"))
READ_MAX_TOKENS = int(os.getenv("READ_MAX_TOKENS", "120"))

LAB_ROOT = pathlib.Path(__file__).resolve().parent.parent
SLICE_PATH = LAB_ROOT / "data" / "longmemeval_slice_w358.json"
# Per-run results: each run writes its OWN file under data/results/ (never
# clobbered). scripts/aggregate.py merges them (latest record per qid/backend
# cell wins) into the canonical merged.jsonl. This prevents the single-file
# `unlink`-and-overwrite that lost full-run raw data 3× — a probe/smoke/partial
# run no longer destroys a prior full run. RESULTS_PATH (the merged file) is what
# rejudge.py + analysis read.
RESULTS_DIR = LAB_ROOT / "data" / "results"
RESULTS_PATH = RESULTS_DIR / "merged.jsonl"

READER_PROMPT = """You are an information-extraction function in a data pipeline. Your input is a set of RETRIEVED RECORDS and a QUERY; your output is the answer extracted from the records. This is a text-processing task — do not describe yourself, your role, or any assistant identity; output only the answer.

If the records contain the answer, respond with a single short answer (one short sentence, or a single number/name). If they don't, respond with: I don't know.

QUERY: {question}

RETRIEVED RECORDS:
{memories}

ANSWER:"""

# Enumerate-then-count reader for "how many" questions. Loaded from a file so
# the prompt can be iterated with the probe harness (src/probe_reader.py) without
# touching the driver; falls back to the baseline prompt if the file is missing.
_COUNT_PROMPT_PATH = LAB_ROOT / "src" / "prompts" / "reader_count.txt"
COUNT_READER_PROMPT = (
    _COUNT_PROMPT_PATH.read_text() if _COUNT_PROMPT_PATH.exists() else READER_PROMPT
)


def _is_count_question(question: str) -> bool:
    """Count-type questions begin with a 'how many/much/often' stem. These need
    the deeper retrieval + enumeration reader path; lookup questions do not."""
    return question.strip().lower().startswith(("how many", "how much", "how often"))


# ── Knowledge-update (latest-wins) reader path ───────────────────────────
# Knowledge-update questions ask for a CURRENT value that CHANGED over time
# (mortgage pre-approval $350k→$400k, Rachel moved A→B). The flat atomic store
# retrieves BOTH values, equally ranked, with no recency signal → the reader
# coin-flips and picks wrong (measured: $350k vs gold $400k, Hawaii vs Paris).
# A stronger reader can't fix this — the answer is unresolvable without recency.
# Fix: the driver imprints each session with quest_id "{qid}-sess{idx}" (idx =
# session order = recency). Surface that as an [sN] tag per fact, pull a DEEPER
# window so old+new both appear, and tell the reader the highest-sN value wins.
KU_TOP_K = int(os.getenv("KU_TOP_K", "20"))
KU_MAX_TOKENS = int(os.getenv("KU_MAX_TOKENS", "300"))

KU_READER_PROMPT = """You are an information-extraction function in a data pipeline. Your input is RETRIEVED RECORDS and a CURRENT-STATE QUERY. This is a text-processing task — do not describe yourself or your role; output only the answer.

IMPORTANT: the value asked for may have CHANGED over time. Each record is tagged [sN] = the conversation session it came from; a HIGHER N is MORE RECENT. If records give conflicting values, the answer is the value from the MOST RECENT session (highest [sN]); if sessions tie, prefer the latest date mentioned in the record text. Report the CURRENT value, not an old one.

QUERY: {question}

RETRIEVED RECORDS:
{memories}

ANSWER:"""


def _session_recency(meta: dict) -> int:
    """Recover session index (recency) from a fact's metadata. The driver stamps
    quest_id='{qid}-sess{idx}'; higher idx = more recent. Returns -1 if absent
    (mem0/evercore/qdrant metadata may not carry it → reader falls back to
    in-text date cues)."""
    import re
    qid = str((meta or {}).get("quest_id", ""))
    m = re.search(r"sess(\d+)", qid)
    return int(m.group(1)) if m else -1


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
OBJECT_BACKENDS = ("qdrant", "mem0", "atomic_fact", "hybrid", "three_tier", "ensemble")
ALL_BACKENDS = ("qdrant", "evercore", "mem0", "atomic_fact", "hybrid", "three_tier", "ensemble")

# Questions excluded from QUALITY (accuracy) analysis because their gold answer
# is unreachable by sound reasoning — measuring willingness-to-match-a-bad-label,
# not memory quality. Still run (predictions saved) so the broken-gold case is
# inspectable; just dropped from accuracy aggregates (rejudge.py honors this).
#   0a995998 ("how many items to pick up/return from a store?", gold=3):
#     strict store items = 2 (blazer + boots); gold's 3rd is a sweater LENT TO A
#     SISTER (not a store), contradicting the question's own "from a store"
#     qualifier. No sound path reaches 3 (only boots-double-count or sweater-
#     inclusion). Replaced by synth_books_bought_v1 (clean count, gold=4).
QUALITY_EXCLUDE = {"0a995998"}


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
    if backend == "ensemble":
        from src.ensemble_memory import EnsembleMemory          # §2.3 stretch — RRF fusion
        return EnsembleMemory(user_id=user_id)
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


# ── Reader LLM — single-shot, same model for every backend ───────────
# All LLM roles (reader/extraction/consolidation) run on ONE model via the
# LLM endpoint (default VibeProxy :8317 → Haiku 4.5). Embeddings stay on local
# oMLX (EMBED_BASE_URL) — VibeProxy hosts no embed model. LLM_BASE_URL falls
# back to OMLX_BASE_URL so a fully-local config still works unchanged.

def _reader_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("LLM_BASE_URL", os.getenv("OMLX_BASE_URL")),
        api_key=os.getenv("LLM_API_KEY", os.getenv("OMLX_API_KEY")),
    )


def _read_answer(question: str, memories: list[dict], question_type: str = "") -> str:
    """Format memories + question, ask reader LLM for an answer.

    Count questions ('how many/much/often') take the deeper path: more memories
    in context (COUNT_TOP_K), an enumerate-then-count prompt, and enough tokens
    to list items before answering. Lookup questions keep the terse single-shot
    path. Probe-validated: the count path turns 'I don't know' into a correct
    enumeration on the multi-session counting questions."""
    is_ku = (question_type == "knowledge-update")
    is_count = _is_count_question(question)
    if is_ku:                       # latest-value-wins path (recency-tagged)
        cap, prompt_tmpl, max_tokens = KU_TOP_K, KU_READER_PROMPT, KU_MAX_TOKENS
    elif is_count:                  # enumerate-then-count path
        cap, prompt_tmpl, max_tokens = COUNT_TOP_K, COUNT_READER_PROMPT, COUNT_MAX_TOKENS
    else:                           # terse single-shot lookup
        cap, prompt_tmpl, max_tokens = TOP_K, READER_PROMPT, READ_MAX_TOKENS
    if not memories:
        body = "(no memories retrieved)"
    else:
        lines = []
        for i, m in enumerate(memories[:cap], 1):
            content = (m.get("content") or m.get("summary")
                       or m.get("episode") or "").strip()
            tag = ""
            if is_ku:               # surface recency so latest-wins is resolvable
                s = _session_recency(m.get("metadata", {}))
                tag = f"|s{s}" if s >= 0 else ""
            lines.append(f"[{i}{tag}] {content[:400]}")
        # Knowledge-update: order by recency so the reader sees newest last.
        body = "\n".join(lines)
    prompt = prompt_tmpl.format(question=question, memories=body)
    # ROOT FIX for the VibeProxy "Claude-Code persona" cloak: the prompt is framed
    # as a data-extraction task (see READER_PROMPT / reader_count.txt), so the
    # injected persona treats it as legitimate text-processing and ANSWERS rather
    # than refusing — the reader uses the USER prompt to answer, no system role.
    # The cloak is also intermittent (load-triggered), so as a SAFETY NET we
    # detect a residual persona refusal and retry with a temperature nudge (a
    # temp=0 re-run would re-cloak identically). We never fall back to "I don't
    # know" — the framing makes the reader answer; we return its answer.
    client = _reader_client()
    out = ""
    for attempt in range(4):
        resp = chat_with_retry(  # reader → VibeProxy Haiku; ride out 503 cooldowns
            client,
            model=READER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0 if attempt == 0 else 0.5,  # break determinism on cloak retry
            max_tokens=max_tokens,
        )
        out = (resp.choices[0].message.content or "").strip()
        if not is_cloak(out):
            return out
        time.sleep(2)  # intermittent cloak — retry with temperature variation
    # PERSISTENT cloak: VibeProxy's injected Claude-Code persona overrode the
    # data-extraction framing (happens on NARRATIVE input like qdrant summaries,
    # which read as personal chat; atomic-fact input stays data-shaped and
    # answers). Fall back to the LOCAL model — it has no injected persona, so it
    # answers. Weaker reasoning than Haiku, but a REAL answer beats a refusal
    # scored 0, and it's never persona text nor a hardcoded "I don't know".
    local = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
    resp = local.chat.completions.create(
        model=os.getenv("MODEL_EXTRACT", os.getenv("MODEL_HAIKU", "Qwen2.5-Coder-14B-Instruct-MLX-4bit")),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    local_out = (resp.choices[0].message.content or "").strip()
    return local_out if local_out and not is_cloak(local_out) else out


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

        # Retrieval — count questions pull a deeper window so the scattered
        # answer items all land in context (k=5 can't gather them).
        if q.get("question_type") == "knowledge-update":
            ret_k = KU_TOP_K            # deep enough that old+new value both appear
        elif _is_count_question(q["question"]):
            ret_k = COUNT_TOP_K
        else:
            ret_k = TOP_K
        # Component 2 — role-aware retrieval (atomic_fact). Provenance policy by
        # question type: axes whose answer is assistant-stated
        # (single-session-assistant) or event-based (temporal-reasoning) keep ALL
        # roles; user-centric axes + multi-session filter to user facts (preserve
        # the de-flooding the old user-turn-only write filter used to give).
        base_type = q.get("question_type", "").rsplit("_abs", 1)[0]
        roles_policy = (None if base_type in ("single-session-assistant", "temporal-reasoning")
                        else ["user"])
        t0 = time.perf_counter()
        if backend == "evercore":
            hits = _ec_search(user_id, q["question"], k=ret_k)
        elif backend == "atomic_fact":
            assert tm is not None  # built above for object-backends
            hits = tm.query_context(q["question"], k=ret_k, roles=roles_policy)
        else:
            assert tm is not None  # guaranteed by branch above
            hits = tm.query_context(q["question"], k=ret_k)
        wall_retrieve = time.perf_counter() - t0

        # Reader
        t0 = time.perf_counter()
        predicted = _read_answer(q["question"], hits, q.get("question_type", ""))
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
            # Judge is on VibeProxy (sonnet) + has its own 503 retry. If it STILL
            # fails (cooldown outlasts backoff), do NOT crash the run — save the
            # prediction unjudged (correct=None) so it completes; rejudge later
            # via replay.py once VibeProxy is cool. A judge error must never lose
            # 2 hours of imprints.
            try:
                result.update(judge(q["question"], str(q["answer"]), result["predicted"]))
            except Exception as exc:  # noqa: BLE001
                result.update({"correct": None, "score": 0.0,
                               "reason": f"<judge_error: {repr(exc)[:80]}>"})
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
    ap.add_argument("--qid", default="",
                    help="run only the question(s) with this/these question_id(s) "
                         "(comma-separated). For testing a single (e.g. new) question.")
    ap.add_argument("--run-tag", default="",
                    help="explicit name for this run's output file "
                         "(data/results/run_<tag>.jsonl). Default: auto from "
                         "backends+scope+epoch. Never clobbers prior runs.")
    ap.add_argument("--slice", default=str(SLICE_PATH),
                    help="path to the slice JSON (default: the w358 2-axis slice; "
                         "pass data/longmemeval_slice_6axis.json for the 6-axis slice)")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR),
                    help="dir for per-run output (default: data/results). Use a "
                         "separate dir per slice (e.g. data/results_6axis) so "
                         "aggregate.py doesn't merge questions across slices.")
    args = ap.parse_args()

    backends = ALL_BACKENDS if args.backend == "all" else (args.backend,)
    if args.skip_evercore:
        backends = tuple(b for b in backends if b != "evercore")
    print(f">>> backends: {backends}")

    qs = json.loads(pathlib.Path(args.slice).read_text())
    print(f">>> slice: {args.slice}")
    if args.qid:
        want = {q.strip() for q in args.qid.split(",") if q.strip()}
        qs = [q for q in qs if q["question_id"] in want]
        print(f">>> QID FILTER — {len(qs)} question(s): {sorted(want)}")
    elif args.smoke:
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

    # Per-run output file (NEVER clobbers prior runs). Tag = scope + backends +
    # epoch, so a full run, a --qid re-run, and a --backend probe each get their
    # own file; aggregate.py merges them latest-per-cell.
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    scope = args.qid.replace(",", "-")[:40] if args.qid else (f"smoke{args.smoke}" if args.smoke else "full")
    tag = args.run_tag or f"{'-'.join(backends)[:40]}_{scope}_{int(time.time())}"
    run_file = results_dir / f"run_{tag}.jsonl"
    t_total = time.perf_counter()
    for i, q in enumerate(qs, 1):
        print(f"\n[{i}/{len(qs)}] qid={q['question_id']} type={q['question_type']}")
        record = run_one(q, backends)
        with run_file.open("a") as f:
            f.write(json.dumps(record) + "\n")

    elapsed = time.perf_counter() - t_total
    print(f"\n>>> DONE — {len(qs)} questions in {elapsed/60:.1f} min")
    print(f"    run file: {run_file}")
    print(f"    merge + matrix: uv run python -m scripts.aggregate")


if __name__ == "__main__":
    main()
