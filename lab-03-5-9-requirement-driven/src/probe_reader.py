"""Reader-probe harness — iterate retrieve→read→judge in seconds, NOT 13 min.

The full eval driver (``run_longmemeval_slice``) re-imprints every backend on
each run: ~13 min wall, dominated by per-message extraction + EverCore flushes.
But imprint writes to a DETERMINISTIC address (``user_id = lme-{qid}-{backend[:2]}``,
Qdrant collection keyed off it). So once a full run has imprinted a question,
the facts persist — and the *read side* (retrieval depth, char-cap, reader
prompt, max_tokens) can be tuned against that persisted store in seconds.

This harness rebuilds the backend object pointing at the SAME store and runs
ONLY retrieve + read + judge. No imprint. Use it to answer the open question:
when a backend undercounts (gold=3, predicts 1), is the needle (a) absent from
the store → extraction-recall bug (imprint side, not fixable here), or (b) in
the store but ranked below top_k → raise k / rerank (read side, fixable here)?

Usage (from lab root, after a full run has imprinted the qid):

    # show what's actually retrieved at depth 30 for the clothing-count Q
    uv run python -m src.probe_reader --qid 0a995998 --backend atomic_fact \\
        --top-k 30 --show-facts

    # sweep top_k to find the recall knee
    uv run python -m src.probe_reader --qid 0a995998 --backend atomic_fact \\
        --sweep 5,10,20,40,80

    # grep the store for the needle terms (is it even in there?)
    uv run python -m src.probe_reader --qid 0a995998 --backend atomic_fact \\
        --grep "pick up,return,blazer,dry clean,tailor"
"""
from __future__ import annotations

import argparse
import json
import time

from openai import OpenAI

from src.judge_sonnet import judge
from src.run_longmemeval_slice import (
    READER_MODEL,
    READER_PROMPT,
    _build_backend,
    _ec_search,
    _reader_client,
    _session_to_scroll,
)
from src.run_longmemeval_slice import SLICE_PATH

# Reader-model override (set via CLI). Lets the probe swap ONLY the answer LLM
# (e.g. local gemma vs Haiku 4.5 via VibeProxy) while retrieval/embedding stay
# fixed on local bge-m3 — isolating "does the reader model matter?" cleanly.
_READER_OVERRIDE: dict[str, str] = {}


def _active_reader() -> tuple[OpenAI, str]:
    """Return (client, model) for the reader. Uses the CLI override if set,
    else the driver's default OMLX client + READER_MODEL."""
    if _READER_OVERRIDE:
        client = OpenAI(
            base_url=_READER_OVERRIDE["base_url"],
            api_key=_READER_OVERRIDE.get("key", "dummy"),
        )
        return client, _READER_OVERRIDE["model"]
    return _reader_client(), READER_MODEL


def _load_q(qid: str) -> dict:
    qs = json.loads(SLICE_PATH.read_text())
    for q in qs:
        if q["question_id"] == qid:
            return q
    raise SystemExit(f"qid {qid!r} not in slice ({len(qs)} questions)")


def _user_id(qid: str, backend: str) -> str:
    """Mirror the driver's deterministic address EXACTLY (lme-{qid}-{backend[:2]})
    so the rebuilt backend reads the store the full run imprinted."""
    return f"lme-{qid}-{backend[:2]}"


def _retrieve(backend: str, qid: str, query: str, k: int,
              uid_override: str = "") -> list[dict]:
    uid = uid_override or _user_id(qid, backend)
    if backend == "evercore":
        return _ec_search(uid, query, k=k)
    tm = _build_backend(backend, uid)
    return tm.query_context(query, k=k)


def _fact_text(m: dict) -> str:
    return (m.get("content") or m.get("summary") or m.get("episode") or "").strip()


def _read(question: str, hits: list[dict], char_cap: int, max_tokens: int,
          prompt_tmpl: str) -> str:
    if not hits:
        body = "(no memories retrieved)"
    else:
        body = "\n".join(f"[{i}] {_fact_text(m)[:char_cap]}" for i, m in enumerate(hits, 1))
    prompt = prompt_tmpl.format(question=question, memories=body)
    client, model = _active_reader()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _one(backend: str, q: dict, k: int, char_cap: int, max_tokens: int,
         prompt_tmpl: str, show_facts: bool, uid_override: str = "") -> None:
    qid, question, gold = q["question_id"], q["question"], str(q["answer"])
    t0 = time.perf_counter()
    hits = _retrieve(backend, qid, question, k, uid_override)
    t_ret = time.perf_counter() - t0
    print(f"\n=== [{backend}] k={k} char_cap={char_cap} max_tokens={max_tokens} ===")
    print(f"    retrieved {len(hits)} hits in {t_ret:.2f}s")
    if show_facts:
        for i, m in enumerate(hits, 1):
            score = m.get("score")
            stag = f"{score:.3f}" if isinstance(score, (int, float)) else "  -  "
            print(f"    [{i:>2}|{stag}] {_fact_text(m)[:160]}")
    t0 = time.perf_counter()
    pred = _read(question, hits, char_cap, max_tokens, prompt_tmpl)
    t_read = time.perf_counter() - t0
    verdict = judge(question, gold, pred)
    print(f"    GOLD={gold!r}  PRED={pred[:120]!r}")
    print(f"    correct={verdict.get('correct')} reason={verdict.get('reason','')[:140]}")
    print(f"    walls: retrieve={t_ret:.2f}s read={t_read:.2f}s")


def _turns_scroll(q: dict, roles: set[str]) -> list[str]:
    """Per-session scrolls keeping only turns whose role is in ``roles``.
    Mirrors the driver's _session_to_scroll shape so AtomicFactMemory.imprint
    splits + extracts per line exactly as in production."""
    out = []
    for sess in q["haystack_sessions"]:
        filtered = [t for t in sess if t["role"] in roles]
        if filtered:
            out.append(_session_to_scroll(filtered))
    return out


def _reimprint(backend: str, q: dict, mode: str, k: int, char_cap: int,
               max_tokens: int, prompt_tmpl: str) -> None:
    """ABLATION: imprint a FRESH store from a subset of turns, then read.
    mode = all | user | assistant. Proves whether dropping assistant-advice
    turns lifts the user-action needles above the distractor flood — a
    production-shaped fix (memory of the USER, not of the assistant), NOT a
    hardcoded answer. Uses a throwaway user_id so it never touches the real
    eval store."""
    from src.atomic_fact_memory import AtomicFactMemory  # only backend this ablates

    roles = {"user", "assistant"} if mode == "all" else {mode}
    uid = f"probe-{mode}-{q['question_id']}"
    mem = AtomicFactMemory(user_id=uid)
    # fresh collection each run so repeated ablations don't accumulate
    try:
        mem._qdrant.delete_collection(mem.collection)
        mem._ensure_collection()
    except Exception:
        pass
    scrolls = _turns_scroll(q, roles)
    t0 = time.perf_counter()
    for s in scrolls:
        mem.imprint(s, metadata={"subject": "probe ablation"})
    t_imp = time.perf_counter() - t0
    n = mem._qdrant.get_collection(mem.collection).points_count
    print(f"\n=== reimprint mode={mode!r} roles={sorted(roles)} ===")
    print(f"    imprinted {n} facts from {len(scrolls)} session-scrolls in {t_imp:.1f}s")
    hits = mem.query_context(q["question"], k=k)
    print(f"    retrieved {len(hits)} hits @k={k}")
    for i, m in enumerate(hits, 1):
        score = m.get("score")
        stag = f"{score:.3f}" if isinstance(score, (int, float)) else "  -  "
        print(f"    [{i:>2}|{stag}] {_fact_text(m)[:140]}")
    pred = _read(q["question"], hits, char_cap, max_tokens, prompt_tmpl)
    verdict = judge(q["question"], str(q["answer"]), pred)
    print(f"    GOLD={str(q['answer'])!r}  PRED={pred[:120]!r}")
    print(f"    correct={verdict.get('correct')} reason={verdict.get('reason','')[:140]}")


def _grep(backend: str, qid: str, query: str, terms: list[str], k: int) -> None:
    """Retrieve deep, then substring-grep the facts for needle terms. Answers
    'is the needle in the store at all?' independent of where it ranks."""
    hits = _retrieve(backend, qid, query, k)
    print(f"\n=== grep [{backend}] over top-{len(hits)} retrieved facts ===")
    for term in terms:
        matches = [(_fact_text(m), m.get("score")) for m in hits
                   if term.lower() in _fact_text(m).lower()]
        print(f"  {term!r}: {len(matches)} match(es)")
        for txt, score in matches[:6]:
            stag = f"{score:.3f}" if isinstance(score, (int, float)) else "-"
            print(f"      ({stag}) {txt[:140]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default="0a995998")
    ap.add_argument("--backend", default="atomic_fact")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--char-cap", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--show-facts", action="store_true",
                    help="print every retrieved fact with its score")
    ap.add_argument("--user-id", default="",
                    help="override the store address (e.g. probe-user-0a995998) to "
                         "read an ablation store without re-extracting")
    ap.add_argument("--sweep", default="",
                    help="comma list of top_k values to sweep, e.g. 5,10,20,40")
    ap.add_argument("--grep", default="",
                    help="comma list of needle terms to substring-search in the store")
    ap.add_argument("--grep-k", type=int, default=200,
                    help="retrieval depth for --grep (how deep to look for needles)")
    ap.add_argument("--reimprint", default="", choices=["", "all", "user", "assistant"],
                    help="ABLATION: rebuild a throwaway atomic_fact store from this "
                         "turn-subset, then read (proves the user-turn-only fix)")
    ap.add_argument("--prompt", default="",
                    help="path to an alternate reader-prompt template file "
                         "(must contain {question} and {memories})")
    ap.add_argument("--reader-model", default="",
                    help="override the answer LLM (e.g. claude-haiku-4-5-20251001). "
                         "Retrieval/embedding stay on local bge-m3 — isolates the "
                         "'does the reader model matter?' variable")
    ap.add_argument("--reader-base-url", default="http://localhost:8317/v1",
                    help="base URL for --reader-model (default: VibeProxy :8317)")
    ap.add_argument("--reader-key", default="dummy",
                    help="API key for --reader-base-url")
    args = ap.parse_args()

    if args.reader_model:
        _READER_OVERRIDE.update(model=args.reader_model,
                                base_url=args.reader_base_url, key=args.reader_key)
        print(f"(reader override: {args.reader_model} @ {args.reader_base_url})")

    q = _load_q(args.qid)
    print(f"Q: {q['question']}")
    print(f"GOLD: {q['answer']}  TYPE: {q['question_type']}  SESSIONS: {len(q['haystack_sessions'])}")

    prompt_tmpl = READER_PROMPT
    if args.prompt:
        prompt_tmpl = open(args.prompt).read()
        print(f"(using alt prompt from {args.prompt}, {len(prompt_tmpl)} chars)")

    if args.reimprint:
        _reimprint(args.backend, q, args.reimprint, args.top_k,
                   args.char_cap, args.max_tokens, prompt_tmpl)
        return

    if args.grep:
        terms = [t.strip() for t in args.grep.split(",") if t.strip()]
        _grep(args.backend, args.qid, q["question"], terms, args.grep_k)
        return

    if args.sweep:
        ks = [int(x) for x in args.sweep.split(",") if x.strip()]
        for k in ks:
            _one(args.backend, q, k, args.char_cap, args.max_tokens,
                 prompt_tmpl, show_facts=False, uid_override=args.user_id)
        return

    _one(args.backend, q, args.top_k, args.char_cap, args.max_tokens,
         prompt_tmpl, show_facts=args.show_facts, uid_override=args.user_id)


if __name__ == "__main__":
    main()
