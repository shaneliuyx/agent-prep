"""Answer-quality eval: compressed vs raw context, judged by a haiku-tier LLM.

For each query:
  1. Retrieve top-50 from Qdrant, rerank to top-5
  2. Build raw context = concat of 5 passages
  3. Build compressed context = compress(query, top-5)  [from 02_compress.py prompt]
  4. Generate answer_raw and answer_compressed via SONNET model from each context
  5. Pass both answers to HAIKU judge (with A/B randomized to mitigate position bias)
  6. Tally compressed_wins / ties / raw_wins

Decision threshold: compressed-context wins+ties ≥ 60% → compression is safe to ship.

Defensive scaffolding (same patterns as 02_compress.py):
  - max_tokens generous to handle reasoning models (6000 for answer, 500 for judge)
  - per-query try/except with failure tracking
  - None-content handling on all LLM calls
  - JSON parsing fallback for the judge response

Output: results/answer_eval.json with per-query winners + summary stats.
"""
import json, os, random, re, time
from pathlib import Path

from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from model_config import BGE_M3_HNSW, BGE_RERANKER_V2_M3

# === Config ===
SONNET = os.getenv("MODEL_SONNET", "gemma-4-26B-A4B-it-heretic-4bit")   # answer generator
HAIKU  = os.getenv("MODEL_HAIKU",  "gpt-oss-20b-MXFP4-Q8")              # judge (lighter, cheaper)
SAMPLE_QUERIES = 30   # per spec — fewer queries because each runs 3 LLM calls (compress + 2 answers + judge)

omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
random.seed(42)   # deterministic A/B assignment for reproducibility

# === Compression prompt (mirrors 02_compress.py) ===
COMPRESS_PROMPT = """You are a context compressor. Given a user query and 5 candidate passages, extract ONLY the sentences that directly answer the query. Preserve original wording for every extracted sentence. If a passage is irrelevant, drop it entirely. Output format:

[doc_id_1] sentence A. sentence B.
[doc_id_2] sentence C.
...

User query: {query}

Passages:
{passages}

Compressed context:"""

# === Answer generation prompt — grounded in retrieved context ===
ANSWER_PROMPT = """Using ONLY the context below, answer the query in 1-3 sentences. If the context doesn't contain the answer, reply exactly: insufficient context.

Context:
{ctx}

Query: {q}
Answer:"""

# === Judge prompt — A/B comparison with structured output ===
JUDGE_PROMPT = """You are a strict grader. Compare two answers to the same query and pick which is more faithful to the evidence and more useful. If they are equally good or both wrong, return "tie".
Return ONLY this JSON: {{"winner": "A", "reason": "<one sentence>"}} where winner is "A", "B", or "tie".

Query: {q}
Answer A: {a}
Answer B: {b}"""


# === LLM helpers — defensive against reasoning-model None content ===
def _llm_text(model, prompt, max_tokens):
    resp = omlx.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    msg = resp.choices[0].message
    content = msg.content
    if content is None:
        finish = resp.choices[0].finish_reason
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning and finish == "length":
            return None, f"reasoning_content truncated by max_tokens (finish_reason={finish})"
        return None, f"content=None (finish_reason={finish}, has_reasoning={bool(reasoning)})"
    return content.strip(), None


def compress(query, docs_with_ids):
    passages = "\n\n".join(f"[{did}] {text}" for did, text, _ in docs_with_ids)
    text, err = _llm_text(SONNET, COMPRESS_PROMPT.format(query=query, passages=passages), max_tokens=6000)
    return text or "", err


def answer(ctx, q):
    text, err = _llm_text(SONNET, ANSWER_PROMPT.format(ctx=ctx, q=q), max_tokens=6000)
    return text or "", err


def judge(q, a, b):
    """Returns (winner_label, reason_str, error). winner_label is 'A', 'B', 'tie', or None on parse failure."""
    # max_tokens=6000: gpt-oss-20b also uses reasoning_content if thinking is on at the model layer.
    # 500 was too small — empirically 2/30 judges hit finish_reason=length. 6000 gives headroom.
    text, err = _llm_text(HAIKU, JUDGE_PROMPT.format(q=q, a=a, b=b), max_tokens=6000)
    if err or text is None:
        return None, "", err or "empty content"
    # Try strict JSON first
    try:
        obj = json.loads(text)
        winner = obj.get("winner", "").lower()
        reason = obj.get("reason", "")
        if winner in {"a", "b", "tie"}:
            return winner, reason, None
    except json.JSONDecodeError:
        pass
    # Fallback: regex-extract winner field from prose
    m = re.search(r'"winner"\s*:\s*"(A|B|tie)"', text, re.IGNORECASE)
    if m:
        return m.group(1).lower(), text[:120], None
    # Last resort: look for "winner: X" or "answer A is better" patterns
    if re.search(r'\b(answer\s+a\s+is\s+(better|more)|winner.*A)\b', text, re.IGNORECASE):
        return "a", text[:120], None
    if re.search(r'\b(answer\s+b\s+is\s+(better|more)|winner.*B)\b', text, re.IGNORECASE):
        return "b", text[:120], None
    return None, text[:200], "could not parse judge output"


# === Retrieve + rerank (verbatim from 01_rerank.py / 02_compress.py) ===
C, M, R = BGE_M3_HNSW, BGE_M3_HNSW.model, BGE_RERANKER_V2_M3
qd = QdrantClient(url="http://127.0.0.1:6333")
encoder  = SentenceTransformer(M.path, device="mps", trust_remote_code=M.trust_remote_code)
reranker = CrossEncoder(R.path, device="mps")
reranker.model.half()


def retrieve(q, top_n=R.max_pairs_per_query):
    qv = encoder.encode([M.query_prefix + q], normalize_embeddings=True)[0]
    return qd.query_points(C.name, query=qv.tolist(), limit=top_n, with_payload=True).points


def rerank(q, cands, k=5):
    pairs = [(q, c.payload["text"]) for c in cands]
    scores = reranker.predict(pairs, batch_size=R.batch_size)
    order = sorted(zip(cands, scores), key=lambda x: -x[1])[:k]
    return [(c.payload["doc_id"], c.payload["text"], float(s)) for c, s in order]


# === Main eval loop ===
queries = json.loads(Path("data/queries.json").read_text())
sample_qids = list(queries.keys())[:SAMPLE_QUERIES]

print(f"Eval: {SAMPLE_QUERIES} queries × 3 LLM calls each (compress + 2 answers + judge)")
print(f"  answer model: {SONNET}")
print(f"  judge model:  {HAIKU}")

rows = []
failures = []
t0 = time.time()
for qid in sample_qids:
    q = queries[qid]
    try:
        # Stage 1: retrieve + rerank
        cands = retrieve(q)
        top5 = rerank(q, cands, 5)
        raw_ctx = "\n\n".join(f"[{did}] {text}" for did, text, _ in top5)

        # Stage 2: compress
        compressed_ctx, comp_err = compress(q, top5)
        if comp_err or not compressed_ctx:
            failures.append({"qid": qid, "stage": "compress", "error": comp_err or "empty"})
            continue

        # Stage 3: answer from each context
        ans_raw, raw_err = answer(raw_ctx, q)
        ans_compressed, comp_a_err = answer(compressed_ctx, q)
        if raw_err or comp_a_err or not ans_raw or not ans_compressed:
            failures.append({"qid": qid, "stage": "answer", "error": f"raw_err={raw_err}, comp_err={comp_a_err}"})
            continue

        # Stage 4: judge with randomized A/B assignment to mitigate position bias
        compressed_is_A = random.choice([True, False])
        ans_A = ans_compressed if compressed_is_A else ans_raw
        ans_B = ans_raw         if compressed_is_A else ans_compressed

        winner, reason, judge_err = judge(q, ans_A, ans_B)
        if judge_err or winner is None:
            failures.append({"qid": qid, "stage": "judge", "error": judge_err or "parse_failed", "raw": reason})
            continue

        # Map A/B back to compressed/raw
        if winner == "tie":
            outcome = "tie"
        elif (winner == "a" and compressed_is_A) or (winner == "b" and not compressed_is_A):
            outcome = "compressed_wins"
        else:
            outcome = "raw_wins"

        rows.append({
            "qid": qid,
            "query": q,
            "outcome": outcome,
            "compressed_was_A": compressed_is_A,
            "judge_reason": reason[:200],
            "ans_raw_words":        len(ans_raw.split()),
            "ans_compressed_words": len(ans_compressed.split()),
        })
    except Exception as e:
        print(f"  [ERROR] qid={qid}: {type(e).__name__}: {e}")
        failures.append({"qid": qid, "stage": "exception", "error": f"{type(e).__name__}: {e}"})
        continue

    n_done = len(rows) + len(failures)
    if n_done % 5 == 0:
        print(f"  {n_done}/{SAMPLE_QUERIES}  (ok={len(rows)}, failed={len(failures)})")

elapsed = time.time() - t0

# === Tally ===
n_ok = len(rows)
compressed_wins = sum(1 for r in rows if r["outcome"] == "compressed_wins")
raw_wins        = sum(1 for r in rows if r["outcome"] == "raw_wins")
ties            = sum(1 for r in rows if r["outcome"] == "tie")
wins_or_ties    = compressed_wins + ties

summary = {
    "n_attempted":    len(sample_qids),
    "n_ok":           n_ok,
    "n_failed":       len(failures),
    "compressed_wins": compressed_wins,
    "ties":           ties,
    "raw_wins":       raw_wins,
    "compressed_win_rate":         round(compressed_wins / max(n_ok, 1), 3),
    "compressed_win_or_tie_rate":  round(wins_or_ties     / max(n_ok, 1), 3),
    "raw_win_rate":                round(raw_wins        / max(n_ok, 1), 3),
    "verdict": (
        "compression_safe_to_ship" if (wins_or_ties / max(n_ok, 1)) >= 0.60
        else "compression_too_aggressive"
    ),
    "total_wall_sec": round(elapsed, 1),
    "per_query_sec":  round(elapsed / max(len(sample_qids), 1), 2),
    "models": {"answer": SONNET, "judge": HAIKU},
}

Path("results").mkdir(exist_ok=True)
Path("results/answer_eval.json").write_text(
    json.dumps({"summary": summary, "rows": rows, "failures": failures}, indent=2)
)
print("\n" + "=" * 60)
print(json.dumps(summary, indent=2))
if failures:
    print(f"\n{len(failures)} failures. First 3:")
    for f in failures[:3]:
        print(f"  qid={f['qid']} [{f['stage']}]: {f['error']}")
