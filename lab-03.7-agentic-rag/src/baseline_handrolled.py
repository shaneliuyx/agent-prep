"""Hand-rolled Self-RAG + CRAG pipeline (Phase 5 baseline for W3.7).

Ported from shaneliuyx/rag@dae7d6f (2025-08-17) graph/ subtree. The original
implemented this on Chroma + Ollama-Gemma2:2b; this port adapts to:

  - Qdrant (via shared/rag_hybrid for encoder + reranker)
  - oMLX OpenAI-compatible client (MODEL_SONNET / MODEL_HAIKU)
  - shared/rag_hybrid.DenseEncoder + CrossEncoderReranker (BGE-M3 + BGE-reranker-v2-m3)

Pipeline (single file, no LangGraph — that's Phase 1-4 of the lab):

    query
      → ComplexityDecider (heuristic)
      → optional LLMDecomposer (Phase 6) → topo-sort sub-queries
      → MultiRetrieve (dense kNN; original + keyword-only variants RRF-fused)
      → Rerank (BGE-reranker)
      → Synthesize (bullets with [#i] citations + drift filter)
      → SelfRAG checks (faithfulness + citation + coverage)
      → Grade hallucination + grade relevance
      → if not pass → CorrectiveRAG (rewrite + retry)
      → if still not pass → suggest web-search fallback
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "shared"))

from rag_hybrid import (  # noqa: E402
    BGE_M3, BGE_RERANKER_V2_M3,
    CrossEncoderReranker, DenseEncoder, autoconfig,
)

load_dotenv()
omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
MODEL = os.getenv("MODEL_SONNET") or ""

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "bge_m3_hnsw")


@dataclass
class Thresholds:
    """Self-RAG / Corrective thresholds (env-driven, ported from old config/settings.py)."""
    selfrag_conf: float = float(os.getenv("SELFRAG_CONF_THRESHOLD", "0.6"))
    relevance: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.2"))
    max_regen: int = int(os.getenv("MAX_REGEN_ATTEMPTS", "1"))
    max_rewrite: int = int(os.getenv("MAX_REWRITE_ATTEMPTS", "2"))


THRESHOLDS = Thresholds()

# Singletons — encoder + reranker load once
_qdrant = QdrantClient(url=QDRANT_URL, timeout=60)
_encoder = DenseEncoder(autoconfig.encoder_config_for(BGE_M3))
_reranker = CrossEncoderReranker(autoconfig.recommend(BGE_M3, BGE_RERANKER_V2_M3).reranker)

# Function words stripped before keyword-overlap scoring (faithfulness / coverage /
# drift filter) so overlap reflects content words, not grammatical glue. Shared so the
# three overlap checks agree on "what counts as a meaningful word". Kept deliberately
# small: overlap scoring should only drop the most content-free words, not question
# words / auxiliaries (those still carry signal when comparing answer vs. passage).
_STOPWORDS = frozenset({"the", "a", "an", "of", "to", "in"})

# Aggressive stop set for building keyword-only RETRIEVAL queries (_keyword_variant).
# Superset of _STOPWORDS: retrieval also drops question words / auxiliaries / pronouns
# so the keyword search keys on distinctive content terms. Different job from scoring,
# hence larger — expressed as `_STOPWORDS | {...}` so the shared core lives in one place.
_KEYWORD_STOPWORDS = _STOPWORDS | {
    "and", "or", "is", "are", "was", "were", "what", "who", "where", "when",
    "why", "how", "did", "do", "does", "with", "for", "on", "at", "by",
    "this", "that", "those", "these", "be", "been", "have", "has", "had",
    "i", "you", "he", "she", "it", "we", "they", "my", "your", "their",
}


# ---------- Node 1: ComplexityDecider ----------------------------------------

_COMPLEX_CUES = [
    "compare", "vs", "versus", "difference", "differences", "and",
    "explain why", "how do", "trade-off", "trade off", "step by step",
    "all", "list every", "each",
]


def decide_complexity(query: str) -> dict[str, Any]:
    """Heuristic classifier — capitalized-entity count + cue keywords + multi-step
    conjunctions. Returns {"label": "Simple" | "Complex", "score": float}."""
    q = query.strip()
    cap_count = len(re.findall(r"\b[A-Z][a-zA-Z]+\b", q))
    cue_hits = sum(1 for c in _COMPLEX_CUES if c in q.lower())
    score = 0.0
    score += min(cap_count, 4) * 0.15      # named entities up to ~0.6
    score += min(cue_hits, 3) * 0.25       # cue keywords up to 0.75
    score += 0.2 if (" and " in q.lower() or "," in q) else 0.0
    return {"label": "Complex" if score >= 0.5 else "Simple", "score": round(score, 3)}


# ---------- Node 2: MultiRetrieve (RRF over original + keyword-only) ---------

def _keyword_variant(query: str) -> str:
    """Strip stopwords / functional words to make a keyword-only variant."""
    return " ".join(w for w in re.findall(r"\w+", query.lower())
                    if w not in _KEYWORD_STOPWORDS)


def _retrieve_qdrant(query: str, k: int = 30) -> list[dict[str, Any]]:
    qv = _encoder.encode([query])[0]
    points = _qdrant.query_points(
        QDRANT_COLLECTION, query=qv.tolist(), limit=k, with_payload=True,
    ).points
    return [
        {"id": str(p.id),
         "text": (p.payload or {}).get("text", ""),
         "payload": p.payload or {}}
        for p in points
    ]


def multi_retrieve(query: str, k: int = 30, rrf_k: int = 60) -> list[dict[str, Any]]:
    """Original-query + keyword-only-query, fuse via RRF (Cormack 1/(k+rank))."""
    variants = [query]
    kw = _keyword_variant(query)
    if kw and kw != query.lower():
        variants.append(kw)
    rrf: dict[str, dict[str, Any]] = {}
    for v in variants:
        hits = _retrieve_qdrant(v, k=k)
        for rank, h in enumerate(hits, start=1):
            cid = h["id"]
            score = 1.0 / (rrf_k + rank)
            if cid in rrf:
                rrf[cid]["_rrf_score"] += score
            else:
                rrf[cid] = {**h, "_rrf_score": score}
    fused = sorted(rrf.values(), key=lambda x: -x["_rrf_score"])
    return fused[:k]


# ---------- Node 3: Rerank ---------------------------------------------------

def rerank(query: str, hits: list[dict[str, Any]], top_k: int = 6) -> list[dict[str, Any]]:
    if not hits:
        return []
    _reranker._ensure_loaded()
    pairs = [(query, h["text"]) for h in hits]
    scores = _reranker._model.predict(pairs, batch_size=_reranker.cfg.spec.batch_size)
    reranked = [h for h, _ in sorted(zip(hits, scores), key=lambda x: -x[1])]
    return reranked[:top_k]


# ---------- Node 4: Synthesize (with [#i] citations + drift filter) ---------

SYNTHESIZE_PROMPT = """Use ONLY the passages below. Answer in 3-5 bullets.
Each bullet MUST include a citation in the form [#N] where N is the passage
number. Do not invent claims. If the passages do not contain the answer, reply
with EXACTLY this and nothing else: I don't know

Passages:
{passages}

Question: {q}

Answer (bullets with [#N] citations, or "I don't know"):"""


def synthesize(query: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    if not hits:
        return {"answer": "I don't know", "drift_filtered": 0}
    passages = "\n\n".join(f"[#{i+1}] {h['text']}" for i, h in enumerate(hits))
    resp = omlx.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=400,
        messages=[{"role": "user", "content": SYNTHESIZE_PROMPT.format(
            passages=passages[:8000], q=query)}],
    )
    text = (resp.choices[0].message.content or "").strip()
    bullets = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Drift filter: drop bullets that share zero keywords with their cited passage
    kept, dropped = [], 0
    for b in bullets:
        m = re.search(r"\[#(\d+)\]", b)
        if not m:
            kept.append(b)
            continue
        n = int(m.group(1)) - 1
        if 0 <= n < len(hits):
            bullet_kws = set(re.findall(r"\w+", b.lower())) - _STOPWORDS
            psg_kws = set(re.findall(r"\w+", hits[n]["text"].lower()))
            if bullet_kws & psg_kws:
                kept.append(b)
            else:
                dropped += 1
        else:
            kept.append(b)
    return {"answer": "\n".join(kept), "drift_filtered": dropped}


# ---------- Node 5: SelfRAG checks (faithfulness/citation/coverage) ---------

def selfrag_checks(answer: str, hits: list[dict[str, Any]], query: str) -> dict[str, Any]:
    bullets = [ln for ln in answer.splitlines() if ln.strip()]
    n_total = len(bullets) or 1
    n_cited = sum(1 for b in bullets if re.search(r"\[#\d+\]", b))
    citation_rate = n_cited / n_total
    # Faithfulness: each bullet's keywords overlap its cited passage
    n_faithful = 0
    for b in bullets:
        m = re.search(r"\[#(\d+)\]", b)
        if not m:
            continue
        n = int(m.group(1)) - 1
        if 0 <= n < len(hits):
            bk = set(re.findall(r"\w+", b.lower())) - _STOPWORDS
            pk = set(re.findall(r"\w+", hits[n]["text"].lower())) - _STOPWORDS
            if len(bk & pk) >= 3:
                n_faithful += 1
    faithfulness_rate = n_faithful / n_total if n_cited else 0.0
    # Coverage: query keywords represented in answer
    qk = set(re.findall(r"\w+", query.lower())) - _STOPWORDS
    ak = set(re.findall(r"\w+", answer.lower()))
    coverage = len(qk & ak) / max(len(qk), 1)
    return {
        "citation_rate": round(citation_rate, 3),
        "faithfulness_rate": round(faithfulness_rate, 3),
        "coverage": round(coverage, 3),
        "confidence": round((citation_rate + faithfulness_rate + coverage) / 3, 3),
    }


# ---------- Node 6/7: Grade hallucination + grade relevance ----------------

def grade_hallucination(answer: str, hits: list[dict[str, Any]],
                        query: str) -> dict[str, Any]:
    sr = selfrag_checks(answer, hits, query)
    ok = sr["faithfulness_rate"] >= 0.5 and sr["citation_rate"] >= 0.5
    return {"pass": ok, "selfrag": sr}


GRADE_RELEVANCE_PROMPT = """Does the ANSWER actually answer the QUESTION with specific information,
or does it decline / say the context is insufficient / fail to provide the answer?
Reply with ONE word: YES (it answers) or NO (it does not).

Question: {q}
Answer: {a}

Verdict (YES/NO):"""


def grade_relevance(answer: str, query: str) -> dict[str, Any]:
    """LLM judge: did we actually ANSWER the question, or abstain?

    The old keyword-overlap version was fooled by abstentions that echo the question
    ("the passages do not contain information regarding <restates query>...") - overlap was
    ~1.0 so it scored 'relevant' and the corrective loop never fired (§3.3). An LLM detects
    'insufficient context' even when it restates the query, so the loop fires when it should -
    matching crag_variant.py's LLM evaluator (apples-to-apples)."""
    # Canonical abstention → not answered, deterministically (no LLM call, no Gemma flip-flop).
    # synthesize() now emits exactly "I don't know" when the passages lack the answer.
    if "i don't know" in answer.lower() or "i do not know" in answer.lower():
        return {"pass": False, "verdict": "abstain"}
    verdict = (omlx.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=5,
        messages=[{"role": "user", "content": GRADE_RELEVANCE_PROMPT.format(q=query, a=answer)}],
    ).choices[0].message.content or "").strip().lower()
    return {"pass": verdict.startswith("y"), "verdict": verdict[:12]}


# ---------- Node 8: CorrectiveRAG (rewrite + retry) -------------------------

REWRITE_PROMPT = """The previous retrieval did not surface the right context.
Rewrite the user's query with synonyms and alternate phrasings to improve recall.
Keep it under 30 words. Return ONLY the rewritten query, no preamble.

Original query: {q}
Rewritten query:"""


def rewrite_query(query: str) -> str:
    resp = omlx.chat.completions.create(
        model=MODEL, temperature=0.3, max_tokens=80,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(q=query)}],
    )
    return (resp.choices[0].message.content or query).strip().split("\n")[0]


def corrective_loop(query: str, top_k: int = 6,
                    max_iters: int = THRESHOLDS.max_rewrite) -> dict[str, Any]:
    current_q = query
    last_out: dict[str, Any] = {}
    for i in range(max_iters):
        rewritten = rewrite_query(current_q) if i > 0 else current_q
        hits = multi_retrieve(rewritten)
        rr = rerank(rewritten, hits, top_k=top_k)
        sy = synthesize(rewritten, rr)
        sr = selfrag_checks(sy["answer"], rr, rewritten)
        gr = grade_relevance(sy["answer"], rewritten)
        last_out = {"hits": rr, "answer": sy["answer"], "selfrag": sr,
                    "grade_relevance": gr, "rewritten_query": rewritten,
                    "iteration": i}
        if gr["pass"] and sr["confidence"] >= THRESHOLDS.selfrag_conf:
            break
        current_q = rewritten
    return last_out


# ---------- Pipeline orchestration -----------------------------------------

def web_search(query: str, k: int = 4) -> list[str]:
    """Real web search for the CRAG fallback. Tavily if TAVILY_API_KEY is set, else DuckDuckGo
    (free, no key), else a clear error. Mirrors crag_variant.py so the two CRAGs are comparable."""
    if os.getenv("TAVILY_API_KEY"):
        from langchain_community.tools.tavily_search import TavilySearchResults
        return [r["content"] for r in TavilySearchResults(max_results=k).invoke(query)]
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    with DDGS() as ddg:
        return [r["body"] for r in ddg.text(query, max_results=k) if r.get("body")]


def _n_bullets(answer_text: str) -> int:
    """Count non-empty lines (bullets) in a synthesized answer."""
    return len([ln for ln in answer_text.splitlines() if ln.strip()])


def answer(query: str, top_k: int = 6) -> dict[str, Any]:
    """Top-level entry. Returns the full pipeline output.

    `out["steps"]` is a step-by-step trace: each pipeline stage in execution order
    with a compact result summary, so the decision path (corpus vs. corrective vs.
    web) is inspectable without re-running. Summaries stay compact (counts / PASS-FAIL
    / scores) — the full objects (hits, selfrag, grades) already live in `out`.
    `steps` is the same list referenced by `out["steps"]`, so appends in the
    conditional CRAG / web branch below still show up in the returned dict.
    """
    steps: list[dict[str, Any]] = []

    decision = decide_complexity(query)
    steps.append({"step": "decide_complexity",
                  "result": f"{decision['label']} (score={decision['score']})"})

    sub_queries = [{"id": "q1", "text": query}]
    decompose_log = "skipped (Simple OR ENABLE_DECOMPOSITION=0)"
    if os.getenv("ENABLE_DECOMPOSITION", "0") == "1" and decision["label"] == "Complex":
        try:
            from decompose import decompose_query  # type: ignore
            plan = decompose_query(query)
            if plan:
                sub_queries = plan
                decompose_log = f"used LLM decomposition: {len(plan)} sub-queries"
        except Exception as e:  # noqa: BLE001
            decompose_log = f"decompose import failed: {type(e).__name__}: {e}"
    steps.append({"step": "decompose",
                  "result": decompose_log, "sub_queries": len(sub_queries)})

    # Run pipeline
    hits = multi_retrieve(query)
    steps.append({"step": "multi_retrieve",
                  "result": f"{len(hits)} candidates after RRF fusion"})

    rr = rerank(query, hits, top_k=top_k)
    steps.append({"step": "rerank",
                  "result": f"top-{len(rr)} kept", "top_ids": [h["id"] for h in rr]})

    sy = synthesize(query, rr)
    steps.append({"step": "synthesize",
                  "result": f"{_n_bullets(sy['answer'])} bullets, "
                            f"{sy.get('drift_filtered', 0)} drift-filtered"})

    sr = selfrag_checks(sy["answer"], rr, query)
    steps.append({"step": "selfrag_checks", "result": sr})

    gh = grade_hallucination(sy["answer"], rr, query)
    steps.append({"step": "grade_hallucination",
                  "result": f"{'PASS' if gh['pass'] else 'FAIL'} "
                            f"(faithful={gh['selfrag']['faithfulness_rate']}, "
                            f"cited={gh['selfrag']['citation_rate']})"})

    gr = grade_relevance(sy["answer"], query)
    steps.append({"step": "grade_relevance",
                  "result": f"{'PASS' if gr['pass'] else 'FAIL'} "
                            f"(verdict={gr.get('verdict', '?')})"})

    out: dict[str, Any] = {
        "query": query, "decision": decision, "sub_queries": sub_queries,
        "decompose_log": decompose_log, "hits": rr, "answer": sy["answer"],
        "selfrag": sr, "grade_hallucination": gh, "grade_relevance": gr,
        "drift_filtered": sy.get("drift_filtered", 0), "steps": steps,
    }

    # CRAG loop if grade_relevance fails
    if not gr["pass"]:
        crag = corrective_loop(query, top_k=top_k)
        out["corrective"] = crag
        crag_pass = crag["grade_relevance"]["pass"]
        steps.append({"step": "corrective_loop",
                      "result": f"{'PASS' if crag_pass else 'FAIL'}"})
        if not crag_pass:
            # REAL web fallback: search the web, then synthesize the answer from the results
            # (was a SUGGESTION; now executed inline so the pipeline actually answers - like CRAG).
            try:
                web_docs = web_search(query)
            except Exception as e:  # noqa: BLE001
                web_docs, out["web_error"] = [], f"{type(e).__name__}: {e}"
            web_hits = [{"id": f"web{i}", "text": d, "payload": {}} for i, d in enumerate(web_docs)]
            wsy = synthesize(query, web_hits)
            out["answer"], out["source"] = wsy["answer"], "web"
            out["web_docs"], out["next_action"] = web_docs, {"type": "web_search", "executed": True}
            steps.append({"step": "web_fallback",
                          "result": f"{len(web_docs)} web docs → re-synthesized "
                                    f"({_n_bullets(wsy['answer'])} bullets)"})

    out.setdefault("source", "corpus")
    steps.append({"step": "finalize", "result": f"source={out['source']}"})
    return out


# 10 post-cutoff questions the 2018 MS-MARCO corpus cannot answer (mirrors src/03_crag_eval.py)
OUT_OF_CORPUS = [
    "What is the most capable Claude model Anthropic released in 2026?",
    "Which team won the 2025 NBA Finals?",
    "What is the official release date of GPT-5?",
    "Who won the 2025 Nobel Prize in Physics?",
    "What AI regulation did the European Union pass in 2025?",
    "What is the newest Apple silicon chip announced in 2025?",
    "What is the latest stable version of Python released in 2026?",
    "Who is the CEO of OpenAI as of 2026?",
    "What was the headline feature of the iPhone announced in 2025?",
    "Which country hosted the 2025 G20 summit?",
]


def run_out_of_corpus() -> None:
    """Run the hand-rolled Self-RAG + CRAG on out-of-corpus queries. Demonstrates that the
    corrective loop FIRES, TERMINATES (bounded `for range(max_rewrite)`, no runaway - contrast
    the LangGraph structural arm that hit GraphRecursionError, §3.2), and escalates to a
    web fallback - which this pipeline now EXECUTES inline (Tavily/DuckDuckGo) and synthesizes a
    real answer from, like crag_variant.py."""
    print("=== hand-rolled Self-RAG + CRAG on 10 out-of-corpus questions (REAL web fallback) ===")
    corr_fired = web_exec = answered = 0
    abstained = lambda a: "i don't know" in (a or "").lower()
    for q in OUT_OF_CORPUS:
        out = answer(q)
        corr = out.get("corrective")               # corrective_loop ran iff grade_relevance FAILED
        n_rewrites = corr["iteration"] if corr else 0   # actual rewrite_query calls (the i>0 passes)
        src = out.get("source", "corpus")
        ans_ok = src == "web" and not abstained(out["answer"])
        corr_fired += bool(corr)
        web_exec += bool(out.get("next_action", {}).get("executed"))
        answered += ans_ok
        print(f"- {q[:46]:46} | grade:{str(out['grade_relevance'].get('verdict', '?')):7} "
              f"| corr:{'y' if corr else 'n'} rw:{n_rewrites} | source:{src:6} "
              f"| {'ANSWERED' if ans_ok else 'abstain '}")
        if src == "web":
            print(f"    web answer: {out['answer'][:118].replace(chr(10), ' ')}")
    print(f"\nout-of-corpus (10 q): corrective fired {corr_fired}/10 | web search EXECUTED {web_exec}/10 "
          f"| answered via web {answered}/10")


if __name__ == "__main__":
    if "--out-of-corpus" in sys.argv or "-b" in sys.argv:
        run_out_of_corpus()
    else:
        q = " ".join(sys.argv[1:]) or "What did Buffett write about non-controlled businesses in 2023?"
        print(json.dumps(answer(q), indent=2, default=str))
