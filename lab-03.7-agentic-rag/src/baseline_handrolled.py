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
    stop = {"the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was",
            "were", "what", "who", "where", "when", "why", "how", "did", "do",
            "does", "with", "for", "on", "at", "by", "this", "that", "those",
            "these", "be", "been", "have", "has", "had", "i", "you", "he",
            "she", "it", "we", "they", "my", "your", "their"}
    return " ".join(w for w in re.findall(r"\w+", query.lower()) if w not in stop)


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
number. Do not invent claims; if context is insufficient, say so.

Passages:
{passages}

Question: {q}

Answer (bullets with [#N] citations):"""


def synthesize(query: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
    if not hits:
        return {"answer": "insufficient context", "drift_filtered": 0}
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
            bullet_kws = set(re.findall(r"\w+", b.lower())) - {"the", "a", "an", "of", "to", "in"}
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
            bk = set(re.findall(r"\w+", b.lower()))
            pk = set(re.findall(r"\w+", hits[n]["text"].lower()))
            if len(bk & pk) >= 3:
                n_faithful += 1
    faithfulness_rate = n_faithful / n_total if n_cited else 0.0
    # Coverage: query keywords represented in answer
    qk = set(re.findall(r"\w+", query.lower())) - {"the", "a", "an", "of"}
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


def grade_relevance(answer: str, query: str) -> dict[str, Any]:
    qk = set(re.findall(r"\w+", query.lower())) - {"the", "a", "an", "of", "to", "in"}
    ak = set(re.findall(r"\w+", answer.lower()))
    overlap = len(qk & ak) / max(len(qk), 1)
    ok = overlap >= THRESHOLDS.relevance or "summarize" in query.lower()
    return {"pass": ok, "overlap": round(overlap, 3)}


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

def answer(query: str, top_k: int = 6) -> dict[str, Any]:
    """Top-level entry. Returns the full pipeline output."""
    decision = decide_complexity(query)
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

    # Run pipeline
    hits = multi_retrieve(query)
    rr = rerank(query, hits, top_k=top_k)
    sy = synthesize(query, rr)
    sr = selfrag_checks(sy["answer"], rr, query)
    gh = grade_hallucination(sy["answer"], rr, query)
    gr = grade_relevance(sy["answer"], query)

    out: dict[str, Any] = {
        "query": query, "decision": decision, "sub_queries": sub_queries,
        "decompose_log": decompose_log, "hits": rr, "answer": sy["answer"],
        "selfrag": sr, "grade_hallucination": gh, "grade_relevance": gr,
        "drift_filtered": sy.get("drift_filtered", 0),
    }

    # CRAG loop if grade_relevance fails
    if not gr["pass"]:
        crag = corrective_loop(query, top_k=top_k)
        out["corrective"] = crag
        if not crag["grade_relevance"]["pass"]:
            out["next_action"] = {
                "type": "web_search",
                "query": query,
                "instructions": "Host can route to a web-search MCP tool, then "
                                "re-feed results to this pipeline as additional context.",
            }

    return out


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What did Buffett write about non-controlled businesses in 2023?"
    print(json.dumps(answer(q), indent=2, default=str))
