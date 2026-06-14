"""CRAG (Corrective RAG, Yan et al. 2024) over the local corpus, exports `crag_app`.

Where the canonical loop *rewrites* when retrieved docs look irrelevant, CRAG handles the harder
case the §2.6 result pointed at: **the local corpus genuinely lacks the answer.** It adds a
retrieval *confidence evaluator* and a **web-search fallback**:

    START -> retrieve_corpus -> evaluate -> { Correct   -> generate            (corpus)
                                              Incorrect -> web_search -> gen   (web only)
                                              Ambiguous -> combine    -> gen   (corpus + web) }

Built on the structural RAG (§2.5.1): retrieval is a guaranteed edge, never an LLM tool-call.
The evaluator is an oMLX LLM judge scoring 0.0-1.0; two thresholds bucket it into Correct /
Ambiguous / Incorrect (the paper uses a T5 evaluator + thresholds - same shape, local model).

Web search is pluggable and degrades gracefully: Tavily if `TAVILY_API_KEY` is set, else
DuckDuckGo (free, no key), else a clear error. No key is required to run this lab.

Import:  from crag_variant import crag_app
Run:     crag_app.invoke({"question": "<q>"})  ->  {..., "answer": str, "source": "corpus|web|both"}
"""
from __future__ import annotations

import os
import re
import sys
from typing import Literal, Required, TypedDict

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/code/agent-prep/lab-03.7-agentic-rag/.env"))
sys.path.insert(0, os.path.expanduser("~/code/agent-prep/shared"))  # for rag_hybrid

from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import PromptTemplate  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from rag_hybrid import (  # noqa: E402
    BGE_M3, BGE_RERANKER_V2_M3, CrossEncoderReranker, DenseEncoder, autoconfig,
)
from web_toolkit import web_search_text as web_search  # noqa: E402  — merged infra (shared/web_toolkit)

# ── thresholds: score >= UPPER -> Correct; <= LOWER -> Incorrect; else Ambiguous ──
CONF_UPPER = float(os.getenv("CRAG_UPPER", "0.7"))
CONF_LOWER = float(os.getenv("CRAG_LOWER", "0.3"))

# ── LLM (oMLX local) ──
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("OMLX_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OMLX_API_KEY", "not-needed")
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("MODEL_SONNET", "gemma-4-26B-A4B-it-heretic-4bit")
os.environ.setdefault("OPENAI_API_KEY", LLM_API_KEY)


def _llm(**kw):
    return ChatOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=LLM_MODEL,
                      temperature=0, **kw)


# ── Corpus retriever: reuse the existing Qdrant collection (no re-index) ──
_qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"), timeout=60)
_encoder = DenseEncoder(autoconfig.encoder_config_for(BGE_M3))
_reranker = CrossEncoderReranker(autoconfig.recommend(BGE_M3, BGE_RERANKER_V2_M3).reranker)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "bge_m3_hnsw")


def retrieve_passages(query: str, k: int = 6, pool: int = 30) -> list[str]:
    qv = _encoder.encode([query])[0]
    pts = _qdrant.query_points(QDRANT_COLLECTION, query=qv.tolist(), limit=pool,
                               with_payload=True).points
    return [text for _doc_id, text, _score in _reranker.rerank(query, pts, top_k=k)]


# ── Web search: shared/web_search.py — SearXNG → Tavily → DuckDuckGo + on-disk cache. ──
# Imported above. Same backend + reproducibility cache as baseline_handrolled.py (one source of
# truth); importing it here also upgrades this file from Tavily/DDG-only to the SearXNG-first chain.


# ── CRAG state ──
class CRAGState(TypedDict, total=False):
    question: Required[str]
    corpus_docs: list[str]
    web_docs: list[str]
    score: float
    source: str          # "corpus" | "web" | "both"
    answer: str


# ── Nodes ──
def retrieve_corpus(state: CRAGState) -> dict:
    return {"corpus_docs": retrieve_passages(state["question"], k=6)}


def evaluate(state: CRAGState) -> dict:
    """CRAG retrieval evaluator: score 0.0-1.0 how well the corpus docs answer the question."""
    docs = state.get("corpus_docs") or []
    if not docs:
        return {"score": 0.0}
    prompt = PromptTemplate(
        template=("On a scale from 0.0 to 1.0, how well do these documents let you fully and "
                  "correctly answer the question? Reply with ONLY a number.\n\n"
                  "Documents:\n{context}\n\nQuestion: {question}\n\nScore:"),
        input_variables=["context", "question"])
    raw = (prompt | _llm() | StrOutputParser()).invoke(
        {"context": "\n\n".join(docs), "question": state["question"]})
    m = re.search(r"\d*\.?\d+", raw)
    score = max(0.0, min(1.0, float(m.group()))) if m else 0.0
    return {"score": score}


def decide(state: CRAGState) -> Literal["generate", "web", "combine"]:
    score = state.get("score", 0.0)
    if score >= CONF_UPPER:
        return "generate"        # Correct: corpus is enough
    if score <= CONF_LOWER:
        return "web"             # Incorrect: corpus failed -> web fallback
    return "combine"             # Ambiguous: corpus + web


def web_node(state: CRAGState) -> dict:
    return {"web_docs": web_search(state["question"], k=3), "source": "web"}


def combine_node(state: CRAGState) -> dict:
    return {"web_docs": web_search(state["question"], k=3), "source": "both"}


def generate(state: CRAGState) -> dict:
    docs = (state.get("corpus_docs") or []) if state.get("source") != "web" else []
    docs = docs + (state.get("web_docs") or [])
    src = state.get("source", "corpus")
    prompt = PromptTemplate(
        template=("Answer the question using only the context below. If the context does not "
                  "contain the answer, say you don't know. Use three sentences maximum.\n\n"
                  "Context:\n{context}\n\nQuestion: {question}\nAnswer:"),
        input_variables=["context", "question"])
    answer = (prompt | _llm() | StrOutputParser()).invoke(
        {"context": "\n\n".join(docs) or "(no documents)", "question": state["question"]})
    return {"answer": answer, "source": src}


# ── Graph ──
_g = StateGraph(CRAGState)
_g.add_node("retrieve_corpus", retrieve_corpus)
_g.add_node("evaluate", evaluate)
_g.add_node("web", web_node)
_g.add_node("combine", combine_node)
_g.add_node("generate", generate)
_g.add_edge(START, "retrieve_corpus")
_g.add_edge("retrieve_corpus", "evaluate")
_g.add_conditional_edges("evaluate", decide,
                         {"generate": "generate", "web": "web", "combine": "combine"})
_g.add_edge("web", "generate")
_g.add_edge("combine", "generate")
_g.add_edge("generate", END)

crag_app = _g.compile()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is the latest model in the Claude 4 family?"
    out = crag_app.invoke({"question": q})
    print(f"source={out.get('source')} score={out.get('score'):.2f}\n{out['answer']}")
