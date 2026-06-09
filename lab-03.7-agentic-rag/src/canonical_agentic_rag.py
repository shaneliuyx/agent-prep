"""CANONICAL Agentic RAG graph (exports `app`) - LangChain's 5-node reference pattern.

This is the Phase-1 canonical graph (§1.4) packaged as an importable module so other scripts
(e.g. 02_comparison_harness.py) can `from canonical_agentic_rag import app`. The notebook file
(01_canonical_agentic_rag.py) can't be imported - module names can't start with a digit.

    START -> agent -> tools_condition -> { retrieve -> grade -> {generate | rewrite}, END }

NOTE (see §2.5.1): the `agent` node lets the LLM decide *whether* to call retrieve_corpus.
On this local model (oMLX/gemma) it skips retrieval on ~30% of questions and answers from
parametric memory - which violates the RAG contract. `structural_rag.py` is the fixed variant
that makes retrieval a guaranteed structural edge. This file is kept as the canonical baseline
the comparison harness measures *against* the fix.

Local assets: oMLX LLM + reused Qdrant collection `bge_m3_hnsw` (BGE-M3 via shared/rag_hybrid
+ BGE-reranker). No OpenAI, no re-indexing.

Import:  from canonical_agentic_rag import app
Run:     app.invoke({"messages": [("user", "<question>")]})
"""
from __future__ import annotations

import os
import sys
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/code/agent-prep/lab-03.7-agentic-rag/.env"))
sys.path.insert(0, os.path.expanduser("~/code/agent-prep/shared"))  # for rag_hybrid

from langchain_core.messages import BaseMessage, HumanMessage  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import PromptTemplate  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode, tools_condition  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from rag_hybrid import (  # noqa: E402
    BGE_M3, BGE_RERANKER_V2_M3, CrossEncoderReranker, DenseEncoder, autoconfig,
)

# ── LLM (oMLX; env-switchable to VibeProxy->Claude for reliable tool-calling) ──
LLM_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("OMLX_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OMLX_API_KEY", "not-needed")
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("MODEL_SONNET", "gemma-4-26B-A4B-it-heretic-4bit")
os.environ.setdefault("OPENAI_API_KEY", LLM_API_KEY)


def _llm(**kw):
    return ChatOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=LLM_MODEL,
                      temperature=0, **kw)


# ── Retriever: reuse the existing Qdrant collection via rag_hybrid (no re-index) ──
_qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"), timeout=60)
_encoder = DenseEncoder(autoconfig.encoder_config_for(BGE_M3))
_reranker = CrossEncoderReranker(autoconfig.recommend(BGE_M3, BGE_RERANKER_V2_M3).reranker)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "bge_m3_hnsw")


def retrieve_passages(query: str, k: int = 6, pool: int = 30) -> list[str]:
    qv = _encoder.encode([query])[0]
    pts = _qdrant.query_points(QDRANT_COLLECTION, query=qv.tolist(), limit=pool,
                               with_payload=True).points
    return [text for _doc_id, text, _score in _reranker.rerank(query, pts, top_k=k)]


@tool
def retrieve_corpus(query: str) -> str:
    """Search the local corpus and return the most relevant passages."""
    passages = retrieve_passages(query, k=6)
    return "\n\n".join(passages) if passages else "No relevant documents found."


tools = [retrieve_corpus]


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def grade_documents(state) -> Literal["generate", "rewrite"]:
    question = state["messages"][0].content
    docs = state["messages"][-1].content
    prompt = PromptTemplate(
        template=("Reply with a single word, 'yes' or 'no': is the document relevant "
                  "to the question?\n\nDocument:\n{context}\n\nQuestion: {question}\n\n"
                  "Relevant (yes/no):"),
        input_variables=["context", "question"])
    out = (prompt | _llm() | StrOutputParser()).invoke(
        {"question": question, "context": docs}).strip().lower()
    return "generate" if (out.startswith("y") or "yes" in out[:6]) else "rewrite"


def agent(state):
    # Canonical pattern: the LLM decides WHETHER to retrieve (emit a retrieve_corpus call) or
    # answer directly. tools_condition then routes the no-tool-call case to END - the skip path
    # that produces ungrounded answers on this local model (see §2.5.1 / structural_rag.py).
    return {"messages": [_llm().bind_tools(tools).invoke(state["messages"])]}


def rewrite(state):
    question = state["messages"][0].content
    msg = [HumanMessage(content=(
        "Look at the input and reason about the underlying intent. "
        f"Initial question: {question}\nFormulate an improved question:"))]
    return {"messages": [_llm().invoke(msg)]}


def generate(state):
    question = state["messages"][0].content
    docs = state["messages"][-1].content
    prompt = PromptTemplate(
        template=("Answer using only the retrieved context. If you don't know, say so. "
                  "Use three sentences maximum.\n\nQuestion: {question}\n"
                  "Context: {context}\nAnswer:"),
        input_variables=["question", "context"])
    resp = (prompt | _llm() | StrOutputParser()).invoke({"context": docs, "question": question})
    return {"messages": [resp]}


# ── Graph: canonical 5-node agentic RAG (retrieval is OPTIONAL - the agent may skip it) ──
_workflow = StateGraph(AgentState)
_workflow.add_node("agent", agent)
_workflow.add_node("retrieve", ToolNode(tools))
_workflow.add_node("rewrite", rewrite)
_workflow.add_node("generate", generate)
_workflow.add_edge(START, "agent")
_workflow.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: END})
_workflow.add_conditional_edges("retrieve", grade_documents)
_workflow.add_edge("generate", END)
_workflow.add_edge("rewrite", "agent")

app = _workflow.compile()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is the main topic of the indexed documents?"
    res = app.invoke({"messages": [("user", q)]})
    print(res["messages"][-1].content)
