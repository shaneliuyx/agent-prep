"""STRUCTURAL RAG graph (exports `app`) - the §2.5.1 fix for the canonical agentic RAG.

Why this exists: the canonical graph (`canonical_agentic_rag.py`) lets the LLM decide
*whether* to retrieve via a tool call; on this local model (oMLX/gemma) it skipped retrieval
on ~30% of questions and answered from parametric memory - violating the RAG contract
("every answer grounded in our data"). tool_choice="required" did NOT fix it (oMLX ignores
tool_choice). So retrieval is enforced in the GRAPH TOPOLOGY instead:

    START -> retrieve (ALWAYS) -> grade -> { generate | rewrite -> retrieve }

No `agent` node, no `tools_condition`, no way for any node or model to skip the store. The
agent's freedom moves from *whether* to retrieve to *how* to query (the rewrite reformulation).

Local assets: oMLX LLM + reused Qdrant collection `bge_m3_hnsw` (BGE-M3 via shared/rag_hybrid
+ BGE-reranker). No OpenAI, no re-indexing.

Import:  from structural_rag import app
Run:     app.invoke({"messages": [("user", "<question>")]})
"""
from __future__ import annotations

import os
import sys
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/code/agent-prep/lab-03.7-agentic-rag/.env"))
sys.path.insert(0, os.path.expanduser("~/code/agent-prep/shared"))  # for rag_hybrid

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import PromptTemplate  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from rag_hybrid import (  # noqa: E402
    BGE_M3, BGE_RERANKER_V2_M3, CrossEncoderReranker, DenseEncoder, autoconfig,
)

# ── LLM (oMLX local) ──
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


def retrieve(state):
    # RAG contract: retrieval is a STRUCTURAL edge, not an LLM tool-call the model can skip.
    # Query = the latest message: the user question on the first pass, or the rewrite node's
    # reformulation after a corrective loop. Emitted as a ToolMessage so grade/generate (which
    # read messages[-1]) and the harness context-capture work unchanged.
    query = state["messages"][-1].content
    passages = retrieve_passages(query, k=6)
    content = "\n\n".join(passages) if passages else "No relevant documents found."
    return {"messages": [ToolMessage(content=content, tool_call_id="retrieve")]}


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


# ── Graph: a TRUE RAG - retrieval is structural, never skippable ──
#   START -> retrieve (always) -> grade -> { generate | rewrite -> retrieve }
_workflow = StateGraph(AgentState)
_workflow.add_node("retrieve", retrieve)
_workflow.add_node("rewrite", rewrite)
_workflow.add_node("generate", generate)
_workflow.add_edge(START, "retrieve")
_workflow.add_conditional_edges("retrieve", grade_documents)
_workflow.add_edge("generate", END)
_workflow.add_edge("rewrite", "retrieve")

app = _workflow.compile()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is the main topic of the indexed documents?"
    res = app.invoke({"messages": [("user", q)]})
    print(res["messages"][-1].content)
