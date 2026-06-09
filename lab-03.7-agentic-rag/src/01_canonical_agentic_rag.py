"""LangChain canonical Agentic RAG, adapted to local oMLX + Week 1 Qdrant collection."""
import os
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.tools import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition

# Local oMLX endpoint (sonnet tier — Gemma 26B)
llm = ChatOpenAI(
    model=os.getenv("MODEL_SONNET", "gemma-4-26B-A4B-it-heretic-4bit"),
    base_url=os.getenv("OMLX_BASE_URL", "http://127.0.0.1:8000/v1"),
    api_key=os.getenv("OMLX_API_KEY", "Shane@7162"),
    temperature=0.0,
)

# Week 1 Qdrant collection (already populated with bge-m3 embeddings)
client = QdrantClient(url="http://127.0.0.1:6333")
# NOTE: vector store needs an embedding function compatible with what was indexed.
# Reuse your Week 1 BGE-M3 wrapper here.
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name=os.path.expanduser("~/models/bge-m3"),
    model_kwargs={"device": "mps"},
)
vectorstore = QdrantVectorStore(
    client=client, collection_name="bge_m3_hnsw", embedding=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
retriever_tool = create_retriever_tool(
    retriever, name="search_corpus", description="Search the corpus for documents relevant to the query."
)

# Build the 5-node graph
# (Following the LangChain official example structure; abridged here — see notebook for full nodes.)
# ... generate_query_or_respond, grade_documents, rewrite_question, generate_answer ...

graph = StateGraph(MessagesState)
# graph.add_node("generate_query_or_respond", ...)
# graph.add_node("retrieve", ToolNode([retriever_tool]))
# graph.add_node("grade_documents", ...)
# graph.add_node("rewrite_question", ...)
# graph.add_node("generate_answer", ...)
# (Wire up edges per the canonical diagram — see official notebook)
app = graph.compile()

# Smoke test
result = app.invoke({"messages": [("user", "Your test query here")]})
print(result["messages"][-1].content)