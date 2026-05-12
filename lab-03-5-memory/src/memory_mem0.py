"""mem0-backed memory shim — same API as src.memory, different backend.
Used in Phase 5's cross-check to compare hand-rolled vs library on
the same 15-Q benchmark.

mem0 v2.0.2 notes (verified 2026-05-12):
  - Memory() with no config defaults to OpenAI cloud — wrong for this lab.
    Use Memory.from_config({...}) with explicit provider + base_url so
    both LLM and embedder route to oMLX on localhost:8000.
  - search() filters changed: user_id is no longer a top-level kwarg;
    pass it inside filters={"user_id": ...} dict.
  - add() still takes user_id= directly.
"""
import os
from typing import Any

from dotenv import load_dotenv
from mem0 import Memory

load_dotenv()

_OMLX_BASE = os.getenv("OMLX_BASE_URL", "http://localhost:8000/v1")
_OMLX_KEY = os.getenv("OMLX_API_KEY", "")
_LLM_MODEL = os.getenv("MEM0_LLM_MODEL", "gpt-oss-20b-MXFP4-Q8")
_EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3-mlx-fp16")

# Explicit mem0 config — route LLM + embedder at oMLX, vector store at
# the lab's existing Qdrant container. Memory() with no config goes to
# OpenAI cloud, which fails with APIConnectionError under our offline
# local-first stack.
_CONFIG: dict[str, Any] = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": _LLM_MODEL,
            "openai_base_url": _OMLX_BASE,
            "api_key": _OMLX_KEY,
            "temperature": 0.0,
            "max_tokens": 400,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": _EMBED_MODEL,
            "openai_base_url": _OMLX_BASE,
            "api_key": _OMLX_KEY,
            "embedding_dims": 1024,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "mem0_memories",
            "embedding_model_dims": 1024,
        },
    },
}

_mem = Memory.from_config(_CONFIG)


def remember_turn(
    user_id: str, session_id: str, user_msg: str, assistant_msg: str
) -> dict[str, Any]:
    """Wrap mem0's add() to match the hand-rolled signature."""
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    result = _mem.add(messages=messages, user_id=user_id)
    return {"semantic": result.get("results", []), "episodic_count": 0}


def recall(user_id: str, query: str, k: int = 5) -> dict[str, Any]:
    """Wrap mem0's search() to match the hand-rolled signature.

    mem0 v2 API:
      - user_id moved into filters dict
      - top_k replaces limit
      - returns {'results': [...]} on hit, may return {} or {'results': None}
        on empty — defend with fallback chain
      - result items can be dicts or strings depending on mem0 version
    """
    r = _mem.search(
        query=query,
        filters={"user_id": user_id},
        top_k=k,
    ) or {}
    memories = r.get("results") or r.get("memories") or []
    # Defensive normalization: drop None entries, coerce strings to dicts
    normalized: list[dict[str, Any]] = []
    for m in memories:
        if m is None:
            continue
        if isinstance(m, str):
            normalized.append({"memory": m, "score": 0.0, "metadata": {}})
        elif isinstance(m, dict):
            normalized.append(m)
    # mem0 collapses episodic + semantic into one list; split heuristically
    # by score so the lab's downstream code (which expects semantic_facts
    # + relevant_episodes) still works.
    return {
        "semantic_facts": [
            {
                "key": (m.get("metadata") or {}).get("category", "fact"),
                "value": m.get("memory", ""),
            }
            for m in normalized
            if m.get("score", 0) > 0.5
        ],
        "relevant_episodes": [
            m.get("memory", "")
            for m in normalized
            if m.get("score", 0) <= 0.5
        ],
    }


def format_memory_block(memory: dict[str, Any]) -> str:
    """Same shape as src.memory.format_memory_block — keeps demos swappable."""
    if not memory.get("semantic_facts") and not memory.get("relevant_episodes"):
        return ""
    lines = ["Known facts about this user:"]
    for f in memory["semantic_facts"]:
        lines.append(f"- {f['key']}: {f['value']}")
    if memory["relevant_episodes"]:
        lines.append("\nRelevant past interactions:")
        for e in memory["relevant_episodes"]:
            lines.append(f"- {e}")
    return "\n".join(lines)
