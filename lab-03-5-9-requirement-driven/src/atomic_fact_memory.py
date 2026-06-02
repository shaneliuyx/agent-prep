# src/atomic_fact_memory.py — Phase 4 1-tier atomic-fact backend (~120 LOC)
"""Per-message atomic-fact extraction → embed → Qdrant upsert.

Write-time primitive: each imprint() call runs ONE LLM call to extract
N atomic facts from the input, embeds each fact, upserts N Qdrant points.
Mimics Mem0's ADD-only architecture in shape (1-tier, per-message extraction,
no consolidation tier) but is homebrewed — full control of prompt + retrieval.

Read-time primitive: cosine top-k over atomic facts. No multi-signal fusion
yet; if the score gap to Mem0 is large, fusion (BM25 + entity match) is the
candidate Phase 4.5 follow-up.
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

ATOMIC_EXTRACT_PROMPT = """Extract atomic facts from this message. An atomic fact is ONE self-contained
proposition about the user, the assistant, an entity, a time, a preference,
or a state. Each fact must be answerable on its own without other facts.

Output JSON array of strings (one fact per string). Output ONLY the array.
If the message contains no atomic facts, output: []

EXAMPLES:
Input: "I love Python. I work at Acme as a senior engineer."
Output: ["User loves Python.", "User works at Acme.", "User is a senior engineer at Acme."]

Input: "Thanks, that's helpful."
Output: []

MESSAGE: {message}"""


def _parse_fact_array(raw: str) -> list[str]:
    """Robustly parse an LLM's JSON fact array. Models often wrap the array in a
    ```json fence or add prose; a bare json.loads then fails (the bug that made
    every non-gpt-oss model look 'broken'). Strip fences, else extract the first
    [...] array. Returns [] only on genuine non-array output."""
    import re

    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    candidates = [s]
    m = re.search(r"\[.*\]", s, re.S)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            facts = json.loads(c)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(facts, list):
            return [str(f).strip() for f in facts if str(f).strip()]
    return []


class AtomicFactMemory:
    """1-tier atomic-fact backend conforming to the lab's TieredMemory interface."""

    def __init__(self, user_id: str, agent_id: str = "lme-eval") -> None:
        self.user_id = user_id
        self.agent_id = agent_id
        self.collection = f"af_{user_id}"
        self._llm = OpenAI(
            base_url=os.getenv("OMLX_BASE_URL"),
            api_key=os.getenv("OMLX_API_KEY", "dummy"),
        )
        self._embed_model = os.getenv("MODEL_EMBED", "bge-m3-mlx-fp16")
        # Dedicated extraction model (benchmarked): gemma-4-26B-A4B gives the best
        # fact recall here and is ~5x faster than gpt-oss-20b. Separate from
        # MODEL_HAIKU so swapping the extraction model doesn't disturb mem0/reader.
        self._chat_model = os.getenv(
            "MODEL_EXTRACT",
            os.getenv("MODEL_HAIKU", "gemma-4-26B-A4B-it-heretic-4bit"),
        )
        self._qdrant = QdrantClient(host="localhost", port=6333)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        cols = {c.name for c in self._qdrant.get_collections().collections}
        if self.collection not in cols:
            self._qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )

    def _extract_facts(self, message: str) -> list[str]:
        """One LLM call → JSON array of atomic-fact strings."""
        resp = self._llm.chat.completions.create(
            model=self._chat_model,
            messages=[{"role": "user", "content": ATOMIC_EXTRACT_PROMPT.format(message=message)}],
            temperature=0.0,
            max_tokens=1200,  # avoid truncating the JSON array when a message is fact-dense
        )
        raw = (resp.choices[0].message.content or "").strip()
        return _parse_fact_array(raw)

    def _embed(self, text: str) -> list[float]:
        resp = self._llm.embeddings.create(model=self._embed_model, input=text)
        return list(resp.data[0].embedding)

    def imprint(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Extract atomic facts PER MESSAGE, embed each, upsert one Qdrant point
        per fact. Returns space-joined fact IDs.

        Per-message extraction (the chapter's design) is the recall fix: one
        extraction call on a whole ~12K-char session scroll yields ~2 facts and
        misses the needles; extracting from each message captures far more (the
        'pick up X' / 'return Y' mentions a count question depends on). The scroll
        arrives as one-message-per-line ('[USER] ...' / '[ASSISTANT] ...'); split
        on lines and extract from each non-trivial one."""
        messages = [ln.strip() for ln in content.splitlines() if len(ln.strip()) > 15]
        if not messages:
            messages = [content]
        points: list[PointStruct] = []
        ids: list[str] = []
        for msg in messages:
            for fact in self._extract_facts(msg):
                pid = str(uuid.uuid4())
                ids.append(pid)
                payload = {"content": fact, "user_id": self.user_id, "agent_id": self.agent_id}
                if metadata:
                    payload.update(metadata)
                points.append(PointStruct(id=pid, vector=self._embed(fact), payload=payload))
        if points:
            self._qdrant.upsert(collection_name=self.collection, points=points)
        return " ".join(ids)

    def query_context(
        self, query: str, k: int = 5, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        vector = self._embed(query)
        # qdrant-client >= 1.12 removed .search(); query_points() is the
        # replacement and returns a response object with a .points list.
        resp = self._qdrant.query_points(
            collection_name=self.collection,
            query=vector,
            limit=k,
            with_payload=True,
        )
        return [
            {"content": h.payload["content"], "score": h.score, "metadata": h.payload}
            for h in resp.points
        ]
