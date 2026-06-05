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
import time
import uuid
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchAny, PointStruct, VectorParams,
)

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


def _qd_retry(fn, *args, **kwargs):
    """Retry a Qdrant client call on the transient `UnexpectedResponse` (an API
    hiccup under concurrent collection load). Covers ALL qdrant entry points
    (get_collections / create_collection / upsert / query_points) — the earlier
    upsert-only retry left collection-setup and query un-retried, and those
    crashed cells under the 14B run's load (measured: 75832dbd, gpt4_70e84552_abs)."""
    for attempt in range(3):
        try:
            return fn(*args, **kwargs)
        except Exception:  # noqa: BLE001 — transient Qdrant error; retry with backoff
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))


def _chunk_text(text: str, max_chars: int = 700) -> list[str]:
    """Split a turn into <=max_chars windows on paragraph/sentence boundaries.

    A turn is the role-tagged unit (correct provenance), but a LONG turn — e.g. a
    2473-char generated children's book the assistant produced — sent as ONE
    extraction call makes 7B return 0 facts, dropping buried details ('the
    Plesiosaur had a blue body'). Chunking restores per-detail extraction WITHOUT
    re-introducing the role mis-tag of raw line-splitting: every chunk of a turn
    inherits that turn's role. Short turns yield a single chunk."""
    import re
    chunks: list[str] = []
    buf = ""
    for part in re.split(r"\n+", text.strip()):
        sents = re.split(r"(?<=[.!?])\s+", part) if len(part) > max_chars else [part]
        for s in sents:
            s = s.strip()
            if not s:
                continue
            if buf and len(buf) + 1 + len(s) > max_chars:
                chunks.append(buf)
                buf = s
            else:
                buf = f"{buf} {s}".strip()
    if buf:
        chunks.append(buf)
    return chunks or [text]


class AtomicFactMemory:
    """1-tier atomic-fact backend conforming to the lab's TieredMemory interface."""

    def __init__(self, user_id: str, agent_id: str = "lme-eval") -> None:
        self.user_id = user_id
        self.agent_id = agent_id
        self.collection = f"af_{user_id}"
        # Extraction → LOCAL oMLX (unmetered, no VibeProxy cooldown under the
        # eval's high per-message call volume). Model = MODEL_EXTRACT (Coder-14B).
        self._llm = OpenAI(
            base_url=os.getenv("OMLX_BASE_URL"),
            api_key=os.getenv("OMLX_API_KEY", "dummy"),
        )
        # Embeddings → local oMLX (bge-m3); the LLM endpoint hosts no embed model.
        self._embedder = OpenAI(
            base_url=os.getenv("EMBED_BASE_URL", os.getenv("OMLX_BASE_URL")),
            api_key=os.getenv("EMBED_API_KEY", os.getenv("OMLX_API_KEY", "dummy")),
        )
        self._embed_model = os.getenv("MODEL_EMBED", "bge-m3-mlx-fp16")
        # Extraction model. MODEL_EXTRACT lets the extraction model be swapped
        # independently; defaults to the shared MODEL_HAIKU (one model for all
        # LLM roles via VibeProxy).
        self._chat_model = os.getenv(
            "MODEL_EXTRACT",
            os.getenv("MODEL_HAIKU", "claude-haiku-4-5-20251001"),
        )
        self._qdrant = QdrantClient(host="localhost", port=6333)
        # Monotonic per-fact insert sequence. We imprint sessions in chronological
        # order (driver date-sorts) and turns/chunks in dialogue order, so `seq`
        # increases with true time at BOTH granularities — unlike the per-session
        # [sN] tag, which is identical for every fact in a session (no intra-session
        # order). Persists across the per-session imprint() calls on this instance.
        self._seq = 0
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        cols = {c.name for c in _qd_retry(self._qdrant.get_collections).collections}
        if self.collection not in cols:
            _qd_retry(self._qdrant.create_collection,
                      collection_name=self.collection,
                      vectors_config=VectorParams(size=1024, distance=Distance.COSINE))

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
        resp = self._embedder.embeddings.create(model=self._embed_model, input=text)
        return list(resp.data[0].embedding)

    def imprint(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Extract atomic facts PER MESSAGE, embed each, upsert one Qdrant point
        per fact. Returns space-joined fact IDs.

        Per-message extraction (the chapter's design) is the recall fix: one
        extraction call on a whole ~12K-char session scroll yields ~2 facts and
        misses the needles; extracting from each message captures far more (the
        'pick up X' / 'return Y' mentions a count question depends on). The scroll
        arrives as one-message-per-line ('[USER] ...' / '[ASSISTANT] ...'); split
        on lines and extract from each non-trivial one.

        ROLE-AWARE (Component 2): extract from BOTH [USER] and [ASSISTANT] turns
        and TAG each fact's payload with its provenance `role`. The earlier
        user-turn-ONLY filter zeroed `single-session-assistant` (the answer lives
        in dropped assistant turns). We no longer DISCARD at write — that loses
        information irrecoverably (evidence-before-belief). Instead we tag role
        and let `query_context(roles=...)` filter at READ time per question
        provenance. The multi-session de-flooding that user-turn-only bought
        (assistant advice burying user-action facts: 783->88 facts on the
        clothing-count probe) is preserved by passing roles=["user"] at read for
        multi-session — NOT by dropping assistant facts at write. Untagged
        scrolls fall back to role 'user' so non-tagged callers still work."""
        # Parse TURNS, not lines. A turn's content can span multiple newlines
        # (multi-line assistant replies); a line WITHOUT a [USER]/[ASSISTANT]
        # prefix is a CONTINUATION of the current turn, not a new user message.
        # Splitting on raw lines + defaulting unprefixed lines to "user" mis-tags
        # all assistant continuation text as user (measured: 283 user / 5
        # assistant facts, assistant replies leaking into the user store →
        # roles=["user"] stops de-flooding → regression). Group continuation
        # lines into their turn and tag by the turn's real role; extract once per
        # turn (also ~halves the call count vs per-line).
        tagged: list[tuple[str, str]] = []
        cur_role: str | None = None
        cur_buf: list[str] = []

        def _flush() -> None:
            if cur_role and cur_buf:
                text = " ".join(cur_buf).strip()
                if len(text) > 15:
                    tagged.append((text, cur_role))

        for raw in content.splitlines():
            ln = raw.strip()
            if not ln:
                continue
            u = ln.upper()
            if u.startswith("[USER]") or u.startswith("[ASSISTANT]"):
                _flush()
                cur_role = "assistant" if u.startswith("[ASSISTANT]") else "user"
                cur_buf = [ln.split("]", 1)[1].strip()]  # drop the role prefix
            elif cur_role is not None:
                cur_buf.append(ln)  # continuation of the current turn
        _flush()
        if not tagged:  # untagged scroll (no role markers) → one user turn
            tagged = [(content.strip(), "user")]
        points: list[PointStruct] = []
        ids: list[str] = []
        for text, role in tagged:
            for chunk in _chunk_text(text):  # long turns → digestible windows
                for fact in self._extract_facts(chunk):
                    pid = str(uuid.uuid4())
                    ids.append(pid)
                    payload: dict[str, Any] = {"content": fact, "user_id": self.user_id,
                                               "agent_id": self.agent_id, "role": role}
                    if metadata:
                        payload.update(metadata)
                    payload["seq"] = self._seq   # after metadata.update so it can't be clobbered
                    self._seq += 1
                    points.append(PointStruct(id=pid, vector=self._embed(fact), payload=payload))
        if points:
            # Qdrant occasionally throws a transient UnexpectedResponse during
            # upsert (API hiccup under concurrent collection load). Unretried, it
            # crashes the whole cell (status='error', n_imprinted=None) and the
            # crash is scored as an incorrect answer — measured inverting the
            # ensemble↔atomic_fact ranking on the KU axis (ensemble crashed 2×,
            # atomic_fact 1×, on questions both backends actually answer right).
            # Mirror the 3× backoff the mem0 adapter already uses. ensemble fans
            # imprint to this method, so one fix hardens both backends.
            _qd_retry(self._qdrant.upsert, collection_name=self.collection, points=points)
        return " ".join(ids)

    def query_context(
        self, query: str, k: int = 5, roles: list[str] | None = None, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        """`roles` filters by provenance (Component 2): e.g. ["user"] for
        multi-session de-flooding, None for all roles (single-session-assistant /
        temporal-reasoning need assistant facts). Facts written before role
        tagging have no `role` payload and are dropped by a roles filter — re-
        imprint after enabling role-aware extraction."""
        vector = self._embed(query)
        qfilter = (
            Filter(must=[FieldCondition(key="role", match=MatchAny(any=list(roles)))])
            if roles else None
        )
        # qdrant-client >= 1.12 removed .search(); query_points() is the
        # replacement and returns a response object with a .points list.
        resp = _qd_retry(
            self._qdrant.query_points,
            collection_name=self.collection,
            query=vector,
            limit=k,
            with_payload=True,
            query_filter=qfilter,
        )
        return [
            {"content": (h.payload or {}).get("content", ""),
             "score": h.score, "metadata": h.payload or {}}
            for h in resp.points
        ]
