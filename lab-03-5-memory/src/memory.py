"""Two-store memory backend:
  - Qdrant holds episodic memories (verbatim-ish: 'user said they love cycling').
  - SQLite holds semantic facts (structured: key='hobby', value='cycling').
Both are written to; retrieval fetches from both and merges.

Contradiction handling: when we store a new semantic fact for a key that
already has a LIVE value, we archive the old row (archived=1) and write
a new one. We NEVER UPDATE IN PLACE — archival preserves audit trail."""
import os, json, sqlite3, uuid
from typing import Literal
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
omlx   = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"))
SQLITE = os.getenv("SQLITE_PATH")
MODEL  = os.getenv("MODEL_SONNET")
HAIKU  = os.getenv("MODEL_HAIKU")
COLLECTION = "user_memories"

# In-process BGE-M3 dense embedder. oMLX doesn't serve embedding
# models by default (only chat models), so we load BGE-M3 directly
# via sentence-transformers. MPS device on Apple Silicon for speed.
_embedder = SentenceTransformer("BAAI/bge-m3", device="mps")

# Bootstrap Qdrant collection (idempotent)
if not qdrant.collection_exists(COLLECTION):
    qdrant.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

# ── Extraction: turn a conversation turn into structured memories ────────────

EXTRACT_PROMPT = """Extract memories from this conversation turn.
Return JSON only: {"semantic": [{"key": str, "value": str}], "episodic": [str]}.

SEMANTIC — durable facts about the user. Structured. Examples:
  {"key": "location", "value": "Taipei"}
  {"key": "diet", "value": "vegan"}
  {"key": "job_role", "value": "cloud infrastructure engineer"}

EPISODIC — noteworthy events. One-sentence summaries. Examples:
  "user asked about setting up LangGraph for a customer-support agent"
  "user mentioned they are preparing for an agent-engineering interview"

Skip trivia. Do not invent facts. If nothing memorable, return empty lists."""


def embed(text: str) -> list[float]:
    vec = _embedder.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()


def extract_memories(user_msg: str, assistant_msg: str) -> dict:
    resp = omlx.chat.completions.create(
        model=HAIKU,   # extraction is cheap; run on haiku
        messages=[
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user",   "content": f"USER: {user_msg}\n\nASSISTANT: {assistant_msg}"},
        ],
        temperature=0.0, max_tokens=400,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return {"semantic": [], "episodic": []}


# ── Write path ───────────────────────────────────────────────────────────────

def write_semantic_fact(user_id: str, key: str, value: str) -> Literal["new", "updated", "unchanged"]:
    conn = sqlite3.connect(SQLITE)
    row = conn.execute(
        "SELECT id, value FROM user_facts WHERE user_id=? AND key=? AND archived=0",
        (user_id, key),
    ).fetchone()

    if row is None:
        conn.execute(
            "INSERT INTO user_facts (user_id, key, value) VALUES (?, ?, ?)",
            (user_id, key, value),
        )
        result = "new"
    elif row[1] == value:
        result = "unchanged"
    else:
        # Archive old, insert new — preserves audit trail
        conn.execute("UPDATE user_facts SET archived=1 WHERE id=?", (row[0],))
        conn.execute(
            "INSERT INTO user_facts (user_id, key, value) VALUES (?, ?, ?)",
            (user_id, key, value),
        )
        result = "updated"

    conn.commit(); conn.close()
    return result


def write_episodic(user_id: str, session_id: str, text: str) -> None:
    qdrant.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(
            id=str(uuid.uuid4()),
            vector=embed(text),
            payload={"user_id": user_id, "session_id": session_id, "text": text},
        )],
    )


def remember_turn(user_id: str, session_id: str, user_msg: str, assistant_msg: str) -> dict:
    mem = extract_memories(user_msg, assistant_msg)
    sem_results = [
        {"key": f["key"], "value": f["value"],
         "status": write_semantic_fact(user_id, f["key"], f["value"])}
        for f in mem.get("semantic", []) if f.get("key") and f.get("value")
    ]
    for ep in mem.get("episodic", []):
        if ep: write_episodic(user_id, session_id, ep)
    return {"semantic": sem_results, "episodic_count": len(mem.get("episodic", []))}


# ── Read path ────────────────────────────────────────────────────────────────

def recall(user_id: str, query: str, k: int = 5) -> dict:
    # Semantic: all live facts
    conn = sqlite3.connect(SQLITE)
    facts = conn.execute(
        "SELECT key, value FROM user_facts WHERE user_id=? AND archived=0",
        (user_id,),
    ).fetchall()
    conn.close()

    # Episodic: top-k by similarity
    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=embed(query),
        query_filter={"must": [{"key": "user_id", "match": {"value": user_id}}]},
        limit=k,
    ).points
    episodes = [h.payload["text"] for h in hits if h.score > 0.35]  # threshold prevents noise

    return {
        "semantic_facts": [{"key": k_, "value": v} for k_, v in facts],
        "relevant_episodes": episodes,
    }


def format_memory_block(memory: dict) -> str:
    if not memory["semantic_facts"] and not memory["relevant_episodes"]:
        return ""
    lines = ["Known facts about this user:"]
    for f in memory["semantic_facts"]:
        lines.append(f"- {f['key']}: {f['value']}")
    if memory["relevant_episodes"]:
        lines.append("\nRelevant past interactions:")
        for e in memory["relevant_episodes"]:
            lines.append(f"- {e}")
    return "\n".join(lines)
