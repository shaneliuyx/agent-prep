"""GraphRAG query: identify seed entities from the query, traverse
2-hop neighbourhood, feed the subgraph to the generator LLM."""
import os, json, re
from openai import OpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
MODEL  = os.getenv("MODEL_SONNET")
HAIKU  = os.getenv("MODEL_HAIKU")
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)


# Non-reasoning fallback: capitalized noun phrases as seeds when LLM extraction fails.
# Reasoning models (gpt-oss-*, qwen3-*) burn their full max_tokens budget on internal
# chain-of-thought before emitting visible content; on long reasoning paths content=None
# and seeds become []. Regex fallback catches "Steve Jobs", "Apple", "NeXT" etc.
_PROPER_NOUN = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")


def _regex_seed_fallback(query: str) -> list[str]:
    seeds = _PROPER_NOUN.findall(query)
    # Drop sentence-initial common words ("Which", "What", "How", "When", "Where", "Who")
    drop = {"Which", "What", "How", "When", "Where", "Who", "Why", "Tell", "List"}
    return [s for s in seeds if s not in drop][:5]


def extract_seed_entities(query: str) -> list[str]:
    """Pick 1-5 candidate entities from the query.

    Use MODEL_SONNET (gemma-4-26B, non-reasoning) for deterministic JSON output.
    MODEL_HAIKU (gpt-oss-20b) is a reasoning model whose chain-of-thought consumes
    max_tokens budget BEFORE visible content is emitted — same query, different
    runs return different results because reasoning length is stochastic even at
    temperature=0. Falls back to regex extraction on empty content.
    """
    resp = omlx.chat.completions.create(
        model=MODEL,  # non-reasoning; fast + deterministic for structured output
        messages=[
            {"role": "system", "content": "Extract 1-5 entities from the query as a JSON object {\"entities\": [...]}. Include any noun phrase a graph could store: people, places, products, organizations, movements, ideologies, events, concepts, time periods. Prefer specific surface forms over generic ones (e.g., 'anarchism' not 'movement'). If the query is generic (e.g. 'tell me about X'), extract X."},
            {"role": "user",   "content": query},
        ],
        temperature=0.0, max_tokens=400,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    if not content:
        finish_reason = resp.choices[0].finish_reason
        print(f"[WARN] LLM returned empty content (finish_reason={finish_reason}); using regex fallback")
        return _regex_seed_fallback(query)
    try:
        data = json.loads(content)
        seeds = data.get("entities", []) if isinstance(data, dict) else data
        return seeds if seeds else _regex_seed_fallback(query)
    except json.JSONDecodeError:
        return _regex_seed_fallback(query)


def fetch_subgraph(seeds: list[str], max_hops: int = 2) -> list[dict]:
    """Fuzzy-match seed names against graph entities, then walk n-hop neighbourhood.

    Splits each seed into words ≥4 chars and matches nodes containing ANY word.
    A seed like "early anarchist thinkers" → ["early", "anarchist", "thinkers"]
    matches "anarchist movement", "anarchist doctrines", etc. Min-length filter
    drops stop words ("the", "a", "in") that would match every entity.
    """
    subgraph = []
    with driver.session() as session:
        for seed in seeds:
            words = [w for w in seed.lower().split() if len(w) >= 4] or [seed.lower()]
            result = session.run(
                f"""
                MATCH (n:Entity)
                WHERE ANY(w IN $words WHERE toLower(n.name) CONTAINS w)
                WITH n LIMIT 5
                MATCH path = (n)-[*1..{max_hops}]-(m)
                WITH DISTINCT relationships(path) AS rels
                UNWIND rels AS r
                RETURN DISTINCT startNode(r).name AS s, r.raw_relation AS rel,
                                endNode(r).name AS o, r.source_title AS src
                LIMIT 50
                """,
                words=words,
            )
            subgraph.extend([dict(record) for record in result])
    return subgraph


def answer(query: str) -> dict:
    seeds    = extract_seed_entities(query)
    subgraph = fetch_subgraph(seeds)

    if not subgraph:
        return {"answer": "No relevant entities found in the graph.", "seeds": seeds, "edges_used": 0}

    context = "\n".join(
        f"- {t['s']} --[{t['rel']}]--> {t['o']}  (source: {t['src']})"
        for t in subgraph[:40]
    )
    resp = omlx.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Answer using ONLY the graph facts below. If the facts do not support an answer, say so. Cite source articles inline."},
            {"role": "user",   "content": f"Query: {query}\n\nGraph facts:\n{context}"},
        ],
        temperature=0.2, max_tokens=400,
    )
    return {
        "answer":     resp.choices[0].message.content,
        "seeds":      seeds,
        "edges_used": len(subgraph),
    }


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Which companies are related to Mark Zuckerberg?"
    print(json.dumps(answer(q), indent=2))