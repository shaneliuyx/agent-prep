"""GraphRAG query: identify seed entities from the query, traverse
2-hop neighbourhood, feed the subgraph to the generator LLM.

Two upgrades over the v1 implementation:
1. Seed-to-entity matching uses Neo4j's full-text index (Lucene tokenisation +
   relevance scoring) instead of `CONTAINS` substring matching. Stops the
   metal/metalloid/Denmark false-positive class entirely.
2. Adds a precondition warning when no seed entity matches any graph node —
   the v1 silently let traversal proceed from false-positive matches and
   returned irrelevant edges. Now we surface the corpus-mismatch case so
   the user knows the corpus does not cover the topic, instead of seeing
   a confidently irrelevant LLM response."""
import json
import os
import re
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()
omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
MODEL = os.getenv("MODEL_SONNET")
HAIKU = os.getenv("MODEL_HAIKU")
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)


# Non-reasoning fallback: capitalized noun phrases as seeds when LLM extraction fails.
# Reasoning models (gpt-oss-*, qwen3-*) burn their full max_tokens budget on internal
# chain-of-thought before emitting visible content; on long reasoning paths content=None
# and seeds become []. Regex fallback catches "Steve Jobs", "Apple", "NeXT" etc.
_PROPER_NOUN = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")
_LUCENE_RESERVED = re.compile(r'[+\-!(){}\[\]^"~*?:\\/]')


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
        print(f"[WARN] LLM returned empty content (finish_reason={finish_reason}); using regex fallback", file=sys.stderr)
        return _regex_seed_fallback(query)
    try:
        data = json.loads(content)
        seeds = data.get("entities", []) if isinstance(data, dict) else data
        return seeds if seeds else _regex_seed_fallback(query)
    except json.JSONDecodeError:
        return _regex_seed_fallback(query)


def _lucene_query(seed: str) -> str:
    """Build a Lucene query string for the full-text index.

    Steps: escape Lucene-reserved characters, drop tokens shorter than 3 chars
    (Lucene's StandardAnalyzer would drop them anyway, but being explicit
    avoids surprising scoring behavior), join remaining tokens with OR.
    Trailing fuzzy operator `~` is NOT added — fuzzy on every term would
    re-introduce the same false-positive class CONTAINS had."""
    cleaned = _LUCENE_RESERVED.sub(" ", seed)
    tokens = [t for t in cleaned.split() if len(t) >= 3]
    return " OR ".join(tokens) if tokens else cleaned.strip()


def fetch_subgraph(seeds: list[str], max_hops: int = 2) -> tuple[list[dict], dict[str, int]]:
    """Match seeds to graph nodes via full-text index, then walk n-hop neighbourhood.

    Returns (subgraph_edges, per_seed_match_counts). The match-count dict is
    consumed by the precondition check in answer() so we can surface the
    corpus-mismatch case to the caller.
    """
    subgraph: list[dict] = []
    matches_per_seed: dict[str, int] = {}
    with driver.session() as session:
        for seed in seeds:
            lucene = _lucene_query(seed)
            if not lucene:
                matches_per_seed[seed] = 0
                continue
            # Two-stage: first find best-scored entity nodes via fulltext index,
            # then expand from those nodes via graph traversal. LIMIT 5 anchors
            # the expansion to a manageable starting set.
            result = session.run(
                f"""
                CALL db.index.fulltext.queryNodes("entity_names", $lucene)
                YIELD node, score
                WITH node, score ORDER BY score DESC LIMIT 5
                MATCH path = (node)-[*1..{max_hops}]-(m)
                WITH DISTINCT relationships(path) AS rels
                UNWIND rels AS r
                RETURN DISTINCT startNode(r).name AS s, r.raw_relation AS rel,
                                endNode(r).name AS o, r.source_title AS src
                LIMIT 50
                """,
                lucene=lucene,
            )
            edges = [dict(record) for record in result]
            subgraph.extend(edges)
            # Also count how many entity nodes the fulltext query matched, so
            # the caller can distinguish "no entities found" from "found but
            # no neighbourhood."
            count = session.run(
                'CALL db.index.fulltext.queryNodes("entity_names", $lucene) '
                "YIELD node RETURN count(node) AS n",
                lucene=lucene,
            ).single()
            matches_per_seed[seed] = count["n"] if count else 0
    return subgraph, matches_per_seed


def answer(query: str) -> dict:
    seeds = extract_seed_entities(query)
    subgraph, matches_per_seed = fetch_subgraph(seeds)

    # Precondition: warn loudly when none of the seeds matched any entity in
    # the graph. v1 silently traversed from false-positive substring matches
    # and returned irrelevant edges; this exposes the corpus-mismatch case.
    unmatched = [s for s, n in matches_per_seed.items() if n == 0]
    if unmatched:
        print(
            f"[WARN] No graph entity matched the following seed(s): {unmatched}. "
            f"Either the corpus does not cover this topic, or the entity is named "
            f"differently in the graph. Run a sanity check in Neo4j Browser:\n"
            f'  MATCH (n:Entity) WHERE toLower(n.name) CONTAINS "<topic>" RETURN n.name LIMIT 20',
            file=sys.stderr,
        )

    if not subgraph:
        return {
            "answer": "No relevant entities found in the graph.",
            "seeds": seeds,
            "matches_per_seed": matches_per_seed,
            "edges_used": 0,
        }

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
        "answer":           resp.choices[0].message.content,
        "seeds":            seeds,
        "matches_per_seed": matches_per_seed,
        "edges_used":       len(subgraph),
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Which companies are related to Mark Zuckerberg?"
    print(json.dumps(answer(q), indent=2))
