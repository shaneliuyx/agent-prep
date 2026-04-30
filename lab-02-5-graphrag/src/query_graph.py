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


def _lucene_tokens(seed: str) -> list[str]:
    """Tokenize + clean a seed into Lucene-safe tokens (lowercased, ≥3 chars).

    Single source of truth for both phrase and OR query builders. Three steps:
    (1) Lucene-reserved-char stripping prevents `+`, `:`, `~` etc. from being
        interpreted as operators.
    (2) Lowercase — Neo4j's full-text index uses StandardAnalyzer which
        lowercases at index time. Phrase queries with capitals (e.g.
        "Jack Dorsey") won't match the lowercased index entries (`jack dorsey`)
        even though the underlying entity name has capitals; running
        `node.name` through StandardAnalyzer at index time strips the case.
    (3) The 3-char minimum drops stop-word-ish fragments that Lucene's
        StandardAnalyzer would drop anyway."""
    cleaned = _LUCENE_RESERVED.sub(" ", seed).lower()
    return [t for t in cleaned.split() if len(t) >= 3]


def _lucene_phrase_query(seed: str) -> str:
    """Lucene required-AND query — every token must be present in the entity.

    Originally implemented as a quoted phrase query (`"jack dorsey"`) but
    that requires *adjacency*, which breaks when the indexed entity name
    has tokens in between (e.g. the canonical name is "Jack Patrick Dorsey"
    — phrase `"jack dorsey"` returns 0). The `+token` syntax requires every
    token to be present without enforcing adjacency, which gives the same
    precision benefit as a phrase (entity must contain all named tokens)
    while accepting middle-name / title / suffix variations.

    Multi-word seeds like "Mark Zuckerberg" still phrase-miss because no
    entity in the corpus contains the token `zuckerberg` at all — exactly
    the precision contract we want."""
    tokens = _lucene_tokens(seed)
    if not tokens:
        return seed.strip()
    if len(tokens) == 1:
        return tokens[0]
    return " ".join(f"+{t}" for t in tokens)


def _lucene_or_query(seed: str) -> str:
    """OR-joined Lucene query — kept as the fallback for phrase misses.

    Trades precision for recall. Used when phrase query returns 0 nodes
    so the caller can decide whether to ground from a weak partial match
    or report the seed as ungrounded."""
    tokens = _lucene_tokens(seed)
    return " OR ".join(tokens) if tokens else seed.strip()


def _count_index_matches(session, lucene: str) -> int:
    """How many entity nodes the index returns for this Lucene query."""
    if not lucene:
        return 0
    row = session.run(
        'CALL db.index.fulltext.queryNodes("entity_names", $lucene) '
        "YIELD node RETURN count(node) AS n",
        lucene=lucene,
    ).single()
    return row["n"] if row else 0


def fetch_subgraph(seeds: list[str], max_hops: int = 2) -> tuple[list[dict], dict[str, dict]]:
    """Match seeds to graph nodes via full-text index, then walk n-hop neighbourhood.

    Phrase-first matching strategy:
    - Multi-word seed → run phrase query first ("mark zuckerberg"). If ≥1
      node matches, use those as traversal seeds — high-precision exact
      match. If 0 match, fall back to OR query (mark OR zuckerberg) so we
      still ground the seed somehow, but flag the strategy as 'or' so the
      caller knows the precision is weaker.
    - Single-word seed → OR query is the same as phrase query, so we just
      run the bare token.

    Returns (subgraph_edges, per_seed_diagnostics). The diagnostic dict per
    seed reports phrase_matches, or_matches, and which strategy was used.
    The precondition check in answer() consumes this to surface the
    corpus-mismatch case (both phrase AND or matches = 0) and the
    weak-match case (phrase=0 but or>0)."""
    subgraph: list[dict] = []
    matches_per_seed: dict[str, dict] = {}
    with driver.session() as session:
        for seed in seeds:
            phrase = _lucene_phrase_query(seed)
            or_form = _lucene_or_query(seed)
            phrase_n = _count_index_matches(session, phrase)
            or_n = _count_index_matches(session, or_form) if or_form != phrase else phrase_n
            if phrase_n > 0:
                lucene_used = phrase
                strategy = "phrase"
            elif or_n > 0:
                lucene_used = or_form
                strategy = "or"
            else:
                matches_per_seed[seed] = {
                    "phrase": phrase_n,
                    "or":     or_n,
                    "strategy": "none",
                }
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
                lucene=lucene_used,
            )
            edges = [dict(record) for record in result]
            subgraph.extend(edges)
            matches_per_seed[seed] = {
                "phrase":   phrase_n,
                "or":       or_n,
                "strategy": strategy,
            }
    return subgraph, matches_per_seed


def answer(query: str) -> dict:
    seeds = extract_seed_entities(query)
    subgraph, matches_per_seed = fetch_subgraph(seeds)

    # Two precondition surfaces:
    # (a) strategy == "none" → corpus does not contain any token of the seed.
    #     This is the corpus-mismatch case the v1 query silently traversed
    #     through false-positive substring matches; precondition warning makes
    #     it loud.
    # (b) strategy == "or" → phrase query missed but OR fallback grounded.
    #     Weaker match — typically means the named entity isn't in the
    #     corpus but some token of the name is. Warning fires too because
    #     downstream answers from this case have low precision.
    unmatched = [s for s, m in matches_per_seed.items() if m["strategy"] == "none"]
    weak_match = [s for s, m in matches_per_seed.items() if m["strategy"] == "or"]
    if unmatched:
        print(
            f"[WARN] No graph entity matched the following seed(s): {unmatched}. "
            f"Either the corpus does not cover this topic, or the entity is named "
            f"differently in the graph. Run a sanity check in Neo4j Browser:\n"
            f'  MATCH (n:Entity) WHERE toLower(n.name) CONTAINS "<topic>" RETURN n.name LIMIT 20',
            file=sys.stderr,
        )
    if weak_match:
        print(
            f"[WARN] Phrase query missed for seed(s): {weak_match}. Falling back to "
            f"OR-over-tokens, which may surface partial-name matches (e.g. 'Mark "
            f"Zuckerberg' falling back to entities tokenizing on 'Mark'). Answer "
            f"quality on these seeds is reduced; verify graph contains the named "
            f"entity before trusting the result.",
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
