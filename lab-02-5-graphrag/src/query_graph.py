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
from collections import defaultdict

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


def fetch_subgraph(seeds: list[str], max_hops: int = 5) -> tuple[list[dict], dict[str, dict]]:
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
            # Two-pass: 1-hop edges first (canonical direct neighbors that
            # the LLM needs for relational + factoid questions), then 2..N-hop
            # fill (for true multi-hop bridges). Concatenating ensures the
            # final 200-edge per-seed budget always surfaces 1-hop edges,
            # even on dense neighborhoods where 5-hop expansion produces
            # 10K+ candidate paths and would otherwise crowd them out.
            #
            # Bug history: a single `MATCH path = (node)-[*1..5]-(m) ... LIMIT 200`
            # without path-length ordering returns edges in Cypher's internal
            # path-traversal order, which prefers BFS-by-anchor expansion over
            # hop-distance ordering. On Microsoft (31 phrase matches → 5
            # anchors → 5-hop expansion), the canonical
            # `Microsoft -[CO_FOUNDED]- Bill Gates` 1-hop edge landed past
            # index 200 and never reached the LLM context — exactly the
            # symptom the v10 hybrid eval was diagnosing as "GraphRAG
            # collapsed to 0.27 ALL recall".
            edges: list[dict] = []
            r1 = session.run(
                """
                CALL db.index.fulltext.queryNodes("entity_names", $lucene)
                YIELD node, score
                WITH node ORDER BY score DESC LIMIT 5
                MATCH (node)-[r]-(m)
                RETURN DISTINCT startNode(r).name AS s, r.raw_relation AS rel,
                                endNode(r).name AS o, r.source_title AS src
                LIMIT 100
                """,
                lucene=lucene_used,
            )
            edges.extend(dict(record) for record in r1)
            if max_hops > 1:
                rn = session.run(
                    f"""
                    CALL db.index.fulltext.queryNodes("entity_names", $lucene)
                    YIELD node, score
                    WITH node ORDER BY score DESC LIMIT 5
                    MATCH path = (node)-[*2..{max_hops}]-(m)
                    WITH DISTINCT relationships(path) AS rels
                    UNWIND rels AS r
                    RETURN DISTINCT startNode(r).name AS s, r.raw_relation AS rel,
                                    endNode(r).name AS o, r.source_title AS src
                    LIMIT 100
                    """,
                    lucene=lucene_used,
                )
                edges.extend(dict(record) for record in rn)
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

    # Per-edge dedup with multi-source aggregation. Each unique directed
    # edge (subject, predicate, object) becomes one fact line; the sources
    # list collapses repeats of the SAME edge across multiple source articles.
    #
    # Why per-edge instead of pair-aggregation (the prior frozenset-keyed
    # bucket approach that regressed eval recall):
    # - **Direction matters.** "Apple --[acquired]--> NeXT" is not the same
    #   as "NeXT --[acquired by]--> Apple". The undirected frozenset key
    #   collapsed both into one bucket and lost subject-object orientation,
    #   which the LLM needs to phrase the answer correctly.
    # - **Per-edge source attribution.** Pair-aggregation listed all sources
    #   under one bucket regardless of which source mentioned which variant,
    #   so the LLM couldn't cite the right article for the right fact.
    # - **No silent edge truncation.** Pair-aggregation sliced
    #   `subgraph[:200]` BEFORE aggregating; on dense neighborhoods (e.g.
    #   Apple matches 24 entities, each walking a 2-hop neighborhood → 400+
    #   edges), the canonical `Apple Inc. --[acquired by]--> NeXT` could land
    #   past index 200 and never reach the LLM context.
    # Empirical: pair-aggregation regressed dense baseline 0.55 → 0.25 ALL
    # recall on the 32-Q eval (relational category collapsed entirely from
    # 0.75 → 0.00). Forward-fix uses per-edge format with no pre-slice.
    edge_groups: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for t in subgraph:
        s, o, rel, src = t.get("s"), t.get("o"), t.get("rel"), t.get("src")
        if not (s and o and rel):
            continue
        edge_groups[(s, rel, o)].add(src or "")

    # Format: cap at 300 unique edges (~10K tokens of context) for
    # Gemma-4-26B's effective window. 264813e used 200; 300 leaves room for
    # richer subgraphs while staying well under the model's prompt limit.
    # For sparse graphs the cap is a no-op.
    context_lines: list[str] = []
    for (s, rel, o), sources in list(edge_groups.items())[:300]:
        srcs = sorted(x for x in sources if x)
        if not srcs:
            src_text = ""
        elif len(srcs) == 1:
            src_text = f"  (source: {srcs[0]})"
        else:
            src_text = f"  (sources: {', '.join(srcs[:4])})"
        context_lines.append(f"- {s} --[{rel}]--> {o}{src_text}")
    context = "\n".join(context_lines)

    # Chain-of-thought + question-type-aware system prompt.
    #
    # Production GraphRAG prompts (Microsoft GraphRAG, LangChain GraphCypherQAChain,
    # Singh et al. 2025 survey) consistently outperform single-shot answer prompts
    # on aggregation/list questions when they:
    #   1. Force the LLM to enumerate matching facts before synthesizing prose.
    #   2. Branch behavior by question type (factoid / list / relational / unknown).
    #   3. Make the citation contract explicit (every claim ↦ a source article).
    #
    # The two-step pattern (extract matching facts as a list, then synthesize)
    # makes the list the load-bearing artifact — empirically lifts list /
    # aggregation recall without hurting factoid or relational answers.
    SYSTEM_PROMPT = """You are a fact synthesizer for a knowledge graph. Answer using ONLY the graph facts below.

GRAPH FACT FORMAT:
Each line is one directed edge: `Subject --[relation]--> Object  (source: ArticleTitle)`.
When the same edge is corroborated by multiple articles, sources are listed:
`Subject --[relation]--> Object  (sources: Article1, Article2)` — treat that
as multi-source evidence for ONE fact, not multiple facts.

Edge direction matters but the SAME relationship can be expressed either
direction in the graph. Examples (treat as the same fact):
  - "Apple --[acquired]--> NeXT" and "NeXT --[acquired by]--> Apple"
  - "Steve Jobs --[co-founded]--> Apple" and "Apple --[was co-founded by]--> Steve Jobs"

REQUIRED PROCESS:
1. **Identify the question type:**
   - LIST/ENUMERATION: "what companies", "which X", "list all", "who founded", "what universities".
   - RELATIONSHIP: "what is the relationship between X and Y", "how is X connected to Y".
   - FACTOID: "who is the CEO of X", "where is X based", "when did X happen".
2. **Extract matching facts.** Scan every graph fact line. For LIST questions, extract EVERY edge that matches the question's category — do not skip any. For RELATIONSHIP, find every edge connecting the two named entities directly OR through shared intermediate entities. For FACTOID, find the most-direct edge.
3. **Synthesize the answer.**
   - **LIST:** produce a bulleted or comma-separated list with one citation per item.
   - **RELATIONSHIP:** gather ALL edges between the named entities (in either direction) and CONSOLIDATE them into 1-3 sentences that capture the canonical relationship plus any supporting details. Don't just list each edge separately — synthesize. The strongest relation (e.g. "acquired") should lead; supporting relations (e.g. "senior employees joined") add color. Multiple edges between the same pair are evidence for ONE consolidated answer.
     Example for "What is the relationship between Apple and NeXT?":
       Edges in graph: `Apple --[ACQUIRED_BY]--> NeXT (Steve Jobs)`, `Apple --[CAME_TO_A_DEAL_WITH]--> NeXT (Steve Jobs)`, `Senior Apple employees --[JOINED]--> NeXT (Steve Jobs)`.
       Consolidated answer: "Apple acquired NeXT, and as part of the deal several senior Apple employees joined NeXT (source: Steve Jobs)."
   - **FACTOID:** 1-2 sentence direct answer.
4. **Cite every claim.** Format: "<fact> (source: <article>)". When multiple sources are listed for one edge, include them all: "(sources: A, B)". When consolidating multiple edges from the same source, cite that source once at the end: "<consolidated sentence> (source: A)".
5. **Refuse on absence.** If the graph facts do not contain the requested information, reply exactly: "The provided graph facts do not contain information about <topic>." Do NOT fabricate or infer beyond the facts."""
    resp = omlx.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Query: {query}\n\nGraph facts:\n{context}"},
        ],
        temperature=0.2, max_tokens=800,
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
