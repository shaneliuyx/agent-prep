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


_DECOMPOSE_SYSTEM = """You are a query planner for a knowledge-graph QA system.

Decide if a question is a MULTI-HOP BRIDGE question that requires:
  Step 1: identify an INTERMEDIATE set of entities (e.g. "founders of PayPal")
  Step 2: follow another edge type from those intermediates to find the answer
         (e.g. "what THEY later started")

If yes, output a 2-step decomposition plan as JSON. If no (single-hop, simple
relational, factoid, or out-of-domain), output {"plan": null}.

The plan format for a 2-step bridge is:
{
  "plan": {
    "step1": {"anchor": "<seed entity>", "edge_filter": "<verb regex pattern>", "yield_var": "<intermediate name>"},
    "step2": {"from_var": "<step1 yield_var>", "edge_filter": "<verb regex pattern>", "exclude_anchor": true|false, "yield_var": "<answer name>"}
  }
}

For an INTERSECTION (must satisfy two filters):
{
  "plan": {
    "type": "intersection",
    "step1a": {"anchor": "<entity 1>", "edge_filter": "<verb pattern>", "yield_var": "<bridge name>"},
    "step1b": {"anchor": "<entity 2>", "edge_filter": "<verb pattern>", "yield_var": "<bridge name>"}
  }
}

Edge_filter is a regex pattern matched against r.raw_relation (case-insensitive substring match).
Use '|' to OR multiple verbs: "found|start|launch|co-found".

Examples:

Q: "Which companies did founders of PayPal later start?"
{"plan": {
  "step1": {"anchor": "PayPal", "edge_filter": "found|co-found|start", "yield_var": "founder"},
  "step2": {"from_var": "founder", "edge_filter": "found|co-found|start|launch", "exclude_anchor": true, "yield_var": "company"}
}}

Q: "What companies were founded by Stanford alumni?"
{"plan": {
  "step1": {"anchor": "Stanford", "edge_filter": "attend|graduate|stud|alumn", "yield_var": "alumnus"},
  "step2": {"from_var": "alumnus", "edge_filter": "found|co-found|start", "yield_var": "company"}
}}

Q: "What companies have been founded by Harvard dropouts?"
{"plan": {
  "step1": {"anchor": "Harvard", "edge_filter": "drop|attend|stud", "yield_var": "dropout"},
  "step2": {"from_var": "dropout", "edge_filter": "found|co-found|start", "yield_var": "company"}
}}

Q: "Who founded both a payments company and a space company?"
{"plan": {
  "type": "intersection",
  "step1a": {"anchor": "payments", "edge_filter": "found|co-found", "yield_var": "founder"},
  "step1b": {"anchor": "space", "edge_filter": "found|co-found", "yield_var": "founder"}
}}

Q: "Who founded Microsoft?"
{"plan": null}

Q: "What is the relationship between Apple and NeXT?"
{"plan": null}

Q: "Where did Bill Gates go to university?"
{"plan": null}

Output strict JSON only. Use null when not multi-hop."""


def _decompose_multihop(query: str) -> dict | None:
    """LLM-based query classifier + decomposition planner. Returns a plan dict
    when the question is multi-hop (bridge or intersection), else None.
    Falls back to None on any LLM/JSON parse error — the caller continues
    with the default fetch_subgraph path."""
    try:
        resp = omlx.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _DECOMPOSE_SYSTEM},
                {"role": "user",   "content": f"Q: {query}"},
            ],
            temperature=0.0, max_tokens=500,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)
        plan = parsed.get("plan")
        if not isinstance(plan, dict):
            return None
        return plan
    except Exception as e:
        print(f"[WARN] decompose_multihop failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _execute_decomposition(plan: dict, max_intermediate: int = 30) -> list[dict]:
    """Execute a 2-step decomposition plan against Neo4j. Returns edges
    suitable for the LLM context (same shape as fetch_subgraph output).

    Two patterns supported:
      - 2-step bridge: anchor → intermediates (filter1) → answers (filter2)
      - intersection: two independent step-1s, intersect intermediates by name,
        return all edges incident to the intersected entities

    Edge filter regex matches r.raw_relation case-insensitively (substring)."""
    edges: list[dict] = []
    plan_type = plan.get("type", "bridge")

    with driver.session() as session:
        if plan_type == "intersection":
            # Run two independent step-1 queries, intersect intermediate names,
            # then collect all edges incident to the intersection.
            step1a, step1b = plan.get("step1a"), plan.get("step1b")
            if not (step1a and step1b):
                return []
            inter_a = _step_one_intermediates(session, step1a["anchor"], step1a["edge_filter"], max_intermediate)
            inter_b = _step_one_intermediates(session, step1b["anchor"], step1b["edge_filter"], max_intermediate)
            common = sorted(set(inter_a) & set(inter_b))
            if not common:
                return []
            # Collect all edges incident to common entities (≤ max_intermediate of each)
            r = session.run("""
                UNWIND $names AS name
                MATCH (n:Entity {name: name})-[r]-(m:Entity)
                RETURN DISTINCT startNode(r).name AS s, r.raw_relation AS rel,
                                endNode(r).name AS o, r.source_title AS src
                LIMIT 200
            """, names=common[:max_intermediate])
            edges.extend(dict(rec) for rec in r)
            return edges

        # Default: 2-step bridge
        step1, step2 = plan.get("step1"), plan.get("step2")
        if not (step1 and step2):
            return []
        intermediates = _step_one_intermediates(session, step1["anchor"], step1["edge_filter"], max_intermediate)
        if not intermediates:
            return []

        # Step 2: from each intermediate entity, follow edge_filter to a target.
        # Collect both the step-1 edges (anchor↔intermediate) AND step-2 edges
        # (intermediate↔target) so the LLM has full chain context.
        anchor = step1["anchor"]
        edge_filter1 = step1["edge_filter"]
        edge_filter2 = step2["edge_filter"]
        exclude_anchor = step2.get("exclude_anchor", False)

        # Step-1 edges (anchor neighborhood, filtered)
        anchor_lucene = _lucene_phrase_query(anchor) or _lucene_or_query(anchor) or anchor
        r_anchor = session.run("""
            CALL db.index.fulltext.queryNodes("entity_names", $lucene)
            YIELD node, score
            WITH node ORDER BY score DESC LIMIT 5
            MATCH (node)-[r]-(m)
            WHERE toLower(r.raw_relation) =~ ('(?i).*(' + $filter + ').*')
            RETURN DISTINCT startNode(r).name AS s, r.raw_relation AS rel,
                            endNode(r).name AS o, r.source_title AS src
            LIMIT 100
        """, lucene=anchor_lucene, filter=edge_filter1)
        edges.extend(dict(rec) for rec in r_anchor)

        # Step-2 edges (each intermediate's outgoing neighborhood, filtered)
        params = {
            "names":  intermediates[:max_intermediate],
            "filter": edge_filter2,
            "anchor": anchor.lower(),
        }
        if exclude_anchor:
            r_step2 = session.run("""
                UNWIND $names AS iname
                MATCH (n:Entity {name: iname})-[r]-(m:Entity)
                WHERE toLower(r.raw_relation) =~ ('(?i).*(' + $filter + ').*')
                  AND NOT toLower(m.name) CONTAINS $anchor
                RETURN DISTINCT startNode(r).name AS s, r.raw_relation AS rel,
                                endNode(r).name AS o, r.source_title AS src
                LIMIT 200
            """, **params)
        else:
            r_step2 = session.run("""
                UNWIND $names AS iname
                MATCH (n:Entity {name: iname})-[r]-(m:Entity)
                WHERE toLower(r.raw_relation) =~ ('(?i).*(' + $filter + ').*')
                RETURN DISTINCT startNode(r).name AS s, r.raw_relation AS rel,
                                endNode(r).name AS o, r.source_title AS src
                LIMIT 200
            """, **params)
        edges.extend(dict(rec) for rec in r_step2)
    return edges


def _step_one_intermediates(session, anchor: str, edge_filter: str, limit: int) -> list[str]:
    """Return up to `limit` distinct entity names connected to `anchor` via
    edges whose r.raw_relation matches edge_filter regex."""
    anchor_lucene = _lucene_phrase_query(anchor) or _lucene_or_query(anchor) or anchor
    try:
        rows = list(session.run("""
            CALL db.index.fulltext.queryNodes("entity_names", $lucene)
            YIELD node, score
            WITH node ORDER BY score DESC LIMIT 5
            MATCH (node)-[r]-(m:Entity)
            WHERE toLower(r.raw_relation) =~ ('(?i).*(' + $filter + ').*')
            RETURN DISTINCT m.name AS name LIMIT $lim
        """, lucene=anchor_lucene, filter=edge_filter, lim=limit))
        return [r["name"] for r in rows]
    except Exception as e:
        print(f"[WARN] step_one_intermediates failed for anchor={anchor!r}: {type(e).__name__}: {e}", file=sys.stderr)
        return []


_GDS_GRAPH = "entity-ppr-graph"


def _ensure_gds_projection() -> bool:
    """Create undirected Entity projection for PPR if not already present.

    Uses a wildcard relationship projection (type='*') so ALL edge types propagate
    — the entire point of using GDS over the LLM decomposer is that no edge-type
    specification is needed.  Returns False and logs a warning on failure so the
    caller can skip PPR gracefully."""
    with driver.session() as s:
        try:
            existing = s.run(
                "CALL gds.graph.list() YIELD graphName RETURN graphName"
            ).data()
            if any(r["graphName"] == _GDS_GRAPH for r in existing):
                return True
            s.run(
                """
                CALL gds.graph.project(
                    $name, 'Entity',
                    {__ALL__: {type: '*', orientation: 'UNDIRECTED'}}
                )
                """,
                name=_GDS_GRAPH,
            ).consume()
            return True
        except Exception as e:
            print(f"[WARN] GDS projection failed: {type(e).__name__}: {e}", file=sys.stderr)
            return False


def _ppr_retrieve(seeds: list[str], top_k: int = 60) -> list[dict]:
    """Personalized PageRank via Neo4j GDS from seed entity names.

    Two-step:
      1. Resolve seed names → exact graph node names via full-text index.
      2. MATCH those nodes → pass as sourceNodes to gds.pageRank.stream.

    PPR propagates through ALL edge types without any regex specification —
    "received a bachelor of science degree from Stanford" and "attended Stanford"
    both reach the same alumni nodes because ALL edges are included.

    Returns edges where both endpoints are in the top-K PPR-ranked nodes,
    ready to be prepended to the LLM context."""
    if not _ensure_gds_projection():
        return []

    seed_node_names: list[str] = []
    with driver.session() as s:
        for seed in seeds:
            lucene = _lucene_phrase_query(seed) or _lucene_or_query(seed) or seed
            rows = s.run(
                "CALL db.index.fulltext.queryNodes(\"entity_names\", $l) YIELD node, score "
                "WITH node ORDER BY score DESC LIMIT 2 RETURN node.name AS name",
                l=lucene,
            ).data()
            seed_node_names.extend(r["name"] for r in rows if r["name"])

    if not seed_node_names:
        return []

    with driver.session() as s:
        try:
            ppr_rows = s.run(
                """
                MATCH (seed:Entity) WHERE seed.name IN $names
                WITH collect(seed) AS seeds
                CALL gds.pageRank.stream($graph, {
                    maxIterations: 20,
                    dampingFactor: 0.85,
                    sourceNodes: seeds
                })
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).name AS name, score
                ORDER BY score DESC LIMIT $top_k
                """,
                names=seed_node_names,
                graph=_GDS_GRAPH,
                top_k=top_k,
            ).data()
        except Exception as e:
            print(f"[WARN] PPR stream failed: {type(e).__name__}: {e}", file=sys.stderr)
            return []

        top_names = [r["name"] for r in ppr_rows if r["name"]]
        if not top_names:
            return []

        edges = s.run(
            """
            MATCH (a:Entity)-[r]-(b:Entity)
            WHERE a.name IN $names AND b.name IN $names
            RETURN DISTINCT startNode(r).name AS s, r.raw_relation AS rel,
                            endNode(r).name   AS o, r.source_title  AS src
            LIMIT 200
            """,
            names=top_names,
        ).data()
    return [dict(r) for r in edges]


_RELATIONAL_KWS = frozenset(
    ["relationship", "connection", "connected", "related", "between", "link"]
)


def _find_bridge_edges(e1: str, e2: str) -> list[dict]:
    """Find edges incident to entities shared by e1's and e2's 1-hop neighborhoods.

    General shared-neighbor intersection — no hardcoded relation types.
    Works for founders, investors, board members, alumni, or any other bridge.

    Strategy: fetch each entity's 1-hop edge set (≤150 edges each), intersect
    neighbor names in Python, return all edges touching shared intermediates.
    The caller prepends these to the main subgraph so the LLM sees bridge
    edges first within the 300-edge context cap."""
    def _get_neighbor_edges(session, lucene: str) -> dict[str, list[dict]]:
        rows = session.run(
            """
            CALL db.index.fulltext.queryNodes("entity_names", $lucene) YIELD node, score
            WITH node ORDER BY score DESC LIMIT 3
            MATCH (node)-[r]-(m:Entity)
            RETURN m.name AS name,
                   startNode(r).name AS s, r.raw_relation AS rel,
                   endNode(r).name   AS o, r.source_title  AS src
            LIMIT 150
            """,
            lucene=lucene,
        )
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            buckets[row["name"]].append(
                {"s": row["s"], "rel": row["rel"], "o": row["o"], "src": row["src"]}
            )
        return dict(buckets)

    l1 = _lucene_phrase_query(e1) or _lucene_or_query(e1) or e1
    l2 = _lucene_phrase_query(e2) or _lucene_or_query(e2) or e2
    with driver.session() as session:
        nbrs_a = _get_neighbor_edges(session, l1)
        nbrs_b = _get_neighbor_edges(session, l2)
    shared = set(nbrs_a.keys()) & set(nbrs_b.keys())
    bridge: list[dict] = []
    for intermediate in sorted(shared):
        bridge.extend(nbrs_a[intermediate])
        bridge.extend(nbrs_b[intermediate])
    return bridge


def answer(query: str) -> dict:
    seeds = extract_seed_entities(query)
    subgraph, matches_per_seed = fetch_subgraph(seeds)

    # Multi-hop query decomposition (Option B from W2.5 follow-up analysis).
    # Detects bridge / intersection question shapes via LLM classifier; if
    # detected, runs targeted 2-step Cypher and concatenates the result with
    # the default subgraph. Default fetch_subgraph already covers 1-hop
    # priority + multi-hop fill; decomposition adds high-precision bridge
    # edges that the shotgun multi-hop fill tends to crowd out.
    # Per-edge dedup later in this function collapses any duplicates.
    decomp_plan = _decompose_multihop(query)
    if decomp_plan:
        decomp_edges = _execute_decomposition(decomp_plan)
        if decomp_edges:
            subgraph.extend(decomp_edges)
            matches_per_seed["__decomposition__"] = {
                "plan_type": decomp_plan.get("type", "bridge"),
                "edges_added": len(decomp_edges),
            }

    # PPR supplement — runs unconditionally for every query.
    # GDS personalized PageRank propagates through ALL edge types (~50ms).
    # No keyword list needed: PPR from seed "Microsoft" naturally ranks
    # Bill Gates / Paul Allen highest regardless of query type.
    # Edges prepended so they surface first within the 300-edge LLM context cap.
    # Overlap with fetch_subgraph is collapsed by per-edge dedup below.
    query_lower = query.lower()
    ppr_edges = _ppr_retrieve(seeds)
    if ppr_edges:
        subgraph = ppr_edges + subgraph
        matches_per_seed["__ppr__"] = {"edges_added": len(ppr_edges)}

    # Relational bridge: when exactly 2 seeds and no decomp plan was found,
    # compute shared-neighbor edges via Python-side intersection.
    # Prepend bridge edges so they land first in the 300-edge LLM context
    # window — prevents them from being crowded out by the larger per-seed
    # neighborhood expansion that fetch_subgraph already ran.
    if (
        len(seeds) == 2
        and decomp_plan is None
        and any(kw in query_lower for kw in _RELATIONAL_KWS)
    ):
        bridge_edges = _find_bridge_edges(seeds[0], seeds[1])
        if bridge_edges:
            subgraph = bridge_edges + subgraph
            matches_per_seed["__bridge__"] = {"edges_added": len(bridge_edges)}

    # Two precondition surfaces:
    # (a) strategy == "none" → corpus does not contain any token of the seed.
    #     This is the corpus-mismatch case the v1 query silently traversed
    #     through false-positive substring matches; precondition warning makes
    #     it loud.
    # (b) strategy == "or" → phrase query missed but OR fallback grounded.
    #     Weaker match — typically means the named entity isn't in the
    #     corpus but some token of the name is. Warning fires too because
    #     downstream answers from this case have low precision.
    unmatched = [
        s for s, m in matches_per_seed.items()
        if not s.startswith("__") and m.get("strategy") == "none"
    ]
    weak_match = [
        s for s, m in matches_per_seed.items()
        if not s.startswith("__") and m.get("strategy") == "or"
    ]
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
