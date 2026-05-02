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
ANSWER_MODEL = os.getenv("MODEL_ANSWER", MODEL)  # prose synthesis; defaults to MODEL if unset
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

# ---------------------------------------------------------------------------
# Composite seed-resolution scorer
# Calibrate weights with: python src/calibrate_scorer.py --quick
# ---------------------------------------------------------------------------
QID_BONUS       = 2.5   # +bonus for QID-keyed canonical entity nodes
                        # Calibrated on 5-probe set (v12.1 graph, pre-rebuild):
                        # (2.5, 0.8, 0.3) → recall=1.000; (1.5, 0.8, 0.3) → recall<1.000
                        # Re-run: python src/calibrate_scorer.py --quick after each rebuild
EXACT_BONUS     = 0.8   # +bonus when seed exactly matches node.name or aliases entry
DEGREE_COEFF    = 0.3   # multiplier on log(degree+1) — activates after graph rebuild
SCORE_THRESHOLD = 2.0   # minimum composite; below this = ungrounded, skip traversal

# Built at module load so changing a constant above auto-updates the query.
# $lucene, $seed, $threshold, $limit are Cypher parameters at call time.
_RERANK_CYPHER = (
    'CALL db.index.fulltext.queryNodes("entity_names", $lucene) '
    'YIELD node, score AS bm25 '
    'WITH node, bm25, '
    f'CASE WHEN node.qid IS NOT NULL THEN {QID_BONUS} ELSE 0 END AS qid_bonus, '
    'CASE WHEN toLower(node.name) = toLower($seed) '
    '  OR $seed IN coalesce(node.aliases, []) '
    f'  THEN {EXACT_BONUS} ELSE 0 END AS exact_bonus, '
    f'log(coalesce(node.degree, 0) + 1) * {DEGREE_COEFF} AS degree_score '
    'WITH node, bm25 + qid_bonus + exact_bonus + degree_score AS composite '
    'WHERE composite >= $threshold '
    # Topology gate: a node is "real" iff externally grounded (QID) OR
    # corpus-internally redundant (degree >= 2). Excludes singleton noise
    # like 'CEO of Apple' (qid=None, degree=1) without enumerating patterns.
    '  AND (node.qid IS NOT NULL OR coalesce(node.degree, 0) >= 2) '
    'WITH node ORDER BY composite DESC LIMIT $limit '
    'RETURN node.name AS name'
)


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
        temperature=0.0, max_tokens=2000,
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
    """OR-joined Lucene query restricted to proper-noun tokens.

    Filters to words that start with a capital letter in the original seed
    before lowercasing. Prevents generic descriptor words from polluting OR
    expansion with semantically unrelated entity matches:

      'Stanford alumni'  → 'stanford'        (not 'stanford OR alumni')
      'Jensen Huang'     → 'jensen OR huang'
      'PayPal founders'  → 'paypal'
      'Harvard dropouts' → 'harvard'

    Falls back to all tokens when no proper nouns exist (handles edge-cases
    like all-lowercase single-word seeds)."""
    proper = [w for w in seed.split() if w and w[0].isupper()]
    seed_filtered = " ".join(proper) if proper else seed
    tokens = _lucene_tokens(seed_filtered)
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


def _resolve_seed_node_names(
    session,
    lucene: str,
    seed: str,
    limit: int = 5,
    threshold: float = SCORE_THRESHOLD,
) -> list[str]:
    """Return up to `limit` entity names ranked by composite BM25+QID+exact+degree score.

    Empty list when no node clears `threshold` — caller treats this as an
    ungrounded seed and skips traversal (no noise expansion).

    Composite formula (weights in _RERANK_CYPHER, constants above):
      composite = bm25 + qid_bonus + exact_bonus + log(degree+1) * DEGREE_COEFF
    """
    try:
        rows = session.run(
            _RERANK_CYPHER,
            lucene=lucene,
            seed=seed,
            threshold=threshold,
            limit=limit,
        ).data()
        return [r["name"] for r in rows if r["name"]]
    except Exception as e:
        print(
            f"[WARN] _resolve_seed_node_names failed for seed={seed!r}: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return []


_degree_checked = False


def _check_degree_coverage_once() -> None:
    """One-shot startup warning when fewer than 50% of Entity nodes have n.degree.

    n.degree is written by build_graph._write_degree_centrality(). Without it
    the composite scorer's degree signal is uniformly 0 — still correct but
    loses hub-node differentiation. Warning fires at most once per process."""
    global _degree_checked
    if _degree_checked:
        return
    _degree_checked = True
    try:
        with driver.session() as s:
            row = s.run(
                "MATCH (n:Entity) "
                "RETURN count(n) AS total, count(n.degree) AS with_degree"
            ).single()
            if row and row["total"] > 0:
                coverage = row["with_degree"] / row["total"]
                if coverage < 0.5:
                    print(
                        f"[WARN] degree coverage {coverage:.1%} "
                        f"({row['with_degree']}/{row['total']} Entity nodes). "
                        "Composite scorer running without degree signal — "
                        "re-run build_graph.py to populate n.degree.",
                        file=sys.stderr,
                    )
    except Exception:
        pass  # DB not available at import time — not fatal


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
            # Two-stage resolution: try phrase first (high precision), fall back
            # to OR (broader recall) if phrase yields no anchors POST-FILTER.
            # _count_index_matches measures raw index hits including noise; the
            # topology gate inside _resolve_seed_node_names can prune everything
            # phrase matched (e.g. seed 'CEO of Apple' hits noise nodes only),
            # so we must retry with OR even when phrase_n > 0.
            anchor_names: list[str] = []
            strategy = "none"
            if phrase_n > 0:
                anchor_names = _resolve_seed_node_names(session, phrase, seed)
                if anchor_names:
                    strategy = "phrase"
            if not anchor_names and or_form != phrase and or_n > 0:
                anchor_names = _resolve_seed_node_names(session, or_form, seed)
                if anchor_names:
                    strategy = "or"
            if not anchor_names:
                matches_per_seed[seed] = {
                    "phrase": phrase_n,
                    "or":     or_n,
                    "strategy": "none",
                }
                continue

            # Substring-token expansion: when the seed is a multi-token proper
            # noun (e.g. "Marc Andreessen", "Tesla Motors"), also surface any
            # entity whose canonical name is one rare token of the seed (e.g.
            # the bare "Andreessen" node, or "Tesla"). The graph may store
            # edges under both the full and substring-only surface form due
            # to extraction-LLM surface-form drift; QID disambiguation +
            # surface-form-drift instructions in the LLM context let the
            # answer LLM merge them based on edge context. Generic across
            # person names, multi-word company names, and any compound proper
            # noun — not surname-specific.
            proper_tokens = [w for w in seed.split() if w and w[0].isupper() and len(w) >= 4]
            if len(proper_tokens) >= 2:
                last_token = proper_tokens[-1]
                if last_token not in anchor_names:
                    extra = session.run(
                        "MATCH (n:Entity) WHERE n.name = $name "
                        "  AND n.qid IS NOT NULL AND coalesce(n.degree, 0) >= 2 "
                        "RETURN n.name AS name LIMIT 1",
                        name=last_token,
                    ).single()
                    if extra and extra["name"] not in anchor_names:
                        anchor_names = anchor_names + [extra["name"]]

            # Two-pass: 1-hop edges first (canonical direct neighbors for relational +
            # factoid questions), then 2..N-hop fill (bridge queries). Concatenating
            # guarantees 1-hop edges appear in the LLM context budget even on dense
            # neighborhoods where multi-hop expansion would otherwise crowd them out.
            edges: list[dict] = []
            r1 = session.run(
                """
                MATCH (node:Entity) WHERE node.name IN $names
                MATCH (node)-[r]-(m)
                RETURN DISTINCT startNode(r).name AS s, r.raw_relation AS rel,
                                endNode(r).name AS o, r.source_title AS src
                LIMIT 100
                """,
                names=anchor_names,
            )
            edges.extend(dict(record) for record in r1)
            if max_hops > 1:
                rn = session.run(
                    f"""
                    MATCH (node:Entity) WHERE node.name IN $names
                    MATCH path = (node)-[*2..{max_hops}]-(m)
                    WITH DISTINCT relationships(path) AS rels
                    UNWIND rels AS r
                    RETURN DISTINCT startNode(r).name AS s, r.raw_relation AS rel,
                                    endNode(r).name AS o, r.source_title AS src
                    LIMIT 100
                    """,
                    names=anchor_names,
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
    "step2": {"from_var": "<step1 yield_var>", "edge_filter": "<verb regex pattern>", "exclude_anchor": true|false, "yield_var": "<answer name>", "expand_terminal": true|false}
  }
}

Set `expand_terminal: true` ONLY when the question's qualifying clauses
reference relations that step-2's edge_filter does NOT match. Examples:
- "...that was later acquired" → step-2 filter is founding-style, but the
  qualifier needs `acquired` edges → expand_terminal = true
- "...and what happened to it" → terminal events not in founding filter
  → expand_terminal = true
- "...what they later started" → step-2 IS the founding edges → expand_terminal = false
- "...companies founded by Stanford alumni" → 2-step is sufficient → expand_terminal = false
Default to false unless the qualifier clearly requires beyond-step-2 edges.

For an INTERSECTION (must satisfy two filters):
{
  "plan": {
    "type": "intersection",
    "step1a": {"anchor": "<entity 1>", "edge_filter": "<verb pattern>", "yield_var": "<bridge name>"},
    "step1b": {"anchor": "<entity 2>", "edge_filter": "<verb pattern>", "yield_var": "<bridge name>"}
  }
}

**Intersection eligibility rule:** Both step1a.anchor AND step1b.anchor MUST
be specific named entities that exist as graph nodes (e.g. "PayPal", "Stanford",
"Apple Inc."). If either would be a category or descriptor (e.g. "technology
company", "enterprise software", "venture capitalists", "founders"), use the
2-step bridge format instead — categories are not graph nodes and intersection
on them returns empty.

Edge_filter is a regex pattern matched against r.raw_relation (case-insensitive substring match).
Use '|' to OR multiple verbs: "found|start|launch|co-found".

Examples:

Q: "Which companies did founders of PayPal later start?"
{"plan": {
  "step1": {"anchor": "PayPal", "edge_filter": "found|co-found|start", "yield_var": "founder"},
  "step2": {"from_var": "founder", "edge_filter": "found|co-found|start|launch", "exclude_anchor": true, "yield_var": "company", "expand_terminal": false}
}}

Q: "What companies were founded by Stanford alumni?"
(Anchor uses canonical entity name "Stanford University", not the disambig
page "Stanford" — the canonical entity has more attendance edges.)
{"plan": {
  "step1": {"anchor": "Stanford University", "edge_filter": "attend|graduate|stud|alum|enroll|earn|receiv|drop|transfer|pursu", "yield_var": "alumnus"},
  "step2": {"from_var": "alumnus", "edge_filter": "found|co-found|start|launch|creat|initiat", "yield_var": "company", "expand_terminal": false}
}}

Q: "What companies have been founded by Harvard dropouts?"
{"plan": {
  "step1": {"anchor": "Harvard", "edge_filter": "drop|attend|stud|enroll|earn|receiv", "yield_var": "dropout"},
  "step2": {"from_var": "dropout", "edge_filter": "found|co-found|start|launch|creat|initiat", "yield_var": "company", "expand_terminal": false}
}}

Q: "Who co-founded Andreessen Horowitz with Marc Andreessen, and what enterprise software company had they previously co-founded together that was later acquired?"
{"plan": {
  "step1": {"anchor": "Andreessen Horowitz", "edge_filter": "co-found|found|start", "yield_var": "co_founder"},
  "step2": {"from_var": "co_founder", "edge_filter": "found|co-found|start|launch|creat", "exclude_anchor": true, "yield_var": "company", "expand_terminal": true}
}}

Q: "Who has worked at both Microsoft and Apple Inc.?"
{"plan": {
  "type": "intersection",
  "step1a": {"anchor": "Microsoft", "edge_filter": "work|join|employ|hired|served", "yield_var": "person"},
  "step1b": {"anchor": "Apple Inc.", "edge_filter": "work|join|employ|hired|served", "yield_var": "person"}
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
            temperature=0.0, max_tokens=2000,
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

        # Step-1 edges (anchor neighborhood, filtered) — supporting context
        anchor_lucene = _lucene_phrase_query(anchor) or _lucene_or_query(anchor) or anchor
        anchor_names = _resolve_seed_node_names(session, anchor_lucene, anchor)
        if anchor_names:
            r_anchor = session.run(
                """
                MATCH (node:Entity) WHERE node.name IN $names
                MATCH (node)-[r]-(m)
                WHERE toLower(r.raw_relation) =~ ('(?i).*(' + $filter + ').*')
                RETURN DISTINCT startNode(r).name AS s, r.raw_relation AS rel,
                                endNode(r).name AS o, r.source_title AS src
                LIMIT 100
                """,
                names=anchor_names, filter=edge_filter1,
            )
            step1_edges = [dict(rec) for rec in r_anchor]
        else:
            step1_edges = []

        # Step-2 edges (each intermediate's outgoing neighborhood, filtered) — direct answer
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
        step2_edges = [dict(rec) for rec in r_step2]

        # Step-3 neighborhood expansion: opt-in per-plan. Compound questions
        # whose qualifying clauses reference relations beyond step-2's filter
        # (e.g. "...that was later acquired") set `expand_terminal: true` in
        # the plan; the decomposition LLM decides based on question structure.
        # When false, skip step3 entirely (default) — avoids flooding simple
        # 2-step chains (e.g. "companies founded by Stanford alumni") with
        # off-topic noise around each step-2 target.
        step3_edges: list[dict] = []
        if step2.get("expand_terminal"):
            step2_target_names = sorted({e["o"] for e in step2_edges if e.get("o")})
            if step2_target_names:
                r_step3 = session.run(
                    """
                    MATCH (n:Entity) WHERE n.name IN $names
                    MATCH (n)-[r]-(m:Entity)
                    RETURN DISTINCT startNode(r).name AS s, r.raw_relation AS rel,
                                    endNode(r).name AS o, r.source_title AS src
                    LIMIT 100
                    """,
                    names=step2_target_names[:max_intermediate],
                )
                step3_edges = [dict(rec) for rec in r_step3]

        # Step-2 first (direct answer), step-3 second (terminal-entity context),
        # step-1 last (supporting context).
        # LLMs attend most reliably to the start of long contexts ("lost in
        # the middle" effect). For bridge queries like "companies founded by
        # Stanford alumni", the founding edges (step-2) are the primary answer
        # and must appear early; the education edges (step-1) are supporting.
        edges.extend(step2_edges)
        edges.extend(step3_edges)
        edges.extend(step1_edges)
    return edges


def _step_one_intermediates(session, anchor: str, edge_filter: str, limit: int) -> list[str]:
    """Return up to `limit` distinct entity names connected to `anchor` via
    edges whose r.raw_relation matches edge_filter regex."""
    anchor_lucene = _lucene_phrase_query(anchor) or _lucene_or_query(anchor) or anchor
    try:
        anchor_names = _resolve_seed_node_names(session, anchor_lucene, anchor)
        if not anchor_names:
            return []
        rows = list(session.run(
            """
            MATCH (node:Entity) WHERE node.name IN $names
            MATCH (node)-[r]-(m:Entity)
            WHERE toLower(r.raw_relation) =~ ('(?i).*(' + $filter + ').*')
            RETURN DISTINCT m.name AS name LIMIT $lim
            """,
            names=anchor_names, filter=edge_filter, lim=limit,
        ))
        return [r["name"] for r in rows]
    except Exception as e:
        print(
            f"[WARN] step_one_intermediates failed for anchor={anchor!r}: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return []


_GDS_GRAPH = "entity-ppr-graph"
_gds_projection_refreshed = False  # session-lifetime flag


def _ensure_gds_projection() -> bool:
    """Create undirected Entity projection for PPR if not already present.

    Uses a wildcard relationship projection (type='*') so ALL edge types propagate
    — the entire point of using GDS over the LLM decomposer is that no edge-type
    specification is needed.  Returns False and logs a warning on failure so the
    caller can skip PPR gracefully.

    On first call per process, drops any pre-existing projection so we never
    inherit stale node IDs from a prior graph build. Subsequent calls reuse
    the in-memory projection from the same session."""
    global _gds_projection_refreshed
    with driver.session() as s:
        try:
            existing = s.run(
                "CALL gds.graph.list() YIELD graphName RETURN graphName"
            ).data()
            already_present = any(r["graphName"] == _GDS_GRAPH for r in existing)
            if already_present and _gds_projection_refreshed:
                return True
            if already_present:
                # Explicit YIELD avoids consuming deprecated `schema` column
                s.run(
                    "CALL gds.graph.drop($name) YIELD graphName RETURN graphName",
                    name=_GDS_GRAPH,
                ).consume()
            s.run(
                """
                CALL gds.graph.project(
                    $name, 'Entity',
                    {__ALL__: {type: '*', orientation: 'UNDIRECTED'}}
                )
                """,
                name=_GDS_GRAPH,
            ).consume()
            _gds_projection_refreshed = True
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
            seed_node_names.extend(
                _resolve_seed_node_names(s, lucene, seed, limit=2)
            )

    if not seed_node_names:
        return []

    with driver.session() as s:
        try:
            ppr_rows = s.run(
                """
                MATCH (seed:Entity) WHERE seed.name IN $names
                  AND coalesce(seed.degree, 0) >= 1
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
    def _get_neighbor_edges(session, lucene: str, seed: str) -> dict[str, list[dict]]:
        anchor_names = _resolve_seed_node_names(session, lucene, seed, limit=3)
        if not anchor_names:
            return {}
        rows = session.run(
            """
            MATCH (node:Entity) WHERE node.name IN $names
            MATCH (node)-[r]-(m:Entity)
            RETURN m.name AS name,
                   startNode(r).name AS s, r.raw_relation AS rel,
                   endNode(r).name   AS o, r.source_title  AS src
            LIMIT 150
            """,
            names=anchor_names,
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
        nbrs_a = _get_neighbor_edges(session, l1, e1)
        nbrs_b = _get_neighbor_edges(session, l2, e2)
    shared = set(nbrs_a.keys()) & set(nbrs_b.keys())
    bridge: list[dict] = []
    for intermediate in sorted(shared):
        bridge.extend(nbrs_a[intermediate])
        bridge.extend(nbrs_b[intermediate])
    return bridge


def answer(query: str) -> dict:
    _check_degree_coverage_once()
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
            # Prepend decomp edges so they appear at the start of the LLM
            # context. The basic subgraph from seed entities is appended after,
            # keeping the high-precision targeted edges visible early (avoids
            # "lost in the middle" suppression on 200+ edge contexts).
            subgraph = decomp_edges + subgraph
            matches_per_seed["__decomposition__"] = {
                "plan_type": decomp_plan.get("type", "bridge"),
                "edges_added": len(decomp_edges),
            }

    # Query-type router — priority chain (mirrors ToG/IRCoT retrieval ordering):
    #   1. Decomposition (already ran above) — multi-hop bridge/intersection questions
    #   2. Relational bridge — two-entity relationship questions
    #   3. PPR — last resort when both structured methods produced nothing
    #
    # PPR fires ONLY if neither __decomposition__ nor __bridge__ produced edges.
    # Unconditional PPR caused multi_hop regression (v13: Q20 0.60→0.20, Q23 1.00→0.33):
    # for queries where decomposition already found targeted edges, PPR's 200 extra
    # edges from global high-PageRank neighbors flooded context and buried the
    # decomposition output. The novelty argument applies equally: if decomp/bridge
    # already found the right neighborhood, PPR returns nodes already covered → no
    # new information, only noise.
    query_lower = query.lower()

    # Steps 2+3: bridge edges (targeted bridge-finding) and PPR (broad
    # PageRank-weighted neighborhood). Final ordering matters: with the
    # 300-edge context cap, PPR's 200+ edges can push targeted bridge
    # edges past the cutoff. Order chosen: Bridge | PPR | Initial.
    # Targeted evidence first, broad signal second, full neighborhood last.
    bridge_edges: list[dict] = []
    if (
        len(seeds) == 2
        and decomp_plan is None
        and any(kw in query_lower for kw in _RELATIONAL_KWS)
    ):
        bridge_edges = _find_bridge_edges(seeds[0], seeds[1])
        if bridge_edges:
            matches_per_seed["__bridge__"] = {"edges_added": len(bridge_edges)}

    ppr_edges: list[dict] = []
    if not matches_per_seed.get("__decomposition__"):
        ppr_edges = _ppr_retrieve(seeds)
        if ppr_edges:
            matches_per_seed["__ppr__"] = {"edges_added": len(ppr_edges)}

    subgraph = bridge_edges + ppr_edges + subgraph

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

    # Wikidata QID lookup so the LLM sees disambig signal in-prompt.
    # Same name + different QID = different real-world entities. Without
    # QID inline, the LLM has to infer from edge context alone, which fails
    # on disambiguation cases (e.g. 'Tesla' the scientist vs 'Tesla, Inc.').
    edge_names: set[str] = set()
    for (s, rel, o), _ in edge_groups.items():
        edge_names.add(s); edge_names.add(o)
    qid_map: dict[str, str] = {}
    if edge_names:
        with driver.session() as s_:
            rows = s_.run(
                "MATCH (n:Entity) WHERE n.name IN $names "
                "RETURN n.name AS name, n.qid AS qid",
                names=list(edge_names),
            ).data()
        qid_map = {r["name"]: r["qid"] for r in rows if r.get("qid")}

    def _label(name: str) -> str:
        q = qid_map.get(name)
        return f"{name} [{q}]" if q else name

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
        context_lines.append(f"- {_label(s)} --[{rel}]--> {_label(o)}{src_text}")
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
Entities may be tagged with a Wikidata QID in brackets: `Tesla, Inc. [Q478214]`.
The QID identifies a specific real-world entity. **Same name + different QID = different
entities.** Example: `Tesla [Q9036]` (the inventor Nikola Tesla) is a different entity
from `Tesla, Inc. [Q478214]` (the car company), even though both render as "Tesla" in
informal English. Use QID to disambiguate when multiple entities share a name.

**Surface-form-drift exception:** Sometimes the SAME real-world entity (person,
organization, product, place, concept) appears under TWO different QIDs because
the extraction step recorded two surface forms (a long form and a substring of
it; e.g. "<Full Name> [Q1]" and "<Substring> [Q2]"). Indicators they refer to
the SAME entity:
  - One name is a substring of the other
  - They share neighbors (collaborators, parents, affiliations, members, etc.)
    in the edges
  - Their edges describe a coherent profile (no contradictions)
When these indicators all hold, MERGE the two entities for the purpose of the
answer. Concretely: if the question asks about Entity X and surface-form drift
adds Entity X' (sharing a substring + overlapping context), then EVERY edge
from X' is ALSO valid evidence for X. Include items from X' in your
LIST/COMPOUND/RELATIONSHIP answer alongside items from X. Do not skip X'
edges just because they have a different QID — the merge instruction makes
them count. Cite both QIDs once at the start of the answer to make the merge
transparent (e.g. "<Canonical Full Name> [Q1, also Q2]").
When the same edge is corroborated by multiple articles, sources are listed:
`Subject --[relation]--> Object  (sources: Article1, Article2)` — treat that
as multi-source evidence for ONE fact, not multiple facts.

Edge direction matters but the SAME relationship can be expressed either
direction in the graph. Examples (treat as the same fact):
  - "Apple --[acquired]--> NeXT" and "NeXT --[acquired by]--> Apple"
  - "Steve Jobs --[co-founded]--> Apple" and "Apple --[was co-founded by]--> Steve Jobs"

REQUIRED PROCESS:
1. **Identify the question type:**
   - LIST/ENUMERATION: question expects multiple items as the answer ("which X",
     "what are all", "list", "who has", "what entities", "what events").
   - RELATIONSHIP: question asks how two named entities relate ("what is the
     relationship between X and Y", "how is X connected to Y", "in what way
     does X relate to Y").
   - FACTOID: question expects a single direct fact ("who is the <role> of X",
     "where is X located", "when did X occur", "what is the <attribute> of X").
   - **COMPOUND**: any question containing multiple sub-clauses joined by "and",
     "with", relative pronouns ("that", "which", "who"), or qualifying phrases
     that introduce additional constraints ("later <verb-ed>", "previously
     <verb-ed>", "ultimately <verb-ed>", "subsequently <verb-ed>"). Treat
     each sub-clause as a separate sub-query. The final answer must address
     EVERY sub-clause, not just the first or main one.
     Example structure: "Who <relation> X with Y, and what <kind-of-entity>
     had they previously <relation> together that was later <relation>?" has
     three sub-clauses: (a) co-actor with Y on X, (b) prior shared entity,
     (c) successor/acquirer/owner of (b). Answer must cover all three.
     Sub-clause (c) requires finding an edge in the graph for the qualifying
     relation that involves the entity from sub-clause (b).
2. **Extract matching facts.** Scan every graph fact line. For LIST questions, extract EVERY edge that matches the question's category — do not skip any. For RELATIONSHIP, find every edge connecting the two named entities directly OR through shared intermediate entities. For FACTOID, find the most-direct edge. For COMPOUND, extract facts for EVERY sub-clause; missing one sub-clause's evidence makes the answer incomplete even if the others are perfect.
3. **Synthesize the answer.**
   - **LIST:** produce a bulleted or comma-separated list with one citation per item.
   - **RELATIONSHIP:** gather ALL edges between the named entities (in either direction) and CONSOLIDATE them into 1-3 sentences that capture the canonical relationship plus any supporting details. Don't just list each edge separately — synthesize. The strongest relation (e.g. "acquired") should lead; supporting relations (e.g. "senior employees joined") add color. Multiple edges between the same pair are evidence for ONE consolidated answer.
   - **COMPOUND:** address every sub-clause sequentially in the answer. Walk the chain: name the answer to sub-clause (a), then connect to (b), then to (c). Cite each sub-clause's source. Do not stop after the first answer — the question is incomplete until every sub-clause is addressed.
     Example for "What is the relationship between Apple and NeXT?":
       Edges in graph: `Apple --[ACQUIRED_BY]--> NeXT (Steve Jobs)`, `Apple --[CAME_TO_A_DEAL_WITH]--> NeXT (Steve Jobs)`, `Senior Apple employees --[JOINED]--> NeXT (Steve Jobs)`.
       Consolidated answer: "Apple acquired NeXT, and as part of the deal several senior Apple employees joined NeXT (source: Steve Jobs)."
     **Bridge inference (no direct edge between the two entities):** when no edge directly links X and Y, look for an intermediate entity that connects to BOTH. The intermediate is the same real-world entity if it shares a QID across both edges.
     Example for "What is the relationship between Apple and Pixar?":
       Edges in graph: `Steven Paul Jobs [Q19837] --[founded]--> Apple Inc. [Q312]`, `Steven Paul Jobs [Q19837] --[purchased]--> Pixar [Q127552]`.
       Consolidated answer: "Apple and Pixar are connected through Steve Jobs, who co-founded Apple Inc. and later purchased Pixar (source: Steve Jobs)."
     Always prefer a bridge answer over refusing when an intermediate entity links both sides.
   - **FACTOID:** 1-2 sentence direct answer.
4. **Cite every claim.** Format: "<fact> (source: <article>)". When multiple sources are listed for one edge, include them all: "(sources: A, B)". When consolidating multiple edges from the same source, cite that source once at the end: "<consolidated sentence> (source: A)".
5. **Refuse on absence.** If the graph facts do not contain the requested information, reply exactly: "The provided graph facts do not contain information about <topic>." Do NOT fabricate or infer beyond the facts.

INTERNAL REASONING (do NOT output — apply silently, then write the final answer):

  LIST queries: mentally walk each candidate edge that could contribute
    to the list. For each: "Edge X → include / exclude / merged from
    drift → why". Apply the surface-form-drift exception explicitly in
    this walk: when an entity's name is a substring of the question's
    target (and overlapping context per the indicators above), MERGE
    them and INCLUDE that entity's items. Do not silently drop them.

  RELATIONSHIP queries: mentally walk each edge between (or through
    bridges connecting) the two named entities. Identify the canonical
    relation and the supporting relations.

  COMPOUND queries: mentally walk each sub-clause: (a), (b), (c)... For
    each, identify the supporting edge in the graph context. Do not skip
    sub-clauses; if a sub-clause has no supporting edge, that's the
    answer to that part.

The internal walk forces consistent reasoning. The visible answer must
reflect every "include" decision from the walk, but DO NOT print the
walk itself — output only the final concise answer.

OUTPUT FORMAT (mandatory; visible output only):

For LIST, emit ONLY:
  `ANSWER:` (bulleted list — one bullet per entity, citation inline in
   the form "- <Entity> (source: X)"; bullets ARE the facts; do not skip
   any candidate entity; do not add prose commentary)

For FACTOID and RELATIONSHIP, use two-pass:
  `RELEVANT FACTS:` (one bullet per edge; verbatim; no commentary)
  `ANSWER:` (synthesized prose; concise)

For COMPOUND, use two-pass:
  `RELEVANT FACTS:` (verbatim edges; one bullet per edge)
  `ANSWER:` (prose addressing every sub-clause; concise)

ENUMERATION COMPLETENESS RULES:
- For LIST: completeness over brevity. List EVERY candidate entity from the
  graph context that satisfies the question's relation, even if the list is
  long. Do not summarize ("...and several others"). Each entity gets its
  own bullet with one citation.
- For RELATIONSHIP / COMPOUND: FACTS section is one bullet per edge;
  ANSWER prose synthesizes them concisely.
- ANSWER must always be emitted last. If you are running low on output
  budget, drop FACTS detail (for non-LIST) before truncating ANSWER list.

Apply the surface-form-drift merge exception by including edges from BOTH
QIDs in your enumeration (LIST) or evidence (RELATIONSHIP/COMPOUND), and
mentioning both in ANSWER. Cite both QIDs once at the start of the
canonical entity reference (e.g. "<Entity> [Q1, also Q2]").

Example shape (RELATIONSHIP):
RELEVANT FACTS:
- Steven Paul Jobs [Q19837] --[founded]--> Apple Inc. [Q312]  (source: Steve Jobs)
- Steven Paul Jobs [Q19837] --[purchased]--> Pixar [Q127552]  (source: Steve Jobs)

ANSWER:
Apple and Pixar are connected through Steve Jobs ...

Example shape (COMPOUND):
THINKING:
Sub-clause (a) "Who co-founded A16Z with Marc Andreessen": graph has
  Ben Horowitz co-founded Andreessen Horowitz → answer (a) = Ben Horowitz
Sub-clause (b) "what enterprise software company had they previously co-founded":
  graph has Marc Andreessen co-founded Opsware AND Ben Horowitz cofounded Loudcloud
  (Loudcloud renamed to Opsware) → answer (b) = Opsware (formerly Loudcloud)
Sub-clause (c) "that was later acquired":
  graph has Hewlett-Packard acquired Opsware → answer (c) = Hewlett-Packard
Final chain: Ben Horowitz → Opsware/Loudcloud → Hewlett-Packard

RELEVANT FACTS:
- Ben Horowitz [Qxxx] --[co-founded]--> Andreessen Horowitz ...
- Marc Andreessen [Qxxx] --[co-founded]--> Opsware ...
- Ben Horowitz --[cofounded]--> Loudcloud ...
- Hewlett-Packard --[acquired]--> Opsware ...

ANSWER:
Ben Horowitz co-founded Andreessen Horowitz with Marc Andreessen. They had
previously co-founded the enterprise software company Opsware (originally
Loudcloud), which was later acquired by Hewlett-Packard."""
    resp = omlx.chat.completions.create(
        model=ANSWER_MODEL,  # prose synthesis model; defaults to MODEL_SONNET if MODEL_ANSWER unset
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Query: {query}\n\nGraph facts:\n{context}"},
        ],
        temperature=0.0, max_tokens=8000,  # Dense LIST queries with intermediate-expansion can need ample budget for full enumeration
    )
    raw = resp.choices[0].message.content or ""
    # Two-pass output: extract just the ANSWER block so the judge scores
    # synthesized prose, not the verbatim FACTS list (which would inflate
    # entity-mention recall via incidental copies of graph edges).
    if "ANSWER:" in raw:
        answer_text = raw.split("ANSWER:", 1)[1].strip()
    else:
        answer_text = raw  # fallback if LLM ignored the format
    return {
        "answer":           answer_text,
        "seeds":            seeds,
        "matches_per_seed": matches_per_seed,
        "edges_used":       len(subgraph),
    }


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Which companies are related to Mark Zuckerberg?"
    print(json.dumps(answer(q), indent=2))
