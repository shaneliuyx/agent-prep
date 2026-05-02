"""Extract (entity, relationship, entity) triples from each article and
write them to Neo4j.

v12 — Wikidata QID linking at extraction time.

Builds on v11 (sentence-aware sliding-window extraction + active-voice
prompt + index-lifecycle fix). v11's main remaining bottleneck was
multi-hop recall ceiling at 0.21-0.34, diagnosed as entity surface-form
fragmentation (Bad-Case Journal Entry 10): "Reid Hoffman" and "Reid
Garrett Hoffman" became separate nodes, splitting edges and breaking
2-hop chains traversing the person.

v12 fix: resolve every extracted entity-name to its canonical Wikidata
QID via wbsearchentities API, then MERGE on QID instead of name. Two
articles mentioning the same person under different surface forms now
write to ONE node — restoring the multi-hop chain.

  Article A: "Reid Hoffman, was_employee_of, PayPal"
             → resolve "Reid Hoffman" → Q211098
             → MERGE (a:Entity {qid: "Q211098"})
  Article B: "Reid Garrett Hoffman, co_founded, LinkedIn"
             → resolve "Reid Garrett Hoffman" → Q211098
             → MERGE matches SAME node (qid match)
  Result: PayPal ← Q211098 → LinkedIn — 2-hop chain reconstructed.

Build budget delta: ~2 min added on first run (~13K unique entity-names
× ~10ms cached API call). Subsequent runs hit disk cache, instant.

Earlier-version backstory (v10): truncated each article to 3500 chars
and asked the LLM for 5-20 triples. This silently dropped ~80% of long
Wikipedia articles (Education, Personal Life, Awards sections past
char 5000) and capped GraphRAG hit rate on bio-event questions. v10
moved to:
  - Sentence-aware sliding windows (~3000 chars, ~500 char overlap)
  - 10-15 triples per window (aggregate ~70-100 for long bios, 3-5× v9)
  - Per-article progress logs with window count + triple density
  - End-of-build proxy metrics so operator distinguishes
    "extraction worked / retrieval bottlenecked" from
    "extraction broken upstream"

Build budget: ~40 min wall on M5 Pro / Gemma-4-26B / MAX_WORKERS=6
for 400 articles × ~7 windows each = ~2800 LLM extraction calls,
plus ~2 min for QID resolution on first build."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from tqdm import tqdm

# Sentence-aware chunker — canonical impl in shared/rag_hybrid/chunking.py.
# Local helpers were removed in phase 7 of the rag_hybrid refactor.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from rag_hybrid.chunking import sentence_window_chunks  # noqa: E402

# Local v12 module — Wikidata QID resolver with disk-backed cache.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wikidata_qid import QIDResolver  # noqa: E402

# LLM-server max_concurrent_requests = 8. Keep workers strictly under that
# so query_graph.py / IDE autocomplete / ad-hoc queries don't queue behind
# the build. If the build runs in isolation, raising MAX_WORKERS to 8 gives
# the full 3-5× speedup; raising past 8 idles threads in server-side queue.
MAX_WORKERS = 6

# Sliding-window chunking parameters.
WINDOW_CHARS = 3000
WINDOW_OVERLAP_CHARS = 500
MIN_WINDOW_CHARS = 200  # tail windows shorter than this are dropped (low signal)

load_dotenv()
omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
MODEL = os.getenv("MODEL_SONNET")
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)

# Updated prompt: per-window cap 10-15, broader category guidance so the
# LLM emits biographical events ("dropped out of", "married", "donated to")
# alongside corporate relations ("founded", "acquired by"). The original
# example list (founded / acquired by / born in) biased the LLM toward
# affiliation/ownership predicates and silently skipped life-event facts.
EXTRACT_SYSTEM = """Extract entities and relationships from the text segment.
Output JSON only: {"triples": [{"subject": str, "relation": str, "object": str}, ...]}.

Rules:
- Use the exact surface form that appears in the text for subject/object.
- **Always emit triples in ACTIVE voice.** If the source text is passive ("Apple was acquired by NeXT"), invert subject/object so the agent is the subject: emit subject="NeXT", relation="acquired", object="Apple". Applies to all by-suffix passives ("X was founded by Y", "X was published by Y", "X was sold to Y"). For passives where the subject is genuinely the patient ("John was awarded the Nobel Prize", "John was named CEO"), keep subject=John — the relationship describes John, not the agent. Use linguistic judgment per triple, not a fixed list.
- Relations are 1-4-word verb phrases. Include BOTH:
  * Corporate / affiliation relations: "founded", "acquired by", "co-founded",
    "led", "merged with", "invested in", "joined", "left".
  * Biographical / life events: "dropped out of", "graduated from", "married",
    "divorced", "donated to", "was sued by", "testified before", "moved to",
    "served as", "studied at".
  * Education / employment: "attended", "earned a degree from", "worked at",
    "interned at", "served on the board of".
- Subject and object must be proper named entities: people, companies, organizations, universities, products, or places. Every entity name must begin with a capital letter (proper noun). Do NOT emit: lowercase descriptive phrases ("the gathering of business intelligence", "integrating payment services"), monetary amounts ("a $22.5 million fundraising effort"), percentages, numeric quantities, event descriptions, bare noun phrases, or job titles as standalone entities. Do NOT emit book titles, article headlines, documentary names, or publication titles — only the author or publisher.
- If a subject or object is a comma-separated list of named entities (e.g. "Yoshua Bengio, Yann LeCun, and Geoffrey Hinton"), emit one triple per entity — never use a comma-separated list as a single subject or object value.
- Drop leading articles from entity names: write "New York Times" not "The New York Times", "Los Angeles" not "the Los Angeles area".
- Include 10-15 triples per text segment. Skip if the segment has no clear entities.
- Do not invent facts. Every triple must be supported by the segment text.
- A single segment may repeat facts that appeared in earlier segments — that's
  fine, MERGE will dedupe."""


def extract_triples(text: str) -> list[dict]:
    """Run one extraction call on a single text window."""
    resp = omlx.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user",   "content": text},
        ],
        temperature=0.1,
        max_tokens=1500,  # bumped from 1200 to fit 10-15 triples per window
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content).get("triples", [])
    except (json.JSONDecodeError, AttributeError, TypeError):
        return []


def write_triples_to_neo4j(
    tx,
    article_id: str,
    article_title: str,
    triples: list[dict],
    qid_map: dict[str, str | None],
):
    """Each entity is a node, each triple creates a relationship.

    v12: MERGE on Wikidata QID when resolvable, else fall back to name MERGE.
    Two surface forms of the same canonical entity (e.g. "Reid Hoffman" /
    "Reid Garrett Hoffman" both → Q211098) write to ONE node — restoring
    multi-hop chains that v11 broke across alias boundaries.

    Implementation uses `apoc.merge.node` for clean dynamic-key MERGE: pass
    `{qid: <Q-id>}` when QID is resolvable, else `{name: <surface form>}`.
    APOC handles the conditional MERGE without Cypher branching gymnastics.

    `qid_map` is built upstream by the worker (one resolver call per unique
    entity name in the article), so this function does no API I/O.
    """
    for t in triples:
        s, r, o = t.get("subject"), t.get("relation"), t.get("object")
        if not (s and r and o):
            continue
        rel_type = re.sub(r'[^A-Z_]', '_', r.upper().replace(' ', '_'))[:40] or "RELATED_TO"

        s_qid = qid_map.get(s)
        o_qid = qid_map.get(o)

        # Build identity properties dynamically. When QID present, MERGE on
        # qid only; the surface form goes into `aliases` so we can audit
        # which forms collapsed into the canonical node. When QID absent,
        # MERGE on name (pre-v12 fallback path — preserves baseline for
        # entities not in Wikidata).
        s_keys = {"qid": s_qid} if s_qid else {"name": s}
        o_keys = {"qid": o_qid} if o_qid else {"name": o}

        # ON-CREATE / ON-MATCH props.  ON-CREATE seeds aliases=[surface_form]
        # and `name` as the first surface form seen.  ON-MATCH appends the
        # surface form to aliases (deduped) and never overwrites the
        # canonical `name` (first writer wins, but the alias list captures
        # every variant).
        s_on_create = (
            {"name": s, "aliases": [s]} if s_qid else {}
        )
        o_on_create = (
            {"name": o, "aliases": [o]} if o_qid else {}
        )

        tx.run(
            f"""
            CALL apoc.merge.node(['Entity'], $s_keys, $s_on_create, {{}}) YIELD node AS a
            CALL apoc.merge.node(['Entity'], $o_keys, $o_on_create, {{}}) YIELD node AS b
            // Append surface form to aliases for QID-keyed nodes (idempotent
            // — only adds if not already present)
            FOREACH (_ IN CASE WHEN $s_qid IS NOT NULL AND NOT $s_name IN coalesce(a.aliases, []) THEN [1] ELSE [] END |
                SET a.aliases = coalesce(a.aliases, []) + $s_name
            )
            FOREACH (_ IN CASE WHEN $o_qid IS NOT NULL AND NOT $o_name IN coalesce(b.aliases, []) THEN [1] ELSE [] END |
                SET b.aliases = coalesce(b.aliases, []) + $o_name
            )
            MERGE (a)-[rel:{rel_type}]->(b)
            ON CREATE SET rel.source_article = $aid,
                          rel.source_title   = $title,
                          rel.raw_relation   = $r
            """,
            s_keys=s_keys, o_keys=o_keys,
            s_on_create=s_on_create, o_on_create=o_on_create,
            s_qid=s_qid, o_qid=o_qid,
            s_name=s, o_name=o,
            aid=article_id, title=article_title, r=r,
        )


def _write_degree_centrality() -> None:
    """Write total relationship count to n.degree on every Entity node.

    query_graph.py's composite scorer uses log(degree+1)*DEGREE_COEFF to
    prefer high-degree canonical nodes over low-degree noise nodes that
    happen to share a BM25-relevant token.

    Tries GDS gds.degree.write first (one-pass O(n)). Falls back to a plain
    Cypher COUNT query if GDS is unavailable — correct but slower. Both paths
    are wrapped in try/except so a GDS failure never aborts the build.
    """
    with driver.session() as s:
        try:
            graph_name = "entity-degree-build"
            existing = s.run(
                "CALL gds.graph.list() YIELD graphName RETURN graphName"
            ).data()
            if not any(r["graphName"] == graph_name for r in existing):
                s.run(
                    """
                    CALL gds.graph.project(
                        $name, 'Entity',
                        {__ALL__: {type: '*', orientation: 'UNDIRECTED'}}
                    )
                    """,
                    name=graph_name,
                ).consume()
            s.run(
                "CALL gds.degree.write($name, {writeProperty: 'degree'}) "
                "YIELD nodePropertiesWritten",
                name=graph_name,
            ).consume()
            s.run("CALL gds.graph.drop($name, false)", name=graph_name).consume()
            print("[build] degree centrality written via GDS gds.degree.write")
        except Exception as gds_err:
            print(
                f"[build] GDS degree.write failed ({type(gds_err).__name__}: {gds_err}); "
                "falling back to Cypher COUNT ...",
                file=sys.stderr,
            )
            try:
                s.run(
                    """
                    MATCH (n:Entity)
                    OPTIONAL MATCH (n)-[r]-()
                    WITH n, count(DISTINCT r) AS deg
                    SET n.degree = deg
                    """
                ).consume()
                print("[build] degree centrality written via Cypher COUNT fallback")
            except Exception as cypher_err:
                print(
                    f"[build] degree fallback also failed: "
                    f"{type(cypher_err).__name__}: {cypher_err}",
                    file=sys.stderr,
                )


def _extract_one(
    article: dict,
    resolver: QIDResolver | None = None,
) -> tuple[dict, list[dict], dict[str, str | None], int, Exception | None]:
    """Worker — runs sliding-window extraction over one article, then
    resolves every unique entity-name to its Wikidata QID.

    Returns (article, all_triples, qid_map, n_windows, exc).
    `qid_map` maps each unique surface form to its QID or None.
    Exception captured rather than raised so a single failure does not
    kill the pool."""
    try:
        windows = sentence_window_chunks(
            article["text"],
            target_chars=WINDOW_CHARS,
            overlap_chars=WINDOW_OVERLAP_CHARS,
            min_window_chars=MIN_WINDOW_CHARS,
        )
        all_triples: list[dict] = []
        for window in windows:
            triples = extract_triples(window)
            all_triples.extend(triples)

        # Collect unique entity names across all triples for batch resolution.
        # ~70-100 triples per article → ~140-200 mentions → ~50-300 unique
        # names typically (subjects + objects share heavily).
        # Use resolve_batch — concurrent HTTPS calls to Wikidata in a thread
        # pool, ~16-way parallelism. Per-name serial resolution made each
        # article ~10-30s slower; batch drops uncached resolution time to
        # ~5-10s/article.
        qid_map: dict[str, str | None] = {}
        if resolver is not None:
            unique_names: set[str] = set()
            for t in all_triples:
                if t.get("subject"):
                    unique_names.add(t["subject"])
                if t.get("object"):
                    unique_names.add(t["object"])
            qid_map = resolver.resolve_batch(unique_names)

        return article, all_triples, qid_map, len(windows), None
    except Exception as exc:  # noqa: BLE001 — we want all errors here
        return article, [], {}, 0, exc


def main() -> None:
    corpus = json.loads(Path("data/corpus.json").read_text())
    t0 = time.time()
    total_triples = 0
    errors: list[tuple[str, Exception]] = []
    triples_per_article: list[int] = []
    windows_per_article: list[int] = []
    predicate_counts: Counter[str] = Counter()
    article_chars = [len(a.get("text", "")) for a in corpus]

    # v12: shared QID resolver across all workers. Disk-backed cache so
    # rebuilds don't re-pay the API cost.
    resolver = QIDResolver(Path("data/wikidata_qid_cache.json"))

    print(
        f"Build start: {len(corpus)} articles, "
        f"avg {sum(article_chars) // max(len(article_chars), 1)} chars/article, "
        f"max {max(article_chars) if article_chars else 0} chars, "
        f"window={WINDOW_CHARS}c overlap={WINDOW_OVERLAP_CHARS}c, "
        f"MAX_WORKERS={MAX_WORKERS}, "
        f"qid_cache_seeded={len(resolver.cache)}"
    )

    with driver.session() as session:
        # Clear previous runs — safe for a lab, not safe for production.
        session.run("MATCH (n) DETACH DELETE n")
        # Create the full-text index BEFORE extraction, not after. Earlier
        # versions did DROP-then-extract-then-CREATE; if extraction crashed
        # mid-loop the DROP had already executed but CREATE never ran,
        # leaving the graph queryable but un-searchable. The IF NOT EXISTS
        # form makes CREATE idempotent — safe to run on a fresh DB or when
        # rebuilding. Index over Entity.name (replaces v6's CONTAINS
        # substring matching that produced "meta" → "metal" false positives).
        session.run(
            "CREATE FULLTEXT INDEX entity_names IF NOT EXISTS "
            "FOR (n:Entity) ON EACH [n.name, n.aliases]"
        )
        # v12: also index Entity.qid (used as canonical key in MERGE). A
        # range index speeds up qid-based MERGE during the build (~13K
        # name lookups otherwise become full label scans). IF NOT EXISTS
        # makes this idempotent across rebuilds.
        session.run(
            "CREATE INDEX entity_qid IF NOT EXISTS "
            "FOR (n:Entity) ON (n.qid)"
        )
        session.run(
            "CREATE INDEX entity_name IF NOT EXISTS "
            "FOR (n:Entity) ON (n.name)"
        )

        # Threaded extraction across articles. Each worker walks all windows
        # of its article serially (no nested parallelism) so we keep the
        # inference server's 8 slots saturated by 6 articles in flight.
        # Resolver is shared across workers (thread-safe).
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(_extract_one, article, resolver)
                for article in corpus
            ]
            done = 0
            for fut in tqdm(as_completed(futures), total=len(corpus), desc="articles"):
                article, triples, qid_map, n_windows, exc = fut.result()
                done += 1
                if exc is not None:
                    errors.append((article["title"], exc))
                    tqdm.write(
                        f"  [{done}/{len(corpus)}] ERROR {article['title']!r}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                if triples:
                    session.execute_write(
                        write_triples_to_neo4j,
                        article["id"], article["title"], triples, qid_map,
                    )
                total_triples += len(triples)
                triples_per_article.append(len(triples))
                windows_per_article.append(n_windows)
                predicate_counts.update(t.get("relation", "") for t in triples if t.get("relation"))
                # Per-article progress line — every article emits one log so
                # the operator sees progress and per-article triple density.
                # tqdm.write doesn't break the progress bar.
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta_s = (len(corpus) - done) / max(rate, 1e-6)
                qid_mapped = sum(1 for v in qid_map.values() if v is not None)
                tqdm.write(
                    f"  [{done:>3}/{len(corpus)}] "
                    f"chars={len(article.get('text', '')):>5} "
                    f"windows={n_windows:>2} "
                    f"triples={len(triples):>3} "
                    f"qid={qid_mapped}/{len(qid_map)} "
                    f"total_triples={total_triples:>5} "
                    f"rate={rate * 60:.1f}/min ETA={eta_s/60:.1f}min  "
                    f"{article['title']!r}"
                )

    # Persist cache one last time (worker writes are best-effort interim;
    # final save is the source of truth for next-run cache).
    resolver.save()

    # Write degree centrality — must run after all MERGE writes are committed
    # so relationship counts are final. Populates n.degree for composite scorer.
    _write_degree_centrality()

    elapsed = time.time() - t0
    rs = resolver.stats()
    print()
    print("=" * 72)
    print("INGEST SUMMARY")
    print("=" * 72)
    print(f"Articles ingested:           {len(corpus)}")
    print(f"Total triples extracted:     {total_triples}")
    if triples_per_article:
        avg_t = sum(triples_per_article) / len(triples_per_article)
        avg_w = sum(windows_per_article) / len(windows_per_article)
        print(f"Triples per article (avg):   {avg_t:.1f}")
        print(f"Triples per article (max):   {max(triples_per_article)}")
        print(f"Windows per article (avg):   {avg_w:.1f}")
        print(f"Windows per article (max):   {max(windows_per_article)}")
    print(f"Unique relation predicates:  {len(predicate_counts)}")
    print(f"Top 10 predicates:           {predicate_counts.most_common(10)}")
    print(f"Wall time:                   {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"Extraction rate:             {total_triples / elapsed:.1f} triples/sec  (MAX_WORKERS={MAX_WORKERS})")
    print(f"Full-text index:             entity_names (over Entity.name, Entity.aliases)")
    print()
    print("WIKIDATA QID RESOLUTION")
    print("-" * 72)
    print(f"Unique entity names seen:    {rs['cache_size']}")
    print(f"Names mapped to QID:         {rs['names_mapped']}  ({rs['qid_coverage'] * 100:.1f}%)")
    print(f"Names without QID:           {rs['names_unmapped']}  (fall back to name MERGE)")
    print(f"API calls:                   {rs['api_calls']}")
    print(f"Cache hits:                  {rs['cache_hits']}  ({rs['cache_hit_rate'] * 100:.1f}%)")
    print(f"API errors:                  {rs['api_errors']}")
    print(f"Cache file:                  data/wikidata_qid_cache.json")
    if errors:
        print(f"\nExtraction errors ({len(errors)}):")
        for title, exc in errors[:10]:
            print(f"  {title!r}: {type(exc).__name__}: {exc}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
