"""Repair pass v2 for v12 build cache — hybrid SPARQL batch + wbsearchentities.

Problem: v12 build hit Wikidata API rate limits at 16-way parallelism (31%
error rate, 9,746 poisoned-None entries). Repair pass v1 used wbsearchentities
at max_workers=4 → 2.10s/it → 13 hr ETA.

Solution: Two-phase hybrid approach (~40× speedup):

  Phase 1 — SPARQL VALUES batch (fast):
    Batch 22,960 names into 459 SPARQL queries of 50 names each.
    Each query hits query.wikidata.org (separate infra from the REST API —
    different rate-limit budget). Returns Wikidata items whose English label
    or altLabel exactly matches a name; picks highest-sitelink item when
    multiple match. Wall: ~5-20 min at 3 concurrent queries.

  Phase 2 — wbsearchentities fuzzy fallback (for SPARQL misses):
    SPARQL only finds exact label/altLabel matches. Names extracted as
    variants not registered as Wikidata altLabels fall back to the fuzzy
    search API. Typically ~20-40% of input. Wall: ~5-15 min at 8 workers.

  Total expected: ~10-35 min vs ~13 hr for all-API approach.

Idempotent: re-running picks up any names still erroring. Graph repair step
is unchanged from v1.

Run from lab-02-5-graphrag:
  ./.venv/bin/python src/repair_qid_cache.py
  ./.venv/bin/python src/repair_qid_cache.py --cache-only
  ./.venv/bin/python src/repair_qid_cache.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

WIKIDATA_API    = "https://www.wikidata.org/w/api.php"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT      = (
    "agent-prep-lab-graphrag-repair/0.2 "
    "(educational; two-phase SPARQL+API; yuxin71.liu@gmail.com)"
)

# SPARQL: 50 names/query × 3 concurrent = 150 names/round; 459 rounds for 22960 names.
# Wikimedia policy asks for polite crawling with a real User-Agent.
SPARQL_BATCH_SIZE  = 50
SPARQL_MAX_WORKERS = 3

# Fuzzy fallback concurrency. wbsearchentities anonymous limit ≈ 50 req/s.
# 8 workers gives ~16 req/s sustained — well within budget, avoids 429 chains.
FUZZY_MAX_WORKERS  = 8


# ── HTTP sessions ─────────────────────────────────────────────────────────────

def make_sparql_session() -> requests.Session:
    """Session for query.wikidata.org SPARQL — lightweight, no retry adapter
    (SPARQL errors handled per-batch, not per-name)."""
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"})
    return sess


def make_retry_session() -> requests.Session:
    """Session for wbsearchentities — retry-on-429/5xx with modest backoff.
    backoff_factor=0.5 is gentler than v1's 2.0: avoids 62-second chains."""
    sess = requests.Session()
    sess.headers["User-Agent"] = USER_AGENT
    retry = Retry(
        total=3,
        backoff_factor=0.5,                   # 0.5, 1.0, 2.0 s — gentle
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET"},
        respect_retry_after_header=True,
    )
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    return sess


# ── SPARQL phase ──────────────────────────────────────────────────────────────

def _sparql_escape(s: str) -> str:
    """Escape a name for use inside a SPARQL double-quoted string literal."""
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", " ").replace("\r", "").replace("\t", " ")
    return s.strip()


def sparql_batch_lookup(names: list[str], session: requests.Session) -> dict[str, str | None]:
    """Resolve a batch of entity names via a single SPARQL VALUES query.

    Returns {name: qid_or_None}. Names with no exact English label/altLabel
    match map to None (will be routed to fuzzy fallback). On network/parse
    error, all names map to 'ERROR' so they fall through to fuzzy as well.

    Disambiguation: when multiple Wikidata items share the same English label,
    picks the one with the most English Wikipedia sitelinks (notability proxy).
    """
    # Filter empty names before building query
    clean = [n for n in names if n and n.strip()]
    if not clean:
        return {n: None for n in names}

    values_str = " ".join(f'"{_sparql_escape(n)}"@en' for n in clean)

    # POST avoids URL-length limits for large VALUES blocks.
    # wikibase:sitelinks is a pre-computed integer on every item — no GROUP BY
    # needed and no risk of the schema:about approach undercounting.
    query = f"""
SELECT ?item ?matchLabel ?slinks WHERE {{
  VALUES ?matchLabel {{ {values_str} }}
  {{
    ?item rdfs:label ?matchLabel .
  }} UNION {{
    ?item skos:altLabel ?matchLabel .
  }}
  ?item wikibase:sitelinks ?slinks .
}}
ORDER BY ?matchLabel DESC(?slinks)
"""
    try:
        r = session.post(
            SPARQL_ENDPOINT,
            data={"query": query},
            timeout=45,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return {n: "ERROR" for n in clean}

    # First row per label = highest sitelinks (ORDER BY DESC)
    found: dict[str, str] = {}
    for row in data["results"]["bindings"]:
        label = row["matchLabel"]["value"]
        qid   = row["item"]["value"].rsplit("/", 1)[-1]   # URI → "Q12345"
        if label not in found:
            found[label] = qid

    # Names missing from SPARQL results → None (fuzzy fallback)
    result: dict[str, str | None] = {}
    for n in clean:
        result[n] = found.get(n, None)

    # Pass through any original names that were empty
    for n in names:
        if n not in result:
            result[n] = None
    return result


# ── wbsearchentities phase ────────────────────────────────────────────────────

def lookup_qid(session: requests.Session, name: str, timeout_s: float = 12.0) -> str | None:
    """One wbsearchentities call. Returns QID, None (no match), or 'ERROR'."""
    try:
        r = session.get(
            WIKIDATA_API,
            params={
                "action":   "wbsearchentities",
                "search":   name,
                "language": "en",
                "type":     "item",
                "limit":    1,
                "format":   "json",
            },
            timeout=timeout_s,
        )
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        return "ERROR"

    results = data.get("search", [])
    if not results:
        return None
    return results[0].get("id")


# ── main repair logic ─────────────────────────────────────────────────────────

def repair_cache(cache_path: Path, dry_run: bool = False) -> dict:
    """Two-phase cache repair: SPARQL batch first, wbsearchentities for misses.

    Returns {recovered, still_null, still_error}."""
    cache: dict[str, str | None] = json.loads(cache_path.read_text())
    nones = sorted(k for k, v in cache.items() if v is None)

    print(f"Cache size:         {len(cache)}")
    print(f"  with QID:         {sum(1 for v in cache.values() if v is not None)}")
    print(f"  with None:        {len(nones)}  (re-resolution candidates)")

    if not nones:
        print("\nNothing to repair.")
        return {"recovered": {}, "still_null": set(), "still_error": set()}

    # ── Phase 1: SPARQL batch ────────────────────────────────────────────────
    sparql_batches = [nones[i:i + SPARQL_BATCH_SIZE]
                      for i in range(0, len(nones), SPARQL_BATCH_SIZE)]
    print(f"\nPhase 1: SPARQL batch")
    print(f"  {len(nones)} names → {len(sparql_batches)} batches "
          f"× {SPARQL_BATCH_SIZE} names, {SPARQL_MAX_WORKERS} concurrent")

    sparql_sess = make_sparql_session()
    sparql_found:  dict[str, str] = {}      # name → QID (exact label match)
    sparql_miss:   list[str]      = []      # name → None (no exact label)
    sparql_errors: list[str]      = []      # name → ERROR (batch failed)

    with ThreadPoolExecutor(max_workers=SPARQL_MAX_WORKERS) as pool:
        futures = {
            pool.submit(sparql_batch_lookup, batch, sparql_sess): batch
            for batch in sparql_batches
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="SPARQL batches"):
            try:
                batch_result = fut.result()
            except Exception:
                sparql_errors.extend(futures[fut])
                continue
            for name, qid in batch_result.items():
                if qid == "ERROR":
                    sparql_errors.append(name)
                elif qid is None:
                    sparql_miss.append(name)
                else:
                    sparql_found[name] = qid

    print(f"\n  SPARQL results:")
    print(f"    Found QID:    {len(sparql_found)}  "
          f"({len(sparql_found) / max(len(nones), 1) * 100:.1f}% of Nones)")
    print(f"    No exact match: {len(sparql_miss)}  → fuzzy fallback")
    print(f"    Batch errors:   {len(sparql_errors)}  → fuzzy fallback")

    # ── Phase 2: wbsearchentities fuzzy fallback ─────────────────────────────
    fuzzy_targets = sparql_miss + sparql_errors
    print(f"\nPhase 2: wbsearchentities fuzzy fallback")
    print(f"  {len(fuzzy_targets)} names, {FUZZY_MAX_WORKERS} workers, "
          f"backoff_factor=0.5")

    api_sess = make_retry_session()
    fuzzy_found:  dict[str, str] = {}
    fuzzy_null:   set[str]       = set()
    fuzzy_errors: set[str]       = set()

    with ThreadPoolExecutor(max_workers=FUZZY_MAX_WORKERS) as pool:
        futures = {pool.submit(lookup_qid, api_sess, name): name
                   for name in fuzzy_targets}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="fuzzy"):
            name = futures[fut]
            try:
                result = fut.result()
            except Exception:
                fuzzy_errors.add(name)
                continue
            if result == "ERROR":
                fuzzy_errors.add(name)
            elif result is None:
                fuzzy_null.add(name)
            else:
                fuzzy_found[name] = result

    print(f"\n  Fuzzy results:")
    print(f"    Found QID:  {len(fuzzy_found)}")
    print(f"    No match:   {len(fuzzy_null)}")
    print(f"    Errors:     {len(fuzzy_errors)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    recovered = {**sparql_found, **fuzzy_found}
    still_null  = fuzzy_null
    still_error = fuzzy_errors

    print(f"\n{'─'*60}")
    print(f"Total recovered:   {len(recovered)} / {len(nones)} "
          f"({len(recovered) / max(len(nones), 1) * 100:.1f}%)")
    print(f"  via SPARQL:      {len(sparql_found)}")
    print(f"  via fuzzy:       {len(fuzzy_found)}")
    print(f"Genuinely null:    {len(still_null)}")
    print(f"Still erroring:    {len(still_error)}")

    if not dry_run and recovered:
        for name, qid in recovered.items():
            cache[name] = qid
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
        tmp.replace(cache_path)
        print(f"\nCache updated: {len(recovered)} entries None → QID")
    elif dry_run:
        print(f"\nDRY RUN — cache not modified")

    return {
        "recovered":   recovered,
        "still_null":  still_null,
        "still_error": still_error,
    }


# ── graph repair (unchanged from v1) ─────────────────────────────────────────

def repair_graph(driver, recovered: dict[str, str], dry_run: bool = False) -> dict:
    """For each name now resolved to a QID, merge the name-keyed Entity into
    the QID-keyed counterpart (creating one if needed).

    Case A — both `Entity {name: X}` and `Entity {qid: Y}` exist:
        apoc.refactor.mergeNodes folds name node into QID node; name added
        to aliases.
    Case B — only `Entity {name: X}` exists:
        Promote in place: set qid + initialize aliases.
    Case C — name node not in graph:
        No-op.

    Returns {merged, promoted, missing}."""
    merged = promoted = missing = 0

    print(f"\nGraph repair: {len(recovered)} names to merge into QID-keyed nodes")
    if dry_run:
        print("DRY RUN — no Cypher writes")

    if not recovered:
        return {"merged": merged, "promoted": promoted, "missing": missing}

    with driver.session() as session:
        for name, qid in tqdm(recovered.items(), desc="graph repair"):
            row = session.run("""
                OPTIONAL MATCH (a:Entity {name: $name}) WHERE a.qid IS NULL
                WITH a
                OPTIONAL MATCH (b:Entity {qid: $qid})
                RETURN a IS NOT NULL AS has_name_only, b IS NOT NULL AS has_qid
            """, name=name, qid=qid).single()

            has_name_only = row["has_name_only"] if row else False
            has_qid       = row["has_qid"]       if row else False

            if dry_run:
                if has_name_only and has_qid:
                    merged   += 1
                elif has_name_only:
                    promoted += 1
                else:
                    missing  += 1
                continue

            if has_name_only and has_qid:
                try:
                    session.run("""
                        MATCH (a:Entity {name: $name}) WHERE a.qid IS NULL
                        MATCH (b:Entity {qid: $qid})
                        CALL apoc.refactor.mergeNodes(
                            [b, a],
                            {properties: 'discard', mergeRels: true}
                        ) YIELD node
                        SET node.aliases = CASE
                            WHEN $name IN coalesce(node.aliases, []) THEN node.aliases
                            ELSE coalesce(node.aliases, []) + $name
                        END
                        RETURN node
                    """, name=name, qid=qid)
                    merged += 1
                except Exception as e:
                    print(f"  [skip] merge {name!r} → {qid}: {type(e).__name__}: {e}")
            elif has_name_only:
                try:
                    session.run("""
                        MATCH (a:Entity {name: $name}) WHERE a.qid IS NULL
                        SET a.qid = $qid,
                            a.aliases = CASE
                                WHEN a.aliases IS NULL OR size(a.aliases) = 0
                                THEN [a.name]
                                ELSE a.aliases
                            END
                    """, name=name, qid=qid)
                    promoted += 1
                except Exception as e:
                    print(f"  [skip] promote {name!r} → {qid}: {type(e).__name__}: {e}")
            else:
                missing += 1

    print(f"\nGraph repair complete:")
    print(f"  Merged   (Case A — both nodes existed): {merged}")
    print(f"  Promoted (Case B — name-only → qid):    {promoted}")
    print(f"  Missing  (Case C — name not in graph):  {missing}")
    return {"merged": merged, "promoted": promoted, "missing": missing}


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Repair v12 QID cache — hybrid SPARQL+API")
    ap.add_argument("--dry-run",    action="store_true",
                    help="show action plan without modifying cache or graph")
    ap.add_argument("--cache-only", action="store_true",
                    help="re-resolve cache only; skip graph repair")
    args = ap.parse_args()

    print("=" * 76)
    print(f"repair_qid_cache v2 — {'DRY RUN' if args.dry_run else 'APPLY MODE'}")
    print(f"Strategy: SPARQL batch (Phase 1) + wbsearchentities fuzzy (Phase 2)")
    print("=" * 76)

    cache_path = Path("data/wikidata_qid_cache.json")
    if not cache_path.exists():
        print(f"ERROR: cache not found at {cache_path}", file=sys.stderr)
        return 1

    t0 = time.time()
    result = repair_cache(cache_path, dry_run=args.dry_run)
    t_cache = time.time() - t0
    print(f"\nCache repair wall: {t_cache:.1f}s ({t_cache / 60:.1f} min)")

    if args.cache_only:
        print("\n--cache-only flag set — skipping graph repair.")
        return 0

    if not result["recovered"]:
        print("\nNo recoveries — graph repair is a no-op. Done.")
        return 0

    load_dotenv()
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
    )

    t0 = time.time()
    repair_graph(driver, result["recovered"], dry_run=args.dry_run)
    t_graph = time.time() - t0
    print(f"\nGraph repair wall: {t_graph:.1f}s ({t_graph / 60:.1f} min)")

    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
