"""One-shot cleanup: flip reversed-direction triples in the existing graph.

Problem: ~190 high-confidence reversed-direction triples sit in the graph
(top: WAS_ACQUIRED_BY 54, WAS_FOUNDED_BY 39, FOUNDED_BY 35, WAS_SOLD_TO 33,
PUBLISHED_BY 32) where the subject is the linguistic patient instead of the
agent. Surfaced when v10c forward-fix removed the pair-aggregation that had
silently masked direction. Apple's `(Apple Inc.)-[ACQUIRED_BY]->(NeXT)` reads
as "Apple was acquired by NeXT" — factually backwards.

Approach: open-vocab compatible. Don't use a static verb table — let the LLM
decide per-predicate-type whether the active form requires inversion.

Pipeline:
  1. Discover all unique passive-voice predicates (rel_type matches WAS_* or
     ENDS WITH _BY). For each, fetch a sample triple.
  2. One LLM call per unique predicate type — gemma-4-26B JSON-mode prompt
     asks: "is this passive voice where subject is the patient? if yes,
     emit active_relation". Decision applies to ALL triples of that type.
  3. Bulk-rewrite Cypher per flipped predicate: swap (s, o), relabel
     rel_type, save `normalized_from` property as audit trail.

~30 LLM calls × ~3-5s = ~2 min. Audit trail is reversible — every flipped
edge has `normalized_from: <original_rel_type>` so the operation can be
undone via Cypher if a flip turns out wrong.

Run from lab-02-5-graphrag:
  ./.venv/bin/python src/normalize_passive_triples.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()
omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
MODEL = os.getenv("MODEL_SONNET", "gemma-4-26B-A4B-it-heretic-4bit")

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)

_SAFE_REL_TYPE = re.compile(r"^[A-Z][A-Z0-9_]*$")

_JUDGE_SYSTEM = """You are a linguistic judge for a knowledge-graph normalization pass.

Given a triple `subject --[relation]--> object` extracted from text, decide
whether the relation is PASSIVE voice where the SUBJECT is the linguistic
patient (action recipient) and OBJECT is the agent — meaning the triple is
stored backwards relative to canonical active-voice convention.

If yes: emit the active form. The triple should be flipped (swap subject
and object) and the predicate relabeled to active.
If no: keep as-is.

Examples:

Triple: subject="Apple Inc.", relation="acquired by", object="NeXT"
→ {"flip": true, "active_relation": "acquired"}
   (NeXT is the agent of acquisition; Apple is the patient. Wait — actually
    Apple acquired NeXT in 1996. The graph stored this BACKWARDS. Flip:
    "NeXT was acquired by Apple", active = "NeXT acquired Apple". Yes flip.)

Triple: subject="Microsoft", relation="founded by", object="Bill Gates"
→ {"flip": true, "active_relation": "founded"}
   (Bill Gates is the agent of founding. Active: "Bill Gates founded
   Microsoft". Flip subject/object and relabel.)

Triple: subject="John", relation="awarded", object="Nobel Prize"
→ {"flip": false, "active_relation": null}
   (John is the genuine recipient. The relationship describes John's
   achievement. Keep as-is.)

Triple: subject="Apple", relation="published by", object="Penguin"
→ {"flip": true, "active_relation": "published"}
   (Penguin is the publisher/agent. Active: "Penguin published Apple".)

Triple: subject="Larry Page", relation="was a member of", object="Stanford"
→ {"flip": false, "active_relation": null}
   (Membership is a state describing Larry Page. Patient-as-subject is
   the canonical form for state-of-being relations.)

Triple: subject="Bill Gates", relation="was born in", object="Seattle"
→ {"flip": false, "active_relation": null}
   (Birth-location is a state describing Bill Gates. Keep as-is.)

Output strict JSON only:
{"flip": bool, "active_relation": str | null, "rationale": str}

Rationale should be one sentence explaining the linguistic call. Use
linguistic judgment per triple — do NOT pattern-match on suffix alone.
WAS_* and *_BY predicates split into both flip and keep cases."""


def _safe_rel_type(active_relation: str) -> str:
    """Normalize a free-form active relation string to a Cypher-safe rel-type
    (uppercase, alphanum + underscore)."""
    cleaned = re.sub(r"[^A-Z_]", "_", active_relation.upper().replace(" ", "_"))[:40]
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "RELATED_TO"


def discover_passive_predicates(session, min_count: int = 1) -> list[dict]:
    """Find all rel_types matching passive patterns, with a sample triple
    for each. Returns list of {rel_type, count, sample_s, sample_raw, sample_o}.

    `min_count` filters the long tail. count=1-2 predicates each affect 1-2
    triples but cost the same per LLM call as count=50 predicates — Pareto
    skew. min_count=5 typically covers 80%+ of affected triples in 30-40
    LLM calls instead of 100+."""
    rows = list(session.run("""
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE (type(r) ENDS WITH '_BY' OR type(r) STARTS WITH 'WAS_')
          AND r.normalized_from IS NULL
        WITH type(r) AS rel_type, head(collect({s: a.name, raw: r.raw_relation, o: b.name})) AS sample, count(*) AS n
        WHERE n >= $min_count
        RETURN rel_type, sample, n
        ORDER BY n DESC
    """, min_count=min_count))
    return [
        {
            "rel_type": r["rel_type"],
            "count":    r["n"],
            "sample_s": r["sample"]["s"],
            "sample_raw": r["sample"]["raw"],
            "sample_o": r["sample"]["o"],
        }
        for r in rows
    ]


def judge_predicate(s: str, raw_rel: str, o: str) -> dict:
    """One LLM call per unique predicate. Returns {flip, active_relation, rationale}."""
    user_msg = (
        f'Triple: subject={s!r}, relation={raw_rel!r}, object={o!r}\n\n'
        f"Decide whether to flip."
    )
    try:
        resp = omlx.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as e:
        return {"flip": False, "active_relation": None, "rationale": f"judge error: {type(e).__name__}: {e}"}


def flip_predicate_in_graph(session, original_rel_type: str, active_relation: str) -> int:
    """Flip ALL triples of a given rel_type. Swap (s, o), relabel rel_type
    to the Cypher-safe active form, save original as `normalized_from`
    audit property. Returns count flipped."""
    new_rel_type = _safe_rel_type(active_relation)
    if not _SAFE_REL_TYPE.match(new_rel_type):
        print(f"  [skip] new rel_type {new_rel_type!r} unsafe — leaving as-is")
        return 0
    if new_rel_type == original_rel_type:
        print(f"  [skip] new rel_type {new_rel_type!r} identical to original — no-op")
        return 0

    # APOC-free pattern: collect old edges, create new flipped edges, delete old.
    # Use parameterized Cypher with dynamic CREATE via apoc.create.relationship
    # if APOC is available; else fall back to a manual loop in Python.
    try:
        # Try APOC path first
        result = session.run(f"""
            MATCH (a:Entity)-[r:{original_rel_type}]->(b:Entity)
            WHERE r.normalized_from IS NULL
            WITH r, a, b, properties(r) AS props
            CALL apoc.create.relationship(b, $new_type, props, a) YIELD rel
            WITH r, rel
            SET rel.normalized_from = $orig_type,
                rel.raw_relation     = $new_raw
            DELETE r
            RETURN count(rel) AS n
        """, new_type=new_rel_type, orig_type=original_rel_type, new_raw=active_relation)
        return result.single()["n"]
    except Exception as e:
        if "apoc" not in str(e).lower():
            raise
        # APOC unavailable — fall back to Python loop
        rows = list(session.run(f"""
            MATCH (a:Entity)-[r:{original_rel_type}]->(b:Entity)
            WHERE r.normalized_from IS NULL
            RETURN id(r) AS rid, a.name AS sname, b.name AS oname,
                   r.raw_relation AS raw, r.source_title AS src
        """))
        n = 0
        for row in rows:
            session.run(f"""
                MATCH (a:Entity {{name: $sname}}), (b:Entity {{name: $oname}})
                MATCH (a)-[r:{original_rel_type}]->(b) WHERE id(r) = $rid
                CREATE (b)-[r2:{new_rel_type} {{
                    raw_relation:    $new_raw,
                    source_title:    $src,
                    normalized_from: $orig_type
                }}]->(a)
                DELETE r
            """, sname=row["sname"], oname=row["oname"], rid=row["rid"],
                 new_raw=active_relation, src=row["src"], orig_type=original_rel_type)
            n += 1
        return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="show decisions without writing")
    ap.add_argument("--min-count", type=int, default=5,
                    help="skip predicates appearing in fewer than this many triples (default: 5)")
    args = ap.parse_args()

    print("=" * 76)
    print(f"normalize_passive_triples — {'DRY RUN' if args.dry_run else 'APPLY MODE'}")
    print("=" * 76)

    with driver.session() as session:
        predicates = discover_passive_predicates(session, min_count=args.min_count)
        print(f"\nFound {len(predicates)} unique passive-voice predicates "
              f"(filtered to count >= {args.min_count})")
        total_triples = sum(p["count"] for p in predicates)
        print(f"Total reversible triples in scope: {total_triples}\n")

        decisions: list[dict] = []
        for p in predicates:
            j = judge_predicate(p["sample_s"], p["sample_raw"], p["sample_o"])
            flip = bool(j.get("flip"))
            active = j.get("active_relation") or ""
            mark = "FLIP" if flip and active else "KEEP"
            print(f"  [{mark}] {p['rel_type']:<35} ({p['count']:>3}× ) "
                  f"sample: {p['sample_s'][:30]!r:<32} -[{p['sample_raw']!r:<25}]-> "
                  f"{p['sample_o'][:25]!r}")
            if flip and active:
                print(f"          → active: {active!r}")
            print(f"          rationale: {j.get('rationale','')[:160]}")
            decisions.append({**p, **j})

        print("\n" + "=" * 76)
        flips = [d for d in decisions if d.get("flip") and d.get("active_relation")]
        print(f"Decision: flip {len(flips)} predicates / {sum(d['count'] for d in flips)} triples")
        print(f"          keep {len(decisions) - len(flips)} predicates / "
              f"{total_triples - sum(d['count'] for d in flips)} triples as-is")
        print("=" * 76)

        if args.dry_run:
            print("\nDry-run mode — no writes. Re-run without --dry-run to apply.")
            return 0

        print("\nApplying flips...")
        t0 = time.time()
        total_flipped = 0
        for d in flips:
            n = flip_predicate_in_graph(session, d["rel_type"], d["active_relation"])
            print(f"  flipped {d['rel_type']} → {_safe_rel_type(d['active_relation'])}: {n} edges")
            total_flipped += n
        print(f"\nDone: {total_flipped} edges flipped in {time.time() - t0:.1f}s")

    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
