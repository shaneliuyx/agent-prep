"""Reverse-edge inference: for each forward predicate, add the inverse direction.

Closes Leak 5 partially — when the extractor surfaces `Apple --[acquired]--> NeXT`
but doesn't also extract `NeXT --[acquired_by]--> Apple`, multi-hop questions
that traverse from NeXT outward miss the connection. This script:

  1. Discovers all forward predicates in the graph (skipping prior-inferred reverses)
  2. LLM judge per predicate type — "what's the natural inverse, or null if no inverse"
  3. Bulk-creates reverse edges via apoc.create.relationship; audit via
     `inferred_reverse_of: <forward_rel_type>` property

Open-vocab. No verb-pair table — LLM applies linguistic judgment per predicate.
Per-predicate (not per-triple): ~30 LLM calls cover thousands of triples.

Decision categories the judge handles:
  - Inversible:    "acquired" → "acquired_by", "co-founded" → "co-founder_of"
  - Symmetric:     "married_to", "merged_with" — no separate inverse needed
  - State-of-being: "was_president_of", "was_born_in" — already on subject
  - Already inverse: "ACQUIRED_BY" itself — caller passes only forward types

Reversible: every inferred edge has `inferred_reverse_of` so they're
distinguishable. To roll back: `MATCH ()-[r]->() WHERE r.inferred_reverse_of
IS NOT NULL DETACH DELETE r`.

Run from lab-02-5-graphrag:
  ./.venv/bin/python src/infer_reverse_edges.py --dry-run
  ./.venv/bin/python src/infer_reverse_edges.py [--min-count 5]
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

_JUDGE_SYSTEM = """You are a linguistic judge for a knowledge-graph reverse-edge inference pass.

Given a forward edge `subject --[relation]--> object`, decide whether to add an
inverse-direction edge `object --[inverse_relation]--> subject` capturing the
SAME fact from the other side.

Three decision categories:

A) **INVERSIBLE** — the forward relation has a clean inverse phrasing.
   Output: {"add_inverse": true, "inverse_relation": "<active inverse>"}

B) **SYMMETRIC** — the relation is reciprocal; querying from either side
   sees the same edge regardless of direction.
   Output: {"add_inverse": false, "inverse_relation": null, "rationale": "symmetric"}

C) **STATE-OF-BEING / NOT-INVERSIBLE** — subject is the natural focal entity;
   inverse phrasing would be awkward or meaningless.
   Output: {"add_inverse": false, "inverse_relation": null, "rationale": "state-of-being"}

Examples:

Triple sample: subject="Apple Inc.", relation="acquired", object="NeXT"
{"add_inverse": true, "inverse_relation": "acquired_by"}

Triple sample: subject="Steve Jobs", relation="co-founded", object="Apple"
{"add_inverse": true, "inverse_relation": "co-founder_of"}

Triple sample: subject="Bill Gates", relation="founded", object="Microsoft"
{"add_inverse": true, "inverse_relation": "founded_by"}

Triple sample: subject="Reid Hoffman", relation="invested in", object="Airbnb"
{"add_inverse": true, "inverse_relation": "received_investment_from"}

Triple sample: subject="Brad Pitt", relation="married to", object="Angelina Jolie"
{"add_inverse": false, "inverse_relation": null, "rationale": "symmetric — querying either side returns the same fact"}

Triple sample: subject="Bill Gates", relation="was born in", object="Seattle"
{"add_inverse": false, "inverse_relation": null, "rationale": "state-of-being on subject — Seattle 'was birthplace of' is grammatically valid but semantically thin"}

Triple sample: subject="Larry Page", relation="was president of", object="Alphabet"
{"add_inverse": false, "inverse_relation": null, "rationale": "role state on subject — 'Alphabet had president' is awkward and downstream queries can already match by role"}

Triple sample: subject="John", relation="was awarded", object="Nobel Prize"
{"add_inverse": false, "inverse_relation": null, "rationale": "state on subject — the relationship is a property of John, not of the prize"}

Triple sample: subject="X.com", relation="merged with", object="Confinity"
{"add_inverse": false, "inverse_relation": null, "rationale": "symmetric — merger is mutual"}

Output strict JSON only. Use false branch when in doubt — adding an awkward
inverse pollutes the graph more than missing one hurts retrieval."""


def _safe_rel_type(active_relation: str) -> str:
    """Cypher-safe rel type from natural-language relation string."""
    cleaned = re.sub(r"[^A-Z_]", "_", active_relation.upper().replace(" ", "_"))[:40]
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "RELATED_TO"


def discover_forward_predicates(session, min_count: int = 5) -> list[dict]:
    """Find all rel_types in graph, with sample triple + count.

    Skips already-inferred-reverse rel_types (those have any edge with
    `inferred_reverse_of` property) so re-running is idempotent."""
    rows = list(session.run("""
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE r.inferred_reverse_of IS NULL
        WITH type(r) AS rel_type,
             head(collect({s: a.name, raw: r.raw_relation, o: b.name})) AS sample,
             count(*) AS n
        WHERE n >= $min_count
        RETURN rel_type, sample, n
        ORDER BY n DESC
    """, min_count=min_count))
    return [
        {
            "rel_type":   r["rel_type"],
            "count":      r["n"],
            "sample_s":   r["sample"]["s"],
            "sample_raw": r["sample"]["raw"],
            "sample_o":   r["sample"]["o"],
        }
        for r in rows
    ]


def judge_predicate(s: str, raw_rel: str, o: str) -> dict:
    """One LLM call per unique forward predicate. Returns
    {add_inverse, inverse_relation, rationale}."""
    user_msg = (
        f"Triple sample: subject={s!r}, relation={raw_rel!r}, object={o!r}\n\n"
        "Decide whether to add the inverse direction edge."
    )
    try:
        resp = omlx.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        return {
            "add_inverse": False,
            "inverse_relation": None,
            "rationale": f"judge error: {type(e).__name__}: {e}",
        }


def add_reverse_edges(
    session, forward_rel_type: str, inverse_relation: str
) -> int:
    """For all forward edges of given rel_type, create a reverse edge with
    the new active phrasing. Audit: `inferred_reverse_of = <forward_rel_type>`."""
    new_rel = _safe_rel_type(inverse_relation)
    if not _SAFE_REL_TYPE.match(new_rel):
        print(f"  [skip] new rel_type {new_rel!r} unsafe — leaving as-is")
        return 0
    if new_rel == forward_rel_type:
        print(f"  [skip] inverse rel_type {new_rel!r} identical to forward — no-op")
        return 0

    try:
        # APOC bulk-create
        result = session.run(f"""
            MATCH (a:Entity)-[r:{forward_rel_type}]->(b:Entity)
            WHERE r.inferred_reverse_of IS NULL
            WITH r, a, b
            CALL apoc.create.relationship(b, $new_type, {{
                raw_relation:        $inverse_raw,
                source_title:        r.source_title,
                inferred_reverse_of: $forward_rel
            }}, a) YIELD rel
            RETURN count(rel) AS n
        """, new_type=new_rel, inverse_raw=inverse_relation, forward_rel=forward_rel_type)
        return result.single()["n"]
    except Exception as e:
        if "apoc" not in str(e).lower():
            raise
        # Fallback: manual loop
        rows = list(session.run(f"""
            MATCH (a:Entity)-[r:{forward_rel_type}]->(b:Entity)
            WHERE r.inferred_reverse_of IS NULL
            RETURN id(r) AS rid, a.name AS sname, b.name AS oname,
                   r.source_title AS src
        """))
        n = 0
        for row in rows:
            session.run(f"""
                MATCH (a:Entity {{name: $sname}}), (b:Entity {{name: $oname}})
                CREATE (b)-[:{new_rel} {{
                    raw_relation:        $inverse_raw,
                    source_title:        $src,
                    inferred_reverse_of: $forward_rel
                }}]->(a)
            """, sname=row["sname"], oname=row["oname"],
                 inverse_raw=inverse_relation, src=row["src"],
                 forward_rel=forward_rel_type)
            n += 1
        return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="show decisions without writing")
    ap.add_argument("--min-count", type=int, default=5,
                    help="skip predicates appearing in fewer than this many triples (default 5)")
    args = ap.parse_args()

    print("=" * 76)
    print(f"infer_reverse_edges — {'DRY RUN' if args.dry_run else 'APPLY MODE'}")
    print("=" * 76)

    with driver.session() as session:
        predicates = discover_forward_predicates(session, min_count=args.min_count)
        print(f"\nFound {len(predicates)} unique forward predicates (count >= {args.min_count})")
        total_triples = sum(p["count"] for p in predicates)
        print(f"Total triples in scope: {total_triples}\n")

        decisions = []
        for p in predicates:
            j = judge_predicate(p["sample_s"], p["sample_raw"], p["sample_o"])
            add = bool(j.get("add_inverse"))
            inv = j.get("inverse_relation") or ""
            mark = "ADD-INVERSE" if (add and inv) else "SKIP"
            print(
                f"  [{mark:<11}] {p['rel_type']:<32} ({p['count']:>3}× ) "
                f"sample: {p['sample_s'][:25]!r:<27} -[{p['sample_raw']!r:<22}]-> "
                f"{p['sample_o'][:22]!r}"
            )
            if add and inv:
                print(f"            → inverse: {inv!r}")
            print(f"            rationale: {(j.get('rationale','') or '')[:160]}")
            decisions.append({**p, **j})

        print("\n" + "=" * 76)
        adds = [d for d in decisions if d.get("add_inverse") and d.get("inverse_relation")]
        print(f"Decision: add inverse on {len(adds)} predicates / "
              f"{sum(d['count'] for d in adds)} new edges to create")
        print(f"          skip {len(decisions) - len(adds)} predicates / "
              f"{total_triples - sum(d['count'] for d in adds)} triples (symmetric/state-of-being)")
        print("=" * 76)

        if args.dry_run:
            print("\nDRY RUN — no writes. Re-run without --dry-run to apply.")
            return 0

        print("\nAdding reverse edges...")
        t0 = time.time()
        total_added = 0
        for d in adds:
            n = add_reverse_edges(session, d["rel_type"], d["inverse_relation"])
            print(f"  {d['rel_type']} → {_safe_rel_type(d['inverse_relation'])}: {n} edges")
            total_added += n
        print(f"\nDone: {total_added} reverse edges added in {time.time()-t0:.1f}s")

    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
