"""Weight calibration for query_graph.py composite scorer.

Probes a fixed set of (seed, lucene, expected_canonical) triples against
the live graph, sweeping QID_BONUS / EXACT_BONUS / DEGREE_COEFF, and
reports which combination maximises top-5 recall.

Usage:
    python src/calibrate_scorer.py --quick   # 3×3×3 coarse sweep (fast)
    python src/calibrate_scorer.py           # 7×5×6 fine sweep

Copy the recommended weights into src/query_graph.py lines:
    QID_BONUS    = <best>
    EXACT_BONUS  = <best>
    DEGREE_COEFF = <best>

Run this script whenever the graph is rebuilt or the corpus changes size.
"""
import argparse
import os
from itertools import product

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)

# Probes: (seed_string, lucene_query, expected_canonical_node_name)
# seed_string   → passed as $seed to the Cypher CASE expressions
# lucene_query  → passed as $lucene to the fulltext CALL
# expected      → must appear in LIMIT-5 result for probe to count as a hit
#
# Lucene mirrors what production query_graph.py builds after proper-noun
# filtering. Noise-trap probes use post-filter lucene (e.g. seed "CEO of Apple"
# in production drops "of"/"CEO" via proper-noun filter → lucene "apple").
PROBES = [
    # ── Alias-gap (canonical name ≠ seed surface form) ──
    ("Jeff Bezos",       "+jeff +bezos",        "Jeffrey Preston Bezos"),
    ("Jensen Huang",     "+jensen +huang",      "Jensen Huang"),

    # ── Direct-name controls ──
    ("Elon Musk",        "+elon +musk",         "Elon Musk"),
    ("Mark Zuckerberg",  "+mark +zuckerberg",   "Mark Zuckerberg"),
    ("Microsoft",        "microsoft",           "Microsoft"),
    ("Pixar",            "+pixar",              "Pixar"),

    # ── Mixed-case / lowercase-prefix brands ──
    ("eBay",             "+ebay",               "eBay"),
    ("iPhone",           "+iphone",             "iPhone"),
    ("iOS",              "+ios",                "iOS"),

    # ── Acronym → full-name (graph stores the spelled-out form) ──
    ("MIT",              "+massachusetts +institute +technology",
                                                "Massachusetts Institute of Technology"),
    ("IBM",              "+ibm",                "IBM"),
    ("GE",               "+general +electric",  "General Electric"),

    # ── Disambig page must lose to canonical entity ──
    ("Apple",            "apple",               "Apple Inc."),
    ("Stanford",         "stanford",            "Stanford University"),
    ("Tesla company",    "tesla",               "Tesla, Inc."),

    # ── Noise traps: seed contains role/descriptor; production proper-noun
    #    filter drops the descriptor → lucene below mirrors that. Expected
    #    canonical MUST NOT be a noise node like 'CEO of Apple'.
    ("CEO of Apple",        "apple",            "Apple Inc."),
    ("president of Microsoft", "microsoft",     "Microsoft"),
    ("founder of Tesla",    "tesla",            "Tesla, Inc."),
    ("former CEO of Pixar", "+pixar",           "Pixar"),
]


def _score_weights(qid_bonus: float, exact_bonus: float, degree_coeff: float) -> float:
    """Return fraction of probes where expected entity is in top-5 composite ranking."""
    cypher = (
        'CALL db.index.fulltext.queryNodes("entity_names", $lucene) '
        'YIELD node, score AS bm25 '
        'WITH node, bm25, '
        f'CASE WHEN node.qid IS NOT NULL THEN {qid_bonus} ELSE 0 END AS qid_bonus, '
        'CASE WHEN toLower(node.name) = toLower($seed) '
        '  OR $seed IN coalesce(node.aliases, []) '
        f'  THEN {exact_bonus} ELSE 0 END AS exact_bonus, '
        f'log(coalesce(node.degree, 0) + 1) * {degree_coeff} AS degree_score '
        'WITH node, bm25 + qid_bonus + exact_bonus + degree_score AS composite '
        # Mirror production topology gate so calibration measures what
        # production actually returns (not unfiltered ranking).
        'WHERE (node.qid IS NOT NULL OR coalesce(node.degree, 0) >= 2) '
        'WITH node, composite ORDER BY composite DESC LIMIT 5 '
        'RETURN node.name AS name'
    )
    hits = 0
    with driver.session() as s:
        for seed, lucene, expected in PROBES:
            names = [r["name"] for r in s.run(cypher, lucene=lucene, seed=seed).data()]
            if expected in names:
                hits += 1
    return hits / len(PROBES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate composite scorer weights")
    parser.add_argument("--quick", action="store_true",
                        help="3×3×3 coarse sweep (~27 combinations)")
    args = parser.parse_args()

    if args.quick:
        qid_vals    = [0.5, 1.5, 2.5]
        exact_vals  = [0.0, 0.8, 1.5]
        degree_vals = [0.0, 0.3, 0.6]
    else:
        qid_vals    = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        exact_vals  = [0.0, 0.4, 0.8, 1.2, 1.6]
        degree_vals = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]

    total = len(qid_vals) * len(exact_vals) * len(degree_vals)
    print(f"Sweeping {total} weight combinations across {len(PROBES)} probes ...")

    results = []
    for qid_b, exact_b, deg_c in product(qid_vals, exact_vals, degree_vals):
        score = _score_weights(qid_b, exact_b, deg_c)
        results.append((score, qid_b, exact_b, deg_c))

    results.sort(reverse=True)
    print(f"\nTop 10 combinations (recall = fraction of {len(PROBES)} probes in top-5):")
    print(f"{'recall':>7}  {'QID_BONUS':>10}  {'EXACT_BONUS':>12}  {'DEGREE_COEFF':>13}")
    for score, qid_b, exact_b, deg_c in results[:10]:
        print(f"{score:>7.3f}  {qid_b:>10.2f}  {exact_b:>12.2f}  {deg_c:>13.2f}")

    best_score, best_qid, best_exact, best_deg = results[0]
    print(f"\nRecommended weights (recall={best_score:.3f}):")
    print(f"  QID_BONUS    = {best_qid}")
    print(f"  EXACT_BONUS  = {best_exact}")
    print(f"  DEGREE_COEFF = {best_deg}")
    print("\nCopy these into src/query_graph.py (QID_BONUS / EXACT_BONUS / DEGREE_COEFF).")


if __name__ == "__main__":
    main()
