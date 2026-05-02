"""Mechanism-level probes for the decomposition path.

Tests `_decompose_multihop` (LLM classifier) AND `_step_one_intermediates`
(intermediate-entity resolution) at the mechanism layer, BEFORE outcome eval.

Per panel critique: outcome-based gates hide mechanism failures. These probes
isolate decomposition correctness from final answer quality.

Usage:
    python src/decomp_probes.py            # full mechanism check
    python src/decomp_probes.py --quick    # subset (5 queries, fast)

Pass criterion: probe_pass_rate >= 0.85 (panel B1' gate).
"""
import argparse
import sys

from query_graph import (
    _decompose_multihop,
    _step_one_intermediates,
    driver,
)

# (query, expected_plan_kind, expected_step1_intermediates_subset)
# expected_plan_kind: "chain" (plan present) | "intersection" | "none"
# expected_step1_intermediates_subset: at least one of these names must
# appear in _step_one_intermediates output (None = skip step1 check).
PROBES = [
    # ── chain queries (should decompose) ──
    ("Which companies did founders of PayPal later start?",
     "chain",
     ["Elon Musk", "Peter Thiel", "Reid Hoffman", "Max Levchin"]),
    ("What companies were founded by Stanford alumni?",
     "chain",
     ["Larry Page", "Sergey Brin", "Jensen Huang", "Jen-Hsun Huang"]),
    ("What companies have been founded by Harvard dropouts?",
     "chain",
     ["Bill Gates", "Mark Zuckerberg"]),
    ("Who co-founded Andreessen Horowitz with Marc Andreessen, and what other companies did they create?",
     "chain",
     ["Ben Horowitz"]),
    ("Which technology company founders are alumni of Stanford University?",
     "chain",
     ["Larry Page", "Sergey Brin", "Jen-Hsun Huang", "Jensen Huang"]),

    # ── single-hop queries (should NOT decompose) ──
    ("Who founded Microsoft?", "none", None),
    ("Who is the CEO of Apple?", "none", None),
    ("Where did Bill Gates go to university?", "none", None),

    # ── relational queries (should NOT decompose; bridge path handles them) ──
    ("What is the relationship between Apple and NeXT?", "none", None),
    ("What is the relationship between Tesla and SpaceX?", "none", None),
    ("What is the relationship between Apple and Pixar?", "none", None),

    # ── out-of-domain (should NOT decompose) ──
    ("What is the boiling point of helium?", "none", None),
    ("Who composed Symphony No. 9?", "none", None),
    ("What is the capital of France?", "none", None),
]


def _kind_of(plan: dict | None) -> str:
    """`_decompose_multihop` returns the unwrapped inner plan dict, or None.
    Inner shape: {'step1': ..., 'step2': ...} for chain, or {'type':
    'intersection', 'step1a': ..., 'step1b': ...} for intersection."""
    if plan is None:
        return "none"
    if not isinstance(plan, dict):
        return "none"
    if plan.get("type") == "intersection":
        return "intersection"
    if "step1" in plan and "step2" in plan:
        return "chain"
    return "none"


def _check_step1(plan: dict, expected_subset: list[str]) -> bool:
    """Run _step_one_intermediates with the plan's anchor + edge_filter and
    check at least one expected intermediate appears. Skips intersection
    plans (different shape)."""
    if not isinstance(plan, dict):
        return False
    if plan.get("type") == "intersection":
        return True  # not testing intersection step1 here
    s1 = plan.get("step1")
    if not isinstance(s1, dict):
        return False
    anchor = s1.get("anchor")
    edge_filter = s1.get("edge_filter")
    if not anchor or not edge_filter:
        return False
    with driver.session() as session:
        intermediates = _step_one_intermediates(session, anchor, edge_filter, limit=30)
    found = any(any(exp.lower() in inter.lower() for inter in intermediates)
                for exp in expected_subset)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Decomposition mechanism probes")
    parser.add_argument("--quick", action="store_true",
                        help="run first 5 probes only")
    args = parser.parse_args()

    probes = PROBES[:5] if args.quick else PROBES
    pass_count = 0
    classify_pass = 0
    step1_pass = 0
    step1_total = 0

    print(f"Running {len(probes)} decomposition mechanism probes ...")
    print()
    for query, expected_kind, expected_step1 in probes:
        plan = _decompose_multihop(query)
        actual_kind = _kind_of(plan)
        kind_ok = actual_kind == expected_kind
        if kind_ok:
            classify_pass += 1

        step1_ok = True
        if kind_ok and expected_step1 and actual_kind == "chain":
            step1_total += 1
            assert plan is not None
            step1_ok = _check_step1(plan, expected_step1)
            if step1_ok:
                step1_pass += 1

        overall_ok = kind_ok and step1_ok
        if overall_ok:
            pass_count += 1

        marker = "PASS" if overall_ok else "FAIL"
        kind_marker = "✓" if kind_ok else "✗"
        step1_marker = "✓" if step1_ok else ("✗" if expected_step1 else "—")
        print(f"  [{marker}] kind={kind_marker} step1={step1_marker} :: {query[:70]}")
        if not kind_ok:
            print(f"        expected kind={expected_kind!r}, got {actual_kind!r}")
        if expected_step1 and not step1_ok and kind_ok:
            print(f"        expected step1 intermediates ⊇ ANY of {expected_step1}")

    print()
    print(f"Pass rate (overall):     {pass_count}/{len(probes)} = {pass_count/len(probes):.3f}")
    print(f"Classification pass:     {classify_pass}/{len(probes)} = {classify_pass/len(probes):.3f}")
    if step1_total:
        print(f"Step1 intermediates:     {step1_pass}/{step1_total} = {step1_pass/step1_total:.3f}")
    gate = 0.85
    print()
    print(f"B1' gate: pass_rate >= {gate}: "
          + ("PASS" if pass_count/len(probes) >= gate else "FAIL"))
    sys.exit(0 if pass_count/len(probes) >= gate else 1)


if __name__ == "__main__":
    main()
