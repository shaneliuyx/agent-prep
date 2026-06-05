"""W3.5.95 — Heat-eviction demo: bound the LEARNING store, keep the right facts.

Seeds a realistic synthetic store (near-duplicates, varied confidence / recency /
recall_count) into a TEMP db — never touches the real data/memory.db — then runs
`heat_eviction.enforce(budget)` and measures what survived. The point is the
POLICY's behavior, not the numbers: dedup collapses redundancy, eviction drops the
coldest low-value facts, and importance-exempt + hot facts survive.

Run: python3 src/heat_eviction_demo.py
"""
from __future__ import annotations

import tempfile

import observability as obs
import heat_eviction as he
from metacog_recall import recall

_DAY = 24 * 3600.0
NOW = 1_780_000_000.0  # fixed epoch → deterministic heat/recency

# (type, pattern_text, confidence, age_days, recall_count) — designed to exercise
# every branch: a near-dup pair, a COLD high-confidence fact (exemption test), a
# HOT low-confidence fact (visits keep it), and a pile of cold low-value facts.
SEED = [
    ("recurring_mistake", "I retry failed API calls immediately with no backoff",         0.70,  2, 9),  # hot, low-conf → survives on visits
    ("recurring_mistake", "I retry failed API calls with no backoff immediately",          0.66, 30, 0),  # LEXICAL near-dup of ^ → deduped
    ("failure_pattern",   "I deadlock when acquiring nested locks in the wrong order",    0.93, 60, 0),  # COLD but high-conf → exempt
    ("success_pattern",   "Writing a failing test first cuts my debug loops in half",     0.88,  1, 4),  # hot + high-conf → survives
    ("tool_preference",   "I reach for grep before ripgrep even though rg is faster",     0.55,  3, 5),  # warm → survives
    ("recurring_mistake", "I forget to close file handles in long-running scripts",       0.60, 45, 0),  # cold low-conf → evicted
    ("recurring_mistake", "I leave database connections open across requests",            0.58, 50, 0),  # cold low-conf → evicted
    ("failure_pattern",   "I misread timezone-naive datetimes as UTC",                    0.52, 70, 0),  # cold low-conf → evicted
    ("tool_preference",   "I default to pandas for tiny CSVs that csv module handles",    0.50, 40, 1),  # cold low-conf → evicted
    ("recurring_mistake", "I over-fetch columns in SELECT * instead of naming them",      0.57, 35, 0),  # cold low-conf → evicted
    ("success_pattern",   "Reading the stack trace bottom-up finds the cause faster",     0.62,  5, 3),  # warm → survives
    ("recurring_mistake", "I ignore linter warnings until they pile up",                  0.48, 25, 0),  # cold low-conf → evicted
    ("failure_pattern",   "I assume list order is stable across set() round-trips",       0.45, 80, 0),  # cold low-conf → evicted
    ("tool_preference",   "I prefer print-debugging over the actual debugger",            0.51, 20, 1),  # cold-ish → evicted
]
BUDGET = 8


def _seed(conn) -> None:
    he.ensure_heat_columns(conn)
    for typ, text, conf, age, visits in SEED:
        ts = NOW - age * _DAY
        last = NOW - (age // 2) * _DAY if visits else None
        conn.execute(
            "INSERT INTO learning (type, pattern_text, confidence, is_self_caused, "
            "source_rows, ts, recall_count, last_recalled_ts) VALUES (?,?,?,1,'[]',?,?,?)",
            (typ, text, conf, ts, visits, last),
        )
    conn.commit()


def main() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        conn = obs.connect(tmp.name)
        _seed(conn)

        before = conn.execute("SELECT COUNT(*) AS n FROM learning").fetchone()["n"]
        stats = he.enforce(conn, BUDGET, now=NOW)

        print(f"seeded {before} facts → budget {BUDGET}")
        print(f"  deduped : {stats['deduped']}")
        print(f"  evicted : {stats['evicted']}")
        print(f"  final   : {stats['final']}\n")

        survivors = conn.execute("SELECT * FROM learning").fetchall()
        survivors = sorted(survivors, key=lambda r: he.heat(r, NOW), reverse=True)
        print(f"{'heat':<7}{'conf':<6}{'visits':<8}pattern")
        print("-" * 78)
        for r in survivors:
            exempt = " [exempt]" if r["confidence"] >= he.IMPORTANCE_FLOOR else ""
            print(f"{he.heat(r, NOW):<7.2f}{r['confidence']:<6.2f}{r['recall_count']:<8}"
                  f"{r['pattern_text'][:44]}{exempt}")

        # sanity: the cold high-confidence fact must have survived (importance-exempt)
        kept = {r["pattern_text"] for r in survivors}
        cold_exempt = "I deadlock when acquiring nested locks in the wrong order"
        print(f"\ncold high-confidence fact survived (importance-exempt): {cold_exempt in kept}")
        # sanity: recall still surfaces a relevant survivor for a live query
        hits = recall(conn, "API call retry backoff", k=2, now=NOW)
        print(f"recall still works post-eviction: top hit = "
              f"{hits[0]['pattern_text'][:48] if hits else 'NONE'}")


if __name__ == "__main__":
    main()
