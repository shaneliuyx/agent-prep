"""Phase 6 benchmark — keyword FTS (`gbrain search`) vs hybrid-RRF (`gbrain query`).

Runs a labeled query set against the live brain, parses GBrain's ranked output
(`[score] slug -- text` lines), and reports hit@3 + MRR per method. The query
set deliberately mixes EXACT-NAME queries (proper nouns — favor keyword/FTS) with
SEMANTIC-PARAPHRASE queries (no shared surface token — favor vector). The whole
point is to show WHERE the two retrievers diverge, not just an aggregate.

Honest-measurement note: this brain is ~19 pages. On a corpus this small both
retrievers nearly saturate, so the RRF lift here is a floor, not the 240-page
number. We report what we measured.
"""
from __future__ import annotations

import os
import re
import subprocess

_BUN = os.path.expanduser("~/.bun/bin")
_GBRAIN = os.path.join(_BUN, "gbrain")

# (query, gold_slug, kind) — gold is the single page that best answers.
QUERIES: list[tuple[str, str, str]] = [
    ("Lin Zhao",                          "people/lin-zhao",             "exact"),
    ("Ridgeline Capital",                 "companies/ridgeline-capital", "exact"),
    ("Northstar Ventures",                "companies/northstar-ventures","exact"),
    ("Marcus Webb",                       "people/marcus-webb",          "exact"),
    ("dinner at Tartine",                 "meetings/tartine-dinner",     "exact"),
    ("who runs serving infrastructure",   "people/lin-zhao",             "semantic"),
    ("protein design foundation models",  "companies/helix-bio",         "semantic"),
    ("inference optimization startup",    "companies/quanta-labs",       "semantic"),
    ("early-stage bio funding round",     "deals/helix-series-a",        "semantic"),
    ("payments company angel investment", "companies/stripe",            "semantic"),
]

K = 3  # hit@K
_LINE = re.compile(r"^\[[-\d.]+\]\s+(\S+)\s+--")


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = _BUN + os.pathsep + env.get("PATH", "")
    env.setdefault("GBRAIN_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/gbrain")
    env.setdefault("OLLAMA_BASE_URL", "http://localhost:8000/v1")
    return env


def _ranked_slugs(cmd: str, query: str) -> list[str]:
    """Run `gbrain <cmd> <query> --json --limit 10`, return ordered slugs."""
    out = subprocess.run(
        [_GBRAIN, cmd, query, "--json", "--limit", "10"],
        capture_output=True, text=True, env=_env(),
    ).stdout
    slugs: list[str] = []
    for line in out.splitlines():
        m = _LINE.match(line.strip())
        if m:
            slugs.append(m.group(1))
    return slugs


def _rank_of(gold: str, slugs: list[str]) -> int | None:
    for i, s in enumerate(slugs, 1):
        if s == gold:
            return i
    return None


def main() -> None:
    methods = {"search (keyword)": "search", "query (hybrid-RRF)": "query"}
    agg: dict[str, dict] = {m: {"hits": 0, "rr": 0.0} for m in methods}
    print(f"{'query':<38}{'kind':<10}{'keyword':<12}{'hybrid-RRF':<12}")
    print("-" * 72)
    for q, gold, kind in QUERIES:
        row = f"{q:<38}{kind:<10}"
        for label, cmd in methods.items():
            rank = _rank_of(gold, _ranked_slugs(cmd, q))
            hit = rank is not None and rank <= K
            agg[label]["hits"] += int(hit)
            agg[label]["rr"] += (1.0 / rank) if rank else 0.0
            row += f"{('@' + str(rank)) if rank else 'MISS':<12}"
        print(row)
    n = len(QUERIES)
    print("-" * 72)
    print(f"\n{'method':<22}{'hit@'+str(K):<10}{'MRR':<8}")
    for m in methods:
        print(f"{m:<22}{agg[m]['hits']}/{n} ({agg[m]['hits']/n:.0%})  {agg[m]['rr']/n:.3f}")


if __name__ == "__main__":
    main()
