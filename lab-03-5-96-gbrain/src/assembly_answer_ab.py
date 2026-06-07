"""assembly_answer_ab.py — does GRAPH-EXPANDED assembly produce better ANSWERS than plain top-C?

assembly_graph_ab.ts showed graph expansion fires on 7/24 questions but lexical entity COVERAGE
doesn't move — because a related entity's page already names the other entity (ridgeline's page
says "Dev Patel"), so the string is in budget even when the page isn't. Coverage is blind to that;
the real question is whether having the neighbor's full PAGE (and the distractors it sits beside)
changes the generated answer. So we judge the answers, pinned to a strong generator (Opus).

Two arms per question (from results/assembly_slugs.json):
  plain     — top-C context
  expanded  — top-C + 1-hop graph-pulled neighbors
Judge entity coverage of each answer. Report pass rates, focused on the questions where expansion
actually pulled a neighbor.

Run: CHAT_MODEL=claude-opus-4-5-20251101 uv run python src/assembly_answer_ab.py
"""
from __future__ import annotations

import json
import os
import pathlib
import time

from openai import APIConnectionError, APIError

from ground_truth_ab import _MEMZERO_TMPL, _ask, _client, gbrain_get

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SLUGS = _ROOT / "results" / "assembly_slugs.json"
_MIN_BODY = 80

_JUDGE_TMPL = (
    "You are grading whether a candidate answer correctly identifies the required facts. "
    "Reply with EXACTLY one word: PASS or FAIL.\n\n"
    "The answer PASSES only if it correctly names/identifies ALL of these (synonyms / full entity "
    "for an abbreviation are fine):\n{entities}\n\nCANDIDATE ANSWER:\n{answer}\n\nVerdict (PASS or FAIL):"
)


class LLMUnavailable(RuntimeError):
    pass


def _resilient(fn, *args, retries: int = 4, backoff: float = 2.0):
    for attempt in range(retries):
        try:
            return fn(*args)
        except (APIConnectionError, APIError) as exc:
            if attempt == retries - 1:
                raise LLMUnavailable(str(exc)) from exc
            time.sleep(backoff * (attempt + 1))
    raise LLMUnavailable("unreachable")


def _context(slugs: list[str]) -> str:
    bodies = [gbrain_get(s) for s in slugs]
    short = [(s, len(b.strip())) for s, b in zip(slugs, bodies) if len(b.strip()) < _MIN_BODY]
    if short:
        raise RuntimeError(f"body too short {short}")
    return "\n\n".join(bodies)


def _answer(client, slugs: list[str], q: str) -> str:
    msg = [{"role": "user", "content": _MEMZERO_TMPL.format(ctx=_context(slugs), q=q)}]
    return _resilient(_ask, client, msg)[0]


def _judge(client, answer: str, entities: list[str]) -> bool:
    msg = [{"role": "user", "content": _JUDGE_TMPL.format(entities="; ".join(entities), answer=answer)}]
    return _resilient(_ask, client, msg)[0].strip().upper().startswith("PASS")


def main() -> None:
    rows = json.loads(_SLUGS.read_text())
    client = _client()
    print(f"generator/judge: {os.getenv('CHAT_MODEL', '(default)')} · {len(rows)} questions\n")

    pp = pe = 0
    fired_pp = fired_pe = fired_n = 0
    for r in rows:
        ents = r["expected_entities"]
        fired = len(r["pulled"]) > 0
        try:
            ap = _judge(client, _answer(client, r["plain_slugs"], r["q"]), ents)
            ae = _judge(client, _answer(client, r["expanded_slugs"], r["q"]), ents)
        except LLMUnavailable as exc:
            print(f"  [skip] {r['q'][:46]} ({exc})")
            continue
        pp += ap; pe += ae
        if fired:
            fired_n += 1; fired_pp += ap; fired_pe += ae
            tag = "FIXED " if (ae and not ap) else ("BROKE " if (ap and not ae) else "same  ")
            print(f"  [{tag}] plain={'P' if ap else 'F'} expanded={'P' if ae else 'F'}  +{len(r['pulled'])}p  \"{r['q'][:44]}\"")

    n = len(rows)
    print(f"\nall {n} questions:        plain {pp / n:.3f}  →  graph-expanded {pe / n:.3f}  (Δ {(pe - pp) / n:+.3f})")
    if fired_n:
        print(f"the {fired_n} where expansion fired: plain {fired_pp / fired_n:.3f}  →  expanded {fired_pe / fired_n:.3f}  (Δ {(fired_pe - fired_pp) / fired_n:+.3f})")


if __name__ == "__main__":
    main()
