"""answer_principle_ab.py — does the per-query ROUTER produce better ANSWERS than global
hybrid (and than a strengthened global), on a PINNED strong generator?

route_principle_ab.ts settled the *grounding* A/B on the balanced set: router 0.862 beats
baseline hybrid 0.823 (+0.039), strengthened-global loses. Grounding takes the max coverage
over the top-C, so it can't see whether the OTHER in-budget chunks are distractors. This
script closes that gap at answer time: for each balanced question, generate an answer from
each policy's context and judge whether it actually identifies the question's expected
entities. Pin a STRONG generator (Opus via VibeProxy) so the verdict reflects retrieval, not
the small-model context-sensitivity confound caught earlier (answer_route_ab.py).

The balanced set carries `expected_entities`, not W2.7 `pass_criteria`, so the judge rubric is
"the answer correctly identifies <entities>" — answer-level coverage, aligned with grounding.

Inputs: results/principle_slugs.json (from `bun src/route_principle_ab.ts`).
Run:    CHAT_MODEL=claude-opus-4-5 JUDGE_MODEL=claude-opus-4-5 uv run python src/answer_principle_ab.py
"""
from __future__ import annotations

import json
import os
import pathlib
import time

from openai import APIConnectionError, APIError

from ground_truth_ab import _MEMZERO_TMPL, _ask, _client, gbrain_get

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SLUGS = _ROOT / "results" / "principle_slugs.json"
_ARMS = ("baseline", "router", "strong_g")
_MIN_BODY_CHARS = 80
_MAX_BODY_CHARS = int(os.getenv("MAX_BODY_CHARS", "0"))

_JUDGE_TMPL = (
    "You are grading whether a candidate answer correctly identifies the required facts. "
    "Reply with EXACTLY one word: PASS or FAIL.\n\n"
    "The answer PASSES only if it correctly names/identifies ALL of these (synonyms and the "
    "full entity for an abbreviation are fine):\n{entities}\n\n"
    "CANDIDATE ANSWER:\n{answer}\n\nVerdict (PASS or FAIL):"
)


class LLMUnavailable(RuntimeError):
    """Chat endpoint refused after retries — skip the question, don't crash the run."""


class SnippetRegression(RuntimeError):
    """A body came back as a truncated `query --json` snippet, not a full `gbrain get` page."""


def _resilient(fn, *args, retries: int = 4, backoff: float = 2.0):
    for attempt in range(retries):
        try:
            return fn(*args)
        except (APIConnectionError, APIError) as exc:
            if attempt == retries - 1:
                raise LLMUnavailable(str(exc)) from exc
            time.sleep(backoff * (attempt + 1))
    raise LLMUnavailable("unreachable")


def build_context(slugs: list[str]) -> str:
    """Assemble reader context from FULL `gbrain get` page bodies (never `query --json`
    snippets — the grounding-loss gotcha). Fail loud if a body is suspiciously short."""
    bodies = [gbrain_get(s) for s in slugs]
    if _MAX_BODY_CHARS:
        bodies = [b[:_MAX_BODY_CHARS] for b in bodies]
    short = [(s, len(b.strip())) for s, b in zip(slugs, bodies) if len(b.strip()) < _MIN_BODY_CHARS]
    if short:
        raise SnippetRegression(f"injected body too short {short} — pull full `gbrain get` bodies")
    return "\n\n".join(bodies)


def _answer(client, slugs: list[str], q: str) -> str:
    msg = [{"role": "user", "content": _MEMZERO_TMPL.format(ctx=build_context(slugs), q=q)}]
    ans, _ = _resilient(_ask, client, msg)
    return ans


def _judge(client, answer: str, entities: list[str]) -> bool:
    rubric = "; ".join(entities)
    msg = [{"role": "user", "content": _JUDGE_TMPL.format(entities=rubric, answer=answer)}]
    verdict, _ = _resilient(_ask, client, msg)
    return verdict.strip().upper().startswith("PASS")


def main() -> None:
    dump = json.loads(_SLUGS.read_text())
    client = _client()
    print(f"generator/judge: {os.getenv('CHAT_MODEL', '(default)')} · {len(dump)} questions\n")

    # per-arm pass flags, and per-class accumulators
    passes: dict[str, list[bool]] = {a: [] for a in _ARMS}
    types: list[str] = []
    for item in dump:
        q = item["q"].strip()
        ents = item["expected_entities"]
        row = {}
        try:
            for arm in _ARMS:
                ans = _answer(client, item[f"{arm}_slugs"], q)
                row[arm] = _judge(client, ans, ents)
        except LLMUnavailable as exc:
            print(f"  [skip] chat endpoint down — {q[:48]} ({exc})")
            continue
        for arm in _ARMS:
            passes[arm].append(row[arm])
        types.append(item["type"])
        differs = item["router_slugs"] != item["baseline_slugs"]
        tag = "DIFF" if differs else "same"
        flags = " ".join(f"{a[:4]}={'P' if row[a] else 'F'}" for a in _ARMS)
        print(f"  [{tag}] {flags}  ({item['type']}) {q[:46]}")

    n = len(types)
    if not n:
        print("\nno questions judged (endpoint down?)")
        return
    rate = lambda a, cls=None: (  # noqa: E731
        sum(p for p, t in zip(passes[a], types) if cls is None or t == cls)
        / max(1, sum(1 for t in types if cls is None or t == cls)))
    base = rate("baseline")
    print(f"\njudged {n} questions on a pinned generator")
    classes = ["kw", "vec", "mixed"]
    hdr = f"  {'policy':12}{'pass':8}" + "".join(f"{'p:'+c:9}" for c in classes) + "Δ vs base"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for a in _ARMS:
        cols = "".join(f"{rate(a, c):<9.3f}" for c in classes)
        print(f"  {a:12}{rate(a):<8.3f}{cols}{rate(a) - base:+.3f}")


if __name__ == "__main__":
    main()
