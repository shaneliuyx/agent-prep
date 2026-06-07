"""reader_ab.py — hold retrieval FIXED, vary the READER; which reader writes better answers?

Retrieval is solved on this corpus (hybrid grounding ~0.93). The leverage is the
generation step — W2.7's 0.96 grounding → 0.25 answer gap, and the "Notes" question that
grounds 1.000 yet answers 0/3. So we fix the context (always the winning arm = hybrid,
full `gbrain get` bodies) and A/B the READER STRATEGY against the same golden pass_criteria:

  - plain          : "answer from the notes" (baseline)
  - authoritative  : frame the notes as AUTHORITATIVE ground truth (memory-os) — don't
                     re-derive, don't hedge
  - extract        : authoritative + extract-then-answer (pull the answer-bearing span
                     first, then compose)
  - cite           : authoritative + require a source citation per claim

Generator = the reader UNDER TEST; judge = a fixed STRONG grader (so the verdict isn't the
generator grading itself). Each role is a NAMED MODEL PRESET — switch with no code change:

  GEN=14b JUDGE=opus uv run python src/reader_ab.py     # weak gen, strong judge
  GEN=haiku JUDGE=opus uv run python src/reader_ab.py   # mid gen, strong judge

Presets resolve endpoint+key+model (see _PROFILES). Raw escape hatch: set <ROLE>_MODEL
(+ optional <ROLE>_BASE_URL / <ROLE>_API_KEY) to bypass the registry. Both roles read .env
for the oMLX key. Context comes from build_context() → full bodies, snippet-guarded.

Inputs: results/arm_scores.json (from `bun src/route_eval.ts`) + the W2.7 ground truth.
"""
from __future__ import annotations

import json
import os
import pathlib

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from answer_route_ab import (
    LLMUnavailable,
    _JUDGE_TMPL,
    _pass_criteria_by_q,
    _resilient,
    build_context,
)

_ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")  # for the oMLX key used by the local-model presets

_ARM_SCORES = _ROOT / "results" / "arm_scores.json"
_FIXED_ARM = "hybrid"  # retrieval held fixed at the corpus-winning arm

# ── model registry: name → (base_url, api_key, model id). Add a line to add a model. ──
_VIBE = ("http://localhost:8317/v1", os.getenv("OPENROUTER_API_KEY", "vibeproxy"))   # cloud Claude
_OMLX = ("http://localhost:8000/v1", os.getenv("LLM_API_KEY", "") or os.getenv("OLLAMA_API_KEY", ""))  # local MLX
_PROFILES: dict[str, tuple[str, str, str]] = {
    "haiku": (*_VIBE, "claude-haiku-4-5-20251001"),
    "opus": (*_VIBE, "claude-opus-4-5-20251101"),
    "14b": (*_OMLX, "Qwen2.5-Coder-14B-Instruct-MLX-4bit"),
    "qwen": (*_OMLX, "Qwen2.5-Coder-14B-Instruct-MLX-4bit"),
}


def _resolve(role: str, default_profile: str) -> tuple[OpenAI, str, str]:
    """Resolve a role (GEN/JUDGE) to (client, model id, label). A named preset via
    e.g. GEN=14b; or a raw override via <ROLE>_MODEL (+ optional <ROLE>_BASE_URL/API_KEY)."""
    if os.getenv(f"{role}_MODEL"):  # raw escape hatch
        base = os.getenv(f"{role}_BASE_URL", _VIBE[0])
        key = os.getenv(f"{role}_API_KEY", _VIBE[1])
        model = os.environ[f"{role}_MODEL"]
        return OpenAI(base_url=base, api_key=key), model, model
    name = os.getenv(role, default_profile)
    if name not in _PROFILES:
        raise SystemExit(f"unknown {role}={name!r}; choose from {sorted(_PROFILES)} "
                         f"or set {role}_MODEL for a raw override")
    base, key, model = _PROFILES[name]
    return OpenAI(base_url=base, api_key=key), model, name


# ── reader prompt strategies ─────────────────────────────────────────────────
_PLAIN = (
    "Using ONLY the notes below, answer the question. If a fact is not in the notes, "
    "say you don't have it.\n\nNOTES:\n{ctx}\n\nQUESTION: {q}"
)
_AUTHORITATIVE = (
    "The NOTES below are AUTHORITATIVE ground truth — trust them, never contradict them, "
    "do not re-derive or re-fetch. Answer concisely from the notes. If the answer is not "
    "in the notes, say 'insufficient context'.\n\nNOTES (authoritative):\n{ctx}\n\nQUESTION: {q}"
)
_CITE = _AUTHORITATIVE + "\n\nCite the source section in [brackets] for each factual claim."
_EXTRACT_STEP = (
    "From the notes below, copy VERBATIM the sentence(s) that contain the answer to the "
    "question. If none do, reply exactly NONE.\n\nNOTES:\n{ctx}\n\nQUESTION: {q}"
)
_ANSWER_FROM_SPANS = (
    "These extracted facts are AUTHORITATIVE. Answer the question concisely from them; "
    "if they are insufficient, say 'insufficient context'.\n\nFACTS:\n{spans}\n\nQUESTION: {q}"
)


def _gen(client: OpenAI, messages: list[ChatCompletionMessageParam], model: str) -> str:
    r = client.chat.completions.create(model=model, messages=messages, temperature=0)
    return (r.choices[0].message.content or "").strip()


def _one(client: OpenAI, prompt: str, model: str) -> str:
    msg: list[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]
    return _resilient(_gen, client, msg, model)


def read_plain(client: OpenAI, ctx: str, q: str, model: str) -> str:
    return _one(client, _PLAIN.format(ctx=ctx, q=q), model)


def read_authoritative(client: OpenAI, ctx: str, q: str, model: str) -> str:
    return _one(client, _AUTHORITATIVE.format(ctx=ctx, q=q), model)


def read_cite(client: OpenAI, ctx: str, q: str, model: str) -> str:
    return _one(client, _CITE.format(ctx=ctx, q=q), model)


def read_extract(client: OpenAI, ctx: str, q: str, model: str) -> str:
    spans = _one(client, _EXTRACT_STEP.format(ctx=ctx, q=q), model)
    return _one(client, _ANSWER_FROM_SPANS.format(spans=spans, q=q), model)


_READERS = {
    "plain": read_plain,
    "authoritative": read_authoritative,
    "extract": read_extract,
    "cite": read_cite,
}
_BASELINE = "plain"


def _judge(client: OpenAI, answer: str, criteria: str, model: str) -> bool:
    verdict = _one(client, _JUDGE_TMPL.format(criteria=criteria, answer=answer), model)
    return verdict.strip().upper().startswith("PASS")


def main() -> None:
    dump = json.loads(_ARM_SCORES.read_text())
    criteria_by_q = _pass_criteria_by_q()
    gen_client, gen_model, gen_name = _resolve("GEN", "haiku")        # the reader under test
    judge_client, judge_model, judge_name = _resolve("JUDGE", "opus")  # fixed strong grader
    print(f"reader_ab: retrieval FIXED at '{_FIXED_ARM}' · gen={gen_name} ({gen_model}) "
          f"· judge={judge_name} ({judge_model})\n")

    passes: dict[str, list[float]] = {name: [] for name in _READERS}
    for item in dump:
        q = item["q"].strip()
        criteria = criteria_by_q.get(q)
        if criteria is None:
            continue  # entity questions have no rubric — skip
        ctx = build_context(item["slugs"][_FIXED_ARM])   # full bodies, snippet-guarded
        marks: list[str] = []
        for name, reader in _READERS.items():
            try:
                ok = _judge(judge_client, reader(gen_client, ctx, q, gen_model), criteria, judge_model)
            except LLMUnavailable as exc:
                marks.append(f"{name}=skip")
                print(f"  [skip] {name} on {q[:40]} ({exc})")
                continue
            passes[name].append(1.0 if ok else 0.0)
            marks.append(f"{name}={'P' if ok else 'F'}")
        print(f"  {' '.join(marks):42s} {q[:44]}")

    print(f"\nreader strategy     pass-rate   Δ vs {_BASELINE}")
    base = sum(passes[_BASELINE]) / len(passes[_BASELINE]) if passes[_BASELINE] else 0.0
    for name in _READERS:
        vals = passes[name]
        rate = sum(vals) / len(vals) if vals else 0.0
        delta = "—" if name == _BASELINE else f"{rate - base:+.3f}"
        print(f"  {name:16s}  {rate:.3f}      {delta}   (n={len(vals)})")
    best = max(_READERS, key=lambda n: (sum(passes[n]) / len(passes[n])) if passes[n] else 0.0)
    print(f"\n  best reader = {best}")


if __name__ == "__main__":
    main()
