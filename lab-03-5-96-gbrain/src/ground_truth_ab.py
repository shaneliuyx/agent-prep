"""Phase 7 — Ground-Truth Hierarchy A/B (memory-os principle, leveraged).

Principle (ClaudioDrews/memory-os): injected memory is AUTHORITATIVE — an agent
must not re-derive or re-fetch facts it already holds. The anti-pattern memory-os
names is "memory-zero": re-establishing context from scratch every turn.

We test it as an A/B over the live GBrain brain — a 5-turn conversation whose
turns chain on overlapping entities (Lin Zhao → Acme AI → its seed → investors):

  - Mode A "memory-zero": every turn issues a FRESH `gbrain query`, fetches the
    top pages, and feeds only that turn's retrieval. Overlapping entities get
    re-retrieved, and per-turn retrieval variance lets the same fact drift.
  - Mode B "ground-truth": retrieve the conversation's subgraph ONCE, inject the
    full pages as authoritative context, and reuse them across turns via history.

Measures: retrieval calls, retrieved-context tokens, total LLM prompt tokens.
Chat LLM via VibeProxy (:8317 → Claude); retrieval + embeddings stay local (oMLX).

Two gotchas this script encodes (both cost a debugging round):
  1. `gbrain query --json` returns only a TRUNCATED snippet — useless as grounding.
     Pull full page bodies with `gbrain get <slug>`.
  2. VibeProxy injects a "you are Claude Code" identity that overrides the system
     role and makes the model REFUSE "questions about people." Frame the task as
     grounded document Q&A in the USER message; don't rely on the system prompt.
"""
from __future__ import annotations

import os
import re
import subprocess

from openai import OpenAI

_BUN = os.path.expanduser("~/.bun/bin")
_GBRAIN = os.path.join(_BUN, "gbrain")
_LINE = re.compile(r"^\[[-\d.]+\]\s+(\S+)\s+--")
_CHAT_MODEL = os.getenv("CHAT_MODEL", "claude-haiku-4-5-20251001")

# Turns deliberately chain on shared entities so a memory-holding agent can reuse.
TURNS = [
    "Who is Lin Zhao?",
    "What company does he lead, and what does it do?",
    "Who invested in that company's seed round?",
    "What other deals is that investor involved in?",
    "Summarize Lin Zhao's professional network in two sentences.",
]

# The instruction lives in the USER turn (system role is overridden by the proxy).
_MEMZERO_TMPL = (
    "You are answering questions from a personal knowledge base (markdown notes). "
    "Using ONLY the notes below, answer the question. If a fact is not in the notes, "
    "say you don't have it.\n\nNOTES:\n{ctx}\n\nQUESTION: {q}"
)
_GROUNDTRUTH_PREAMBLE = (
    "You are answering a short series of questions about a personal knowledge base. "
    "The NOTES below are AUTHORITATIVE ground truth — trust them, never contradict "
    "them, and do not ask to re-fetch anything already present. Answer concisely "
    "from the notes and the conversation so far.\n\nNOTES (authoritative):\n{ctx}"
)


def _server_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = _BUN + os.pathsep + env.get("PATH", "")
    env.setdefault("GBRAIN_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/gbrain")
    env.setdefault("OLLAMA_BASE_URL", "http://localhost:8000/v1")
    return env


def _run(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, env=_server_env()).stdout


def gbrain_query_slugs(q: str, limit: int) -> list[str]:
    """Hybrid retrieval — one call. Returns ranked slugs (snippets are too thin)."""
    slugs: list[str] = []
    for line in _run([_GBRAIN, "query", q, "--json", "--limit", str(limit)]).splitlines():
        m = _LINE.match(line.strip())
        if m:
            slugs.append(m.group(1))
    return slugs


def gbrain_get(slug: str) -> str:
    """Full page body — the actual grounding `query`'s snippet lacks."""
    body = _run([_GBRAIN, "get", slug])
    return "\n".join(ln for ln in body.splitlines() if not ln.startswith(("Starting", "[gbrain")))


def _context(slugs: list[str]) -> str:
    return "\n\n".join(gbrain_get(s) for s in slugs)


def _client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "http://localhost:8317/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY", "vibeproxy"),
    )


def _ask(client: OpenAI, messages: list[dict]) -> tuple[str, int]:
    r = client.chat.completions.create(model=_CHAT_MODEL, messages=messages, temperature=0)
    return (r.choices[0].message.content or "").strip(), (r.usage.prompt_tokens if r.usage else 0)


def run_memory_zero(client: OpenAI) -> dict:
    """Re-query + re-fetch every turn; feed only that turn's retrieval."""
    calls, ctx_chars, prompt_tokens, answers = 0, 0, 0, []
    for q in TURNS:
        ctx = _context(gbrain_query_slugs(q, limit=3))
        calls += 1
        ctx_chars += len(ctx)
        ans, ptok = _ask(client, [{"role": "user", "content": _MEMZERO_TMPL.format(ctx=ctx, q=q)}])
        prompt_tokens += ptok
        answers.append(ans)
    return {"calls": calls, "ctx_tokens": ctx_chars // 4, "prompt_tokens": prompt_tokens, "answers": answers}


def run_ground_truth(client: OpenAI) -> dict:
    """Retrieve the subgraph ONCE, inject full pages as authoritative, reuse."""
    ctx = _context(gbrain_query_slugs("Lin Zhao Acme AI seed round investors network", limit=6))
    calls, ctx_chars = 1, len(ctx)
    history: list[dict] = [{"role": "user", "content": _GROUNDTRUTH_PREAMBLE.format(ctx=ctx)},
                           {"role": "assistant", "content": "Understood — I'll answer from those notes."}]
    prompt_tokens, answers = 0, []
    for q in TURNS:
        history.append({"role": "user", "content": q})
        ans, ptok = _ask(client, history)
        prompt_tokens += ptok
        history.append({"role": "assistant", "content": ans})
        answers.append(ans)
    return {"calls": calls, "ctx_tokens": ctx_chars // 4, "prompt_tokens": prompt_tokens, "answers": answers}


def main() -> None:
    client = _client()
    a = run_memory_zero(client)
    b = run_ground_truth(client)

    def row(name: str, d: dict) -> str:
        return f"{name:<16}{d['calls']:<14}{d['ctx_tokens']:<16}{d['prompt_tokens']:<14}"

    print(f"{'mode':<16}{'retrievals':<14}{'retr. ctx tok':<16}{'LLM prompt tok':<14}")
    print("-" * 60)
    print(row("memory-zero", a))
    print(row("ground-truth", b))
    print(
        f"\nre-query waste avoided by treating memory as ground truth: "
        f"{a['calls'] - b['calls']} retrievals, ~{a['ctx_tokens'] - b['ctx_tokens']} retrieval-context tokens"
    )
    for i, q in enumerate(TURNS):
        print(f"\nQ{i + 1}: {q}")
        print(f"  [memory-zero ] {a['answers'][i][:170]}")
        print(f"  [ground-truth] {b['answers'][i][:170]}")


if __name__ == "__main__":
    main()
