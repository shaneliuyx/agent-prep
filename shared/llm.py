"""shared/llm.py — cross-chapter LLM utilities: client, model presets, resilient calls, judge.

Introduced by W3.5.96 (reader_ab.py model presets + resilient retry + LLM judge;
answer_route_ab.py / verify_arch.py). W3.5.95 (self-observability) builds its OpenAI client
the same way (OMLX_BASE_URL / LLM_BASE_URL) — this generalizes both so either chapter's env
names resolve. GBrain-specific plumbing lives in `gbrain_cli.py`, not here (this module is
provider-agnostic: any OpenAI-compatible endpoint — local oMLX :8000, VibeProxy :8317 → Claude).

Use from a consuming lab:
    import sys, pathlib
    sys.path.insert(0, "/Users/yuxinliu/code/agent-prep/shared")
    from llm import make_client, resolve, chat, judge, resilient, LLMUnavailable, load_pass_criteria
"""
from __future__ import annotations

import json
import os
import pathlib
import time

from openai import APIConnectionError, APIError, OpenAI
from openai.types.chat import ChatCompletionMessageParam


# ── endpoint defaults: walk an env chain so every chapter's names resolve ─────
def _default_base() -> str:
    return (os.getenv("LLM_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")
            or os.getenv("OMLX_BASE_URL") or "http://localhost:8000/v1")


# Non-empty sentinel: the OpenAI SDK rejects an empty api_key at construction; local endpoints
# (oMLX) authenticate per-request, so a placeholder lets the client build and fail (gracefully)
# at call time rather than crashing at construction when no .env is loaded.
_KEY_SENTINEL = "EMPTY"


def _default_key() -> str:
    return (os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OLLAMA_API_KEY") or _KEY_SENTINEL)


def make_client(base_url: str | None = None, api_key: str | None = None) -> OpenAI:
    """An OpenAI-compatible client; unset args fall back through the env chain above."""
    return OpenAI(base_url=base_url or _default_base(), api_key=api_key or _default_key())


# ── named model presets: name → (base_url, api_key, model id). Add a line to add one. ──
_VIBE = ("http://localhost:8317/v1", os.getenv("OPENROUTER_API_KEY", "vibeproxy"))   # cloud Claude
_OMLX = ("http://localhost:8000/v1",
         os.getenv("LLM_API_KEY") or os.getenv("OLLAMA_API_KEY") or _KEY_SENTINEL)  # local MLX
PROFILES: dict[str, tuple[str, str, str]] = {
    "haiku": (*_VIBE, "claude-haiku-4-5-20251001"),
    "opus": (*_VIBE, "claude-opus-4-5-20251101"),
    "14b": (*_OMLX, "Qwen2.5-Coder-14B-Instruct-MLX-4bit"),
    "qwen": (*_OMLX, "Qwen2.5-Coder-14B-Instruct-MLX-4bit"),
}


def resolve(role: str, default_profile: str) -> tuple[OpenAI, str, str]:
    """Resolve a ROLE (e.g. "GEN"/"JUDGE") to (client, model id, label) with no code change:
      - named preset via `<ROLE>=haiku|opus|14b|...`
      - raw override via `<ROLE>_MODEL` (+ optional `<ROLE>_BASE_URL` / `<ROLE>_API_KEY`)."""
    if os.getenv(f"{role}_MODEL"):
        base = os.getenv(f"{role}_BASE_URL", _default_base())
        key = os.getenv(f"{role}_API_KEY", _default_key())
        model = os.environ[f"{role}_MODEL"]
        return OpenAI(base_url=base, api_key=key), model, model
    name = os.getenv(role, default_profile)
    if name not in PROFILES:
        raise SystemExit(f"unknown {role}={name!r}; choose from {sorted(PROFILES)} "
                         f"or set {role}_MODEL for a raw override")
    base, key, model = PROFILES[name]
    return OpenAI(base_url=base, api_key=key), model, name


# ── resilient calls (flaky local servers drop connections under load) ─────────
class LLMUnavailable(RuntimeError):
    """Endpoint refused after retries — the caller should SKIP this item, not crash the run."""


def resilient(fn, *args, retries: int = 4, backoff: float = 2.0):
    """Retry an LLM call through transient connection drops; raise LLMUnavailable if it never recovers."""
    for attempt in range(retries):
        try:
            return fn(*args)
        except (APIConnectionError, APIError) as exc:
            if attempt == retries - 1:
                raise LLMUnavailable(str(exc)) from exc
            time.sleep(backoff * (attempt + 1))
    raise LLMUnavailable("unreachable")


# ── chat + LLM judge ──────────────────────────────────────────────────────────
def chat_usage(client: OpenAI, prompt_or_messages, model: str,
               temperature: float = 0.0,
               max_tokens: int | None = None) -> tuple[str, dict[str, int]]:
    """One completion, returning (text, usage). `usage` is a dict with
    prompt_tokens / completion_tokens / total_tokens (zeros if the endpoint omits
    usage). Use this when you need REAL token counts — cost metering, throughput
    benches — instead of re-reading `response.usage` off a raw client. Plain
    chat() drops usage for the common text-only path. `max_tokens` caps generation
    when set (left to the server default when None)."""
    messages: list[ChatCompletionMessageParam] = (
        [{"role": "user", "content": prompt_or_messages}]
        if isinstance(prompt_or_messages, str) else prompt_or_messages)
    if max_tokens is not None:
        r = client.chat.completions.create(model=model, messages=messages,
                                           temperature=temperature, max_tokens=max_tokens)
    else:
        r = client.chat.completions.create(model=model, messages=messages,
                                           temperature=temperature)
    text = (r.choices[0].message.content or "").strip()
    u = getattr(r, "usage", None)
    usage = {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        "total_tokens": getattr(u, "total_tokens", 0) or 0,
    }
    return text, usage


def chat(client: OpenAI, prompt_or_messages, model: str, temperature: float = 0.0) -> str:
    """One completion → text only (usage discarded). Accepts a raw prompt string or
    a messages list. temp 0 by default. For real token counts use chat_usage()."""
    text, _ = chat_usage(client, prompt_or_messages, model, temperature)
    return text


JUDGE_TMPL = (
    "You are grading a candidate answer against a rubric. Reply with EXACTLY one word: "
    "PASS or FAIL.\n\nRUBRIC (what makes the answer correct):\n{criteria}\n\n"
    "CANDIDATE ANSWER:\n{answer}\n\nVerdict (PASS or FAIL):"
)


def judge(client: OpenAI, answer: str, criteria: str, model: str) -> bool:
    """LLM judge: PASS/FAIL of `answer` against `criteria`. Retries through drops."""
    verdict = resilient(chat, client, JUDGE_TMPL.format(criteria=criteria, answer=answer), model)
    return verdict.strip().upper().startswith("PASS")


# ── eval data ──────────────────────────────────────────────────────────────────
def load_pass_criteria(ground_truth_path: str | pathlib.Path) -> dict[str, str]:
    """question text → pass_criteria, from a W2.7-style eval_ground_truth.json
    (skips `_`-prefixed keys and entries without a rubric)."""
    raw = json.loads(pathlib.Path(ground_truth_path).expanduser().read_text())
    out: dict[str, str] = {}
    for key, entry in raw.items():
        if key.startswith("_") or not isinstance(entry, dict):
            continue
        if entry.get("q") and entry.get("pass_criteria"):
            out[entry["q"].strip()] = entry["pass_criteria"]
    return out
