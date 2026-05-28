# code/llm.py — single chat() helper; swap providers via env vars
"""Tiny LLM provider abstraction. All chapter code calls llm.chat(prompt, system).

Providers (selected via LLM_PROVIDER env var):
  - "anthropic-proxy" — Claude-Sonnet-4.6 via local :8317 proxy (curriculum default)
  - "openai"          — OpenAI-compatible endpoint (Azure OpenAI / local oMLX / vLLM)
  - "mock"            — deterministic stub for offline tests (see tests/conftest.py)

Environment: python-dotenv loads `.env` automatically at import time.
Walks from cwd up to filesystem root looking for `.env` — finds the lab's
`.env` AND `~/code/agent-prep/.env` (umbrella) without `source .env`.
"""
from __future__ import annotations
import os
import httpx

# Auto-load .env on module import. find_dotenv() walks up the directory tree.
# Existing process env (real shell exports) takes precedence over .env values.
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    # python-dotenv optional — caller can still `source .env` manually.
    pass

def _provider() -> str:
    """Resolve provider at CALL time, not import time. Allows pytest
    monkeypatch.setenv to override LLM_PROVIDER per-test."""
    return os.getenv("LLM_PROVIDER", "anthropic-proxy")


def _timeout_s() -> float:
    return float(os.getenv("LLM_TIMEOUT_S", "60"))


def chat(prompt: str, system: str | None = None, max_tokens: int = 1024) -> str:
    """Send (system, prompt) → return assistant text. Sync; uses httpx.

    `max_tokens` defaults to 1024. Bump higher for synthesis-heavy calls
    when the model is a reasoning model (e.g. gpt-oss-20b) whose CoT
    consumes the budget before the final answer emits — see W3.5.8 BCJ
    Entry 8 for the canonical 'finish_reason=length, content=None' trap.
    """
    provider = _provider()
    if provider == "anthropic-proxy":
        return _chat_anthropic_proxy(prompt, system, max_tokens)
    if provider == "openai":
        return _chat_openai(prompt, system, max_tokens)
    if provider == "mock":
        return _chat_mock(prompt, system)
    raise ValueError(f"unknown LLM_PROVIDER: {provider}")


def _chat_anthropic_proxy(prompt: str, system: str | None, max_tokens: int = 1024) -> str:
    """Claude-Sonnet-4.6 via local :8317 proxy. User-only payload avoids the
    proxy's system-field overwrite (see W3.5.8 BCJ Entry 19)."""
    url = os.getenv("ANTHROPIC_BASE_URL", "http://localhost:8317") + "/v1/messages"
    body = {
        "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        "max_tokens": max_tokens,
        "messages": [{
            "role": "user",
            "content": (f"[INSTRUCTIONS]\n{system}\n\n[USER MESSAGE]\n{prompt}"
                        if system else prompt),
        }],
    }
    headers = {
        "x-api-key": os.getenv("ANTHROPIC_API_KEY", "dummy"),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    r = httpx.post(url, json=body, headers=headers, timeout=_timeout_s())
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def _chat_openai(prompt: str, system: str | None, max_tokens: int = 1024) -> str:
    """OpenAI-compatible chat.completions endpoint (Azure / vLLM / oMLX).

    Env-var precedence (agent-prep convention):
      OMLX_*   — local oMLX server (canonical for the curriculum's labs)
      OPENAI_* — generic OpenAI-compatible (Azure, public OpenAI, etc.)
    Whichever is set wins; OMLX_* takes precedence when BOTH are set.
    """
    base_url = (
        os.getenv("OMLX_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "http://localhost:8000/v1"
    )
    api_key = (
        os.getenv("OMLX_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "sk-local"
    )
    model = (
        os.getenv("OMLX_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-oss-20b-MXFP4-Q8"
    )
    url = base_url.rstrip("/") + "/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    r = httpx.post(url, json=body, headers=headers, timeout=_timeout_s())
    r.raise_for_status()
    data = r.json()
    # Defensive: reasoning models (gpt-oss-20b, DeepSeek-R1, o1-class) emit
    # chain-of-thought into `reasoning_content` and the final answer into
    # `content`. On heavy prompts the CoT can exhaust max_tokens entirely,
    # leaving `content=null` + `finish_reason=length` + `reasoning_content`
    # holding the truncated thinking. Falling back to reasoning_content
    # salvages SOMETHING for the caller — even if it's just incomplete CoT,
    # it's better than a silently empty string. See W3.5.5.5 BCJ Entry 6
    # for the diagnostic probe + full trap analysis.
    try:
        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content")
        if content:
            return content
        # Fall back to reasoning_content if final content is empty/null.
        reasoning = msg.get("reasoning_content")
        if reasoning:
            finish_reason = choice.get("finish_reason", "unknown")
            return f"[reasoning_only — finish_reason={finish_reason}]\n{reasoning}"
        return ""
    except (KeyError, IndexError, TypeError):
        return ""


def _chat_mock(prompt: str, system: str | None) -> str:
    """Format-aware deterministic stub. Inspects (system, prompt) and returns
    a response that satisfies the calling code's parse expectations.

    Recognized formats:
      - decompose plan      → JSON {"sub_questions": [...]}
      - synthesize          → multi-sentence answer
      - triage / handoff    → "HANDOFF: <tool_name>"
      - group-chat selector → single agent name
      - LLM-judge           → "BEST: <id>\\nREASON: ..."
      - default             → "Mock response."
    """
    import re
    sys = (system or "").lower()
    p = prompt.lower()

    # Supervisor decomposition: needs JSON shape with sub_questions list
    if "decompose" in sys and ("sub-question" in sys or "json" in sys):
        return '{"sub_questions": ["mock sub-q 1", "mock sub-q 2", "mock sub-q 3"]}'

    # Synthesis prompt → non-trivial multi-sentence answer
    if "synthesi" in sys:
        return ("Mock synthesized answer. Combines worker outputs into one summary. "
                "Surfaces no disagreement because workers are mocked.")

    # Triage handoff: emit HANDOFF: <tool> based on USER MESSAGE keywords.
    # Important: tool docstrings are embedded in the prompt; extract just the
    # USER MESSAGE line to avoid false-matching on tool descriptions.
    if "triage" in sys:
        user_msg_match = re.search(r"USER MESSAGE:\s*(.+?)(?:\n|$)", prompt)
        umsg = (user_msg_match.group(1) if user_msg_match else prompt).lower()
        if any(k in umsg for k in ("refund", "money", "billing", "credit card", "charge")):
            return "HANDOFF: transfer_to_refunds"
        if any(k in umsg for k in ("plan", "upgrade", "pricing", "enterprise", "subscribe", "difference between")):
            return "HANDOFF: transfer_to_sales"
        return "I can help with that directly."

    # Group-chat speaker selector: "Pick ONE of: coder/reviewer/tester"
    m = re.search(r"[Pp]ick\s+(?:one\s+of)?:\s*([\w/]+)", prompt)
    if m:
        return m.group(1).split("/")[0]

    # LLM-judge: "Which solver's ANSWER is most accurate?"
    if "which solver" in p or "best:" in p:
        return "BEST: 0\nREASON: mock judge picks solver 0"

    # Worker / specialist: 3-sentence factual response
    if any(k in sys for k in ("worker", "refund specialist", "sales specialist")):
        return "Mock answer line one. Line two. Line three end."

    # Solver agents — return one of "42" / "yes" / "Paris" with ANSWER: prefix
    if "answer:" in sys:
        return "Reasoning step. Reasoning step. ANSWER: 42"

    # Default fallback
    return "Mock response."