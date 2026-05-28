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


# Back-compat shims for any older callers reading module-level constants.
_PROVIDER = _provider()
_TIMEOUT_S = _timeout_s()


def chat(prompt: str, system: str | None = None) -> str:
    """Send (system, prompt) → return assistant text. Sync; uses httpx."""
    provider = _provider()
    if provider == "anthropic-proxy":
        return _chat_anthropic_proxy(prompt, system)
    if provider == "openai":
        return _chat_openai(prompt, system)
    if provider == "mock":
        return _chat_mock(prompt, system)
    raise ValueError(f"unknown LLM_PROVIDER: {provider}")


def _chat_anthropic_proxy(prompt: str, system: str | None) -> str:
    """Claude-Sonnet-4.6 via local :8317 proxy. User-only payload avoids the
    proxy's system-field overwrite (see W3.5.8 BCJ Entry 19)."""
    url = os.getenv("ANTHROPIC_BASE_URL", "http://localhost:8317") + "/v1/messages"
    body = {
        "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        "max_tokens": 1024,
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
    r = httpx.post(url, json=body, headers=headers, timeout=_TIMEOUT_S)
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def _chat_openai(prompt: str, system: str | None) -> str:
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
        "max_tokens": 1024,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    r = httpx.post(url, json=body, headers=headers, timeout=_timeout_s())
    r.raise_for_status()
    data = r.json()
    # Defensive: some OpenAI-compat servers return null content when model
    # emits tool_calls or hits a stop sequence; others return empty string.
    # Caller expects str; coerce None/missing to empty string.
    try:
        return data["choices"][0]["message"]["content"] or ""
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