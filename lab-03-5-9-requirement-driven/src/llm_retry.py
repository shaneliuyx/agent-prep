"""Retry wrapper for VibeProxy (:8317) chat calls.

VibeProxy routes to Claude via a metered subscription auth that COOLS DOWN under
sustained load, returning HTTP 503 `auth_unavailable: ... cooldown state`. The
cooldown is short (seconds): a 503 is followed by a clean response moments later.
So the right client behavior is exponential backoff + retry, not failing the call.

Used by every VibeProxy-bound call (reader + complex jobs: consolidation, dedup,
mem0). Local oMLX calls don't need it (unmetered), but the wrapper is harmless
there too. Pure stdlib, no Date.now/random dependency on wall-clock semantics
beyond sleeping.
"""
from __future__ import annotations

import time
from typing import Any

# 503/auth cooldown markers VibeProxy emits. Match on the message text so we
# only retry the cooldown case, not genuine 4xx/contract errors.
_COOLDOWN_MARKERS = ("auth_unavailable", "cooldown", "503", "no auth available")

# Backoff schedule (seconds). ~total 2+4+8+16+30+30 ≈ 90s of patience before
# giving up — enough to ride out a typical cooldown without hanging forever.
_BACKOFFS = (2, 4, 8, 16, 30, 30)


def _is_cooldown(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(m in s for m in _COOLDOWN_MARKERS)


# VibeProxy intermittently returns a 200-OK Claude-Code PERSONA REFUSAL instead
# of answering (the cloak — it routes through Claude Code's interactive system
# prompt). These are not answers; detect + retry, and never let the persona text
# become a prediction.
_CLOAK_MARKERS = (
    "i'm claude code", "i am claude code", "anthropic's cli", "anthropic's official cli",
    "i appreciate you sharing", "i need to clarify my role", "clarify my role",
    "as claude code",
)


def is_cloak(text: str) -> bool:
    s = (text or "").lower()
    return any(m in s for m in _CLOAK_MARKERS)


def call_with_retry(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call fn(*args, **kwargs) with backoff on VibeProxy 503 cooldowns.
    Re-raises immediately on non-cooldown errors and after backoff is exhausted.
    Generic so SDK-internal call sites (e.g. mem0's OpenAILLM.generate_response)
    can reuse the same cooldown handling."""
    last: Exception | None = None
    for delay in (0, *_BACKOFFS):
        if delay:
            time.sleep(delay)
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — classify then re-raise
            if not _is_cooldown(exc):
                raise
            last = exc
    assert last is not None
    raise last


def chat_with_retry(client: Any, **create_kwargs: Any) -> Any:
    """client.chat.completions.create(**kwargs) with VibeProxy cooldown backoff."""
    return call_with_retry(client.chat.completions.create, **create_kwargs)
