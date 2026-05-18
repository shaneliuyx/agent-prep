"""Phase 4 — signature-validated MagicMock injection.

Prevents the agent's generated test from passing wrong args silently.

3 mock styles in production preference order:
  1. dependency injection (cleanest, no `patch`)
  2. unittest.mock.patch (most common)
  3. monkey-patch (avoid — order-dependent + flaky)

This module ships (1) + a verification helper for (2).
"""
from __future__ import annotations

from inspect import signature
from typing import Any, Callable
from unittest.mock import MagicMock


def make_validated_mock(real_callable: Callable,
                        return_value: Any = None,
                        side_effect: Any = None) -> MagicMock:
    """Create a MagicMock that ENFORCES the real callable's signature
    via `spec=`. Catches arg-mismatch at call time, not at runtime later."""
    mock = MagicMock(spec=real_callable, name=real_callable.__name__)
    if return_value is not None:
        mock.return_value = return_value
    if side_effect is not None:
        mock.side_effect = side_effect
    return mock


def assert_called_matching_signature(mock: MagicMock,
                                     real_callable: Callable) -> None:
    """After test runs, verify the mock was called with args compatible
    with the real callable's signature. Catches signature drift."""
    sig = signature(real_callable)
    for call in mock.call_args_list:
        try:
            sig.bind(*call.args, **call.kwargs)
        except TypeError as e:
            raise AssertionError(
                f"Mock for {real_callable.__name__} called with args "
                f"incompatible with real signature: {e}"
            ) from e
