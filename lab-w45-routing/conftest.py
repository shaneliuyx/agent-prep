"""Root conftest.

Two jobs:
1. Its mere presence at the repo root makes pytest add this directory to
   sys.path[0] (prepend import mode), so `from src.router import ...` resolves
   when tests run from anywhere. Without it, pytest only puts tests/ on the path
   and `src` is not importable.
2. Gate `@pytest.mark.integration` tests (they hit the live oMLX endpoint) behind
   RUN_INTEGRATION=1, matching the agent-prep convention of skip-by-default so a
   plain `uv run pytest` stays offline and fast.
"""
import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: hits a live LLM endpoint; skipped unless RUN_INTEGRATION=1",
    )
    config.addinivalue_line("markers", "slow: long-running benchmark (minutes)")


def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="integration test; set RUN_INTEGRATION=1 to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
