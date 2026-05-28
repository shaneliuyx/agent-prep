# tests/conftest.py
import os
import sys
import pytest

# Make code/ importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))


def pytest_configure(config):
    """Register custom markers so pytest doesn't warn about them."""
    config.addinivalue_line(
        "markers",
        "integration: test requires a real LLM endpoint (set LLM_PROVIDER=openai or anthropic-proxy)",
    )


@pytest.fixture(autouse=True)
def _mock_llm_for_unit_tests(monkeypatch, request):
    """Default unit tests use mock LLM (deterministic, fast).
    Tests marked @pytest.mark.integration use the real configured provider."""
    if "integration" not in request.keywords:
        monkeypatch.setenv("LLM_PROVIDER", "mock")
