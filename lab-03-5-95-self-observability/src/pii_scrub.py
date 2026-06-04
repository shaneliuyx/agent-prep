"""W3.5.95 — PII / secret scrubbing for the OBSERVABILITY write boundary.

Two backends behind one `scrub_pii(text) -> str` call:

  * PRIMARY — Microsoft Presidio (presidio-analyzer + presidio-anonymizer). An
    NLP recognizer engine (spaCy NER + pattern recognizers) DETECTS PII spans,
    then the anonymizer REPLACES each span with a typed placeholder. This catches
    what a fixed regex can't — PERSON names, locations, phone numbers, credit
    cards, IPs, SSNs — because detection is contextual, not literal.
  * FALLBACK — the original regex scrubber. Used when Presidio (or its spaCy
    model) isn't installed, so the lab stays runnable without the heavy dependency
    and degrades instead of crashing.

Custom recognizers add the secret shapes Presidio has no built-in for: OpenAI-style
API keys, Bearer tokens, /Users/ home paths, long hex tokens.

DESIGN TRADE-OFF (chapter Phase 1 / BCJ Entry 7): Presidio loads a spaCy model
once (lazy singleton here) and runs NER per call — tens of ms, vs the regex's
microseconds. On a high-volume OBSERVABILITY hot path that cost is real; the
singleton amortizes the model load, and for extreme volume scrubbing would move
to a batched/async lane. The accuracy gain (named-entity PII) is usually worth it.
"""
from __future__ import annotations

import re

# ── Regex fallback (also the original write-boundary scrubber) ───────────────
_REGEX_SCRUBBERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "<API_KEY>"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{16,}"), "Bearer <TOKEN>"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<EMAIL>"),
    (re.compile(r"/Users/[^/\s\"']+"), "/Users/<USER>"),
    (re.compile(r"\b[A-Fa-f0-9]{32,}\b"), "<HEX>"),
]


def _regex_scrub(text: str) -> str:
    for pat, repl in _REGEX_SCRUBBERS:
        text = pat.sub(repl, text)
    return text


# ── Presidio backend (lazy singleton: spaCy model load is expensive) ─────────
# Custom secret recognizers + the operators that name their placeholders.
_SECRET_PATTERNS = [
    ("API_KEY", r"sk-[A-Za-z0-9_-]{16,}", 0.9),
    ("BEARER_TOKEN", r"(?i)bearer\s+[A-Za-z0-9._-]{16,}", 0.85),
    ("USER_PATH", r"/Users/[^/\s\"']+", 0.8),
    ("HEX_TOKEN", r"\b[A-Fa-f0-9]{32,}\b", 0.6),
]
# Placeholder overrides; entities not listed fall back to Presidio's default
# `<ENTITY_TYPE>` (so PERSON → <PERSON>, CREDIT_CARD → <CREDIT_CARD>, etc.).
_PLACEHOLDERS = {
    "API_KEY": "<API_KEY>",
    "BEARER_TOKEN": "Bearer <TOKEN>",
    "USER_PATH": "/Users/<USER>",
    "HEX_TOKEN": "<HEX>",
    "EMAIL_ADDRESS": "<EMAIL>",
}

_ENGINES: dict | None = None  # {"analyzer", "anonymizer", "operators"} or sentinel


def _build_engines() -> dict | None:
    """Construct the Presidio analyzer (predefined recognizers + our secret
    recognizers, on a SMALL spaCy model to keep the lab light) and anonymizer.
    Returns None if Presidio or the spaCy model isn't available (→ regex fallback)."""
    try:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig
    except ImportError:
        return None

    try:
        # en_core_web_sm: ~12MB vs en_core_web_lg ~560MB. Weaker NER recall but
        # ample for the lab and far lighter on disk / first-call latency.
        nlp_engine = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }).create_engine()
    except Exception:
        return None  # spaCy model not downloaded → fall back

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()  # EMAIL_ADDRESS, PERSON, CREDIT_CARD, IP, …
    for entity, regex, score in _SECRET_PATTERNS:
        registry.add_recognizer(PatternRecognizer(
            supported_entity=entity,
            patterns=[Pattern(name=entity.lower(), regex=regex, score=score)]))

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
    operators = {ent: OperatorConfig("replace", {"new_value": ph})
                 for ent, ph in _PLACEHOLDERS.items()}
    return {"analyzer": analyzer, "anonymizer": AnonymizerEngine(), "operators": operators}


def _engines() -> dict | None:
    global _ENGINES
    if _ENGINES is None:
        _ENGINES = _build_engines() or {}  # {} = "tried, unavailable" → don't retry
    return _ENGINES or None


def scrub_pii(text: str) -> str:
    """Redact PII/secrets. Presidio (NER + pattern recognizers) when available;
    regex fallback otherwise. Idempotent enough for an append-only write boundary."""
    eng = _engines()
    if eng is None:
        return _regex_scrub(text)
    results = eng["analyzer"].analyze(text=text, language="en")
    if not results:
        return text
    return eng["anonymizer"].anonymize(
        text=text, analyzer_results=results, operators=eng["operators"]).text


def backend() -> str:
    """Which scrubber is active — 'presidio' or 'regex'. For the lab's RESULTS."""
    return "presidio" if _engines() is not None else "regex"
