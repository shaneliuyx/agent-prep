"""W4.5 Phase 6 — privacy/sensitivity router (the third routing axis).

Ports ClawXRouter's two-phase detect->route into the Python lab idiom:
a fast deterministic rule pass runs first; only on a CLEAN rule result do
we pay for the local LLM detector. A sensitive hit short-circuits straight
to the local fleet -- it never reaches the tier classifier, never the cloud.

Local-first framing: the MLX fleet IS the edge. 'redirect' (S3) is the safe
default for this stack; the privacy router's real job is gating what is
ALLOWED to escalate to a cloud tier ('passthrough' / 'desensitize').
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Sensitivity = Literal["S1", "S2", "S3"]
Action = Literal["passthrough", "desensitize", "redirect"]


# Trimmed from ClawXRouter src/rules.ts. S3 (most sensitive) checked before S2.
PRIVACY_RULES: dict[str, dict[str, list[str]]] = {
    "S3": {  # secrets / credentials / regulated PII -> force local, never cloud
        "keywords": ["id_rsa", "private_key", ".pem", ".env", "master_password",
                     "身份证", "银行卡", "病历", "密钥"],
        "patterns": [
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            r"AKIA[0-9A-Z]{16}",                              # AWS access key id
            r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{8,}",
            r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",    # card number
        ],
    },
    "S2": {  # internal / commercial PII -> redact then allow cloud
        "keywords": ["password", "api_key", "secret", "token", "credential",
                     "salary", "合同", "客户", "intranet"],
        "patterns": [
            r"\b(?:10|172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b",  # private IP
            r"(?:mysql|postgres|mongodb|redis)://[^\s]+",                          # db dsn
            r"\b(?:sk|key|token)-[A-Za-z0-9]{16,}\b",
            r"(?i)ghp_[a-zA-Z0-9]{36}",                                            # github pat
        ],
    },
}

_ACTION: dict[str, Action] = {"S3": "redirect", "S2": "desensitize", "S1": "passthrough"}


@dataclass(frozen=True)
class PrivacyVerdict:
    level: Sensitivity
    action: Action
    matched: str  # the rule that fired ("" for S1) — audit trail


def rule_detect(text: str) -> PrivacyVerdict | None:
    """Fast, deterministic, free. Check S3 before S2 (highest sensitivity wins).
    Returns None on a clean pass so the caller decides whether to run llm_detect().
    """
    low = text.lower()
    for level in ("S3", "S2"):                       # order matters — most sensitive first
        rule = PRIVACY_RULES[level]
        for kw in rule["keywords"]:
            if kw.lower() in low:
                return PrivacyVerdict(level, _ACTION[level], f"keyword:{kw}")
        for pat in rule["patterns"]:
            if re.search(pat, text):
                return PrivacyVerdict(level, _ACTION[level], f"pattern:{pat[:24]}")
    return None


def privacy_route(text: str, use_llm_detector: bool = False) -> PrivacyVerdict:
    """Two-phase: rules first (cheap), LLM detector only on a clean rule pass.
    Defense-in-depth — regex catches known shapes; the LLM detector is the
    backstop for novel phrasings the patterns miss (BCJ Entry 7).
    """
    verdict = rule_detect(text)
    if verdict is not None:
        return verdict
    if use_llm_detector and _llm_flags_sensitive(text):
        return PrivacyVerdict("S2", "desensitize", "llm-detector")
    return PrivacyVerdict("S1", "passthrough", "")


def _llm_flags_sensitive(text: str) -> bool:
    """Local Qwen3.5-4B as a yes/no sensitivity backstop. Stubbed False here;
    wire to FLEET['classifier'] with a strict 'reply Y or N' prompt in the lab
    so Phase 6 stays runnable without the detector enabled.
    """
    return False