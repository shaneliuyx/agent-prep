"""W4.5 Phase 6 — privacy router verification (offline, deterministic, $0).

No LLM, no network: exercises rule_detect/privacy_route on a small hand-built
probe set and asserts the HEADLINE SAFETY METRIC — zero false-negative leaks
(a sensitive prompt that scored S1 and would have escaped to a cloud tier).
The false-POSITIVE rate (a benign prompt over-restricted to local) is measured
and printed but NOT gated: over-restriction is safe-but-costly, a leak is
irreversible (§6 insight). Runs in the default `uv run pytest` lane — no
integration marker, since the LLM detector ships stubbed to False.

Run:  uv run pytest tests/test_privacy_router.py -v -s
"""
from src.privacy_router import privacy_route

# (prompt, expected_level). The set is deliberately mixed:
#   S3/S2 POSITIVES carry real-shaped secrets -> they probe the LEAK rate.
#   S1 HARD-NEGATIVES name a sensitive TOPIC but contain no secret -> they
#     probe the false-POSITIVE rate (the keyword list over-triggering).
PRIVACY_PROBES = [
    # --- S3: secrets / credentials / regulated PII -> must force LOCAL ---
    ("here's my id_rsa, debug the ssh config", "S3"),
    ("-----BEGIN OPENSSH PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...", "S3"),
    ("my aws key is AKIAIOSFODNN7EXAMPLE, rotate it", "S3"),
    ("card on file 4111 1111 1111 1111, issue a refund", "S3"),
    ("我的身份证号能在系统里查到吗", "S3"),
    # --- S2: internal / commercial PII -> redact then allow cloud ---
    ("prod db is at postgres://app:hunter2@10.2.3.4:5432/main", "S2"),
    ("rotate the github token ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "S2"),
    ("what's a fair salary band for a staff eng in Dubai", "S2"),
    # --- S1: sensitive TOPIC, no actual secret -> PASSTHROUGH to classifier ---
    ("explain how RSA private keys are generated", "S1"),
    ("explain TCP vs UDP for a backend dev", "S1"),
    ("what does a 401 vs 403 status code mean", "S1"),
    ("how do password managers store credentials safely", "S1"),  # known FP: trips keyword
]

_SENSITIVE = {"S2", "S3"}


def test_privacy_router_no_leaks():
    """Hard gate: NO sensitive prompt may score S1 (leak to cloud). Also asserts
    the most-critical class (S3) is detected exactly, and measures the
    false-positive (over-restriction) rate for the record."""
    leaks, false_pos, s3_miss = [], [], []

    for prompt, expected in PRIVACY_PROBES:
        v = privacy_route(prompt)  # use_llm_detector defaults False -> rules-only
        if expected in _SENSITIVE and v.level == "S1":
            leaks.append((prompt, v))                       # false negative — DANGER
        if expected == "S3" and v.level != "S3":
            s3_miss.append((prompt, v))                     # critical class mis-detected
        if expected == "S1" and v.level in _SENSITIVE:
            false_pos.append((prompt, v))                   # over-restriction — safe but costly

    n = len(PRIVACY_PROBES)
    n_neg = sum(1 for _, e in PRIVACY_PROBES if e == "S1")
    n_pos = sum(1 for _, e in PRIVACY_PROBES if e in _SENSITIVE)
    print(f"\n=== privacy router (rules-only, n={n}) ===")
    print(f"leak_rate (FN):        {len(leaks)}/{n_pos}")
    print(f"false_positive (FP):   {len(false_pos)}/{n_neg}  (over-restricted to local)")
    for p, v in false_pos:
        print(f"  FP -> {v.level} ({v.matched}): {p[:48]!r}")

    # The safety guarantee: zero leaks, and the critical S3 class detected exactly.
    assert not leaks, f"LEAK: sensitive prompt(s) scored S1 -> {leaks}"
    assert not s3_miss, f"S3 mis-detected (critical class) -> {s3_miss}"
    # FP is measured, not gated — over-restriction is the safe failure direction.
