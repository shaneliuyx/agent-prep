"""autoconfig smoke + invariant tests.

Doesn't validate that recommendations are *optimal* (that's an empirical
question, settled by re-running evals). Validates that the probes return
sensible values and the recommendation logic is self-consistent.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "shared"))

from rag_hybrid.autoconfig import (
    SystemProbe,
    probe_system,
    recommend,
    recommend_encode_batch,
)
from rag_hybrid.models import BGE_M3, BGE_RERANKER_V2_M3


def test_probe_returns_sensible_values() -> None:
    p = probe_system()
    assert p.device in ("cuda", "mps", "cpu"), f"unknown device {p.device!r}"
    assert p.total_memory_gb > 0, "memory probe failed"
    assert p.cpu_count >= 1, "cpu_count must be at least 1"
    # fp16 invariants:
    if p.device == "cuda":
        assert p.fp16_safe_for_encoder, "fp16 should be safe on CUDA"
        assert p.fp16_safe_for_reranker, "fp16 should be safe on CUDA reranker"
    if p.device == "mps":
        # Empirically verified clean across 10K passages (5K MS MARCO + 5K
        # tech corpus, all 10 length deciles, batch=128) on 2026-05-01.
        # Re-run shared/tests/probe_bge_m3_fp16_mps.py to refresh evidence.
        assert p.fp16_safe_for_encoder, "fp16 should be enabled on MPS (verified clean)"
        assert p.fp16_safe_for_reranker, "fp16 should be safe on MPS reranker"
    if p.device == "cpu":
        assert not p.fp16_safe_for_encoder, "fp16 not useful on CPU"
        assert not p.fp16_safe_for_reranker, "fp16 not useful on CPU"


def test_encode_batch_tiers_monotonic() -> None:
    """More memory should never produce a smaller batch."""
    tiny = SystemProbe(device="cpu", fp16_safe_for_encoder=False,
                       fp16_safe_for_reranker=False, total_memory_gb=8.0, cpu_count=4)
    mid = SystemProbe(device="mps", fp16_safe_for_encoder=False,
                      fp16_safe_for_reranker=True, total_memory_gb=16.0, cpu_count=8)
    big = SystemProbe(device="mps", fp16_safe_for_encoder=False,
                      fp16_safe_for_reranker=True, total_memory_gb=48.0, cpu_count=12)
    a = recommend_encode_batch(tiny)
    b = recommend_encode_batch(mid)
    c = recommend_encode_batch(big)
    assert a <= b <= c, f"non-monotonic: {a}, {b}, {c}"
    assert a == 32, f"low-memory tier should be 32, got {a}"
    assert b == 64, f"mid-memory tier should be 64, got {b}"
    assert c == 128, f"high-memory tier should be 128, got {c}"


def test_recommend_assembles_consistent_config() -> None:
    cfg = recommend(BGE_M3, BGE_RERANKER_V2_M3)
    # encoder + ingest must agree on batch size
    assert cfg.encoder.default_batch_size == cfg.ingest.encode_batch, (
        f"batch-size drift between encoder and ingest configs: "
        f"encoder={cfg.encoder.default_batch_size}, ingest={cfg.ingest.encode_batch}"
    )
    # device propagated everywhere
    assert cfg.encoder.device == cfg.probe.device
    assert cfg.reranker.device == cfg.probe.device
    # specs propagated
    assert cfg.encoder.spec is BGE_M3
    assert cfg.reranker.spec is BGE_RERANKER_V2_M3
    # fp16 policies match probe
    assert cfg.encoder.use_fp16 == cfg.probe.fp16_safe_for_encoder
    assert cfg.reranker.fp16 == cfg.probe.fp16_safe_for_reranker
    # notes are populated and human-readable
    assert len(cfg.notes) >= 4
    assert any("device=" in n for n in cfg.notes)
    assert any("encode_batch=" in n for n in cfg.notes)


def main() -> int:
    failures = []
    tests = [
        ("test_probe_returns_sensible_values", test_probe_returns_sensible_values),
        ("test_encode_batch_tiers_monotonic", test_encode_batch_tiers_monotonic),
        ("test_recommend_assembles_consistent_config",
         test_recommend_assembles_consistent_config),
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
        except AssertionError as e:
            failures.append(f"{name}: {e}")
            print(f"FAIL: {name}: {e}")
    if failures:
        return 1
    print(f"OK — {len(tests)} autoconfig tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
