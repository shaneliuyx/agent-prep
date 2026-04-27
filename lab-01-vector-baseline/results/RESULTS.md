# Lab 01 — Vector Retrieval Baseline (RESULTS)

**Date:** 2026-04-27
**Corpus:** MS MARCO passage dev set, 10K-doc slice, 6,980 labeled queries
**Hardware:** Apple Silicon (MPS via sentence-transformers), Qdrant v1.15 on OrbStack
**Embedding models:** BGE-M3 (1024-dim, dense+sparse hybrid) · Nomic Embed v2 MoE (768-dim, 8 experts top-2 routing)
**Raw artifact:** [`retrieval_metrics.json`](./retrieval_metrics.json)

---

## Numbers

| Collection         | Embedding model     | HNSW config       | Recall@10 | MRR@10 | nDCG@10 | Embed (s) | Search (s) |
|--------------------|---------------------|-------------------|-----------|--------|---------|-----------|------------|
| bge_m3_hnsw        | BGE-M3              | m=16, ef=128      | **0.9933** | 0.9556 | 0.9637  | 10.01     | 29.02      |
| bge_m3_hnsw_fast   | BGE-M3              | m=8,  ef=64       | **0.9933** | 0.9556 | 0.9637  | 9.73      | 27.22      |
| nomic_hnsw         | Nomic Embed v2 MoE  | m=16, ef=128      | **0.9966** | 0.9629 | 0.9702  | 23.60     | 25.15      |

**Per-query latency** (over 6,980 queries):

| Operation | BGE-M3 (per query) | Nomic v2 (per query) |
|---|---|---|
| Query embed | ~1.4 ms | ~3.4 ms |
| HNSW search (m=16) | ~4.2 ms | ~3.6 ms |
| HNSW search (m=8) | ~3.9 ms | — |

**Receipts (raw float precision):** BGE-baseline and BGE-fast recall@10 are **bit-identical** at `0.9932664756446992` — same top-10 returned for every single one of 6,980 queries. Confirmed via the raw JSON, not just rounded display. Nomic recall@10 is `0.9965616045845272`.

---

## What I learned

**HNSW config is irrelevant at 10K-doc scale, and the null result is the data.** The two BGE-M3 collections — one with `m=16, ef_construct=128` and one with `m=8, ef_construct=64` — produced bit-identical recall, MRR, and nDCG to full float precision. The only measurable difference is search latency: 27.2s vs 29.0s, a ~6% speedup on the smaller graph. At this corpus size both HNSW graphs are over-connected enough that greedy descent lands on the exact same top-10 neighbors for every query in the dev set. The decision becomes meaningful only at 1M+ vectors where the m=8 graph's lower memory footprint forces tradeoffs — at 10K, "tune your HNSW config" is premature optimization. The actionable rule: don't bother tuning HNSW config below ~1M vectors; ship the simpler config and document the threshold where the comparison would matter.

**Nomic v2 MoE marginally beats BGE-M3 on quality (+0.33pp recall, +0.65pp nDCG) but costs 2.4× more embed time** because of MoE routing overhead. The router + 8-expert dispatch with top-2 activation roughly doubles the per-token FLOPs even with the gated expert pattern, and the extra forward-pass complexity shows up in the embed-time column (23.6s vs 9.7s for the same 6,980 queries). For production retrieval where embed cost dominates at scale, BGE-M3 is the right ship; Nomic v2 stays useful as a quality ceiling for offline reranking experiments or when the +0.3pp recall on hard queries actually matters (e.g., regulated-document retrieval where misses are costly). Search latency on the Nomic side is *lower* (25.2s vs 29.0s) because the 768-dim vectors are 33% smaller than BGE-M3's 1024-dim, and HNSW search is bandwidth-bound on Apple Silicon — but that doesn't recover the 14-second embed-time gap.

**Recall@10 above 0.99 on a 10K MS MARCO slice is a ceiling-effect warning, not a victory lap.** The dev set has ~1-2 gold passages per query in a 10K-doc pool, so the random-chance baseline is ~0.001 — and the queries themselves were originally designed to be answerable by these passages. Hitting 0.99+ means the retrievers are saturated against an artificially clean test, not that they'd survive a harder corpus. To get real signal on the BGE-vs-Nomic question, the next experiment would be on **BEIR-FiQA** (financial Q&A, retrieval recall typically 0.4-0.6) or **BEIR-SciFact** (scientific claim verification, similar range). At those quality levels, 0.4pp differences become statistically meaningful instead of in-the-noise.

---

## Bad-case journal

**2026-04-27 — Nomic recall came back at 0.021 on first eval run.** Initial Phase 4 output showed both BGE collections at 0.993 (looked correct) but Nomic at 0.021 — basically random chance baseline (~0.001 for 10K/top-10). Nothing in the logs flagged an error; ingest had completed cleanly, eval ran to completion, all three collections had `points_count = 10000` in Qdrant.

Diagnosis path:
1. Suspected first: my code in `04_eval.py` (forgotten `trust_remote_code=True` on the Nomic side) — added it, re-ran, recall stayed at 0.021. Necessary fix but not sufficient.
2. Inspected the LOAD REPORT in stderr more carefully — saw `MISSING` for dense MLP weights (`mlp.up_proj.weight`, `mlp.gate_proj.weight`, `mlp.down_proj.weight`) and `UNEXPECTED` for MoE expert weights (`mlp.experts.mlp.w1`, `mlp.router.layer.weight`). This meant the MoE weights on disk were being discarded and the dense layers were being randomly initialized — the model's class definition was the wrong one for the weights file.
3. Traced to `~/.cache/huggingface/modules/transformers_modules/nomic_hyphen_ai/nomic_hyphen_bert_hyphen_2048/<oid>/modeling_hf_nomic_bert.py` — cached at commit `7710840340a098...`, an older version that had partial MoE classes (NomicRouter, NomicExperts, NomicMoELayer all defined) but missing the layer-integration code that wires MoE into NomicBertBlock. Upstream-current was `46cf2dead046...`, a different commit.
4. Fix: `rm -rf ~/.cache/huggingface/modules/transformers_modules/nomic_hyphen_ai/`. Re-ran ingest (forced fresh modeling-code download). Re-ran eval. Recall jumped from 0.021 → 0.997.

Total debug time: ~30 minutes (mostly chasing the wrong hypothesis on `trust_remote_code` first). Captured as Phase 3.2 gotcha #4 in the curriculum so future-me hits a 2-minute fix instead.

**Lesson:** silent embedding-model failures usually trace to stale `~/.cache/huggingface/modules/` content. Diagnostic is a 5-second sanity test (encode the same string twice, check for bit-identity); if outputs differ, model has random-init layers and the cache is stale. Nuke the cache before deeper debugging.

---

## Infra bridge

The HNSW null result IS the data — the kind of finding cloud infra teaches you to recognize and ship: ablations producing "no measurable difference" are valuable signals, not failed experiments. They tell you where the inflection point isn't yet, so you stop optimizing in the noise. Same instinct that makes me leave a CDN config alone when the A/B shows identical RPS, or skip a Kubernetes resource-request tune when both pod sizes hit the same p99.

The Nomic cache-staleness incident is exactly the operational pattern I've debugged on the platform side: a system reports `success` everywhere (ingest exited 0, Qdrant `points_count = 10000`, eval ran to completion) while producing semantically meaningless output. Same shape as a misconfigured Argo Workflow that succeeds on every node but produces empty artifacts, or a Lambda that returns 200 with a malformed payload. The diagnostic muscle is the same: **don't trust exit codes; verify outputs against expected semantics**. The 5-second "encode twice, compare" test is the same idea as a smoke-test endpoint with a known input/output contract — it catches silent failures that no exit-code check ever will.

The full stack here (Qdrant + sentence-transformers + HuggingFace cache + custom modeling code via `trust_remote_code`) is denser than typical infra layers but follows the same principles: explicit version pinning, cache invalidation as a first-class debug step, and observability that distinguishes "code ran" from "code produced correct output." Bringing those instincts to LLM-engineering work is the actual differentiator — most ML practitioners debug at the model layer; I debug at the system layer first.

---

## Exit criteria check

- [x] Three embedding/index combinations measured on the same 6,980-query dev set
- [x] HNSW config ablation: m=16 vs m=8 isolates the index variable; same vectors, different graph
- [x] Recall@10, MRR@10, nDCG@10 computed for each
- [x] Comparison table + 3-paragraph reflection committed
- [x] Can whiteboard "how HNSW works" in 60 seconds (greedy descent through layered navigable small-world graph; m = neighbors per node, ef_construct = build-time search budget)
- [x] Phase 3.2 gotcha #4 captured in curriculum so the cache failure doesn't recur

---

## What's next

Move to **Week 2** (Rerank & Context Compression). The `bge_m3_hnsw` collection (10K vectors, m=16 graph) is the one Week 2 reuses for the BGE-reranker-v2-m3 work. Don't drop it.

— end —
