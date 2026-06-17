# agent-prep

A 3-month, lab-driven curriculum to convert a cloud-infrastructure background into AI Agent / LLM Engineer skills.

Each `lab-NN-*/` subdirectory is a self-contained week. Every lab follows the same shape: scaffold → instrumented implementation → measured comparison → committed `RESULTS.md`. The point is *measured* engineering — every claim is grounded in numbers from a runnable artifact, not vibes.

## Curriculum spine

| Week | Lab | Status |
|---|---|---|
| 0 | `lab-00-env-setup` — local-first MLX stack bring-up (oMLX + vMLX + Qdrant + Phoenix); chapter shipped (W0 Environment Setup), no lab dir | pending |
| 1 | [`lab-01-vector-baseline`](./lab-01-vector-baseline) — embedding + HNSW config ablation on MS MARCO 10K-doc slice | ✅ complete |
| 2 | [`lab-02-rerank-compress`](./lab-02-rerank-compress) — BGE-reranker lift + context compression A/B + chunking sweep | ✅ complete |
| 2b | [`lab-02b-production-libs`](./lab-02b-production-libs) — port lab-02 to `langchain-qdrant` + `rerankers` + `ranx` | ✅ complete |
| 2.5 | [`lab-02-5-graphrag`](./lab-02-5-graphrag) — GraphRAG on tech-founder Wikipedia subset, 32-Q head-to-head vs vector RAG, **v12.4m: 0.96 judge / 32-0-0 W-L-T** | ✅ complete |
| 2.7 | [`lab-02-7-pageindex`](./lab-02-7-pageindex) — PageIndex / tree-index RAG on Berkshire 2023 10-K (152 pages), 4-index architecture (LLM tree + K-means cluster + entity reverse-index + BGE-M3 hybrid page-vector fallback) + agentic multi-iter loop + GT-judge methodology, **Phase 9 final: 16/16 = 1.000 vs Vector 0.500 / Graph 0.375** | ✅ complete |
| 3 | [`lab-03-rag-eval`](./lab-03-rag-eval) — RAGAS harness + HyDE A/B + multi-query fusion + Phoenix tracing | ✅ complete |
| 3.5 | [`lab-03-5-memory`](./lab-03-5-memory) — single-agent cross-session memory: hand-rolled Python extraction + Qdrant episodic + SQLite semantic (SCD-2 archival, partial unique index, WAL + try/finally), `src/lab_init.py` guided setup, **15/15 recall benchmark + Phase 5 mem0 cross-check 10/14 (4 measured architectural differences)**; [`RESULTS.md`](./lab-03-5-memory/RESULTS.md) | ✅ complete |
| 3.5.5 | [`lab-03-5-5-guild`](./lab-03-5-5-guild) — multi-agent shared memory via `mathomhaus/guild` (Go MCP), atomic-claim race demo, 3-act cross-session handoff, 15-Q multi-agent recall benchmark, `RESULTS.md` dated 2026-05-12 | ✅ complete |
| 3.5.5.5 | [`lab-03-5-5-5-topology`](./lab-03-5-5-5-topology) — five multi-agent topology patterns (supervisor + hierarchical + group-chat + handoffs + voting) implemented as standalone runnable Python; LLM-provider abstraction (anthropic-proxy / openai-compatible / mock) with .env autoload + integration-marker test gating; **17/17 PASS against real LLM (gpt-oss-20b via oMLX) in 95s**; [`RESULTS.md`](./lab-03-5-5-5-topology/RESULTS.md) | ✅ complete |
| 3.5.8 | [`lab-03-5-8-two-tier`](./lab-03-5-8-two-tier) — two-tier production architecture (guild operational + EverCore semantic) with consolidation pipeline (hippocampus + neocortex + REM-sleep analogy); Phase 9+ longmemeval slice harness (Sonnet judge + memory tools + replay rejudge); **N=100 judge-controlled board: Qwen3.5-27B-Opus-distill 77% / Opus-4.7-proxy 68% / Sonnet-4.6 60% vs EverCore-published 83%**; commitment-bias + lifecycle-bound atomisation findings; [`RESULTS.md`](./lab-03-5-8-two-tier/RESULTS.md) shipped | ✅ complete |
| 3.5.9 | [`lab-03-5-9-requirement-driven`](./lab-03-5-9-requirement-driven) — requirement-driven memory architecture: LongMemEval decomposition + multi-backend matrix (1-tier / 2-tier / hybrid / three-tier HyperMem L3) + topic-presence abstention gate; **7-backend × 20-Q matrix + 6-axis slice base 37%→83%**; [`RESULTS.md`](./lab-03-5-9-requirement-driven/RESULTS.md) | ✅ complete |
| 3.5.95 | [`lab-03-5-95-self-observability`](./lab-03-5-95-self-observability) — self-facing memory (PAI v7.6 OBSERVABILITY + LEARNING): append-only behavioral log + LLM consolidation + metacognitive recall (BM25 × recency × confidence) + paired-trial **divergence 6/6** + Phase 7 heat-scored eviction (BAI-LAB/MemoryOS; importance-exempt) + smolagents integration; [`RESULTS.md`](./lab-03-5-95-self-observability/RESULTS.md) | ✅ complete |
| 3.5.96 | [`lab-03-5-96-gbrain`](./lab-03-5-96-gbrain) — self-wiring markdown knowledge graph over GBrain via smolagents + MCP; deterministic edge extraction + measured keyword/vector/hybrid-RRF (**pure-vector > RRF on the 19-page corpus** — the 83→95 lift is corpus-dependent, not universal) + Ground-Truth Hierarchy A/B (ClaudioDrews/memory-os); [`RESULTS.md`](./lab-03-5-96-gbrain/RESULTS.md) | ✅ complete |
| 3.7 | [`lab-03.7-agentic-rag`](./lab-03.7-agentic-rag) — hand-rolled Self-RAG + CRAG vs canonical/structural LangGraph + FastMCP server (first MCP-server pattern). **Measured: the canonical skip-allowed graph is mis-built — faithfulness 0.876 vs single-pass 0.980, 15/50 retrieval skips, 1.93× latency; a structural always-retrieve edge recovers to 1.000 at parity.** CRAG web fallback decomposes comparison queries (per-sub-query rerank + interleave) over a SearXNG backend, answers 10/10 out-of-corpus where single-pass abstains; [`RESULTS.md`](./lab-03.7-agentic-rag/RESULTS.md) | ✅ complete |
| 4 | [`lab-04-react-from-scratch`](./lab-04-react-from-scratch) — ReAct loop in ~150 lines, 15-scenario bad-case suite | in progress |
| 4.6 | [`lab-04-6-durable-runtime`](./lab-04-6-durable-runtime) — durable agent runtime: SQLite-backed DAG store + append-only event log, external-trigger scheduler (manual / webhook / cron), asyncio worker pool with file-lock mutex, cost meter; four 5-node process topologies (sequential / parallel / hierarchical / workflow) benchmarked at constant model + node count so **structure is the only variable**, plus a real SIGKILL-mid-run recovery proof (`recover_run` resets orphaned RUNNING → READY, zero lost / double-done work). **Measured (repeats=5, M5 Pro): 195 tokens/topology, peak concurrency 1/4/3/1, recovery 1.227s (topology-agnostic single probe)**; [`RESULTS.md`](./lab-04-6-durable-runtime/RESULTS.md) | ✅ complete |
| 5 | `lab-05-pattern-zoo` — ReAct vs Plan-and-Solve vs Reflexion vs Orchestrator-Worker | pending |
| 6 | `lab-06-claude-code-map` — Claude Code source-dive subsystem study sheets | pending |
| 7 | `lab-07-tool-harness` — generic ToolHarness with 20-scenario bad-case suite | pending |
| 7.3 | `lab-07-3-prod-infra` — LiteLLM gateway routing Claude + GPT + local oMLX through one endpoint; Anthropic + OpenAI prompt caching + GPTCache semantic cache + LangSmith cost-attribution metadata + circuit-breaker provider fallback + end-to-end re-run of W3 RAG eval through gateway; fills Akshay 6-area rubric areas 2+5 (inference + production infra); chapter shipped | pending |
| 8 | `lab-08-schema-bench` — 5-strategy × 5-model schema reliability matrix | pending |
| 9 | `lab-09-faithfulness-checker` — claim split + NLI + SelfCheckGPT-lite + abstention | pending |
| 9.5 | `lab-09-5-agentic-rl` — agentic RL fine-tuning (SFT + GRPO) on small open model; chapter shipped (W9.5 Agentic RL Fine-Tuning), cloud-GPU optional ($0–30) | pending |
| 10 | `lab-10-framework-shootout` — same task in LangGraph / LlamaIndex / OpenAI Agents SDK | pending |
| 11 | `lab-11-system-design` — system-design interview drills + reference architectures (multi-tenant agent platform, cost-bounded RAG, low-latency tool-use); chapter shipped (W11 System Design) | pending |
| 12 | `lab-12-capstone` — capstone project + mock interviews; lives in separate repo for portfolio framing ([`shaneliuyx/capstone`](../capstone) parallel to agent-prep); chapter shipped (W12 Capstone and Mocks) | pending |

Companion narrative + interview-prep chapters live in [`shaneliuyx/agent-development-curriculum`](https://github.com/shaneliuyx/agent-development-curriculum) (Obsidian vault). The capstone (Week 12) lives in a separate repo for portfolio framing.

### Akshay 6-area hiring-rubric coverage (2026)

The curriculum maps onto Akshay Pachaar's 6-area AI-engineer rubric (verified by 12 May 2026 audit of the teach_fireworks 11-section reading list). Coverage split into **today** (lab dirs with tracked source + measurements) vs **planned** (chapter shipped, lab pending):

| # | Area | Covered today | Planned |
|---|---|---|---|
| 1 | Harness engineering (loop / tool registry / budget / scratchpad / **multi-agent topology** / **durable execution**) | W3.5.5.5 (5 topology patterns, 17/17 PASS), W4.6 (durable runtime + 4 process topologies + SIGKILL recovery proof) | W4 (in progress, 14 files), W5, W7 |
| 2 | Inference serving (KV cache, paged attention, spec decoding, quantization) | W2.7 BCJ #23 (single quantization deep-dive) | W0 (env setup), W9.5 (agentic RL) |
| 3 | Structured output reliability (FSM-guided decoding, schema-first, post-validation) | — | W8 |
| 4 | Evals + observability (LLM-as-judge bias, RAGAS, Phoenix, OpenTelemetry GenAI) | W3 (RAGAS + HyDE), W2.7 (GT-judge methodology), W3.5 (15/15 recall) | — |
| 5 | Production LLM infrastructure (gateway, prompt + semantic caching, cost attribution, provider fallback) | — | W7.3 |
| 6 | Fine-tune vs in-context decision-making | — | W9 (faithfulness baseline), W9.5 (RL fine-tune) |

**Honest read today**: areas 4 + part of 1 + sliver of 2 are measured. That's a 2024-vintage LLM-engineering profile with multi-agent depth added.

**Roadmap claim**: when areas 3, 5, 6 + the rest of 1 + 2 land, the profile matches the 2026 staff-track AI engineer rubric. W7.3 is the unlock — it converts areas 2/5 from theory citations into measured artifacts. The "1+3+4 = 2024 / 1+2+3+4+5+6 = 2026 staff" framing is the destination, not the current state.

## Shared libraries

- [`shared/llm.py`](./shared/llm.py) — provider-agnostic LLM plumbing used across labs: `make_client` / `PROFILES` / `resolve(role, default)` (swap models via env, no code change), `chat` (text only), `chat_usage` (text + real `{prompt, completion, total}` token usage — for cost metering and throughput benches; `chat` is the thin wrapper over it), `judge` (LLM PASS/FAIL), `resilient` retry / `LLMUnavailable`. W4.6's cost meter reads real tokens through `chat_usage`.
- [`shared/rag_hybrid`](./shared/rag_hybrid) — lab-02-5, lab-02b, lab-03 retrieval primitives (encoder + reranker + retriever + chunker). `autoconfig` probes host (mps / cuda / cpu + memory tier 32 / 64 / 128) and emits a `RecommendedConfig` consumed by downstream labs without hardcoded device flags.
- [`shared/tree_index`](./shared/tree_index) — lab-02-7 structure-aware-RAG primitives: `TreeIndex` (hierarchical), `SummaryIndex` (K-means RAPTOR Level-2 cluster routing with top-K δ=0.07 tiebreak), `EntityIndex` (regex-extracted reverse index), `PageVectorIndex` (BGE-M3 dense+sparse hybrid fallback), `AgenticTreeRetriever` (multi-iter agentic loop with `get_page_content` tool + BUDGET-EXHAUSTED 5-rule synthesis + chunk-level fallback). Powers the 16/16 GT-judge result on Berkshire 2023.
- [`shared/phoenix_tracing`](./shared/phoenix_tracing) — observability primitives distilled from W3's `src/05_trace.py`. Three ergonomic tiers: `trace_run()` one-call wrapper, `@traced` decorator, raw `OpenInference` spans. Consumed by any lab that needs Phoenix tracing without re-implementing OTel boilerplate.
- [`shared/agent_loop_tools`](./shared/agent_loop_tools) — `interrupt_state` (atomic file-based pause/resume signal) + `token_accounting` (budget-aware token counter with provider-specific tokenizers). Patterns lifted from `kunchenguid/gnhf` (1.8K-star TypeScript overnight-agent orchestrator); minimal Python ports of the highest-leverage primitives for use in W4 ReAct + W5 pattern zoo + W7 tool harness.
- [`shared/parity`](./shared/parity) + [`shared/parity_baseline.py`](./shared/parity_baseline.py) — refactor-safety harness. Freezes pre-refactor ground truth as three signal classes (Qdrant point counts, sample vector signatures, result-file content hashes) so mechanical diff after each refactor step catches encoder drift, ingest breakage, and eval changes. Used during lab-02-5-graphrag v12 series refactors.
- [`shared/web_search.py`](./shared/web_search.py) — lab-03.7 web-fallback backend: `web_search` (precedence `SEARXNG_URL` → `TAVILY_API_KEY` → DuckDuckGo) + on-disk reproducibility cache (`cache_lookup`/`cache_store`) + `rerank_results` (cross-encoder rerank of result strings — the reranker is passed in, so the module stays torch-free). Imported by **both** the hand-rolled (`baseline_handrolled.py`) and LangGraph (`crag_variant.py`) CRAGs. [`shared/searxng/`](./shared/searxng) ships a `docker-compose.yml` for the free local SearXNG backend.

## Stack

- **Local-first inference**: oMLX serving Qwen3.6-35B-A3B-nvfp4 (opus tier; MoE, agent loops) / gemma-4-26B-A4B-it-heretic-4bit (sonnet tier; RAG synthesis) / gpt-oss-20b-MXFP4-Q8 (haiku tier; workers, classifiers) on `:8000` (Anthropic + OpenAI API surface). vMLX as a second backend on `:8003`. Cloud APIs scoped to: **W7–8** (frontier-model reliability comparisons, ~$8), **W7.3** (cross-provider gateway routing, ~$3), **W9.5** (optional cloud GPU for SFT+GRPO run, $0–30). **Total program cloud cap: ~$13** (with $20 diagnostic threshold — if you exceed $20, audit which lab is leaking, usually a missed `max_tokens` cap or a forgotten cache breakpoint).
- **Vector DB**: Qdrant via OrbStack (Docker) on `:6333`.
- **Memory infra (Weeks 3.5.5 / 3.5.8 / 3.5.9)**: `mathomhaus/guild` (Go MCP, single binary, embedded SQLite) for operational tier; EverMind-AI's EverCore (Python + Postgres via Docker compose, port 1995) for semantic tier; HyperMem (Docker compose, port 1996) for relational L3 tier. Benchmarked via LongMemEval `oracle` subset anchored to EverCore's published 83%.
- **Observability**: Phoenix on `:6006`.
- **Embeddings**: BGE-M3 (oMLX-served `bge-m3-mlx-fp16` for embedding API; `sentence-transformers` MPS fallback when oMLX has no embedding model), BGE-reranker-v2-m3, Nomic Embed v2 MoE — all running locally on Apple Silicon.

See each lab's `RESULTS.md` for the per-lab measured findings.

## License

MIT (see [LICENSE](./LICENSE) when added). Curriculum content is original; companion-text references in each `RESULTS.md` cite their original authors (Anthropic, agentway.dev, Gerred, Gulli, Singh et al., NousResearch, etc.).
