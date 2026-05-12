# agent-prep

A 3-month, lab-driven curriculum to convert a cloud-infrastructure background into AI Agent / LLM Engineer skills.

Each `lab-NN-*/` subdirectory is a self-contained week. Every lab follows the same shape: scaffold → instrumented implementation → measured comparison → committed `RESULTS.md`. The point is *measured* engineering — every claim is grounded in numbers from a runnable artifact, not vibes.

## Curriculum spine

| Week | Lab | Status |
|---|---|---|
| 1 | [`lab-01-vector-baseline`](./lab-01-vector-baseline) — embedding + HNSW config ablation on MS MARCO 10K-doc slice | ✅ complete |
| 2 | [`lab-02-rerank-compress`](./lab-02-rerank-compress) — BGE-reranker lift + context compression A/B + chunking sweep | ✅ complete |
| 2b | [`lab-02b-production-libs`](./lab-02b-production-libs) — port lab-02 to `langchain-qdrant` + `rerankers` + `ranx` | ✅ complete |
| 2.5 | [`lab-02-5-graphrag`](./lab-02-5-graphrag) — GraphRAG on tech-founder Wikipedia subset, 32-Q head-to-head vs vector RAG, **v12.4m: 0.96 judge / 32-0-0 W-L-T** | ✅ complete |
| 2.7 | [`lab-02-7-pageindex`](./lab-02-7-pageindex) — PageIndex / tree-index RAG on Berkshire 2023 10-K (152 pages), 4-index architecture (LLM tree + K-means cluster + entity reverse-index + BGE-M3 hybrid page-vector fallback) + agentic multi-iter loop + GT-judge methodology, **Phase 9 final: 16/16 = 1.000 vs Vector 0.500 / Graph 0.375** | ✅ complete |
| 3 | [`lab-03-rag-eval`](./lab-03-rag-eval) — RAGAS harness + HyDE A/B + multi-query fusion + Phoenix tracing | ✅ complete |
| 3.5 | [`lab-03-5-memory`](./lab-03-5-memory) — single-agent cross-session memory: hand-rolled Python extraction + Qdrant episodic + SQLite semantic (SCD-2 archival, partial unique index, WAL + try/finally), `src/lab_init.py` guided setup, **15/15 recall benchmark + Phase 5 mem0 cross-check 10/14 (4 measured architectural differences)** | ✅ complete |
| 3.5.5 | `lab-03-5-5-guild` — multi-agent shared memory via `mathomhaus/guild` (Go MCP), atomic-claim race demo, 3-act cross-session handoff, 15-Q multi-agent recall benchmark; chapter shipped (1037 lines), lab pending | pending |
| 3.5.8 | `lab-03-5-8-two-tier` — two-tier production architecture (guild operational + EverCore semantic) with consolidation pipeline (hippocampus + neocortex + REM-sleep analogy), 4-way benchmark; chapter shipped, lab pending | pending |
| 3.5.9 | `lab-03-5-9-bench-hypergraph` — three-tier (guild + EverCore + HyperMem) + LongMemEval `oracle` subset 5-way comparison anchored to EverCore's published 83%; chapter shipped, lab pending | pending |
| 3.7 | [`lab-03.7-agentic-rag`](./lab-03.7-agentic-rag) — LangChain 5-node Agentic RAG + CRAG variant + Self-RAG hand-roll + FastMCP wrapper (first MCP-server pattern); chapter Phases 1-8 | in progress |
| 4 | [`lab-04-react-from-scratch`](./lab-04-react-from-scratch) — ReAct loop in ~150 lines, 15-scenario bad-case suite | in progress |
| 5 | `lab-05-pattern-zoo` — ReAct vs Plan-and-Solve vs Reflexion vs Orchestrator-Worker | pending |
| 6 | `lab-06-claude-code-map` — Claude Code source-dive subsystem study sheets | pending |
| 7 | `lab-07-tool-harness` — generic ToolHarness with 20-scenario bad-case suite | pending |
| 7.3 | `lab-07-3-prod-infra` — LiteLLM gateway routing Claude + GPT + local oMLX through one endpoint; Anthropic + OpenAI prompt caching + GPTCache semantic cache + LangSmith cost-attribution metadata + circuit-breaker provider fallback + end-to-end re-run of W3 RAG eval through gateway; fills Akshay 6-area rubric areas 2+5 (inference + production infra); chapter shipped | pending |
| 8 | `lab-08-schema-bench` — 5-strategy × 5-model schema reliability matrix | pending |
| 9 | `lab-09-faithfulness-checker` — claim split + NLI + SelfCheckGPT-lite + abstention | pending |
| 10 | `lab-10-framework-shootout` — same task in LangGraph / LlamaIndex / OpenAI Agents SDK | pending |

Companion narrative + interview-prep chapters live in [`shaneliuyx/agent-development-curriculum`](https://github.com/shaneliuyx/agent-development-curriculum) (Obsidian vault). The capstone (Week 12) lives in a separate repo for portfolio framing.

### Akshay 6-area hiring-rubric coverage (2026)

The curriculum maps onto Akshay Pachaar's 6-area AI-engineer rubric — verified by 12 May 2026 audit of the teach_fireworks 11-section reading list:

| # | Area | Anchor week(s) |
|---|---|---|
| 1 | Harness engineering (loop / tool registry / budget / scratchpad) | W4, W5, W7 |
| 2 | Inference serving (KV cache, paged attention, spec decoding, quantization) | W0, W2.7 BCJ #23, W9.5 |
| 3 | Structured output reliability (FSM-guided decoding, schema-first, post-validation) | W8 |
| 4 | Evals + observability (LLM-as-judge bias, RAGAS, Phoenix, OpenTelemetry GenAI) | W3, W2.7, W3.5 |
| 5 | Production LLM infrastructure (gateway, prompt + semantic caching, cost attribution, provider fallback) | W7.3 |
| 6 | Fine-tune vs in-context decision-making | W9, W9.5 |

A candidate covering 1+3+4 looks like a 2024 LLM engineer. Covering 1+2+3+4+5+6 looks like a 2026 staff-track AI engineer. W7.3 is the bridge — converts areas 2/5 from theory citations into measured lab artifacts.

## Shared libraries

- [`shared/rag_hybrid`](./shared/rag_hybrid) — lab-02-5, lab-02b, lab-03 retrieval primitives (encoder + reranker + retriever + chunker). `autoconfig` probes host (mps / cuda / cpu + memory tier 32 / 64 / 128) and emits a `RecommendedConfig` consumed by downstream labs without hardcoded device flags.
- [`shared/tree_index`](./shared/tree_index) — lab-02-7 structure-aware-RAG primitives: `TreeIndex` (hierarchical), `SummaryIndex` (K-means RAPTOR Level-2 cluster routing with top-K δ=0.07 tiebreak), `EntityIndex` (regex-extracted reverse index), `PageVectorIndex` (BGE-M3 dense+sparse hybrid fallback), `AgenticTreeRetriever` (multi-iter agentic loop with `get_page_content` tool + BUDGET-EXHAUSTED 5-rule synthesis + chunk-level fallback). Powers the 16/16 GT-judge result on Berkshire 2023.

## Stack

- **Local-first inference**: oMLX serving Qwen3.6-35B-A3B / gemma-4-26B-heretic / gpt-oss-20b on `:8000` (Anthropic + OpenAI API surface). vMLX as a second backend on `:8003`. Cloud APIs scoped to: **W7–8** (frontier-model reliability comparisons, ~$8), **W7.3** (cross-provider gateway routing, ~$3), **W9.5** (optional cloud GPU for SFT+GRPO run, $0–30). **Total program cloud cap: ~$13** (with $20 diagnostic threshold — if you exceed $20, audit which lab is leaking, usually a missed `max_tokens` cap or a forgotten cache breakpoint).
- **Vector DB**: Qdrant via OrbStack (Docker) on `:6333`.
- **Memory infra (Weeks 3.5.5 / 3.5.8 / 3.5.9)**: `mathomhaus/guild` (Go MCP, single binary, embedded SQLite) for operational tier; EverMind-AI's EverCore (Python + Postgres via Docker compose, port 1995) for semantic tier; HyperMem (Docker compose, port 1996) for relational L3 tier. Benchmarked via LongMemEval `oracle` subset anchored to EverCore's published 83%.
- **Observability**: Phoenix on `:6006`.
- **Embeddings**: BGE-M3 (oMLX-served `bge-m3-mlx-fp16` for embedding API; `sentence-transformers` MPS fallback when oMLX has no embedding model), BGE-reranker-v2-m3, Nomic Embed v2 MoE — all running locally on Apple Silicon.

See each lab's `RESULTS.md` for the per-lab measured findings.

## License

MIT (see [LICENSE](./LICENSE) when added). Curriculum content is original; companion-text references in each `RESULTS.md` cite their original authors (Anthropic, agentway.dev, Gerred, Gulli, Singh et al., NousResearch, etc.).
