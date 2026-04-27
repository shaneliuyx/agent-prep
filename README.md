# agent-prep

A 3-month, lab-driven curriculum to convert a cloud-infrastructure background into AI Agent / LLM Engineer skills.

Each `lab-NN-*/` subdirectory is a self-contained week. Every lab follows the same shape: scaffold → instrumented implementation → measured comparison → committed `RESULTS.md`. The point is *measured* engineering — every claim is grounded in numbers from a runnable artifact, not vibes.

## Curriculum spine

| Week | Lab | Status |
|---|---|---|
| 1 | [`lab-01-vector-baseline`](./lab-01-vector-baseline) — embedding + HNSW config ablation on MS MARCO 10K-doc slice | ✅ complete |
| 2 | `lab-02-rerank-compress` — BGE-reranker lift + context compression A/B | pending |
| 3 | `lab-03-rag-eval` — RAGAS harness + HyDE / multi-query A/B | pending |
| 4 | `lab-04-react-from-scratch` — ReAct loop in ~150 lines, 15-scenario bad-case suite | pending |
| 5 | `lab-05-pattern-zoo` — ReAct vs Plan-and-Solve vs Reflexion vs Orchestrator-Worker | pending |
| 6 | `lab-06-claude-code-map` — Claude Code source-dive subsystem study sheets | pending |
| 7 | `lab-07-tool-harness` — generic ToolHarness with 20-scenario bad-case suite | pending |
| 8 | `lab-08-schema-bench` — 5-strategy × 5-model schema reliability matrix | pending |
| 9 | `lab-09-faithfulness-checker` — claim split + NLI + SelfCheckGPT-lite + abstention | pending |
| 10 | `lab-10-framework-shootout` — same task in LangGraph / LlamaIndex / OpenAI Agents SDK | pending |

The capstone (Week 12) lives in a separate repo for portfolio framing.

## Stack

- **Local-first inference**: oMLX serving Qwen3.6-35B-A3B / gemma-4-26B-heretic / gpt-oss-20b on `:8000` (Anthropic + OpenAI API surface). vMLX as a second backend on `:8003`. Cloud APIs only used in Weeks 7–8 for benchmarking comparisons (~$8 total budget across all 12 weeks).
- **Vector DB**: Qdrant via OrbStack (Docker) on `:6333`.
- **Observability**: Phoenix on `:6006`.
- **Embeddings**: BGE-M3, BGE-reranker-v2-m3, Nomic Embed v2 MoE — all running locally on Apple Silicon via MPS.

See each lab's `RESULTS.md` for the per-lab measured findings.

## License

MIT (see [LICENSE](./LICENSE) when added). Curriculum content is original; companion-text references in each `RESULTS.md` cite their original authors (Anthropic, agentway.dev, Gerred, Gulli, Singh et al., NousResearch, etc.).
