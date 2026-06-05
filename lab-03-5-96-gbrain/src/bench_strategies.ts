/**
 * Phase 6 benchmark (CORRECT path) — keyword FTS vs pure vector vs hybrid-RRF.
 *
 * WHY this exists: `gbrain search` and `gbrain query` CLI commands fall through
 * to the SAME hybrid handler (cli.ts:771-772), so they cannot be A/B'd from the
 * shell — they return byte-identical rankings. The real keyword/vector/hybrid
 * split lives one layer down in gbrain's own eval harness (src/core/search/eval.ts),
 * which calls engine.searchKeyword / engine.searchVector / hybridSearch directly.
 * This script bootstraps the engine + AI gateway exactly as the CLI does, then
 * runs runEval() three times on one labeled qrel set.
 *
 * Run: bun src/bench_strategies.ts   (needs GBRAIN_DATABASE_URL + OLLAMA_* env)
 */
const GB = "/Users/yuxinliu/code/agent-prep/gbrain/src";

const { loadConfig, toEngineConfig } = await import(`${GB}/core/config.ts`);
const { createEngine } = await import(`${GB}/core/engine-factory.ts`);
const { connectWithRetry } = await import(`${GB}/core/db.ts`);
const { configureGateway } = await import(`${GB}/core/ai/gateway.ts`);
const { buildGatewayConfig } = await import(`${GB}/core/ai/build-gateway-config.ts`);
const { runEval } = await import(`${GB}/core/search/eval.ts`);

// (query, relevant-slug, kind) — single gold per query; kind documents intent.
const QRELS = [
  { query: "Lin Zhao",                          relevant: ["people/lin-zhao"],             kind: "exact" },
  { query: "Ridgeline Capital",                 relevant: ["companies/ridgeline-capital"], kind: "exact" },
  { query: "Northstar Ventures",                relevant: ["companies/northstar-ventures"],kind: "exact" },
  { query: "Marcus Webb",                       relevant: ["people/marcus-webb"],          kind: "exact" },
  { query: "dinner at Tartine",                 relevant: ["meetings/tartine-dinner"],     kind: "exact" },
  { query: "who runs serving infrastructure",   relevant: ["people/lin-zhao"],             kind: "semantic" },
  { query: "protein design foundation models",  relevant: ["companies/helix-bio"],         kind: "semantic" },
  { query: "inference optimization startup",    relevant: ["companies/quanta-labs"],       kind: "semantic" },
  { query: "early-stage bio funding round",     relevant: ["deals/helix-series-a"],        kind: "semantic" },
  { query: "payments company angel investment", relevant: ["companies/stripe"],            kind: "semantic" },
];

const K = 3;

const config = loadConfig();
configureGateway(buildGatewayConfig(config));
const engine = await createEngine(toEngineConfig(config));
await connectWithRetry(engine, toEngineConfig(config), { noRetry: true });
const { reconfigureGatewayWithEngine } = await import(`${GB}/core/ai/gateway.ts`);
await reconfigureGatewayWithEngine(engine);

const qrels = QRELS.map(({ query, relevant }) => ({ query, relevant }));
const strategies = ["keyword", "vector", "hybrid"] as const;

// Per-query rank table (rank of the gold slug under each strategy).
const reports: Record<string, any> = {};
for (const strategy of strategies) {
  reports[strategy] = await runEval(engine, qrels, { strategy, expand: false }, K);
}

const pad = (s: string, n: number) => s.padEnd(n);
console.log(pad("query", 38) + pad("kind", 10) + pad("keyword", 10) + pad("vector", 10) + pad("hybrid", 10));
console.log("-".repeat(78));
QRELS.forEach((q, i) => {
  let row = pad(q.query, 38) + pad(q.kind, 10);
  for (const s of strategies) {
    const hits: string[] = reports[s].queries[i].hits;
    const rank = hits.indexOf(q.relevant[0]) + 1; // 0 → not found
    row += pad(rank > 0 ? `@${rank}` : "MISS", 10);
  }
  console.log(row);
});
console.log("-".repeat(78));
console.log("\n" + pad("strategy", 12) + pad(`recall@${K}`, 12) + pad("MRR", 10) + pad(`nDCG@${K}`, 10));
for (const s of strategies) {
  const r = reports[s];
  console.log(pad(s, 12) + pad(r.mean_recall.toFixed(3), 12) + pad(r.mean_mrr.toFixed(3), 10) + pad(r.mean_ndcg.toFixed(3), 10));
}

await engine.disconnect?.();
process.exit(0);
