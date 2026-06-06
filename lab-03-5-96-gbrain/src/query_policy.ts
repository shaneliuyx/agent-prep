/**
 * query_policy.ts — the APPLY/actuator half of the auto-eval loop.
 *
 * Stock `gbrain query` is hybrid-only (Phase 6 trap), so it cannot honor a
 * corpus-selected strategy. This helper does: it reads the policy artifact
 * `results/search_policy.json` (written by auto_eval.ts after it measures the
 * current corpus) and routes the query to the WINNING engine path —
 * keyword / vector / hybrid — instead of bare hybrid. That closes the loop:
 *
 *   ingest → reconcile → auto_eval (measure+decide+write policy) → THIS (apply)
 *
 * The agent calls this for its retrieval; the per-corpus verdict steers it.
 *
 * Fallback: if no policy file exists yet, default to hybrid (GBrain's own default).
 * Usage: bun src/query_policy.ts "<query text>" [limit]
 */
import { existsSync, readFileSync } from "node:fs";

const GB = "/Users/yuxinliu/code/agent-prep/gbrain/src";
const POLICY = `${import.meta.dir}/../results/search_policy.json`;

type Strategy = "keyword" | "vector" | "hybrid";

const query = process.argv[2];
if (!query) {
  console.error('usage: bun src/query_policy.ts "<query text>" [limit]');
  process.exit(1);
}
const limit = Number(process.argv[3] ?? "5");

// ── load the corpus-selected policy (fallback: hybrid) ──────────────────────
let strategy: Strategy = "hybrid";
let rrfK = 60;
let source = "default (no policy file yet)";
if (existsSync(POLICY)) {
  const p = JSON.parse(readFileSync(POLICY, "utf-8"));
  if (p.strategy === "keyword" || p.strategy === "vector" || p.strategy === "hybrid") {
    strategy = p.strategy;
  }
  if (typeof p.rrf_k === "number") rrfK = p.rrf_k;
  source = `results/search_policy.json (n_pages=${p.n_pages}, recall=${p.recall})`;
}

// ── bootstrap engine (identical sequence to auto_eval.ts / the CLI) ──────────
const { loadConfig, toEngineConfig } = await import(`${GB}/core/config.ts`);
const { createEngine } = await import(`${GB}/core/engine-factory.ts`);
const { connectWithRetry } = await import(`${GB}/core/db.ts`);
const { configureGateway, reconfigureGatewayWithEngine } = await import(`${GB}/core/ai/gateway.ts`);
const { buildGatewayConfig } = await import(`${GB}/core/ai/build-gateway-config.ts`);
const { embed } = await import(`${GB}/core/embedding.ts`);
const { hybridSearch } = await import(`${GB}/core/search/hybrid.ts`);

const config = loadConfig();
configureGateway(buildGatewayConfig(config));
const engine = await createEngine(toEngineConfig(config));
await connectWithRetry(engine, toEngineConfig(config), { noRetry: true });
await reconfigureGatewayWithEngine(engine);

// ── route to the policy-selected strategy ───────────────────────────────────
async function search(): Promise<{ slug: string }[]> {
  if (strategy === "keyword") return engine.searchKeyword(query, { limit });
  if (strategy === "vector") return engine.searchVector(await embed(query), { limit });
  return hybridSearch(engine, query, { limit, expansion: false, rrfK });
}

const results = await search();
const knob = strategy === "hybrid" ? ` rrf_k=${rrfK}` : "";
console.log(`policy: strategy=${strategy}${knob}  ←  ${source}`);
console.log(`query: ${JSON.stringify(query)}  (limit=${limit})`);
for (const [i, r] of results.entries()) console.log(`  ${i + 1}. ${r.slug}`);

await engine.disconnect?.();
process.exit(0);
