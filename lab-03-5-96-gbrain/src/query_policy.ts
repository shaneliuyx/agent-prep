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
 *
 * Per-query ROUTER switch (opt-in): set `QUERY_ROUTER=on` to route each query to the arm its
 * TYPE favours (kw→OR-preprocessed keyword · vec→vector · mixed→hybrid) instead of the single
 * global policy. DEFAULT OFF — routing is a *conditional* win (route_principle_ab.ts: +0.039
 * grounding / +0.083 answer-quality ONLY when the workload spans query types AND a cheap
 * classifier detects them; on a single-type corpus a perfect router merely ties global and a
 * real classifier nets negative). So it ships off; turn it on when the traffic is mixed-type
 * and the extra accuracy is worth the per-query classify step.
 *
 * Usage: bun src/query_policy.ts "<query text>" [limit]      (global policy — default)
 *        QUERY_ROUTER=on bun src/query_policy.ts "<query>" [limit]   (per-query router)
 */
import { existsSync, readFileSync } from "node:fs";

import { type Strategy, classifyType, preprocessOR, TYPE_TO_STRATEGY } from "./query_routing.ts";

const GB = "/Users/yuxinliu/code/agent-prep/gbrain/src";
const POLICY = `${import.meta.dir}/../results/search_policy.json`;

const ROUTER_ON = ["1", "on", "true", "yes"].includes((process.env.QUERY_ROUTER ?? "").toLowerCase());

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

// ── pick the strategy: per-query router (opt-in) or the global policy (default) ──
// Router ON: classify the query's type and route to the arm it favours; the keyword arm is
// fed the OR-preprocessed query (the validated path). Router OFF: the single global strategy.
let routeNote = "";
function pickStrategy(): { strategy: Strategy; kwQuery: string } {
  if (!ROUTER_ON) {
    routeNote = `router OFF (global policy)  ←  ${source}`;
    return { strategy, kwQuery: query };
  }
  const type = classifyType(query);
  const routed = TYPE_TO_STRATEGY[type];
  routeNote = `router ON: type=${type} → ${routed}`;
  return { strategy: routed, kwQuery: routed === "keyword" ? preprocessOR(query) : query };
}

async function search(): Promise<{ slug: string }[]> {
  const { strategy: strat, kwQuery } = pickStrategy();
  if (strat === "keyword") return engine.searchKeyword(kwQuery, { limit });
  if (strat === "vector") return engine.searchVector(await embed(query), { limit });
  return hybridSearch(engine, query, { limit, expansion: false, rrfK });
}

const results = await search();
console.log(`policy: ${routeNote}`);
console.log(`query: ${JSON.stringify(query)}  (limit=${limit})`);
for (const [i, r] of results.entries()) console.log(`  ${i + 1}. ${r.slug}`);

await engine.disconnect?.();
process.exit(0);
