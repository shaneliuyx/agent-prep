/**
 * policy_eval.ts — REAL-question policy generator (the trustworthy policy source).
 *
 * Replaces auto_eval.ts's known-item PROXY as the thing that writes the policy.
 * The proxy used page titles as queries and mis-selected `keyword`; real questions
 * refuted that ~5×. So the policy must come from a real, labeled golden set.
 *
 * Reads data/golden_eval.json (real questions + expected_entities, domain-tagged),
 * runs keyword/vector/hybrid over the CURRENT corpus, scores substring grounding@K
 * (do the retrieved sections CONTAIN the expected entities), picks the overall
 * winner, and WRITES results/search_policy.json. Re-run after every ingest →
 * the policy ADAPTS as the corpus drifts (new pages = distractors that can shift
 * which arm retrieves best).
 *
 *   grounding@K  = mean over questions of the best top-K section's entity coverage
 *   answerable@K = fraction of questions where some top-K section covers ALL entities
 *
 * The golden set is a stable measuring stick (version-controlled); keep it
 * representative of the real query distribution as the workload shifts.
 *
 * Run: bun src/policy_eval.ts        (needs the corpus loaded + OLLAMA_* up)
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

const GB = "/Users/yuxinliu/code/agent-prep/gbrain/src";
const GOLDEN = `${import.meta.dir}/../data/golden_eval.json`;
const POLICY = `${import.meta.dir}/../results/search_policy.json`;
const K = Number(process.env.POLICY_K ?? "5");

interface GoldenQ { q: string; expected_entities: string[]; domain: string }
type Strategy = "keyword" | "vector" | "hybrid";

if (!existsSync(GOLDEN)) {
  console.error(`no golden set at ${GOLDEN} — run auto_eval.ts (known-item fallback) instead.`);
  process.exit(1);
}
const golden: GoldenQ[] = JSON.parse(readFileSync(GOLDEN, "utf-8")).questions;

// ── engine bootstrap (identical sequence to auto_eval.ts / the CLI) ─────────
const { loadConfig, toEngineConfig } = await import(`${GB}/core/config.ts`);
const { createEngine } = await import(`${GB}/core/engine-factory.ts`);
const { connectWithRetry } = await import(`${GB}/core/db.ts`);
const { configureGateway, reconfigureGatewayWithEngine } = await import(`${GB}/core/ai/gateway.ts`);
const { buildGatewayConfig } = await import(`${GB}/core/ai/build-gateway-config.ts`);
const { runEval } = await import(`${GB}/core/search/eval.ts`);

const config = loadConfig();
configureGateway(buildGatewayConfig(config));
const engine = await createEngine(toEngineConfig(config));
await connectWithRetry(engine, toEngineConfig(config), { noRetry: true });
await reconfigureGatewayWithEngine(engine);
const nPages = (await engine.listPages()).length;

// section text cache (slug → lowercased title+body+timeline)
const textCache = new Map<string, string>();
async function sectionText(slug: string): Promise<string> {
  if (textCache.has(slug)) return textCache.get(slug)!;
  const p = await engine.getPage(slug);
  const t = p ? `${p.title}\n${p.compiled_truth}\n${p.timeline}`.toLowerCase() : "";
  textCache.set(slug, t);
  return t;
}
const coverage = (text: string, ents: string[]) =>
  ents.length ? ents.filter(e => text.includes(e.toLowerCase())).length / ents.length : 0;

const domains = [...new Set(golden.map(g => g.domain))];
const strategies: readonly Strategy[] = ["keyword", "vector", "hybrid"];

// per strategy: best-section coverage for each question
const perQ = {} as Record<Strategy, number[]>;
for (const strategy of strategies) {
  const report = await runEval(
    engine,
    golden.map(g => ({ query: g.q, relevant: [] as string[] })),
    { strategy, expand: false, limit: K },
    K,
  );
  const scores: number[] = [];
  for (let i = 0; i < golden.length; i++) {
    const hits: string[] = report.queries[i].hits.slice(0, K);
    const ents = golden[i].expected_entities;
    let best = 0;
    for (const slug of hits) best = Math.max(best, coverage(await sectionText(slug), ents));
    scores.push(best);
  }
  perQ[strategy] = scores;
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
const groundingFor = (s: Strategy, domain?: string) =>
  mean(perQ[s].filter((_, i) => !domain || golden[i].domain === domain));
const answerableFor = (s: Strategy) =>
  mean(perQ[s].map(v => (v === 1 ? 1 : 0)));

// ── report: overall + per-domain grounding@K ────────────────────────────────
const pad = (s: string, n: number) => s.padEnd(n);
const f = (x: number) => x.toFixed(3);
console.log(`policy_eval: golden set = ${golden.length} real questions ` +
  `(${domains.map(d => `${d}:${golden.filter(g => g.domain === d).length}`).join(", ")}) ` +
  `· corpus = ${nPages} pages · K=${K}\n`);
console.log(pad("strategy", 10) + pad(`grounding@${K}`, 14) + pad(`answerable@${K}`, 15) +
  domains.map(d => pad(`g@${K}:${d}`, 13)).join(""));
console.log("-".repeat(40 + domains.length * 13));
for (const s of strategies) {
  console.log(pad(s, 10) + pad(f(groundingFor(s)), 14) + pad(f(answerableFor(s)), 15) +
    domains.map(d => pad(f(groundingFor(s, d)), 13)).join(""));
}

// ── decide + write policy (winner = overall grounding, tie → answerable) ────
const winner = [...strategies].sort((a, b) =>
  groundingFor(b) - groundingFor(a) || answerableFor(b) - answerableFor(a))[0];

const policy = {
  strategy: winner,
  rrf_k: 60,
  k: K,
  source: "golden_eval",
  n_questions: golden.length,
  n_pages: nPages,
  grounding: groundingFor(winner),
  per_domain: Object.fromEntries(domains.map(d => [d, groundingFor(winner, d)])),
  note: "auto-selected from data/golden_eval.json (REAL questions, grounding@K). "
    + "Read by query_policy.ts; does NOT change stock `gbrain query` (hybrid-only).",
};
mkdirSync(dirname(POLICY), { recursive: true });
writeFileSync(POLICY, JSON.stringify(policy, null, 2) + "\n");
console.log(`\napplied policy → strategy=${winner} (grounding@${K}=${f(groundingFor(winner))})  ` +
  `← results/search_policy.json [source: golden_eval]`);

await engine.disconnect?.();
process.exit(0);
