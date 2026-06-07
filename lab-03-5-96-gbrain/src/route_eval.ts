/**
 * route_eval.ts — does PER-QUERY routing beat the single global policy?
 *
 * The global loop (policy_eval.ts) picks ONE arm for the whole corpus. But the best
 * arm is query-dependent: proper-noun lookups favour keyword/hybrid, semantic factoids
 * favour dense vector. This script measures whether routing each query to its own best
 * arm beats committing to the global winner — and how close a CHEAP heuristic classifier
 * gets to that ceiling.
 *
 * It scores three routers with the same budget-aware discounted grounding@C as the policy:
 *   - global   : every query → the global winner (here hybrid). The baseline to beat.
 *   - heuristic : a zero-LLM classifier picks an arm per query (the shippable router).
 *   - oracle   : every query → its own best arm. The CEILING; unattainable in production
 *                (it peeks at the labels) but it bounds how much routing can ever help.
 *
 * If oracle ≈ global, routing has no headroom on this corpus — report it and stop.
 *
 * Run: bun src/route_eval.ts        (needs the corpus loaded + OLLAMA_* up)
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

import { budgetScore, coverage } from "./grounding.ts";

const GB = "/Users/yuxinliu/code/agent-prep/gbrain/src";
const GOLDEN = process.env.GOLDEN_EVAL ?? `${import.meta.dir}/../data/golden_eval.json`;
const SLUG_DUMP = `${import.meta.dir}/../results/route_slugs.json`;
const ARM_DUMP = `${import.meta.dir}/../results/arm_scores.json`; // per-arm grounding + slugs (verify_arch.py)
const K = Number(process.env.POLICY_K ?? "5");
const C = Number(process.env.POLICY_C ?? "3");

interface GoldenQ { q: string; expected_entities: string[]; domain: string }
type Strategy = "keyword" | "vector" | "hybrid";
const strategies: readonly Strategy[] = ["keyword", "vector", "hybrid"];
const GLOBAL: Strategy = "hybrid"; // current global policy winner on the mixed corpus

const golden: GoldenQ[] = JSON.parse(readFileSync(GOLDEN, "utf-8")).questions;

// ── cheap, deterministic query classifier (the shippable router) ─────────────
// Signal: proper-noun lookups (capitalised entity tokens, short query) are exact-term
// territory → keyword's strength; everything else stays on the global hybrid default.
const STOP = new Set(["What", "Who", "Where", "Which", "When", "How", "Why", "Does",
  "Did", "Is", "Are", "The", "In", "Of", "A", "An"]);
function classify(q: string): Strategy {
  const properNouns = (q.match(/\b[A-Z][a-zA-Z]+\b/g) ?? []).filter(w => !STOP.has(w));
  const words = q.split(/\s+/).length;
  if (properNouns.length >= 2 && words <= 9) return "keyword"; // short proper-noun lookup
  return "hybrid";
}

// ── engine bootstrap (identical sequence to policy_eval.ts / the CLI) ────────
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

const textCache = new Map<string, string>();
async function sectionText(slug: string): Promise<string> {
  if (textCache.has(slug)) return textCache.get(slug)!;
  const p = await engine.getPage(slug);
  const t = p ? `${p.title}\n${p.compiled_truth}\n${p.timeline}`.toLowerCase() : "";
  textCache.set(slug, t);
  return t;
}

// ── per-arm, per-question discounted grounding@C + top-C slugs ───────────────
const score = {} as Record<Strategy, number[]>;      // score[arm][i] = gDisc for question i
const slugsByArm = {} as Record<Strategy, string[][]>; // slugsByArm[arm][i] = top-C slugs (for the answer A/B)
for (const strategy of strategies) {
  const report = await runEval(
    engine,
    golden.map(g => ({ query: g.q, relevant: [] as string[] })),
    { strategy, expand: false, limit: K },
    K,
  );
  const row: number[] = [];
  const slugs: string[][] = [];
  for (let i = 0; i < golden.length; i++) {
    const topC: string[] = report.queries[i].hits.slice(0, C);
    const ents = golden[i].expected_entities;
    const covs = await Promise.all(topC.map(async s => coverage(await sectionText(s), ents)));
    row.push(budgetScore(covs, C).gDisc);
    slugs.push(topC);
  }
  score[strategy] = row;
  slugsByArm[strategy] = slugs;
}

// ── routers ──────────────────────────────────────────────────────────────────
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
const bestArm = (i: number): Strategy =>
  strategies.reduce((best, s) => (score[s][i] > score[best][i] ? s : best), strategies[0]);

const globalScores    = golden.map((_, i) => score[GLOBAL][i]);
const heuristicPicks  = golden.map(g => classify(g.q));
const heuristicScores = golden.map((_, i) => score[heuristicPicks[i]][i]);
const oracleScores    = golden.map((_, i) => score[bestArm(i)][i]);

// ── report ───────────────────────────────────────────────────────────────────
const pad = (s: string, n: number) => s.padEnd(n);
const f = (x: number) => x.toFixed(3);
console.log(`route_eval: ${golden.length} questions · ${nPages} pages · K=${K} C=${C}\n`);
console.log(pad("#", 3) + pad("dom", 7) + pad("key", 7) + pad("vec", 7) + pad("hyb", 7) +
  pad("best", 9) + pad("heur", 9) + "question");
console.log("-".repeat(86));
golden.forEach((g, i) => {
  const best = bestArm(i);
  const flag = best !== GLOBAL ? " *" : "";              // * = routing could beat global here
  console.log(pad(String(i), 3) + pad(g.domain, 7) +
    pad(f(score.keyword[i]), 7) + pad(f(score.vector[i]), 7) + pad(f(score.hybrid[i]), 7) +
    pad(best + flag, 9) + pad(heuristicPicks[i], 9) + g.q.slice(0, 40));
});
console.log("-".repeat(86));
const wins       = golden.filter((_, i) => bestArm(i) !== GLOBAL).length;
const mGlobal    = mean(globalScores);
const mHeuristic = mean(heuristicScores);
const mOracle    = mean(oracleScores);
console.log(`\nrouter             grounding@${C}   Δ vs global`);
console.log(`global (${GLOBAL})    ${f(mGlobal)}        —`);
console.log(`heuristic          ${f(mHeuristic)}        ${f(mHeuristic - mGlobal)}`);
console.log(`oracle (ceiling)   ${f(mOracle)}        ${f(mOracle - mGlobal)}`);
console.log(`\n${wins}/${golden.length} questions have a non-global best arm (routing headroom).`);

// ── dump per-question slugs for the answer-quality A/B (answer_route_ab.py) ──
const dump = golden.map((g, i) => {
  const best = bestArm(i);
  return {
    q: g.q,
    domain: g.domain,
    best_arm: best,
    global_arm: GLOBAL,
    global_slugs: slugsByArm[GLOBAL][i],
    routed_slugs: slugsByArm[best][i],
  };
});
mkdirSync(dirname(SLUG_DUMP), { recursive: true });
writeFileSync(SLUG_DUMP, JSON.stringify(dump, null, 2) + "\n");
console.log(`wrote per-question slugs → ${SLUG_DUMP}`);

// ── dump per-arm grounding + slugs for architecture verification (verify_arch.py) ──
const armDump = golden.map((g, i) => ({
  q: g.q,
  domain: g.domain,
  grounding: Object.fromEntries(strategies.map(s => [s, score[s][i]])),
  slugs: Object.fromEntries(strategies.map(s => [s, slugsByArm[s][i]])),
}));
writeFileSync(ARM_DUMP, JSON.stringify(armDump, null, 2) + "\n");
console.log(`wrote per-arm grounding+slugs → ${ARM_DUMP}`);

await engine.disconnect?.();
process.exit(0);
