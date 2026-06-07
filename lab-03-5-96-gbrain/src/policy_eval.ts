/**
 * policy_eval.ts — REAL-question policy generator (the trustworthy policy source).
 *
 * Replaces auto_eval.ts's known-item PROXY as the thing that writes the policy.
 * The proxy used page titles as queries and mis-selected `keyword`; real questions
 * refuted that ~5×. So the policy must come from a real, labeled golden set.
 *
 * Reads data/golden_eval.json (real questions + expected_entities, domain-tagged),
 * runs keyword/vector/hybrid over the CURRENT corpus, scores DISCOUNTED grounding
 * over the context budget, picks the overall winner, and WRITES
 * results/search_policy.json. Re-run after every ingest → the policy ADAPTS as the
 * corpus drifts (new pages = distractors that can shift which arm retrieves best).
 *
 * Why discounted-over-budget, not the old rank-blind max-over-top-K:
 *   The real objective is ANSWER quality, not raw retrieval. Hybrid RRF can fuse a
 *   keyword distractor above the dense answer chunk — the answer is still "in top-K"
 *   (rank-blind grounding unchanged) but it now reads later, or falls past the chunks
 *   the generator is actually given. Rank-blind grounding cannot see that; it's the
 *   exact RRF failure we care about. So we score the prompt the generator really sees:
 *
 *   grounding@C  = mean over questions of  max_i coverage_i · disc(i)   over top-C
 *                  (position-weighted best coverage; primacy via disc, budget via C).
 *                  Answer demoted by RRF → lower; answer pushed past C → 0.
 *   answerable@C = fraction of questions where some top-C section covers ALL entities.
 *
 * K = retrieval depth (how many hits we pull); C = context budget (how many chunks
 * the generator reads, C ≤ K). Set C to the production injected-chunk count so the
 * metric measures "did the answer survive into the prompt?". The golden set is a
 * stable measuring stick (version-controlled); keep it representative of the workload.
 *
 * Run: bun src/policy_eval.ts        (needs the corpus loaded + OLLAMA_* up)
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { budgetScore, coverage } from "./grounding.ts";

const GB = "/Users/yuxinliu/code/agent-prep/gbrain/src";
const GOLDEN = `${import.meta.dir}/../data/golden_eval.json`;
const POLICY = `${import.meta.dir}/../results/search_policy.json`;
const K = Number(process.env.POLICY_K ?? "5");   // retrieval depth (hits pulled)
const C = Number(process.env.POLICY_C ?? "3");   // context budget (chunks the generator reads; C ≤ K)

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
const domains = [...new Set(golden.map(g => g.domain))];
const strategies: readonly Strategy[] = ["keyword", "vector", "hybrid"];

// Per strategy, per question: discounted grounding@C (drives policy) + raw best
// coverage in budget (feeds answerable@C). Retrieval pulls K hits; we score only the
// C the generator actually reads, so an RRF reorder that demotes the answer — or
// pushes it past C — is penalised. See grounding.ts for the scoring contract.
const perQ = {} as Record<Strategy, number[]>;       // discounted grounding@C
const perQFull = {} as Record<Strategy, number[]>;   // raw best coverage within budget
for (const strategy of strategies) {
  const report = await runEval(
    engine,
    golden.map(g => ({ query: g.q, relevant: [] as string[] })),
    { strategy, expand: false, limit: K },
    K,
  );
  const gDisc: number[] = [];
  const gFull: number[] = [];
  for (let i = 0; i < golden.length; i++) {
    const topC: string[] = report.queries[i].hits.slice(0, C); // limit fetches to the budget window
    const ents = golden[i].expected_entities;
    const coverages = await Promise.all(topC.map(async slug => coverage(await sectionText(slug), ents)));
    const { gDisc: d, gFull: full } = budgetScore(coverages, C);
    gDisc.push(d);
    gFull.push(full);
  }
  perQ[strategy] = gDisc;
  perQFull[strategy] = gFull;
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
const groundingFor = (s: Strategy, domain?: string) =>
  mean(perQ[s].filter((_, i) => !domain || golden[i].domain === domain));
// flatFor = the OLD rank-blind metric (mean raw best-coverage in budget). Shown next
// to the discounted score: where flat > discounted, the answer sits below rank 0 —
// i.e. an arm (often hybrid RRF) demoted it. The gap is the penalty we added.
const flatFor = (s: Strategy, domain?: string) =>
  mean(perQFull[s].filter((_, i) => !domain || golden[i].domain === domain));
const answerableFor = (s: Strategy) =>
  mean(perQFull[s].map(v => (v === 1 ? 1 : 0)));

// ── report: overall + per-domain grounding@C ────────────────────────────────
const pad = (s: string, n: number) => s.padEnd(n);
const f = (x: number) => x.toFixed(3);
console.log(`policy_eval: golden set = ${golden.length} real questions ` +
  `(${domains.map(d => `${d}:${golden.filter(g => g.domain === d).length}`).join(", ")}) ` +
  `· corpus = ${nPages} pages · retrieval K=${K} · context budget C=${C}\n`);
console.log(pad("strategy", 10) + pad(`grnd@${C}↓`, 12) + pad(`flat@${C}`, 11) + pad(`answ@${C}`, 11) +
  domains.map(d => pad(`g@${C}:${d}`, 13)).join(""));
console.log("-".repeat(44 + domains.length * 13));
for (const s of strategies) {
  console.log(pad(s, 10) + pad(f(groundingFor(s)), 12) + pad(f(flatFor(s)), 11) + pad(f(answerableFor(s)), 11) +
    domains.map(d => pad(f(groundingFor(s, d)), 13)).join(""));
}

// ── decide + write policy (winner = overall discounted grounding, tie → answerable) ──
const winner = [...strategies].sort((a, b) =>
  groundingFor(b) - groundingFor(a) || answerableFor(b) - answerableFor(a))[0];

const policy = {
  strategy: winner,
  rrf_k: 60,
  k: K,
  c: C,
  metric: "discounted_grounding@C (position-weighted best coverage over the context budget)",
  source: "golden_eval",
  n_questions: golden.length,
  n_pages: nPages,
  grounding: groundingFor(winner),
  per_domain: Object.fromEntries(domains.map(d => [d, groundingFor(winner, d)])),
  note: "auto-selected from data/golden_eval.json (REAL questions, discounted grounding@C: "
    + "rank- and budget-aware, so an RRF reorder that demotes the answer is penalized). "
    + "Read by query_policy.ts; does NOT change stock `gbrain query` (hybrid-only).",
};
mkdirSync(dirname(POLICY), { recursive: true });
writeFileSync(POLICY, JSON.stringify(policy, null, 2) + "\n");
console.log(`\napplied policy → strategy=${winner} (grnd@${C}↓=${f(groundingFor(winner))})  ` +
  `← results/search_policy.json [source: golden_eval]`);

await engine.disconnect?.();
process.exit(0);
