/**
 * bench_w27_questions.ts — the REAL W2.7 comparison (Path A, honest version).
 *
 * Runs W2.7's own labeled financial questions (data/eval_v2.json + eval.json:
 * `{q, expected_entities}`) through GBrain retrieval over the brk 10-K, and scores
 * by whether the RETRIEVED sections actually contain the expected entities
 * (substring grounding@K). This replaces the known-item PROXY queries with real
 * human questions.
 *
 * Metric note (kept honest): W2.7's three-way scored ANSWER quality across INDEX
 * TYPES (vector / GraphRAG / tree-index). This scores RETRIEVAL grounding across
 * GBrain's ARMS (keyword / vector / hybrid). Same corpus + same questions, but a
 * retrieval metric, not an answer-generation metric — so compare the SHAPE of the
 * findings (which method wins which query class), not the absolute numbers.
 *
 *   grounding@K   = mean over questions of the BEST top-K section's entity coverage
 *                   (fraction of expected_entities found as substrings in it)
 *   answerable@K  = fraction of questions where some top-K section covers ALL entities
 *
 * Run: bun src/bench_w27_questions.ts   (needs gbrain_brk loaded + OLLAMA_* up)
 */
import { readFileSync } from "node:fs";

const GB = process.env.GBRAIN_SRC ?? `${import.meta.dir}/../../gbrain/src`;
const EVAL_DIR = "/Users/yuxinliu/code/agent-prep/lab-02-7-pageindex/data";
const K = Number(process.env.BENCH_K ?? "5");

interface W27Q { q: string; expected_entities: string[]; type?: string }
type Strategy = "keyword" | "vector" | "hybrid";

// ── load + dedup W2.7's labeled questions ───────────────────────────────────
const raw: W27Q[] = ["eval_v2.json", "eval.json"].flatMap(f =>
  JSON.parse(readFileSync(`${EVAL_DIR}/${f}`, "utf-8")));
const seen = new Set<string>();
const questions = raw.filter(q => q.q && q.expected_entities?.length &&
  !seen.has(q.q) && (seen.add(q.q), true));

// ── engine bootstrap (same sequence as auto_eval.ts / the CLI) ──────────────
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

// section text cache (slug → "title + compiled_truth + timeline", lowercased)
const textCache = new Map<string, string>();
async function sectionText(slug: string): Promise<string> {
  if (textCache.has(slug)) return textCache.get(slug)!;
  const p = await engine.getPage(slug);
  const t = p ? `${p.title}\n${p.compiled_truth}\n${p.timeline}`.toLowerCase() : "";
  textCache.set(slug, t);
  return t;
}

// fraction of expected_entities present as substrings in a section
function coverage(text: string, entities: string[]): number {
  const hit = entities.filter(e => text.includes(e.toLowerCase())).length;
  return entities.length ? hit / entities.length : 0;
}

const strategies: readonly Strategy[] = ["keyword", "vector", "hybrid"];
const qrels = questions.map(q => ({ query: q.q, relevant: [] as string[] }));

const summary: Record<Strategy, { grounding: number; answerable: number }> =
  {} as Record<Strategy, { grounding: number; answerable: number }>;

for (const strategy of strategies) {
  // reuse runEval purely to get ranked hits per query (metrics ignored — no gold slug)
  const report = await runEval(engine, qrels, { strategy, expand: false, limit: K }, K);
  let groundingSum = 0;
  let answerable = 0;
  for (let i = 0; i < questions.length; i++) {
    const hits: string[] = report.queries[i].hits.slice(0, K);
    const ents = questions[i].expected_entities.map(e => e.toLowerCase());
    let best = 0;
    for (const slug of hits) best = Math.max(best, coverage(await sectionText(slug), ents));
    groundingSum += best;
    if (best === 1) answerable++;
  }
  summary[strategy] = {
    grounding: groundingSum / questions.length,
    answerable: answerable / questions.length,
  };
}

const pad = (s: string, n: number) => s.padEnd(n);
console.log(`W2.7 real-question retrieval over brk 10-K — ${questions.length} questions · K=${K}\n`);
console.log(pad("strategy", 12) + pad(`grounding@${K}`, 15) + pad(`answerable@${K}`, 15));
console.log("-".repeat(42));
for (const s of strategies) {
  console.log(pad(s, 12) + pad(summary[s].grounding.toFixed(3), 15) + pad(summary[s].answerable.toFixed(3), 15));
}
console.log("\nmetric: retrieval grounding (do retrieved sections CONTAIN the expected entities), " +
  "NOT answer quality. Compare the winner SHAPE vs W2.7's three-way, not absolute numbers.");

await engine.disconnect?.();
process.exit(0);
