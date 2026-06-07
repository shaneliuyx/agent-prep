/**
 * auto_eval.ts — post-ingest retrieval-health regression check.
 *
 * WIRED INTO THE INGEST FLOW: ingest_agent.py / resumable_ingest.py call this
 * via `bun src/auto_eval.ts` right after reconcile_graph(), so every corpus
 * write is followed by a keyword-vs-vector-vs-hybrid measurement on the LIVE
 * brain. It runs at the ENGINE layer (engine.searchKeyword / searchVector /
 * hybridSearch via gbrain's runEval) — the CLI cannot A/B these (Phase 6 trap:
 * `gbrain search` and `gbrain query` share one handler).
 *
 * AUTO-LABELING (no hand-built qrels): this is a KNOWN-ITEM eval. We sample Q
 * pages and synthesize one query per page whose gold answer IS that page's own
 * slug:
 *   - exact    query = the page TITLE          → exercises the keyword arm
 *   - semantic query = body words MINUS the title tokens → exercises the vector
 *                      arm (the gold page's name is NOT a literal substring, so
 *                      keyword cannot trivially win it)
 * Q scales with corpus size: Q = clamp(ceil(ratio·N), Q_MIN, Q_MAX), sampled
 * with a SEEDED shuffle so a given page-set is reproducible run-to-run.
 *
 * HONEST LIMITATION (do not oversell): known-item queries are drawn from each
 * doc's OWN vocabulary, so they are a PROXY for real user queries, not a
 * substitute for a hand-labeled gold set that reflects the real query
 * distribution. This harness answers "did retrieval regress / which arm wins on
 * THIS corpus shape?" — it is a guardrail + trend signal, not a quality verdict.
 *
 * Config (env): AUTO_EVAL_RATIO=0.30  AUTO_EVAL_QMIN=5  AUTO_EVAL_QMAX=50
 *   AUTO_EVAL_K=3  AUTO_EVAL_SEED=42  AUTO_EVAL_REGRESS_EPS=0.05  AUTO_EVAL_STRICT=0
 * Run: bun src/auto_eval.ts   (needs GBRAIN_DATABASE_URL + OLLAMA_* env)
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

const GB = process.env.GBRAIN_SRC ?? `${import.meta.dir}/../../gbrain/src`;
const RESULTS = `${import.meta.dir}/../results/auto_eval.jsonl`;
// The APPLY target: the decision artifact the policy-aware query path reads to
// honor the corpus-selected strategy (closes the measure→decide→apply loop).
const POLICY = `${import.meta.dir}/../results/search_policy.json`;

const RATIO = Number(process.env.AUTO_EVAL_RATIO ?? "0.30");
const Q_MIN = Number(process.env.AUTO_EVAL_QMIN ?? "5");
const Q_MAX = Number(process.env.AUTO_EVAL_QMAX ?? "50");
const K = Number(process.env.AUTO_EVAL_K ?? "3");
const SEED = Number(process.env.AUTO_EVAL_SEED ?? "42") || 42;
const REGRESS_EPS = Number(process.env.AUTO_EVAL_REGRESS_EPS ?? "0.05");
const STRICT = process.env.AUTO_EVAL_STRICT === "1";

export interface PageLite { slug: string; title: string; compiled_truth: string }
export type Kind = "exact" | "semantic";
export interface AutoQrel { query: string; relevant: string[]; kind: Kind }
type Strategy = "keyword" | "vector" | "hybrid";

// Q scales with corpus size: clamp(ceil(ratio·N), Q_MIN, Q_MAX), never above N.
export function computeQ(n: number, ratio: number, qMin: number, qMax: number): number {
  return Math.min(n, qMax, Math.max(qMin, Math.ceil(ratio * n)));
}

// ── deterministic sampling ────────────────────────────────────────────────
// mulberry32: tiny seeded PRNG so the sampled page-set is reproducible for a
// given (corpus, seed). Adding pages reshuffles — the eval is a per-run trend,
// not a frozen gold set.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function seededShuffle<T>(arr: readonly T[], seed: number): T[] {
  const rand = mulberry32(seed);
  const out = [...arr];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

// ── known-item query synthesis ────────────────────────────────────────────
const WIKILINK = /\[\[([^\]]+)\]\]/g;

// Replace [[dir/slug]] with its readable tail ("companies/acme-ai" → "acme ai")
// so the body becomes plain words, not link markup.
function deLink(text: string): string {
  return text.replace(WIKILINK, (_m, inner: string) => {
    const tail = inner.split("/").pop() ?? "";
    return tail.replace(/-/g, " ");
  });
}

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
}

// Semantic probe: first body sentence, de-linked, with the page's TITLE tokens
// removed — so the gold page's name is never a literal substring of the query
// (keyword cannot trivially match it; the vector arm has to earn the hit).
// Returns null when too few content words survive to be a meaningful query.
export function semanticQuery(page: PageLite): string | null {
  const body = (page.compiled_truth ?? "").replace(/^#.*$/m, "").trim();
  const plain = deLink(body).replace(/\s+/g, " ").trim();
  const firstSentence = (plain.split(/(?<=[.!?])\s+/)[0] ?? "").trim();
  if (!firstSentence) return null;
  const titleTokens = new Set(tokenize(page.title));
  const kept = tokenize(firstSentence).filter(t => t.length > 2 && !titleTokens.has(t));
  return kept.length >= 3 ? kept.join(" ") : null;
}

export function buildQrels(sample: readonly PageLite[]): AutoQrel[] {
  const qrels: AutoQrel[] = [];
  for (const page of sample) {
    const exact = page.title?.trim();
    if (exact) qrels.push({ query: exact, relevant: [page.slug], kind: "exact" });
    const semantic = semanticQuery(page);
    if (semantic) qrels.push({ query: semantic, relevant: [page.slug], kind: "semantic" });
  }
  return qrels;
}

// ── live run (engine layer) — only when executed directly, not on import ────
async function main(): Promise<void> {
// engine bootstrap (identical sequence to bench_strategies.ts / the CLI)
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

// ── sample + label ─────────────────────────────────────────────────────────
const allPages: PageLite[] = await engine.listPages();
const N = allPages.length;
if (N === 0) {
  console.log("auto-eval: empty brain (0 pages) — nothing to evaluate.");
  await engine.disconnect?.();
  process.exit(0);
}

const Q = computeQ(N, RATIO, Q_MIN, Q_MAX);
// Sort by slug first so the seeded shuffle is order-independent of listPages().
const bySlug = [...allPages].sort((a, b) => (a.slug < b.slug ? -1 : 1));
const sample = seededShuffle(bySlug, SEED).slice(0, Q);
const qrels = buildQrels(sample);

const nExact = qrels.filter(q => q.kind === "exact").length;
const nSemantic = qrels.filter(q => q.kind === "semantic").length;
console.log(
  `auto-eval: N=${N} pages · ratio=${RATIO} · Q=${Q} sampled (seed=${SEED}) · ` +
  `${qrels.length} known-item queries (${nExact} exact / ${nSemantic} semantic) · K=${K}`,
);

// ── run the three strategies ────────────────────────────────────────────────
const strategies: readonly Strategy[] = ["keyword", "vector", "hybrid"];
const plain = qrels.map(({ query, relevant }) => ({ query, relevant }));
const reports: Record<Strategy, any> = {} as Record<Strategy, any>;
for (const strategy of strategies) {
  reports[strategy] = await runEval(engine, plain, { strategy, expand: false }, K);
}

// per-kind recall: where does each arm actually win?
function recallByKind(strategy: Strategy, kind: Kind): number {
  const vals = qrels
    .map((q, i) => (q.kind === kind ? reports[strategy].queries[i].recall_at_k : null))
    .filter((v): v is number => v !== null);
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : NaN;
}

const f = (x: number) => (Number.isNaN(x) ? "  n/a" : x.toFixed(3));
const pad = (s: string, n: number) => s.padEnd(n);

console.log("\n" + pad("strategy", 10) + pad(`recall@${K}`, 11) + pad("MRR", 9) +
  pad(`nDCG@${K}`, 10) + pad("R@K exact", 11) + pad("R@K seman", 11));
console.log("-".repeat(62));
for (const s of strategies) {
  const r = reports[s];
  console.log(
    pad(s, 10) + pad(f(r.mean_recall), 11) + pad(f(r.mean_mrr), 9) +
    pad(f(r.mean_ndcg), 10) + pad(f(recallByKind(s, "exact")), 11) +
    pad(f(recallByKind(s, "semantic")), 11),
  );
}

// winner by recall@K, tie-broken by MRR
const winner = [...strategies].sort((a, b) =>
  reports[b].mean_recall - reports[a].mean_recall ||
  reports[b].mean_mrr - reports[a].mean_mrr)[0];
console.log(`\nwinner on this corpus: ${winner} ` +
  `(recall@${K}=${f(reports[winner].mean_recall)}, MRR=${f(reports[winner].mean_mrr)})`);

// ── persist + regression check vs the previous run ──────────────────────────
const record = {
  ts: new Date().toISOString(),
  n_pages: N, q: Q, ratio: RATIO, seed: SEED, k: K, n_queries: qrels.length,
  winner,
  strategies: Object.fromEntries(strategies.map(s => [s, {
    recall: reports[s].mean_recall, mrr: reports[s].mean_mrr, ndcg: reports[s].mean_ndcg,
    recall_exact: recallByKind(s, "exact"), recall_semantic: recallByKind(s, "semantic"),
  }])),
};

let previous: typeof record | null = null;
if (existsSync(RESULTS)) {
  const lines = readFileSync(RESULTS, "utf-8").trim().split("\n").filter(Boolean);
  if (lines.length) {
    try { previous = JSON.parse(lines[lines.length - 1]); } catch { previous = null; }
  }
}

mkdirSync(dirname(RESULTS), { recursive: true });
appendFileSync(RESULTS, JSON.stringify(record) + "\n");

// ── APPLY: write the corpus-selected policy the query path will honor ────────
// Winner = recall@K winner (tie-broken by MRR). This is the actuator half: the
// next agent query reads search_policy.json and routes to this strategy instead
// of bare hybrid. Stock `gbrain query` is hybrid-only and is NOT affected.
const policy = {
  strategy: winner,
  rrf_k: 60,                       // GBrain RRF default; only used when strategy==="hybrid"
  k: K,
  n_pages: N,
  recall: reports[winner].mean_recall,
  per_kind: { exact: recallByKind(winner, "exact"), semantic: recallByKind(winner, "semantic") },
  ts: record.ts,
  note: "auto-selected by auto_eval.ts (known-item eval). Read by query_policy.ts; "
    + "does NOT change stock `gbrain query` (hybrid-only).",
};
writeFileSync(POLICY, JSON.stringify(policy, null, 2) + "\n");
console.log(`applied search policy → strategy=${winner}  (results/search_policy.json)`);

let regressed = false;
if (previous) {
  console.log(`\nvs previous run (${previous.ts}, N=${previous.n_pages}):`);
  for (const s of strategies) {
    const d = reports[s].mean_recall - previous.strategies[s].recall;
    const tag = d < -REGRESS_EPS ? "  ⚠ REGRESSION" : "";
    if (d < -REGRESS_EPS) regressed = true;
    console.log(`  ${pad(s, 10)} Δrecall@${K} = ${d >= 0 ? "+" : ""}${d.toFixed(3)}${tag}`);
  }
  if (!regressed) console.log("  no strategy regressed beyond ε=" + REGRESS_EPS);
} else {
  console.log("\n(no previous run — this is the baseline)");
}

console.log("\nnote: known-item proxy eval (queries drawn from each page's own text). " +
  "Measures retrieval health + per-arm fit, NOT real-query quality.");

await engine.disconnect?.();
process.exit(regressed && STRICT ? 2 : 0);
}

if (import.meta.main) {
  await main();
}
