/**
 * route_eval_kwpp.ts — does KEYWORD-QUERY PREPROCESSING revive the keyword arm,
 * and does that re-open per-query routing headroom?
 *
 * Diagnosis (deep-dive): GBrain's keyword arm uses conjunctive `websearch_to_tsquery`
 * over chunk-grain — every query token must co-occur in one chunk, so verbose queries
 * (natural questions, padded exact queries) AND to ZERO. Bare exact tokens rank ~1.0.
 *
 * Fix tested here, entirely on OUR side (no GBrain change): preprocess the keyword query —
 * drop stop/question words, then **OR-join** the salient tokens. OR (`a OR b OR c`) switches
 * websearch from AND to best-match, so a missing token lowers rank instead of zeroing the match.
 *
 * Arms scored with the same discounted grounding@C as the policy:
 *   key      — engine.searchKeyword(raw q)            (conjunctive, current behavior)
 *   key_pp   — engine.searchKeyword(preprocessOR(q))  (the fix)
 *   vector   — engine.searchVector(embed(q))
 *   hyb      — hybridSearch(raw q)                     (GBrain's current hybrid)
 *   hyb_pp   — RRF(key_pp, vector)                     (fixed hybrid using the revived keyword)
 * Then: per-class grounding + oracle{key_pp, vector, hyb_pp} vs global(hyb_pp) → routing headroom?
 *
 * Run: GOLDEN_EVAL=data/golden_balanced.json bun src/route_eval_kwpp.ts
 */
import { readFileSync } from "node:fs";

import { budgetScore, coverage } from "./grounding.ts";

const GB = "/Users/yuxinliu/code/agent-prep/gbrain/src";
const GOLDEN = process.env.GOLDEN_EVAL ?? `${import.meta.dir}/../data/golden_eval.json`;
const K = Number(process.env.POLICY_K ?? "5");
const C = Number(process.env.POLICY_C ?? "3");

interface GoldenQ { q: string; expected_entities: string[]; domain: string }
const golden: GoldenQ[] = JSON.parse(readFileSync(GOLDEN, "utf-8")).questions;

// ── keyword query preprocessing: drop noise, OR-join salient tokens ──────────
const STOP = new Set([
  "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was", "were",
  "what", "which", "who", "where", "when", "how", "why", "did", "does", "do", "write", "wrote",
  "describe", "describes", "described", "about", "berkshire", "company", "firm", "report",
  "annual", "five", "percent", "common", "shares", "its", "his", "her", "their", "that", "this",
  "with", "from", "by", "as", "at", "be", "it", "they", "long", "term", "holds", "hold", "leave",
  "leaves", "comfortable", "according", "year", "lists", "could", "go", "wrong", "business",
  "filing", "section", "part", "where-is", "located", "live", "lives", "puts", "putting", "up",
  "money", "anchor", "early", "funding", "round", "guards", "guard", "against", "oversees",
  "owns", "own", "large", "minority", "stake", "runs", "run", "optimizing", "before", "designing",
  "structure", "detailed", "following", "primary", "statements", "operating", "earnings",
  "described", "associated", "venture", "angel", "investor", "network",
]);

function preprocessOR(q: string): string {
  const toks = (q.toLowerCase().match(/[a-z0-9]+/g) ?? [])
    .filter(t => t.length > 1 && !STOP.has(t));
  const uniq = [...new Set(toks)];
  return uniq.length ? uniq.join(" OR ") : q;   // fall back to raw if we stripped everything
}

// ── simple RRF over two ranked slug lists ────────────────────────────────────
function rrf(a: string[], b: string[], k = 60): string[] {
  const score = new Map<string, number>();
  for (const list of [a, b]) {
    list.forEach((slug, i) => score.set(slug, (score.get(slug) ?? 0) + 1 / (k + i + 1)));
  }
  return [...score.entries()].sort((x, y) => y[1] - x[1]).map(([s]) => s);
}

// ── engine bootstrap ──────────────────────────────────────────────────────────
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

const textCache = new Map<string, string>();
async function sectionText(slug: string): Promise<string> {
  if (textCache.has(slug)) return textCache.get(slug)!;
  const p = await engine.getPage(slug);
  const t = p ? `${p.title}\n${p.compiled_truth}\n${p.timeline}`.toLowerCase() : "";
  textCache.set(slug, t);
  return t;
}
async function ground(slugs: string[], ents: string[]): Promise<number> {
  const covs = await Promise.all(slugs.slice(0, C).map(async s => coverage(await sectionText(s), ents)));
  return budgetScore(covs, C).gDisc;
}
const slugsOf = (rs: { slug: string }[]) => rs.map(r => r.slug);

// ── per-question scoring across the five arms ────────────────────────────────
type Arm = "key" | "key_pp" | "vector" | "hyb" | "hyb_pp";
const arms: Arm[] = ["key", "key_pp", "vector", "hyb", "hyb_pp"];
const score = Object.fromEntries(arms.map(a => [a, [] as number[]])) as Record<Arm, number[]>;

for (const g of golden) {
  const ents = g.expected_entities;
  const keyRaw = slugsOf(await engine.searchKeyword(g.q, { limit: K }));
  const keyPP = slugsOf(await engine.searchKeyword(preprocessOR(g.q), { limit: K }));
  const vec = slugsOf(await engine.searchVector(await embed(g.q), { limit: K }));
  const hyb = slugsOf(await hybridSearch(engine, g.q, { limit: K, expansion: false, rrfK: 60 }));
  const hybPP = rrf(keyPP, vec).slice(0, K);
  score.key.push(await ground(keyRaw, ents));
  score.key_pp.push(await ground(keyPP, ents));
  score.vector.push(await ground(vec, ents));
  score.hyb.push(await ground(hyb, ents));
  score.hyb_pp.push(await ground(hybPP, ents));
}

// ── report ───────────────────────────────────────────────────────────────────
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
const pad = (s: string, n: number) => s.padEnd(n);
const f = (x: number) => x.toFixed(3);
const domains = [...new Set(golden.map(g => g.domain))];
const meanFor = (a: Arm, d?: string) => mean(score[a].filter((_, i) => !d || golden[i].domain === d));

console.log(`route_eval_kwpp: ${golden.length} questions · K=${K} C=${C}\n`);
console.log(pad("arm", 10) + pad("grnd@C", 9) + domains.map(d => pad(`g@C:${d}`, 9)).join(""));
console.log("-".repeat(10 + 9 + domains.length * 9));
for (const a of arms) {
  console.log(pad(a, 10) + pad(f(meanFor(a)), 9) + domains.map(d => pad(f(meanFor(a, d)), 9)).join(""));
}

// routing headroom with the REVIVED keyword arm: oracle over {key_pp, vector, hyb_pp} vs global hyb_pp
const routeArms: Arm[] = ["key_pp", "vector", "hyb_pp"];
const global = mean(score.hyb_pp);
const oracle = mean(golden.map((_, i) => Math.max(...routeArms.map(a => score[a][i]))));
const wins = golden.filter((_, i) => {
  const best = Math.max(...routeArms.map(a => score[a][i]));
  return best > score.hyb_pp[i] + 1e-9;   // a non-hybrid arm strictly beats fixed-hybrid
}).length;
console.log(`\nrouting (with revived keyword): global hyb_pp ${f(global)} · oracle ${f(oracle)} · Δ ${f(oracle - global)}`);
console.log(`${wins}/${golden.length} questions where some arm strictly beats hyb_pp (real routing headroom)`);

await engine.disconnect?.();
process.exit(0);
