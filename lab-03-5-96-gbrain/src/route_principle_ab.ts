/**
 * route_principle_ab.ts — honor the per-query principle in the LIVE path, three ways, A/B'd.
 *
 * The design intent of per-query routing: pick the arm that fits the QUERY TYPE, because a
 * single global policy compromises queries whose best arm isn't the global winner. The earlier
 * rejection (route_eval.ts) was unfair — its router targeted the RAW keyword arm, which GBrain's
 * conjunctive FTS had crippled (route_eval_kwpp.ts: raw key g@C:kw 0.500 vs preprocessed 1.000).
 * Route to the *preprocessed* keyword arm and the routing decision should finally pay on kw-class.
 *
 * Three contestants, measured with the same discounted grounding@C, on the balanced 24-Q set:
 *   baseline  — every query → global hybrid `hyb`         (the current shipped default)
 *   router    — classify query type, route kw→key_pp · vec→vector · mixed→hyb   (the principle)
 *   strong_g  — every query → strengthened global, no router:
 *                 g1 = RRF(key_pp, vector)        (hybrid w/ preprocessed keyword leg)
 *                 g2 = hybridSearch(preprocessOR(q))  (whole query preprocessed)
 *               we report whichever global variant scores higher.
 *
 * Decision: does either honor-the-principle arm beat baseline by a margin a real classifier can
 * keep? Router needs its classifier to actually detect type; strong_g needs no classifier at all.
 * Dumps per-arm top-C slugs → results/principle_slugs.json for the answer-quality A/B.
 *
 * Run: GOLDEN_EVAL=data/golden_balanced.json bun src/route_principle_ab.ts
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

import { budgetScore, coverage } from "./grounding.ts";

const GB = "/Users/yuxinliu/code/agent-prep/gbrain/src";
const GOLDEN = process.env.GOLDEN_EVAL ?? `${import.meta.dir}/../data/golden_balanced.json`;
const SLUG_DUMP = `${import.meta.dir}/../results/principle_slugs.json`;
const K = Number(process.env.POLICY_K ?? "5");
const C = Number(process.env.POLICY_C ?? "3");

interface GoldenQ { q: string; expected_entities: string[]; domain: string }
const golden: GoldenQ[] = JSON.parse(readFileSync(GOLDEN, "utf-8")).questions;

// ── keyword query preprocessing (same fix as route_eval_kwpp.ts) ─────────────
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
  "associated", "venture", "angel", "investor", "network",
]);

function preprocessOR(q: string): string {
  const toks = (q.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter(t => t.length > 1 && !STOP.has(t));
  const uniq = [...new Set(toks)];
  return uniq.length ? uniq.join(" OR ") : q;
}

// ── deterministic query-TYPE classifier (the shippable router's brain) ───────
// Signals, no LLM: a natural question (ends '?') is mixed; otherwise a low function-word
// ratio + distinctive exact tokens (Item N, ALL-CAPS, capitalised entity runs, digit+%) marks
// a keyword-focus probe, while a high function-word ratio / leading lowercase connective marks a
// semantic paraphrase. Returns the arm the type should route to.
type Type = "kw" | "vec" | "mixed";
const FUNCTION = new Set([
  "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was", "were",
  "that", "this", "with", "from", "by", "as", "at", "be", "it", "they", "its", "his", "her",
  "their", "how", "who", "which", "where", "what", "when", "could", "the", "following",
]);
function classify(q: string): Type {
  const raw = q.trim();
  if (/\?\s*$/.test(raw)) return "mixed";                       // natural question
  const words = raw.split(/\s+/);
  const distinctive = raw.match(
    /\bItem\s+\d+[A-Z]?\b|\b[A-Z]{2,}\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b|\b\d+\s*(?:percent|%)\b/g,
  ) ?? [];
  const fnRatio = words.filter(w => FUNCTION.has(w.toLowerCase())).length / words.length;
  const leadsLowerFn = FUNCTION.has(words[0].toLowerCase()) && /^[a-z]/.test(words[0]);
  if (leadsLowerFn || fnRatio >= 0.4) return "vec";            // prose paraphrase
  if (distinctive.length >= 1 || /^[A-Z]/.test(words[0])) return "kw"; // exact-token probe
  return "vec";
}

// ── RRF over two ranked slug lists ───────────────────────────────────────────
function rrf(a: string[], b: string[], k = 60): string[] {
  const score = new Map<string, number>();
  for (const list of [a, b]) {
    list.forEach((slug, i) => score.set(slug, (score.get(slug) ?? 0) + 1 / (k + i + 1)));
  }
  return [...score.entries()].sort((x, y) => y[1] - x[1]).map(([s]) => s);
}

// ── engine bootstrap (identical sequence to the other probes / the CLI) ──────
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

// ── per-question: compute every arm's top-C slugs, then the three policies ───
type Arm = "key_pp" | "vector" | "hyb" | "g1" | "g2";
interface Row {
  q: string; domain: string; type: Type;
  baseline: string[]; router: string[]; strong_g: string[];
  arm: Record<Arm, number>;       // grounding of each candidate arm (for transparency)
}
const rows: Row[] = [];
let gWhich: "g1" | "g2" = "g1";    // decided after the loop by mean grounding

const perArm = { key_pp: [] as number[], vector: [] as number[], hyb: [] as number[],
                 g1: [] as number[], g2: [] as number[] };
const raw: { q: string; domain: string; type: Type; slugs: Record<Arm, string[]>; ents: string[] }[] = [];

for (const g of golden) {
  const ents = g.expected_entities;
  const keyPP = slugsOf(await engine.searchKeyword(preprocessOR(g.q), { limit: K }));
  const vec = slugsOf(await engine.searchVector(await embed(g.q), { limit: K }));
  const hyb = slugsOf(await hybridSearch(engine, g.q, { limit: K, expansion: false, rrfK: 60 }));
  const g1 = rrf(keyPP, vec).slice(0, K);
  const g2 = slugsOf(await hybridSearch(engine, preprocessOR(g.q), { limit: K, expansion: false, rrfK: 60 }));
  const slugs: Record<Arm, string[]> = { key_pp: keyPP, vector: vec, hyb, g1, g2 };
  perArm.key_pp.push(await ground(keyPP, ents));
  perArm.vector.push(await ground(vec, ents));
  perArm.hyb.push(await ground(hyb, ents));
  perArm.g1.push(await ground(g1, ents));
  perArm.g2.push(await ground(g2, ents));
  raw.push({ q: g.q, domain: g.domain, type: classify(g.q), slugs, ents });
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
gWhich = mean(perArm.g1) >= mean(perArm.g2) ? "g1" : "g2";  // strengthened-global = better variant

// route each question by its detected type → the arm the principle assigns
const armForType: Record<Type, Arm> = { kw: "key_pp", vec: "vector", mixed: "hyb" };
for (let i = 0; i < raw.length; i++) {
  const r = raw[i];
  const routerArm = armForType[r.type];
  rows.push({
    q: r.q, domain: r.domain, type: r.type,
    baseline: r.slugs.hyb, router: r.slugs[routerArm], strong_g: r.slugs[gWhich],
    arm: { key_pp: perArm.key_pp[i], vector: perArm.vector[i], hyb: perArm.hyb[i],
           g1: perArm.g1[i], g2: perArm.g2[i] },
  });
}

// ── scoring helpers ──────────────────────────────────────────────────────────
const score = {
  baseline: rows.map((_, i) => perArm.hyb[i]),
  router:   rows.map((r, i) => perArm[armForType[r.type]][i]),
  strong_g: rows.map((_, i) => perArm[gWhich][i]),
};
const classes: Type[] = ["kw", "vec", "mixed"];
const meanFor = (xs: number[], cls?: Type) => mean(xs.filter((_, i) => !cls || rows[i].type === cls));

// ── report ───────────────────────────────────────────────────────────────────
const pad = (s: string, n: number) => s.padEnd(n);
const f = (x: number) => x.toFixed(3);
console.log(`route_principle_ab: ${rows.length} questions · K=${K} C=${C} · strengthened-global = ${gWhich}\n`);

// classifier confusion vs gold domain
const correct = rows.filter(r => r.type === r.domain).length;
console.log(`classifier vs gold domain: ${correct}/${rows.length} correct`);
console.log("  per-q:  " + rows.map(r => (r.type === r.domain ? r.type : `${r.domain}->${r.type}`)).join(" "));
console.log();

console.log(pad("policy", 12) + pad("grnd@C", 9) + classes.map(c => pad(`g@C:${c}`, 9)).join(""));
console.log("-".repeat(12 + 9 + classes.length * 9));
for (const [name, xs] of Object.entries(score)) {
  console.log(pad(name, 12) + pad(f(meanFor(xs)), 9) + classes.map(c => pad(f(meanFor(xs, c)), 9)).join(""));
}
const base = mean(score.baseline);
console.log(`\nΔ vs baseline(hyb ${f(base)}):  router ${f(mean(score.router) - base)}  ·  strong_g(${gWhich}) ${f(mean(score.strong_g) - base)}`);

// how often each honor-the-principle arm strictly beats baseline (where the principle pays)
const rWins = rows.filter((_, i) => score.router[i] > score.baseline[i] + 1e-9).length;
const rLoses = rows.filter((_, i) => score.router[i] < score.baseline[i] - 1e-9).length;
console.log(`router vs baseline per-question:  +${rWins} better  ·  -${rLoses} worse  ·  ${rows.length - rWins - rLoses} tie`);

// ── dump slugs for the answer-quality A/B ────────────────────────────────────
const dump = rows.map(r => ({
  q: r.q, domain: r.domain, type: r.type,
  expected_entities: golden.find(g => g.q === r.q)!.expected_entities,
  baseline_slugs: r.baseline, router_slugs: r.router, strong_g_slugs: r.strong_g,
  strong_g_variant: gWhich,
}));
mkdirSync(dirname(SLUG_DUMP), { recursive: true });
writeFileSync(SLUG_DUMP, JSON.stringify(dump, null, 2) + "\n");
console.log(`\nwrote per-arm slugs → ${SLUG_DUMP}`);

await engine.disconnect?.();
process.exit(0);
