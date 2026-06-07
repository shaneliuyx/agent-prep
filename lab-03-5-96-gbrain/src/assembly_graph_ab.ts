/**
 * assembly_graph_ab.ts — does GRAPH-EXPANDED assembly get more answer entities into the C-chunk
 * budget than plain top-C? Measured on the balanced 24-Q set.
 *
 * For each question: route it to its type's arm (same classifier as production), take that arm's
 * top-K ranked candidates, then assemble two contexts —
 *   plain     = ranked.slice(0, C)                 (rank-only, the current behaviour)
 *   expanded  = graphExpand(ranked, C, maxPull)    (top-C + 1-hop retrieved-but-demoted neighbors)
 * — and score entity COVERAGE of each (fraction of the question's expected_entities present in the
 * assembled pages' text). Reports where expansion pulls a neighbor and whether coverage rises,
 * plus the context-size cost.
 *
 * Run: GOLDEN_EVAL=data/golden_balanced.json bun src/assembly_graph_ab.ts
 */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

import { graphExpand } from "./assembly.ts";
import { coverage } from "./grounding.ts";
import { classifyType, preprocessOR, TYPE_TO_STRATEGY } from "./query_routing.ts";

const GB = process.env.GBRAIN_SRC ?? `${import.meta.dir}/../../gbrain/src`;
const GOLDEN = process.env.GOLDEN_EVAL ?? `${import.meta.dir}/../data/golden_balanced.json`;
const K = Number(process.env.POLICY_K ?? "5");
const C = Number(process.env.POLICY_C ?? "3");
const MAX_PULL = Number(process.env.MAX_PULL ?? "2");

interface GoldenQ { q: string; expected_entities: string[]; domain: string }
const golden: GoldenQ[] = JSON.parse(readFileSync(GOLDEN, "utf-8")).questions;

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
async function setCoverage(slugs: string[], ents: string[]): Promise<number> {
  const text = (await Promise.all(slugs.map(sectionText))).join("\n");
  return coverage(text, ents);
}
const slugsOf = (rs: { slug: string }[]) => rs.map(r => r.slug);

// route a query to its type's arm and return that arm's ranked candidates (top-K)
async function rankedFor(q: string): Promise<string[]> {
  const strat = TYPE_TO_STRATEGY[classifyType(q)];
  if (strat === "keyword") return slugsOf(await engine.searchKeyword(preprocessOR(q), { limit: K }));
  if (strat === "vector") return slugsOf(await engine.searchVector(await embed(q), { limit: K }));
  return slugsOf(await hybridSearch(engine, q, { limit: K, expansion: false, rrfK: 60 }));
}

const SLUG_DUMP = `${import.meta.dir}/../results/assembly_slugs.json`;
interface Row {
  q: string; type: string; expected_entities: string[];
  plain_slugs: string[]; expanded_slugs: string[];
  covPlain: number; covExp: number; pulled: string[]; size: number;
}
const rows: Row[] = [];
for (const g of golden) {
  const ranked = await rankedFor(g.q);
  const plain = ranked.slice(0, C);
  const { slugs: expanded, pulled } = await graphExpand(engine, ranked, { C, maxPull: MAX_PULL });
  rows.push({
    q: g.q, type: classifyType(g.q), expected_entities: g.expected_entities,
    plain_slugs: plain, expanded_slugs: expanded,
    covPlain: await setCoverage(plain, g.expected_entities),
    covExp: await setCoverage(expanded, g.expected_entities),
    pulled, size: expanded.length,
  });
}
mkdirSync(dirname(SLUG_DUMP), { recursive: true });
writeFileSync(SLUG_DUMP, JSON.stringify(rows.map(r => ({
  q: r.q, type: r.type, expected_entities: r.expected_entities,
  plain_slugs: r.plain_slugs, expanded_slugs: r.expanded_slugs, pulled: r.pulled,
})), null, 2) + "\n");

// ── report ───────────────────────────────────────────────────────────────────
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
const f = (x: number) => x.toFixed(3);
console.log(`assembly_graph_ab: ${rows.length} questions · K=${K} C=${C} maxPull=${MAX_PULL}\n`);

const expandedRows = rows.filter(r => r.pulled.length > 0);
console.log(`graph expansion fired on ${expandedRows.length}/${rows.length} questions:`);
for (const r of expandedRows) {
  const tag = r.covExp > r.covPlain + 1e-9 ? "↑ COVERAGE" : r.covExp < r.covPlain - 1e-9 ? "↓" : "= (already covered)";
  console.log(`  ${tag}  cov ${f(r.covPlain)}→${f(r.covExp)}  +${r.pulled.length}p (size ${r.size})  pulled[${r.pulled.join(", ")}]  "${r.q.slice(0, 44)}"`);
}

const improved = rows.filter(r => r.covExp > r.covPlain + 1e-9).length;
const hurt = rows.filter(r => r.covExp < r.covPlain - 1e-9).length;
console.log(`\nentity coverage:  plain ${f(mean(rows.map(r => r.covPlain)))}  →  graph-expanded ${f(mean(rows.map(r => r.covExp)))}`);
console.log(`questions improved: ${improved}  ·  hurt: ${hurt}  ·  unchanged: ${rows.length - improved - hurt}`);
console.log(`avg context size:  plain ${C}  →  expanded ${f(mean(rows.map(r => r.size)))}  (pulls only where an edge says a retrieved page completes an in-budget entity)`);

await engine.disconnect?.();
process.exit(0);
