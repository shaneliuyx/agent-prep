/**
 * assembly.ts — entity-aware context assembly via 1-hop GRAPH EXPANSION.
 *
 * The gap it closes (measured): a 2-hop question needs TWO entity pages in the same C-chunk
 * reader budget. Rank-only assembly (`ranked.slice(0, C)`) can drop the second one when retrieval
 * ranks it at C+1 — e.g. "which venture firm is the angel investor associated with" retrieved both
 * `ridgeline-capital` (#3, in budget) and `people/dev-patel` (#4, OUT), so the generator saw the
 * firm but not the investor and failed (route_principle_ab / answer_principle_ab).
 *
 * GBrain is a knowledge graph, so the fix is one edge traversal: `dev-patel` is a 1-hop neighbor
 * of the in-budget `ridgeline-capital` (`invested_in` / `founded` edges, both directions). Pull it
 * into the context. We pull a neighbor only when it is (a) graph-connected to an in-budget page AND
 * (b) itself retrieved-but-demoted (in the top-K pool, ranked past C) — high precision: the page was
 * relevant enough to retrieve, and the graph says it completes an in-budget entity. `maxPull` caps
 * the spend so single-hop queries stay at C.
 */

interface GraphEngine {
  getLinks(slug: string, opts?: { sourceId?: string }): Promise<{ to_slug: string }[]>;
  getBacklinks(slug: string, opts?: { sourceId?: string }): Promise<{ from_slug: string }[]>;
}

/** 1-hop neighbors of `slug`: union of outgoing (`getLinks.to`) and incoming (`getBacklinks.from`). */
export async function neighborSlugs(engine: GraphEngine, slug: string): Promise<string[]> {
  const [out, back] = await Promise.all([engine.getLinks(slug), engine.getBacklinks(slug)]);
  const s = new Set<string>();
  for (const l of out) s.add(l.to_slug);
  for (const l of back) s.add(l.from_slug);
  s.delete(slug);
  return [...s];
}

export interface ExpandOpts {
  C: number;                 // reader budget (injected-chunk count)
  maxPull?: number;          // max neighbors to pull beyond C (default 2)
  poolOnly?: boolean;        // pull only retrieved-but-demoted neighbors (default true, high precision)
}

export interface ExpandResult {
  slugs: string[];           // assembled context: top-C + pulled neighbors (length C..C+maxPull)
  pulled: string[];          // the graph-pulled neighbor slugs (empty = no expansion happened)
}

/**
 * Graph-expanded assembly. Takes the full ranked candidate list (top-K from one search arm),
 * keeps the top-C, then appends up to `maxPull` 1-hop neighbors of the in-budget pages.
 *
 * poolOnly=true  → only neighbors that were themselves retrieved (in `ranked`) but demoted past C.
 * poolOnly=false → any 1-hop neighbor of an in-budget page (pure graph reach; lower precision).
 */
export async function graphExpand(
  engine: GraphEngine,
  ranked: string[],
  { C, maxPull = 2, poolOnly = true }: ExpandOpts,
): Promise<ExpandResult> {
  const topC = ranked.slice(0, C);
  const inBudget = new Set(topC);

  // 1-hop neighbor set of the in-budget pages
  const nbr = new Set<string>();
  for (const ns of await Promise.all(topC.map(p => neighborSlugs(engine, p)))) {
    for (const n of ns) nbr.add(n);
  }

  // candidates to pull, in priority order
  const candidates = poolOnly
    ? ranked.slice(C).filter(s => nbr.has(s))         // retrieved-but-demoted AND graph-connected
    : [...nbr].filter(s => !inBudget.has(s));         // any graph neighbor (incl. never-retrieved)

  const pulled: string[] = [];
  for (const s of candidates) {
    if (pulled.length >= maxPull) break;
    if (!inBudget.has(s) && !pulled.includes(s)) pulled.push(s);
  }
  return { slugs: [...topC, ...pulled], pulled };
}
