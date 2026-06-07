/**
 * query_routing.ts — shared per-query TYPE classifier + keyword preprocessing.
 *
 * ONE canonical definition, imported by BOTH the production actuator (query_policy.ts, behind
 * the QUERY_ROUTER switch) and the A/B that validated it (route_principle_ab.ts). If the two
 * diverged, the measured win would not transfer — production would route by an unvalidated
 * classifier. Co-locating them guarantees production == tested.
 *
 * The principle this implements: a single global policy compromises queries whose best arm
 * isn't the global winner. Route by query TYPE instead —
 *   exact-token probe  → keyword, OR-preprocessed (so GBrain's conjunctive FTS doesn't AND a
 *                         verbose query to zero; route_eval_kwpp.ts: g@C:kw 0.500 → 1.000)
 *   semantic paraphrase → vector
 *   natural question    → hybrid
 * Measured on the balanced 24-Q set (route_principle_ab.ts): classifier 24/24, router +0.039
 * grounding and +0.083 answer-quality (pinned Opus) over global hybrid.
 */
export type QueryType = "kw" | "vec" | "mixed";
export type Strategy = "keyword" | "vector" | "hybrid";

// Noise words dropped before OR-joining the keyword query (domain + question + function words).
export const STOP = new Set([
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

// Function words used to detect a prose paraphrase (high ratio → semantic, not exact-token).
export const FUNCTION = new Set([
  "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was", "were",
  "that", "this", "with", "from", "by", "as", "at", "be", "it", "they", "its", "his", "her",
  "their", "how", "who", "which", "where", "what", "when", "could", "following",
]);

/**
 * Drop stop/question words, then OR-join the salient tokens. OR switches GBrain's
 * `websearch_to_tsquery` from AND (every token must co-occur in one chunk → verbose queries
 * score 0) to best-match (a missing token lowers rank instead of zeroing the match).
 */
export function preprocessOR(q: string): string {
  const toks = (q.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter(t => t.length > 1 && !STOP.has(t));
  const uniq = [...new Set(toks)];
  return uniq.length ? uniq.join(" OR ") : q;   // fall back to raw if we stripped everything
}

/**
 * Deterministic, zero-LLM query-type classifier. A natural question (ends '?') is mixed;
 * a high function-word ratio / leading lowercase connective marks a semantic paraphrase;
 * otherwise distinctive exact tokens (Item N, ALL-CAPS, capitalised entity runs, digit+%) or a
 * capitalised lead marks an exact-token keyword probe.
 */
export function classifyType(q: string): QueryType {
  const raw = q.trim();
  if (/\?\s*$/.test(raw)) return "mixed";
  const words = raw.split(/\s+/);
  const distinctive = raw.match(
    /\bItem\s+\d+[A-Z]?\b|\b[A-Z]{2,}\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b|\b\d+\s*(?:percent|%)\b/g,
  ) ?? [];
  const fnRatio = words.filter(w => FUNCTION.has(w.toLowerCase())).length / words.length;
  const leadsLowerFn = FUNCTION.has(words[0].toLowerCase()) && /^[a-z]/.test(words[0]);
  if (leadsLowerFn || fnRatio >= 0.4) return "vec";
  if (distinctive.length >= 1 || /^[A-Z]/.test(words[0])) return "kw";
  return "vec";
}

// Arm each query type routes to. `kw` → keyword arm, fed the OR-preprocessed query.
export const TYPE_TO_STRATEGY: Record<QueryType, Strategy> = {
  kw: "keyword",
  vec: "vector",
  mixed: "hybrid",
};
