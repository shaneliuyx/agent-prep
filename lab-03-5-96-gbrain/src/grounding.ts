/**
 * grounding.ts — pure scoring primitives for the policy eval (no I/O, unit-testable).
 *
 * The policy's objective is ANSWER quality, not raw retrieval. The generator reads
 * only the top-C chunks, in order — so the metric scores the prompt it actually sees:
 * position-weighted best coverage over the context budget. An RRF reorder that demotes
 * the answer chunk, or pushes it past C, lowers the score; the old rank-blind
 * max-over-top-K could not see either.
 */

/** Substring coverage of a section's (lowercased) text against expected entities, ∈ [0,1]. */
export const coverage = (lowercasedText: string, ents: string[]): number =>
  ents.length ? ents.filter(e => lowercasedText.includes(e.toLowerCase())).length / ents.length : 0;

/** Position discount: rank-0 = 1.0, decaying with depth. disc(0)=1, disc(1)=0.63, disc(2)=0.5. */
export const disc = (rank0: number): number => 1 / Math.log2(rank0 + 2);

export interface BudgetScore {
  /** max_i coverage_i · disc(i) over the top-C window — drives the policy. */
  gDisc: number;
  /** max_i coverage_i over the top-C window — raw best coverage (feeds answerable@C). */
  gFull: number;
}

/**
 * Score one question from the per-rank coverage of its retrieved hits.
 * `coverages` is coverage at each retrieved rank (length ≤ K); only the first `c`
 * (the context budget) are read — anything past C contributes nothing, modelling
 * "the answer fell out of the prompt".
 */
export function budgetScore(coverages: number[], c: number): BudgetScore {
  let gDisc = 0;
  let gFull = 0;
  for (let r = 0; r < Math.min(coverages.length, c); r++) {
    gDisc = Math.max(gDisc, coverages[r] * disc(r));
    gFull = Math.max(gFull, coverages[r]);
  }
  return { gDisc, gFull };
}
