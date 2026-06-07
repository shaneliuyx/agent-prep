/**
 * Verifies the budget-aware discounted grounding metric behaves where the old
 * rank-blind max-over-top-K was blind: RRF reorders and budget overflow.
 *
 * Run: bun test src/grounding.test.ts
 */
import { expect, test } from "bun:test";

import { budgetScore, coverage, disc } from "./grounding.ts";

const C = 3;
const oldRankBlind = (covs: number[], k: number) => Math.max(0, ...covs.slice(0, k)); // the metric we replaced

test("disc encodes primacy: 1.0, 0.63, 0.5 at ranks 0,1,2", () => {
  expect(disc(0)).toBe(1);
  expect(disc(1)).toBeCloseTo(0.6309, 3);
  expect(disc(2)).toBe(0.5);
});

test("coverage is fraction of expected entities present (case-insensitive)", () => {
  expect(coverage("sam okafor anchors northstar", ["Sam Okafor", "Northstar"])).toBe(1);
  expect(coverage("sam okafor only", ["Sam Okafor", "Northstar"])).toBe(0.5);
  expect(coverage("nothing relevant", ["Sam Okafor"])).toBe(0);
});

test("answer at rank 0 scores a perfect 1.0", () => {
  expect(budgetScore([1, 0, 0], C).gDisc).toBe(1);
});

test("RRF demotion (distractor at rank 0, answer at rank 1) is penalised — old metric was blind", () => {
  const demoted = [0, 1, 0]; // answer pushed to rank 1 by a fused distractor
  const top = [1, 0, 0];
  // The failure the user named: old rank-blind max says both are identical...
  expect(oldRankBlind(demoted, 5)).toBe(oldRankBlind(top, 5)); // both 1.0 — blind
  // ...the new metric sees the demotion.
  expect(budgetScore(demoted, C).gDisc).toBeCloseTo(0.6309, 3);
  expect(budgetScore(demoted, C).gDisc).toBeLessThan(budgetScore(top, C).gDisc);
});

test("answer pushed PAST the context budget scores 0 — retrieved (in K) but out of the prompt", () => {
  const pastBudget = [0, 0, 0, 1]; // answer at rank 3, budget C=3 reads ranks 0..2
  expect(oldRankBlind(pastBudget, 5)).toBe(1); // old metric: still "in top-K", looks fine
  expect(budgetScore(pastBudget, C).gDisc).toBe(0); // new metric: it never reaches the generator
  expect(budgetScore(pastBudget, C).gFull).toBe(0);
});

test("partial coverage is preserved, not rounded up (answer-quality magnitude kept)", () => {
  expect(budgetScore([0.5, 0, 0], C).gDisc).toBe(0.5); // half the entities, at rank 0
  expect(budgetScore([0.5, 1, 0], C).gDisc).toBeCloseTo(0.6309, 3); // full@1 (0.63) beats half@0 (0.5)
});

test("gFull (answerable source) ignores position; gDisc (policy) respects it", () => {
  const s = budgetScore([0, 1, 0], C);
  expect(s.gFull).toBe(1); // a fully-covering section exists in budget → answerable
  expect(s.gDisc).toBeCloseTo(0.6309, 3); // but it's demoted → graded down for the policy
});
