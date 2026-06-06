/**
 * Offline unit tests for auto_eval.ts's pure auto-labeling logic.
 * No DB / oMLX needed — auto_eval.ts guards its engine run behind
 * `import.meta.main`, so importing it here triggers no connection.
 *
 * Run: bun test tests/auto_eval.test.ts
 */
import { describe, expect, test } from "bun:test";
import {
  buildQrels,
  computeQ,
  seededShuffle,
  semanticQuery,
  type PageLite,
} from "../src/auto_eval.ts";

describe("computeQ — Q scales with N, clamped", () => {
  test("ratio applies in the normal range", () => {
    expect(computeQ(100, 0.3, 5, 50)).toBe(30);
    expect(computeQ(19, 0.3, 5, 50)).toBe(6); // ceil(5.7)
  });
  test("floor at Q_MIN for tiny corpora", () => {
    expect(computeQ(10, 0.3, 5, 50)).toBe(5); // ceil(3)=3 → floored to 5
  });
  test("never exceeds N", () => {
    expect(computeQ(3, 0.3, 5, 50)).toBe(3); // Q_MIN 5 capped to N=3
  });
  test("ceiling at Q_MAX for huge corpora", () => {
    expect(computeQ(100_000, 0.3, 5, 50)).toBe(50);
  });
});

describe("seededShuffle — deterministic permutation", () => {
  const items = Array.from({ length: 20 }, (_, i) => i);
  test("same seed → identical order (reproducible sampling)", () => {
    expect(seededShuffle(items, 42)).toEqual(seededShuffle(items, 42));
  });
  test("different seed → different order", () => {
    expect(seededShuffle(items, 42)).not.toEqual(seededShuffle(items, 7));
  });
  test("is a permutation (same multiset, no loss/dup)", () => {
    const out = seededShuffle(items, 42);
    expect(out.length).toBe(items.length);
    expect([...out].sort((a, b) => a - b)).toEqual(items);
  });
  test("does not mutate the input", () => {
    const copy = [...items];
    seededShuffle(items, 99);
    expect(items).toEqual(copy);
  });
});

describe("semanticQuery — title tokens masked so keyword can't trivially win", () => {
  const page: PageLite = {
    slug: "people/alice-chen",
    title: "Alice Chen",
    compiled_truth:
      "# Alice Chen\n\nFounder and CEO of [[companies/acme-ai]], an inference " +
      "optimization startup; previously led serving infrastructure.\n\n---\n## Timeline\n- 2026-05-12 — dinner",
  };
  const q = semanticQuery(page) ?? "";

  test("excludes every title token", () => {
    expect(q).not.toContain("alice");
    expect(q).not.toContain("chen");
  });
  test("keeps semantic content words from the body", () => {
    expect(q).toContain("founder");
    expect(q).toContain("inference");
  });
  test("de-links wikilinks into plain words (no [[ ]] markup)", () => {
    expect(q).not.toContain("[[");
    expect(q).toContain("acme"); // from [[companies/acme-ai]] tail
  });
  test("returns null when too few content words survive", () => {
    expect(semanticQuery({ slug: "x/y", title: "Foo Bar", compiled_truth: "# Foo Bar\n\nFoo bar." }))
      .toBeNull();
  });
});

describe("buildQrels — known-item labels (gold = the page's own slug)", () => {
  const pages: PageLite[] = [
    { slug: "people/lin-zhao", title: "Lin Zhao",
      compiled_truth: "# Lin Zhao\n\nRuns serving infrastructure at [[companies/acme-ai]] for model inference." },
    { slug: "companies/helix-bio", title: "Helix Bio",
      compiled_truth: "# Helix Bio\n\nProtein design foundation models for early drug discovery." },
  ];
  const qrels = buildQrels(pages);

  test("every qrel's gold is the source page slug", () => {
    for (const q of qrels) expect(pages.some(p => p.slug === q.relevant[0])).toBe(true);
  });
  test("emits an exact (title) probe per page", () => {
    const exact = qrels.filter(q => q.kind === "exact");
    expect(exact.map(q => q.query)).toEqual(["Lin Zhao", "Helix Bio"]);
  });
  test("emits semantic probes that drop the title surface form", () => {
    const sem = qrels.filter(q => q.kind === "semantic");
    expect(sem.length).toBeGreaterThan(0);
    for (const q of sem) {
      const gold = pages.find(p => p.slug === q.relevant[0])!;
      for (const tok of gold.title.toLowerCase().split(/\s+/)) {
        expect(q.query.toLowerCase()).not.toContain(tok);
      }
    }
  });
});
