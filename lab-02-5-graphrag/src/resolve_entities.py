"""Entity resolution: merge similar `Entity` nodes via BGE-M3 cosine similarity.

Surfaces v11's identified multi_hop ceiling driver — the graph contains
duplicate entities like `Reid Hoffman` + `Reid Garrett Hoffman` that split
edges across separate nodes and break multi-hop chains (Bad-Case Journal
Entry 10). This script runs a one-shot resolution pass:

  1. Discover all `Entity` names not already merged
  2. Embed every name via BGE-M3 (one forward pass, ~50 MB matrix for ~13K entities)
  3. Chunked pairwise cosine similarity (no full N² matrix materialization)
  4. Union-find clustering above threshold (default 0.92)
  5. Pick canonical per cluster (max edges, then longest name, alphabetic tie)
  6. Bulk-merge aliases via apoc.refactor.mergeNodes; audit via
     `merged_aliases` property listing the original name strings

Reversible: every merged-into node carries `merged_aliases: [<original_name>, ...]`.
To audit or roll back, query `MATCH (e:Entity) WHERE e.merged_aliases IS NOT NULL`.

Run from lab-02-5-graphrag:
  ./.venv/bin/python src/resolve_entities.py --dry-run
  ./.venv/bin/python src/resolve_entities.py [--threshold 0.92] [--max-cluster-size 10]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Use shared library for BGE-M3 embedding
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from rag_hybrid import BGE_M3, EncoderConfig, HybridEncoder  # noqa: E402

load_dotenv()
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)


def discover_entities(session, max_count: int = 50000) -> list[dict]:
    """Return Entity names + edge counts. Edge count is the canonical-pick tiebreaker
    — entities with more edges are likelier to be the canonical name (canonical
    Wikipedia entities are central to the corpus; aliases are mentioned in passing)."""
    rows = list(session.run("""
        MATCH (e:Entity)
        WHERE e.merged_aliases IS NULL
        OPTIONAL MATCH (e)-[r]-()
        RETURN e.name AS name, count(r) AS edge_count
        ORDER BY edge_count DESC
        LIMIT $limit
    """, limit=max_count))
    return [{"name": r["name"], "edges": r["edge_count"]} for r in rows]


def find_clusters_chunked(
    embeddings: np.ndarray, threshold: float, chunk_size: int = 100
) -> list[set[int]]:
    """Union-find clustering using chunked pairwise cosine similarity.

    Avoids materializing the full N² similarity matrix — for 13K entities with
    1024-dim embeddings, full matrix is ~656 MB. Chunked: process 100 entities
    at a time, compare to all entities, ~5 MB peak RSS for the slice.

    BGE-M3 dense vectors are unit-normalized (cosine = dot product directly).
    """
    n = len(embeddings)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    pairs_added = 0
    for chunk_start in range(0, n, chunk_size):
        chunk = embeddings[chunk_start : chunk_start + chunk_size]
        sim = chunk @ embeddings.T  # (chunk_size, n)
        # Mask self-comparisons (will be 1.0 on the diagonal)
        for i in range(len(chunk)):
            global_i = chunk_start + i
            sim[i, global_i] = 0.0
        # Find above-threshold pairs
        rows, cols = np.where(sim >= threshold)
        for i, j in zip(rows, cols):
            global_i = chunk_start + int(i)
            global_j = int(j)
            if global_i < global_j:  # dedupe symmetric pairs
                union(global_i, global_j)
                pairs_added += 1

    # Group indices by root
    clusters: dict[int, set[int]] = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, set()).add(i)
    return [c for c in clusters.values() if len(c) > 1]


def pick_canonical(cluster: set[int], names: list[str], edge_counts: list[int]) -> int:
    """Heuristic: max edges (most-cited entity is the canonical one),
    tiebreak on longest name (longer = more specific, e.g. "Reid Garrett Hoffman"
    over "Reid Hoffman"), final tiebreak alphabetic."""
    return max(cluster, key=lambda i: (edge_counts[i], len(names[i]), names[i]))


def merge_aliases(
    session, canonical_name: str, alias_names: list[str]
) -> int:
    """Merge alias entities into canonical via apoc.refactor.mergeNodes.
    Returns count of successful merges. Each merged-into node accumulates
    `merged_aliases: [<original_alias_name>, ...]` for audit/reverse."""
    n = 0
    for alias in alias_names:
        if alias == canonical_name:
            continue
        try:
            session.run("""
                MATCH (canonical:Entity {name: $can}), (alias:Entity {name: $ali})
                CALL apoc.refactor.mergeNodes([canonical, alias], {properties: 'discard', mergeRels: true})
                YIELD node
                SET node.merged_aliases = coalesce(node.merged_aliases, []) + [$ali]
                RETURN node
            """, can=canonical_name, ali=alias)
            n += 1
        except Exception as e:
            print(f"  [skip] merge {alias!r} -> {canonical_name!r}: {type(e).__name__}: {e}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="show clusters without writing")
    ap.add_argument("--threshold", type=float, default=0.92,
                    help="cosine similarity threshold for cluster membership (default 0.92)")
    ap.add_argument("--max-cluster-size", type=int, default=10,
                    help="skip clusters bigger than this (likely false positives, default 10)")
    args = ap.parse_args()

    print("=" * 76)
    print(f"resolve_entities — {'DRY RUN' if args.dry_run else 'APPLY MODE'}")
    print(f"threshold={args.threshold}  max_cluster_size={args.max_cluster_size}")
    print("=" * 76)

    with driver.session() as session:
        entities = discover_entities(session)
        print(f"\nDiscovered {len(entities)} entities (post-prior-merges)")

        names = [e["name"] for e in entities]
        edges = [e["edges"] for e in entities]

        # Embed
        print(f"\nEmbedding via BGE-M3 (this is heavy — ~30s on M5 Pro for 13K entities)...")
        t0 = time.time()
        encoder = HybridEncoder(EncoderConfig(spec=BGE_M3, default_batch_size=128))
        out = encoder.encode(names, with_sparse=False)
        embeddings = out["dense_vecs"]
        # BGE-M3 dense is already unit-normalized. Sanity-check first row norm.
        norm0 = float(np.linalg.norm(embeddings[0]))
        if abs(norm0 - 1.0) > 1e-3:
            print(f"  [WARN] embedding norm deviation: {norm0:.4f} (expected ~1.0); cosine assumption may be off")
        print(f"Embedded {len(names)} entities in {time.time()-t0:.1f}s")

        # Cluster
        t0 = time.time()
        clusters = find_clusters_chunked(embeddings, args.threshold)
        clusters_kept = [c for c in clusters if len(c) <= args.max_cluster_size]
        clusters_dropped = [c for c in clusters if len(c) > args.max_cluster_size]
        print(f"\nFound {len(clusters)} clusters in {time.time()-t0:.1f}s")
        print(f"  kept (size <= {args.max_cluster_size}): {len(clusters_kept)}")
        print(f"  dropped (size > {args.max_cluster_size}): {len(clusters_dropped)}")
        if clusters_dropped:
            print("  (dropped clusters are likely false-positive hub-entity matches; "
                  "raise --max-cluster-size or --threshold to recover)")

        # Sort clusters by canonical entity edge count (most-impactful first)
        cluster_summaries = []
        for cluster in clusters_kept:
            canonical_idx = pick_canonical(cluster, names, edges)
            cluster_summaries.append({
                "indices": sorted(cluster),
                "canonical_idx": canonical_idx,
                "canonical_edges": edges[canonical_idx],
            })
        cluster_summaries.sort(key=lambda c: -c["canonical_edges"])

        total_aliases = sum(len(c["indices"]) - 1 for c in cluster_summaries)
        print(f"\n{len(cluster_summaries)} merge-clusters covering {total_aliases} alias entities to merge\n")

        for cs in cluster_summaries:
            ci = cs["canonical_idx"]
            print(f"  CANONICAL: {names[ci]!r} ({edges[ci]} edges)")
            for idx in cs["indices"]:
                if idx == ci:
                    continue
                sim = float(embeddings[ci] @ embeddings[idx])
                print(f"    ← {names[idx]!r} (sim={sim:.3f}, {edges[idx]} edges)")
            print()

        if args.dry_run:
            print(f"DRY RUN — no writes. Re-run without --dry-run to apply.")
            return 0

        print("Applying merges...")
        t0 = time.time()
        total_merged = 0
        for cs in cluster_summaries:
            canonical = names[cs["canonical_idx"]]
            aliases = [names[idx] for idx in cs["indices"] if idx != cs["canonical_idx"]]
            n = merge_aliases(session, canonical, aliases)
            total_merged += n
        print(f"\nDone: {total_merged} aliases merged in {time.time()-t0:.1f}s")

    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
