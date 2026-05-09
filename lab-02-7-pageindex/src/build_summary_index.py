"""Build the Level-2 summary index from data/tree.json.

Pipeline (across multiple tasks — Task 4 is skeleton):
  1. Extract leaf summaries from primary tree                   <- Task 4
  2. Embed via BGE-M3 (or injected embedder)                    <- Task 7
  3. K-means cluster (auto-K via silhouette)                    <- Task 4
  4. LLM-generate cluster title + summary + tags per cluster    <- Task 6
  5. Atomic write to data/summary_index.json with tree_hash     <- Task 5
                       binding

Resume: per-cluster journaling to .partial; on restart skip completed
clusters. Idempotent given fixed random_state."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

_LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LAB_ROOT.parents[0] / "shared"))

from tree_index._hashing import tree_hash  # noqa: E402


def extract_summary_nodes(tree: dict) -> list[dict]:
    """Walk the tree, return all nodes with non-empty summary (root,
    internals, leaves alike). Used as input to clustering."""
    out: list[dict] = []

    def walk(node: dict) -> None:
        if node.get("summary"):
            out.append({
                "node_id": node["node_id"],
                "title": node.get("title", ""),
                "summary": node["summary"],
                "tags": node.get("tags", []),
                "start_page": node.get("start_page"),
                "end_page": node.get("end_page", node.get("start_page")),
            })
        for c in node.get("nodes", []):
            walk(c)

    walk(tree)
    return out


def kmeans_cluster(
    embeddings: "np.ndarray", k: int, random_state: int = 42,
) -> "np.ndarray":
    """Deterministic k-means with sklearn. Returns cluster labels.

    Raises ValueError if k > len(embeddings) — sklearn's own error
    is opaque in a multi-step pipeline."""
    if k > len(embeddings):
        raise ValueError(
            f"k={k} exceeds number of embeddings ({len(embeddings)}). "
            f"Reduce k or supply more leaf nodes."
        )
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    return km.fit_predict(embeddings)


def auto_k_silhouette(
    embeddings: "np.ndarray",
    k_range: tuple[int, int] = (5, 12),
    min_improvement: float = 0.01,
    random_state: int = 42,
) -> tuple[int, list[tuple[int, float]]]:
    """Pick `k` for K-means by maximizing cosine silhouette over `k_range`.

    Returns (chosen_k, [(k, score), ...]) — full curve for diagnostics.

    Tie-break: prefers smaller k (Occam). A larger k only wins if its
    silhouette beats the current best by at least `min_improvement`.

    k_range upper bound is silently truncated to `n - 1` (silhouette
    requires k < n_samples and k >= 2).

    Use case: replacement for hardcoded k=8 in build pipeline. Adds
    one-time ~5-10s build cost (8 K-means fits + 8 silhouettes on
    typical 45 × 1024 BGE-M3 embeddings); zero query-time cost.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(embeddings)
    lo, hi = max(2, k_range[0]), min(k_range[1], n - 1)
    if lo > hi:
        raise ValueError(
            f"k_range ({k_range}) empty after truncation to n-1={n-1}"
        )

    scores: list[tuple[int, float]] = []
    for k in range(lo, hi + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(embeddings)
        s = float(silhouette_score(embeddings, labels, metric="cosine"))
        scores.append((k, s))

    best_k, best_score = scores[0]
    for k, s in scores[1:]:
        if s > best_score + min_improvement:
            best_k, best_score = k, s
    return best_k, scores


def resolve_k(
    requested_k: int, auto_k: bool, embeddings: "np.ndarray",
) -> tuple[int, list[tuple[int, float]] | None]:
    """Resolve final k for the cluster build.

    auto_k=False: return requested_k as-is, no curve.
    auto_k=True : run silhouette over k in [2, requested_k]; return chosen
                  k + full curve for build_meta forensics.

    requested_k acts as the upper bound on the silhouette sweep when
    auto_k=True — prevents the auto path from exploring k larger than
    the user's stated max.
    """
    if not auto_k:
        return requested_k, None
    return auto_k_silhouette(embeddings, k_range=(2, requested_k))


def _partial_path(out_path: Path) -> Path:
    return out_path.parent / (out_path.name + ".partial")


def _write_json_atomic(dest: Path, payload: dict, sort_keys: bool = False) -> None:
    """Write JSON to dest atomically: temp file in same dir, fsync, rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=dest.name + ".", suffix=".tmp", dir=str(dest.parent),
    )
    try:
        try:
            f = os.fdopen(fd, "w", encoding="utf-8")
        except Exception:
            os.close(fd)
            raise
        with f:
            json.dump(payload, f, indent=2, sort_keys=sort_keys)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def write_atomic(out_path: Path, payload: dict) -> None:
    """Write JSON atomically: temp file in same dir, fsync, rename."""
    out_path = Path(out_path)
    _write_json_atomic(out_path, payload, sort_keys=True)
    # Clean up any stale partial via the centralized helper
    _partial_path(out_path).unlink(missing_ok=True)


def journal_partial(out_path: Path, clusters_so_far: list[dict]) -> None:
    """Per-cluster journal — atomic full-file rewrite of .partial."""
    payload = {"clusters_completed": clusters_so_far,
               "journal_ts": time.time()}
    _write_json_atomic(_partial_path(out_path), payload)


def load_partial(out_path: Path) -> list[dict]:
    """Return list of already-completed clusters from .partial, [] if none."""
    pp = _partial_path(out_path)
    if not pp.exists():
        return []
    try:
        data = json.loads(pp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        # Corrupt journal — log but don't crash. Build restarts from
        # scratch on next run.
        import logging
        logging.warning(
            f"load_partial: ignoring corrupt journal at {pp}: "
            f"{type(e).__name__}: {e}"
        )
        return []
    if not isinstance(data, dict):
        return []
    completed = data.get("clusters_completed", [])
    return completed if isinstance(completed, list) else []


def _llm_call_with_retry(
    client, model: str, messages: list, max_tokens: int,
    response_format: dict | None = None,
) -> str:
    """LLM call with backoff retry on transient errors + post-call cooldown.

    vMLX returns 503 'Metal GPU working set too full' under accumulated load.
    Strategy:
      - Pre-call: progressive backoff retry on transient errors (up to 8
        attempts: 15, 30, 60, 90, 120, 180, 240, 300 s).
      - Post-call: brief cooldown sleep (2s) so GPU can release before the
        next call. Trades ~5 min total build time for OOM elimination.

    Only retries on TRANSIENT signals ('503', 'GPU working set', 'Connection').
    Non-transient errors (e.g. RuntimeError('rate limit')) re-raise on first
    attempt so unit tests with mocked failures don't burn the full schedule.
    Returns string content. Raises if all retries exhaust.
    """
    last_err = None
    backoff_schedule = [15, 30, 60, 90, 120, 180, 240, 300]
    for attempt, sleep_s in enumerate(backoff_schedule):
        try:
            kwargs = dict(model=model, messages=messages,
                          temperature=0.0, max_tokens=max_tokens)
            if response_format:
                kwargs["response_format"] = response_format
            r = client.chat.completions.create(**kwargs)
            content = (r.choices[0].message.content or "").strip()
            time.sleep(2)  # cooldown: let GPU release before next call
            return content
        except Exception as e:                                # noqa: BLE001
            last_err = e
            msg = str(e)
            transient = ("503" in msg or "GPU working set" in msg
                         or "ConnectionError" in type(e).__name__
                         or "Connection" in msg)
            if not transient:
                raise
            print(f"  [summarize_cluster] transient {type(e).__name__} on "
                  f"attempt {attempt+1}/{len(backoff_schedule)}; sleep "
                  f"{sleep_s}s before retry. detail={msg[:120]}", flush=True)
            time.sleep(sleep_s)
    raise RuntimeError(f"_llm_call_with_retry exhausted "
                       f"{len(backoff_schedule)} attempts; last_err={last_err}")


_CLUSTER_SUMMARIZE_SYSTEM = """You receive 2-10 document section summaries
that share a thematic cluster. Generate a SINGLE cluster meta-summary that
preserves verbatim entities + quoted phrases from the inputs.

OUTPUT: strict JSON with exactly these keys:
  {
    "title":   "<3-8 word cluster theme — distinctive phrase from inputs>",
    "summary": "<100-180 word prose summary covering ALL member sections.
                 PRESERVE: every named entity verbatim (Coca-Cola, Itochu,
                 BNSF), every distinctive quoted phrase ('not-so-secret
                 weapon', 'patience pays'), every numeric fact with units
                 ($364.5 billion, 27.8%). Do NOT paraphrase distinctive
                 vocabulary>",
    "tags":    ["<15-30 lookup tokens — entities, aliases, numeric anchors,
                 quoted phrases>"]
  }

Output ONLY this JSON. No prose preamble."""


def summarize_cluster(client, model: str,
                      member_summaries: list[str]) -> dict:
    """One LLM call → cluster {title, summary, tags}.

    Defensive contract: returns empty-fields dict on ANY failure (LLM error,
    JSON parse fail, type drift). NEVER raises. NEVER retries — Task 7's
    orchestrator handles retry policy.

    Empty-fields signal: caller checks ``if not result["title"]:`` to detect
    failure. Tags list is element-coerced to strings.
    """
    user = "\n\n---\n\n".join(member_summaries)
    try:
        raw = _llm_call_with_retry(
            client, model,
            messages=[
                {"role": "system", "content": _CLUSTER_SUMMARIZE_SYSTEM},
                {"role": "user", "content": user},
            ],
            max_tokens=1500,   # bumped 500 → 1500 (2026-05-09): large
                              # clusters (5-10 members) overflowed at 500,
                              # producing truncated JSON (Unterminated string
                              # / missing comma). 5/8 clusters had empty
                              # title in initial v4 build. 1500 gives 3×
                              # headroom; cost is ~2× wall per cluster.
            response_format={"type": "json_object"},
        )
        parsed = json.loads(raw or "{}")
        if isinstance(parsed, list):
            # DWQ schema-disagreement pattern (Bad-Case Entry 16):
            # the model returns a JSON ARRAY of entity strings instead of
            # the {title, summary, tags} dict, even with response_format=
            # json_object. The content is good — preserve it as tags-only.
            # main() will synthesize title + summary from member nodes.
            tags = [t for t in parsed if isinstance(t, str)]
            print(f"  [summarize_cluster] WARNING: list-response (likely "
                  f"DWQ schema-disagreement); preserving {len(tags)} tags, "
                  f"title+summary deferred to main() salvage", flush=True)
            return {"title": "", "summary": "", "tags": tags}
        if not isinstance(parsed, dict):
            print("  [summarize_cluster] WARNING: non-dict, non-list JSON; "
                  "using empty fields", flush=True)
            return {"title": "", "summary": "", "tags": []}
        for k in ("title", "summary"):
            if not isinstance(parsed.get(k), str):
                parsed[k] = ""
        # Coerce tags element-by-element — model can return [1, null, "..."]
        # and downstream tag consumers (entity index, search) require strings.
        tags = parsed.get("tags")
        if not isinstance(tags, list):
            parsed["tags"] = []
        else:
            parsed["tags"] = [t for t in tags if isinstance(t, str)]
        if not parsed["title"]:
            print("  [summarize_cluster] WARNING: empty title from LLM",
                  flush=True)
        return parsed
    except Exception as e:                                    # noqa: BLE001
        print(f"  [summarize_cluster] WARNING: {type(e).__name__}: {e}; "
              f"using empty fields", flush=True)
        return {"title": "", "summary": "", "tags": []}


def _embed_summaries(summaries: list[str]) -> "np.ndarray":
    """BGE-M3 embedding — reuses lab-02-3-bge_m3_hnsw infrastructure.

    Lazy import: only loads the BGE-M3 model when build runs."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-m3", device="mps")
    return np.asarray(
        model.encode(summaries, batch_size=8, normalize_embeddings=True,
                     show_progress_bar=False),
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Level-2 summary index over data/tree.json"
    )
    parser.add_argument("--force", action="store_true",
                        help="Ignore .partial and rebuild from scratch")
    parser.add_argument("--k", type=int, default=8,
                        help="Number of clusters (default: 8). When --auto-k is "
                             "set, this acts as the upper bound for the "
                             "silhouette sweep.")
    parser.add_argument("--auto-k", action="store_true",
                        help="Pick k via cosine silhouette over [2, --k]; "
                             "logs full curve to build_meta.silhouette_scores")
    parser.add_argument("--check", action="store_true",
                        help="Verify tree_hash match without rebuilding; "
                             "exit 0 if fresh, 1 if stale")
    parser.add_argument("--output", type=str,
                        default=str(_LAB_ROOT / "data" / "summary_index.json"),
                        help="Output path")
    args = parser.parse_args()

    out_path = Path(args.output)
    tree_path = _LAB_ROOT / "data" / "tree.json"

    # --check mode
    if args.check:
        if not out_path.exists():
            print(f"NOT FRESH: {out_path} does not exist")
            sys.exit(1)
        try:
            existing = json.loads(out_path.read_text())
            stored = existing.get("build_meta", {}).get("tree_hash", "")
            actual = tree_hash(tree_path)
            if stored != actual:
                print(f"STALE: tree_hash mismatch ({stored[:12]} != {actual[:12]})")
                sys.exit(1)
            print(f"FRESH: tree_hash {actual[:12]} matches")
            sys.exit(0)
        except Exception as e:
            print(f"NOT FRESH: {e}")
            sys.exit(1)

    load_dotenv(_LAB_ROOT / ".env")
    client = OpenAI(base_url=os.getenv("OMLX_BASE_URL"),
                    api_key=os.getenv("OMLX_API_KEY"))
    model = os.getenv("MODEL_BUILD") or os.getenv("MODEL_SONNET") or ""
    if not model:
        print("FATAL: MODEL_BUILD / MODEL_SONNET unset", file=sys.stderr)
        sys.exit(2)

    # Skip-if-fresh
    if not args.force and out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            if existing.get("build_meta", {}).get("tree_hash", "") == tree_hash(tree_path):
                print(f"summary_index up to date — skip rebuild "
                      f"(tree_hash {tree_hash(tree_path)[:12]})")
                return
        except Exception:
            pass

    # Load tree + leaves
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    leaves = extract_summary_nodes(tree)
    print(f"[1/4] {len(leaves)} primary leaves loaded", flush=True)

    # Embed
    print(f"[2/4] Embedding leaf summaries via BGE-M3 ...", flush=True)
    embeddings = _embed_summaries([n["summary"] for n in leaves])

    # Cluster — k may be auto-selected by silhouette
    requested_k = min(args.k, len(leaves) - 1)
    k, silhouette_curve = resolve_k(
        requested_k=requested_k, auto_k=args.auto_k, embeddings=embeddings,
    )
    if silhouette_curve is not None:
        curve_str = ", ".join(f"k={kv}:{s:.3f}" for kv, s in silhouette_curve)
        print(f"[2.5/4] Auto-K silhouette curve: {curve_str}", flush=True)
        print(f"[2.5/4] Auto-K chose k={k}", flush=True)
    labels = kmeans_cluster(embeddings, k=k, random_state=42)
    print(f"[3/4] K-means k={k} → {len(set(labels))} clusters", flush=True)

    # Compute cluster centroids (used as cluster_embeddings)
    centroids = np.zeros((k, embeddings.shape[1]), dtype=np.float32)
    for i in range(k):
        mask = labels == i
        if mask.any():
            centroids[i] = embeddings[mask].mean(axis=0)

    # Resume from .partial if present (and not --force)
    completed = [] if args.force else load_partial(out_path)
    completed_ids = {c["cluster_id"] for c in completed}
    print(f"[4/4] Generating cluster summaries "
          f"(resume: {len(completed_ids)}/{k} already done)", flush=True)

    clusters: list[dict] = list(completed)
    for ci in range(k):
        cid = f"C{ci+1}"
        if cid in completed_ids:
            continue
        member_idx = np.where(labels == ci)[0]
        member_summaries = [leaves[j]["summary"] for j in member_idx]
        meta = summarize_cluster(client, model, member_summaries)
        # Salvage: if LLM returned tags-only (list-response from DWQ schema
        # disagreement), synthesize title + summary from member nodes.
        # Three-tier fallback for title:
        #   1. LLM-generated title (preferred)
        #   2. First 2 member titles concatenated (when only tags returned)
        #   3. Generic "Cluster CN" (when both LLM AND members provide nothing)
        member_titles = [leaves[j]["title"] for j in member_idx
                         if leaves[j].get("title")]
        if meta["title"]:
            title = meta["title"]
        elif meta["tags"] and member_titles:
            title = " + ".join(member_titles[:2])[:80]
        else:
            title = f"Cluster {cid}"

        # Same fallback ladder for summary: first sentence of each member,
        # joined. Caps at 400 chars to avoid over-bloating the index.
        if meta["summary"]:
            summary = meta["summary"]
        elif meta["tags"]:
            summary_parts = [
                leaves[j]["summary"].split(".")[0].strip()[:120]
                for j in member_idx[:5]
                if leaves[j].get("summary")
            ]
            summary = ". ".join(p for p in summary_parts if p)[:400]
        else:
            summary = ""

        cluster_obj = {
            "cluster_id": cid,
            "title": title,
            "summary": summary,
            "tags": meta["tags"],
            "member_node_ids": [leaves[j]["node_id"] for j in member_idx],
            "primary_pages": [
                [leaves[j]["start_page"], leaves[j]["end_page"]]
                for j in member_idx
            ],
        }
        clusters.append(cluster_obj)
        journal_partial(out_path, clusters)
        print(f"    cluster {cid} ({len(member_idx)} members): "
              f"{cluster_obj['title']!r}", flush=True)

    # Final atomic write
    build_meta: dict[str, Any] = {
        "tree_hash": tree_hash(tree_path),
        "k": k,
        "embedding_model": "BGE-M3",
        "cluster_embeddings": centroids.tolist(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if silhouette_curve is not None:
        build_meta["k_selection"] = "auto-silhouette"
        build_meta["silhouette_scores"] = [[kv, s] for kv, s in silhouette_curve]
    payload = {"build_meta": build_meta, "clusters": clusters}
    write_atomic(out_path, payload)
    print(f"\nWrote {out_path} — {k} clusters", flush=True)


if __name__ == "__main__":
    main()
