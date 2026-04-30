"""Generate PNG plots for RESULTS.md from the JSON files in results/.

Reads:
  results/langchain_fiqa_metrics.json
  results/throughput_stack_fiqa_metrics.json
  results/rerankers_metrics.json
  results/throughput_stack_rerank_metrics.json
  results/tei_fiqa_metrics.json

Writes:
  results/plots/wall_time_per_query_vs_throughput.png
  results/plots/recall_vs_hit_rate.png
  results/plots/rerank_lift.png
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results")
PLOTS = RESULTS / "plots"
PLOTS.mkdir(exist_ok=True)


def _load(name: str):
    return json.loads((RESULTS / name).read_text())


def plot_walltime():
    """Per-query vs throughput-stack wall_sec across dense/sparse/hybrid."""
    pq = {r["mode"]: r["wall_sec"] for r in _load("langchain_fiqa_metrics.json")}
    ts = {r["mode"]: r["wall_sec"] for r in _load("throughput_stack_fiqa_metrics.json")}
    modes = ["dense", "sparse", "hybrid"]
    pq_vals = [pq[m] for m in modes]
    ts_vals = [ts[m] for m in modes]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(modes))
    w = 0.38
    bars1 = ax.bar([i - w/2 for i in x], pq_vals, w,
                   label="per-query (langchain-qdrant)", color="#cf6679")
    bars2 = ax.bar([i + w/2 for i in x], ts_vals, w,
                   label="throughput-stack (qdrant.query_batch_points)", color="#03dac6")
    ax.set_xticks(list(x))
    ax.set_xticklabels(modes)
    ax.set_ylabel("Wall time (seconds, log scale)")
    ax.set_yscale("log")
    ax.set_title("FiQA eval wall-time — per-query vs throughput-stack (M5 Pro, 648 queries)")
    for b, v in list(zip(bars1, pq_vals)) + list(zip(bars2, ts_vals)):
        ax.text(b.get_x() + b.get_width()/2, v * 1.08, f"{v:.1f}s",
                ha="center", va="bottom", fontsize=9)
    # speedup annotations
    for i, m in enumerate(modes):
        speedup = pq[m] / ts[m]
        ax.text(i, max(pq[m], ts[m]) * 3.5, f"{speedup:.0f}× faster",
                ha="center", color="#03dac6", fontweight="bold", fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    out = PLOTS / "wall_time_per_query_vs_throughput.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_recall_vs_hit_rate():
    """ranx recall@10 vs hand-rolled hit_rate@10 — exposes formula divergence."""
    tei = _load("tei_fiqa_metrics.json")
    modes = ["dense", "sparse", "hybrid"]
    recall_vals = [next(r["ranx"]["recall@10"] for r in tei if r["mode"] == m) for m in modes]
    hit_vals    = [next(r["hand_rolled"]["hit_rate@10"] for r in tei if r["mode"] == m) for m in modes]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(modes))
    w = 0.38
    b1 = ax.bar([i - w/2 for i in x], recall_vals, w,
                label="ranx recall@10  (IR-canonical)", color="#6200ee")
    b2 = ax.bar([i + w/2 for i in x], hit_vals, w,
                label="hand-rolled hit_rate@10  (lab-02 mislabeled as recall)", color="#ffb300")
    ax.set_xticks(list(x))
    ax.set_xticklabels(modes)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.85)
    ax.set_title("FiQA — ranx recall vs hand-rolled hit_rate (same retrieval results)")
    for b, v in list(zip(b1, recall_vals)) + list(zip(b2, hit_vals)):
        ax.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)
    # delta annotations
    for i, m in enumerate(modes):
        delta = hit_vals[i] - recall_vals[i]
        ax.text(i, max(recall_vals[i], hit_vals[i]) + 0.07,
                f"Δ +{delta*100:.1f}pp", ha="center", color="#b00020",
                fontweight="bold", fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = PLOTS / "recall_vs_hit_rate.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_rerank_lift():
    """Baseline (dense top-5) vs reranked top-5: recall@5 + nDCG@5, two stacks."""
    pq = _load("rerankers_metrics.json")
    ts = _load("throughput_stack_rerank_metrics.json")

    metrics = ["recall@5", "ndcg@5"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, metric in zip(axes, metrics):
        pq_b = pq["baseline_dense_top5"][metric]
        pq_r = pq["rerank_top5"][metric]
        ts_b = ts["baseline_dense_top5"][metric]
        ts_r = ts["rerank_top5"][metric]
        labels = ["rerankers\n(per-query API)", "throughput-stack\n(cross-query batch)"]
        baseline = [pq_b, ts_b]
        reranked = [pq_r, ts_r]
        x = range(len(labels))
        w = 0.38
        ax.bar([i - w/2 for i in x], baseline, w, label="dense top-5 (no rerank)", color="#80cbc4")
        ax.bar([i + w/2 for i in x], reranked, w, label="rerank top-5", color="#1976d2")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylim(0.95, 1.005)
        ax.set_ylabel(metric)
        ax.set_title(f"MS MARCO 6,980 queries · {metric}")
        for i, (a, b) in enumerate(zip(baseline, reranked)):
            ax.text(i - w/2, a + 0.0006, f"{a:.4f}", ha="center", fontsize=8)
            ax.text(i + w/2, b + 0.0006, f"{b:.4f}", ha="center", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    # wall_sec subtitle
    fig.suptitle(f"Reranker lift — wall-time: rerankers={pq['wall_sec']:.0f}s · "
                 f"throughput-stack={ts['stage_seconds']['total']:.0f}s "
                 f"({pq['wall_sec']/ts['stage_seconds']['total']:.2f}×)",
                 fontsize=11)
    fig.tight_layout()
    out = PLOTS / "rerank_lift.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for fn in (plot_walltime, plot_recall_vs_hit_rate, plot_rerank_lift):
        out = fn()
        print(f"wrote {out}")
