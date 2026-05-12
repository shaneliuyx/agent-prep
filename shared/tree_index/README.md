# tree_index — PageIndex-pattern tree-index RAG primitives

Distilled from `lab-02-7-pageindex` (W2.7) where layered patterns lifted tree
backend judge from **0.44 → 0.885** on Berkshire Hathaway 2023 (16-question eval).
See `lab-02-7-pageindex/RESULTS.md` for the full measurement chain.

## What's in here

| File | Purpose |
|---|---|
| `agentic.py` | `AgenticTreeRetriever` class — multi-turn tool-calling loop. 4 tools: `get_page_content`, `find_nodes_mentioning`, `get_subtree_text`, `find_cluster_for_synthesis`. Hermes-template parser fallback for vMLX-quantized Qwen models. |
| `builder.py` | `split_large_nodes` — recursive node-split helper. Splits any leaf > 5 pages or > 20K chars into 2-5 sub-sections via LLM. |
| `index.py` | `TreeIndex` 3-dict primitive (id→node, page→nodes, parent→children). Used by EntityIndex + EnsembleTreeRetriever. |
| `entity_index.py` | `EntityIndex` regex over node body + tags merge. Multi-query expansion + RRF for semantic-equivalent matching. |
| `summary_index.py` | `SummaryIndex` Level-2 RAPTOR-style cluster index over the primary tree. `find_clusters_for_query()` returns top-K with delta-band tiebreak for cluster-first routing on synthesis questions. |
| `_hashing.py` | `tree_hash(tree_path)` sha256 of canonical JSON. Binds summary_index.json to tree.json state — fail-fast on stale index. |
| `ensemble.py` | `EnsembleTreeRetriever` — runs v1 (greedy) + v2 (entity-graph + cluster) paths in parallel, LLM synthesis picks best. Best-of-both at the cost of 2× retrieval. |
| `prompts.py` | `AGENTIC_SYSTEM_TEMPLATE` (v1), `AGENTIC_SYSTEM_TEMPLATE_V2` (v2/v3 with cluster-first Rule -1), `FACT_RICH_SUMMARIZE_SYSTEM` (multi-pass build), `SPLIT_SYSTEM` (recursive split). |

## Why these matter

The W2.7 lab measured each pattern's contribution. Cumulative chain from naive tree-walk to current champion:

| Pattern | What it fixes | Measured lift |
|---|---|---|
| Agentic loop with `get_page_content` tool | Navigator's "only sees titles" architectural blind spot — factoid queries with non-keyword titles unreachable | factoid 0.00 → 1.00 (+1.00) |
| Recursive node split (`split_large_nodes`) | Monolithic leaves (e.g., 18-page Chairman's Letter) hide content from agentic fetches | citation 0.25 → 0.67 (+0.42) |
| Fact-rich summary contract | Vague summaries ("various financial metrics") confuse the navigator's keyword matching | improves first-pick accuracy; not isolated-measured |
| Three-rule agentic system prompt | TOC-trap guard, explained refusal, synthesis-from-fragments | refusal 0.67 → 1.00 (+0.33), synthesis 0.12 → 0.50 (+0.38) |
| Multi-pass tree build (verbatim title preservation) | Single-pass loses distinctive phrases ("Our Not-So-Secret Weapon" → "competitive advantages") | Q-ENTITY worst 0.00 → 0.50, mean 0.33 → 0.67 |
| Hermes-format tool-call parser fallback | vMLX doesn't extract `<function=NAME>...</function>` text from DWQ-style models | DWQ retriever 0.39 → 0.67 (+0.28) |
| EntityIndex + multi-query expansion + RRF | Regex literal-string match misses semantic equivalents | +0.10-0.20 on entity-graph queries |
| Synthesis-question guard | Forces ≥2 fetches on "what did X say about Y" queries | synthesis 0.12 → 0.50 |
| Level-2 cluster pre-fetch (`SummaryIndex` + `find_cluster_for_synthesis`) | Synthesis questions spend 2-3 iters locating page ranges; pre-fetch routes in 1 cosine call | Q4 0.00 → 0.75 |
| Top-K cluster routing with delta-band tiebreak | Top-1 demands embedding precision below noise floor; top-K with δ=0.07 returns both ambiguous candidates for LLM tiebreak | Q4 + others recovered without breaking single-cluster routing |
| 2-model split discipline (MODEL_TREE = 9B-GLM, MODEL_SONNET = Gemma-26B) | Single MoE model breaks under sustained tool-call load (Issue #1011); judge swap is one-way door (4/16 disagree ≥0.25) | Eval wall-clock 30→12 min, judge baseline preserved |

**Current champion: agg_judge=0.885, agg_lat=46.6s/question (16-Q eval).**

## Usage

### Minimal example — agentic retrieval over an existing tree

```python
import json
from openai import OpenAI
from pypdf import PdfReader

from tree_index import AgenticTreeRetriever, AGENTIC_SYSTEM_TEMPLATE

client = OpenAI(base_url="http://localhost:8000/v1", api_key="...")
tree = json.loads(open("data/tree.json").read())
reader = PdfReader("data/document.pdf")
pages_text = [p.extract_text() or "" for p in reader.pages]

def page_provider(start: int, end: int) -> str:
    return "\n\n".join(
        f"[page {i+1}]\n{pages_text[i]}"
        for i in range(max(0, start - 1), min(len(pages_text), end))
    )

retriever = AgenticTreeRetriever(
    tree=tree,
    page_provider=page_provider,
    model_client=client,
    model_name="Qwen3.6-35B-A3B-UD-MLX-4bit",
    system_prompt=AGENTIC_SYSTEM_TEMPLATE,
)

result = retriever.answer("What was the company's net earnings in 2023?")
print(result["answer"])           # "$96,223 million [page 96]"
print(result["iterations"])       # 2
print(result["tool_calls"])       # [{iter: 0, tool: get_page_content, args: {...}}]
```

### Recursive split during tree build

```python
from tree_index import split_large_nodes, SPLIT_SYSTEM

# After your initial heuristic + LLM tree build:
split_large_nodes(
    tree, pages,
    model_client=client, model_name="...",
    split_system_prompt=SPLIT_SYSTEM,
    max_pages=5, max_chars=20_000,
)
# Tree now has finer-grained leaves where they were too coarse.
```

## Model isolation (W2.7 lesson — current 2-model split)

When running a tree-index backend ALONGSIDE non-agentic backends (vector,
graph) on the same inference server, route the tree backend to a **different
model** than the non-agentic backends. Server-side KV-cache reuse can pollute
tool-routing across heterogeneous request shapes.

Beyond cache isolation, W2.7 measured a second discipline: **judge-model
identity is a one-way door**. Swapping the judge invalidates all prior
baselines — 4/16 judge disagreements ≥0.25 between Gemma-26B and 9B-GLM
on the same answer set (mean |Δ|=0.141, max 0.75 on out-of-document
refusals). Keep the judge fixed for the life of the eval.

Current 2-model split pattern in `.env`:

```
# Hot path — agentic retriever loop, ~14s/call avg, 6/6 capability probe
MODEL_TREE=models/MLX-Qwen3.5-9B-GLM5.1-Distill-v1-8bit
# Judge baseline — non-negotiable, preserves comparability with all prior runs
MODEL_SONNET=models/gemma-4-26B-A4B-it-heretic-4bit
# Build path — one-shot per cluster (8 calls), retry helper absorbs vMLX 503s
MODEL_BUILD=models/gemma-4-26B-A4B-it-heretic-4bit
MODEL_HAIKU=models/gemma-4-26B-A4B-it-heretic-4bit
```

Total VRAM ~24 GB on Apple Silicon unified memory. Eval wall-clock 12 min.

Cluster routing is env-tunable:

```
SUMMARY_INDEX_THRESHOLD=0.5   # cosine floor for any candidate
SUMMARY_INDEX_TOP_K=2         # max candidates
SUMMARY_INDEX_DELTA=0.07      # band — 2nd candidate kept if (best - score) ≤ delta
```

`δ=0.07` calibrated to BGE-M3 noise floor on ~1k-token cluster centroids.
`δ=0.10` is too wide (paralyzes 9B-GLM on Q3-class wider gaps); `δ=0.05`
is too tight (misses Q4-class 0.052 noise-band ties).

## What's NOT in here (yet)

- **`build_tree(pdf_path) -> tree.json`** — the heuristic + LLM tree-builder
  entry point. W2.7's `build_tree.py` does this lab-specifically; will be
  promoted to `tree_index.builder.build_tree(...)` when a second lab needs it.
- **`build_summary_index(tree.json) -> summary_index.json`** — the Level-2
  cluster index builder lives at `lab-02-7-pageindex/src/build_summary_index.py`.
  Same promotion path as `build_tree`. Includes opt-in `--method llm` flag for
  LLM-grouping cluster build with auto-place orphans (Approach B, currently
  net-negative under existing AMBIGUOUS-hint pattern; reserved for future when
  hint format is title-injecting).
- **Vector retrieval as a fourth tool** — wire BGE-M3 from
  `lab-02-3-bge_m3_hnsw` as `find_nodes_by_semantic_match`. Closes the regex
  semantic gap structurally. Estimated +0.10-0.15 aggregate.

## Reference implementation

- `lab-02-7-pageindex/src/query_tree.py` — uses `AgenticTreeRetriever`.
- `lab-02-7-pageindex/src/build_tree.py` — uses `split_large_nodes` + multi-pass.
- `lab-02-7-pageindex/src/build_summary_index.py` — builds Level-2 cluster index
  (K-means default, `--method llm` opt-in).
- `lab-02-7-pageindex/scripts/run_one_variant.py` — wires `SummaryIndex` into v2.
- `lab-02-7-pageindex/RESULTS.md` — full measurement chain. Section
  "Level-2 Summary Index Cluster Pre-Fetch + 2-Model Split" covers today's
  champion architecture with bundled per-block walkthroughs (mermaid diagrams,
  code excerpts, results, insights). Section "Comparison vs Original PageIndex"
  documents what this lab kept, improved, and could leverage further.
- W2.7 RESULTS.md is the canonical runbook reference per-Python-block bundle
  pattern.
