# tree_index — PageIndex-pattern tree-index RAG primitives

Distilled from `lab-02-7-pageindex` (W2.7) where these patterns lifted tree
backend judge from **0.44 → 0.79** on Berkshire Hathaway 2023 (8-question eval).

## What's in here

| File | Purpose |
|---|---|
| `agentic.py` | `AgenticTreeRetriever` class — multi-turn tool-calling loop with `get_page_content` tool. Replaces greedy descent. |
| `builder.py` | `split_large_nodes` — recursive node-split helper. Splits any leaf > 5 pages or > 20K chars into 2-5 sub-sections via LLM. |
| `prompts.py` | `AGENTIC_SYSTEM_TEMPLATE`, `FACT_RICH_SUMMARIZE_SYSTEM`, `SPLIT_SYSTEM` — battle-tested system prompts. |

## Why these matter

The W2.7 lab measured each pattern's contribution:

| Pattern | What it fixes | Measured lift |
|---|---|---|
| Agentic loop with `get_page_content` tool | Navigator's "only sees titles" architectural blind spot — factoid queries with non-keyword titles unreachable | factoid 0.00 → 1.00 (+1.00) |
| Recursive node split | Monolithic leaves (e.g., 18-page Chairman's Letter) hide content from agentic fetches | citation 0.25 → 0.67 (+0.42) |
| Fact-rich summary contract | Vague summaries ("various financial metrics") confuse the navigator's keyword matching | shape: navigator's first-pick accuracy improves; not measured in isolation |
| Three-rule agentic system prompt | (a) TOC-trap guard, (b) explained refusal, (c) synthesis-from-fragments | refusal 0.67 → 1.00 (+0.33), synthesis 0.12 → 0.50 (+0.38) |

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

## Model isolation (W2.7 lesson)

When running a tree-index backend ALONGSIDE non-agentic backends (vector,
graph) on the same inference server, route the tree backend to a **different
model** than the non-agentic backends. Server-side KV-cache reuse can pollute
tool-routing across heterogeneous request shapes.

Pattern in `.env`:

```
MODEL_SONNET=gemma-4-26B-A4B-it-heretic-4bit  # vector + graph, no-tools calls
MODEL_TREE=Qwen3.6-35B-A3B-UD-MLX-4bit         # tree, tools calls
```

Different model = different KV cache pool on the server.

## What's NOT in here (yet)

- **`build_tree(pdf_path) -> tree.json`** — the heuristic + LLM tree-builder
  entry point. W2.7's `build_tree.py` does this lab-specifically; will be
  promoted to `tree_index.builder.build_tree(...)` when a second lab needs it.
- **`get_subtree_text(parent_id)`** — a third tool that fetches all leaves
  under a parent in one call, restoring synthesis-as-single-fetch while
  keeping per-leaf precision. Flagged as W2.7 follow-up; not in v1.

## Reference implementation

- `lab-02-7-pageindex/src/query_tree.py` — uses `AgenticTreeRetriever`.
- `lab-02-7-pageindex/src/build_tree.py` — uses `split_large_nodes`.
- `lab-02-7-pageindex/RESULTS.md` §"PageIndex Optimization Run" — measured
  lifts per pattern.
- W2.7 chapter §4.3.3 — narrative rationale + shared-lib design.
