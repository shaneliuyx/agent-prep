"""Tree-index RAG primitives — PageIndex pattern, distilled from W2.7.

Reusable across labs that build hierarchical-structure retrievers over long
PDFs (10-Ks, contracts, textbooks, technical specs). The lab-specific work
becomes: supply a `page_provider`, a built `tree.json`, and a model client.

Top-level imports:

    from tree_index import (
        AgenticTreeRetriever,
        split_large_nodes,
        AGENTIC_SYSTEM_TEMPLATE,
        FACT_RICH_SUMMARIZE_SYSTEM,
        SPLIT_SYSTEM,
    )

See README.md for usage examples + the W2.7 lab for a worked end-to-end
implementation that achieves judge=0.79 on Berkshire 2023 (vs 0.44 pre-opt).
"""

from .agentic import AgenticTreeRetriever, PageProvider
from .builder import split_large_nodes
from .prompts import (
    AGENTIC_SYSTEM_TEMPLATE,
    FACT_RICH_SUMMARIZE_SYSTEM,
    SPLIT_SYSTEM,
)

__all__ = [
    "AgenticTreeRetriever",
    "PageProvider",
    "split_large_nodes",
    "AGENTIC_SYSTEM_TEMPLATE",
    "FACT_RICH_SUMMARIZE_SYSTEM",
    "SPLIT_SYSTEM",
]
