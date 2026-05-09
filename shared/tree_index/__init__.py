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
from .ensemble import ENSEMBLE_SYNTHESIS_PROMPT, EnsembleTreeRetriever
from .entity_index import EntityIndex, extract_entities
from .index import TreeIndex
from .prompts import (
    AGENTIC_SYSTEM_TEMPLATE,
    AGENTIC_SYSTEM_TEMPLATE_V2,
    FACT_RICH_SUMMARIZE_SYSTEM,
    SPLIT_SYSTEM,
)
from .page_vector_index import PageVectorIndex
from .summary_index import SummaryIndex

__all__ = [
    "AgenticTreeRetriever",
    "EnsembleTreeRetriever",
    "PageProvider",
    "PageVectorIndex",
    "SummaryIndex",
    "TreeIndex",
    "EntityIndex",
    "extract_entities",
    "split_large_nodes",
    "AGENTIC_SYSTEM_TEMPLATE",
    "AGENTIC_SYSTEM_TEMPLATE_V2",
    "ENSEMBLE_SYNTHESIS_PROMPT",
    "FACT_RICH_SUMMARIZE_SYSTEM",
    "SPLIT_SYSTEM",
]
