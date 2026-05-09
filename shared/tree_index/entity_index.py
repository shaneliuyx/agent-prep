"""EntityIndex — entity-graph layer ON TOP of a tree (BookRAG pattern).

Build-time pass over tree.json that extracts entities from each node's
title + summary, builds two indexes:

    node_to_entities: node_id -> set[entity]
    entity_to_nodes:  entity  -> set[node_id]

Cheap because the tree's `FACT_RICH_SUMMARIZE_SYSTEM` already requires
each summary to quote 5 named entities verbatim. We just regex-extract
them. No additional LLM calls at build time.

Used at query-time by `find_nodes_mentioning(entity_or_phrase)` to
augment the agentic loop's candidate set for synthesis questions where
content spans multiple sub-sections after recursive split.

Pairs with TreeIndex (`shared/tree_index/index.py`) — that's structure
+ subtree fetch; this is content + entity lookup. Together they close
the W2.7 §4.3.3 split-merge tradeoff (auto-merge via subtree, entity
lookup for cross-section synthesis discovery).

Reference: arXiv:2512.* BookRAG (CUHK 2025-12) which adds entity-graph
layer on top of structural tree. Their entity layer needs full
text + LLM extraction; ours is summary-derived + regex (cheaper, fits
~100 KB tree.json scale).
"""
from __future__ import annotations

import re
from collections import defaultdict


# Capitalized multi-word noun phrases (e.g., "Berkshire Hathaway",
# "Coca-Cola", "American Express", "Form 10-K", "Item 1A"). Conservatively
# requires 2+ words OR a clear all-caps acronym, OR a numbered Item.
_ENTITY_PATTERNS = [
    # Multi-word capitalized: "Berkshire Hathaway", "American Express",
    # "Burlington Northern Santa Fe", "Tim Berners-Lee"
    re.compile(r"\b([A-Z][a-zA-Z'’]+(?:\s+(?:and|&|of|the)?\s*[A-Z][a-zA-Z'’\-]+){1,4})\b"),
    # Item identifiers in 10-Ks: "Item 1A", "Item 7", "Note 16"
    re.compile(r"\b((?:Item|Note|Schedule|Section|Part|Article|Chapter)\s+\d+[A-Z]?(?:\.\d+)?)\b"),
    # Form codes: "Form 10-K", "10-Q", "S-1"
    re.compile(r"\b((?:Form\s+)?\d{1,2}-[A-Z]{1,2})\b"),
    # All-caps acronyms 2-6 chars: "BNSF", "MD&A", "GAAP"
    re.compile(r"\b([A-Z]{2,6}(?:&[A-Z])?)\b"),
    # Hyphenated proper nouns: "Coca-Cola", "Berkshire-Hathaway"
    re.compile(r"\b([A-Z][a-z]+-[A-Z][a-z]+(?:-[A-Z][a-z]+)*)\b"),
]


_STOPLIST = {
    "The", "This", "These", "Those", "That", "Their", "There", "They",
    "When", "Where", "Why", "How", "Who", "What", "Which",
    # Common short words that catch on 2-word capitalization but aren't entities:
    "I", "II", "III", "IV", "V",
}


def extract_entities(text: str) -> set[str]:
    """Extract candidate entity strings from a single text. Used per-node
    over title + summary — cheap regex, no LLM call.

    Returns canonicalized strings (whitespace-collapsed). Caller is
    responsible for further normalization (lowercase / Wikidata QID
    linking) if needed.
    """
    if not text:
        return set()
    out: set[str] = set()
    for pat in _ENTITY_PATTERNS:
        for m in pat.finditer(text):
            ent = re.sub(r"\s+", " ", m.group(1)).strip()
            if ent and ent not in _STOPLIST:
                out.add(ent)
    return out


class EntityIndex:
    """Entity → [node_ids] inverted index built from a tree.json's
    title + summary fields.

    Build cost: O(N) regex passes where N is total node count. ~50ms
    for a 62-node tree. No LLM calls.

    Query cost: O(1) keyword lookup via case-insensitive normalized
    match. Substring fallback for partial matches.
    """

    def __init__(
        self, tree_index_or_tree,
        page_provider=None,           # callable(start, end) -> str — full body text
    ):
        """Build the entity index.

        Args:
            tree_index_or_tree: a TreeIndex or raw tree dict.
            page_provider:      optional callable that, given a leaf's
                                start_page/end_page, returns the raw body
                                text. Increases entity recall dramatically
                                (summary-only misses entities the
                                FACT_RICH_SUMMARIZE_SYSTEM didn't quote).
                                Pass None to fall back to title+summary
                                only (cheaper, lower recall).
        """
        # Accept either a TreeIndex instance or a raw tree dict
        if hasattr(tree_index_or_tree, "id_map"):
            self.id_map = tree_index_or_tree.id_map
        else:
            from .index import TreeIndex
            self.id_map = TreeIndex(tree_index_or_tree).id_map

        self.node_to_entities: dict[str, set[str]] = {}
        self.entity_to_nodes: dict[str, set[str]] = defaultdict(set)
        # Lowercase index for case-insensitive lookup
        self._lc_to_canonical: dict[str, str] = {}

        for nid, node in self.id_map.items():
            text_parts = [node.get("title", ""), node.get("summary", "")]
            # Optionally pull body text via page_provider for higher recall
            if page_provider is not None:
                sp = node.get("start_page")
                ep = node.get("end_page", sp)
                if sp is not None and ep is not None:
                    try:
                        # NO truncation — long sections like Chairman's Letter
                        # (~46K chars over 16 pages) put critical entities like
                        # "Coca-Cola" at char 17K+. Regex extraction is fast
                        # (~10ms per 50K chars × 62 nodes ≈ 600ms total build),
                        # and entity-graph recall is the W2.7 §4.3.3 split-merge
                        # follow-up's load-bearing fix. Trade ~0.5s build time
                        # for full-document entity coverage.
                        body = page_provider(sp, ep)
                        text_parts.append(body)
                    except Exception:  # noqa: BLE001
                        pass
            text = " ".join(p for p in text_parts if p)
            # Regex-extracted entities (recall layer)
            entities = extract_entities(text)
            # LLM-curated tags from build_tree.py multi-pass summarization
            # (precision layer — preserves quoted phrases + aliases that
            # regex would miss, e.g. "not-so-secret weapon", "Bertie",
            # "Rip Van Winkle slumber"). Tags are stored verbatim under
            # node["tags"] when available.
            for raw_tag in node.get("tags", []) or []:
                tag = (raw_tag or "").strip()
                if tag and len(tag) >= 2:
                    entities.add(tag)
            self.node_to_entities[nid] = entities
            for ent in entities:
                self.entity_to_nodes[ent].add(nid)
                self._lc_to_canonical[ent.lower()] = ent

    def __len__(self) -> int:
        return len(self.entity_to_nodes)

    def all_entities(self) -> list[str]:
        return list(self.entity_to_nodes.keys())

    def entities_in(self, node_id: str) -> set[str]:
        """Entities mentioned in a single node's title+summary."""
        return self.node_to_entities.get(node_id, set())

    def nodes_with(self, entity: str) -> set[str]:
        """Exact entity → set of node_ids that mention it. Case-insensitive."""
        canon = self._lc_to_canonical.get(entity.lower())
        if canon:
            return self.entity_to_nodes.get(canon, set())
        return set()

    def find_nodes_mentioning(self, query: str, *, max_nodes: int = 10) -> list[str]:
        """Multi-strategy lookup for the agentic-loop tool:
        1. exact case-insensitive entity match
        2. substring match on canonical entity strings
        3. token overlap (every query token must appear as substring of an entity)

        Returns a deduplicated list of node_ids ranked by hit count.
        """
        if not query.strip():
            return []
        # Strategy 1 — exact match
        exact = self.nodes_with(query)
        if exact:
            return list(exact)[:max_nodes]

        # Strategy 2 — substring match
        ql = query.lower()
        candidates_substr: dict[str, int] = defaultdict(int)
        for ent, nids in self.entity_to_nodes.items():
            if ql in ent.lower():
                for nid in nids:
                    candidates_substr[nid] += 1
        if candidates_substr:
            ranked = sorted(candidates_substr.items(), key=lambda x: -x[1])
            return [nid for nid, _ in ranked[:max_nodes]]

        # Strategy 3 — token overlap on the query
        tokens = [t for t in re.findall(r"\w+", ql) if len(t) > 2]
        if not tokens:
            return []
        candidates_tok: dict[str, int] = defaultdict(int)
        for ent, nids in self.entity_to_nodes.items():
            ent_lc = ent.lower()
            hits = sum(1 for t in tokens if t in ent_lc)
            if hits > 0:
                for nid in nids:
                    candidates_tok[nid] += hits
        ranked = sorted(candidates_tok.items(), key=lambda x: -x[1])
        return [nid for nid, _ in ranked[:max_nodes]]
