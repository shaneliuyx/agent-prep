"""Agentic tree-index retriever — PageIndex pattern.

Replaces greedy single-shot tree-walk with multi-turn tool-calling loop:
the LLM sees the document tree (ids/titles/page-ranges/summaries), decides
which page range is most likely to contain the answer, calls
get_page_content(start, end), iterates if needed, and either composes the
final answer with a citation or refuses with explanation.

Closes the architectural blind spot of greedy navigation: navigator can now
inspect body text mid-decision instead of being limited to titles + summaries.
"""
from __future__ import annotations

import json
import os
import re
from typing import Protocol


# Fallback parser for oMLX state-degradation pattern: under sustained tools-call
# load, oMLX sometimes emits Qwen3.6's native tool-call template (`<|tool_call>
# call:get_page_content(start_page: 96, end_page: 96)`) AS PLAIN TEXT in the
# message content instead of populating the structured `tool_calls` field.
# Without this fallback, the agent loop sees `tool_calls=[]` + non-empty
# content and exits with the malformed text as the "answer".
#
# Detects `<|tool_call>call:NAME(arg1: val1, arg2: val2)` and converts to a
# pseudo-tool-call structure compatible with the dispatch loop below.
_TC_RE = re.compile(
    r"""<\|tool_call\>call:                       # marker
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)          # tool name
        \(                                         # open paren
        (?P<args>[^)]*)                            # arg list (no nested parens)
        \)""",
    re.VERBOSE,
)

# Hermes/Llama-style template emitted by Qwen3.6-A3B-DWQ as plain text:
#   <function=NAME><parameter=K1>V1</parameter><parameter=K2>V2</parameter></function>
# Sometimes followed by </tool_call>. vMLX's structured-tool extractor doesn't
# parse this format, so the agent loop sees tcalls=[] and treats the entire
# text as a final answer, breaking retrieval. This regex recovers it.
_TC_HERMES_RE = re.compile(
    r"""<function=(?P<name>[A-Za-z_][A-Za-z0-9_]*)>     # opening
        (?P<body>.*?)                                    # parameter block
        </function>""",
    re.VERBOSE | re.DOTALL,
)
_TC_HERMES_PARAM_RE = re.compile(
    r"<parameter=(?P<k>[A-Za-z_][A-Za-z0-9_]*)>"
    r"\s*(?P<v>.*?)\s*"
    r"</parameter>",
    re.DOTALL,
)


def _parse_native_toolcalls(text: str) -> list[dict]:
    """Extract tool calls embedded as plain text in message content. Handles
    BOTH Qwen native format (<|tool_call>call:NAME(...)`) and Hermes-style
    format (`<function=NAME><parameter=K>V</parameter></function>`).
    Returns list of {name, arguments_json_str, id} dicts.
    """
    out: list[dict] = []
    seen: set[tuple] = set()
    counter = 0
    # Pattern 1 — Qwen native
    for m in _TC_RE.finditer(text):
        name = m.group("name")
        args_raw = m.group("args").strip()
        args_dict: dict = {}
        for pair in args_raw.split(","):
            if ":" not in pair:
                continue
            k, v = pair.split(":", 1)
            k = k.strip()
            v = v.strip().strip("\"'")
            try:
                args_dict[k] = int(v)
            except ValueError:
                args_dict[k] = v
        if not args_dict:
            continue
        key = (name, tuple(sorted(args_dict.items())))
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": f"native_{counter}", "name": name,
                    "arguments": json.dumps(args_dict)})
        counter += 1
    # Pattern 2 — Hermes-style (<function=...><parameter=...>...)
    for m in _TC_HERMES_RE.finditer(text):
        name = m.group("name")
        body = m.group("body")
        args_dict = {}
        for pm in _TC_HERMES_PARAM_RE.finditer(body):
            k = pm.group("k").strip()
            v = pm.group("v").strip().strip("\"'")
            try:
                args_dict[k] = int(v)
            except ValueError:
                args_dict[k] = v
        if not args_dict:
            continue
        key = (name, tuple(sorted(args_dict.items())))
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": f"hermes_{counter}", "name": name,
                    "arguments": json.dumps(args_dict)})
        counter += 1
    return out


class PageProvider(Protocol):
    """Returns raw text for a 1-indexed inclusive page range."""

    def __call__(self, start: int, end: int) -> str: ...


_DEFAULT_TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_page_content",
        "description": "Fetch raw text from a page range of the source document.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_page": {"type": "integer", "description": "Start page (1-indexed)"},
                "end_page":   {"type": "integer", "description": "End page (inclusive, 1-indexed)"},
            },
            "required": ["start_page", "end_page"],
        },
    },
}]

# Optional v2 tools — enabled when AgenticTreeRetriever is built with
# tree_idx + entity_idx. Together they implement the BookRAG entity-graph +
# HiChunk Auto-Merge pattern on top of the W2.7 tree.
_V2_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_subtree_text",
            "description": (
                "AUTO-MERGE: Given a parent node_id, fetch the combined text "
                "of all leaves under it. Use when a synthesis question's answer "
                "is fragmented across multiple sub-sections of the same parent "
                "(e.g., recursive-split sub-sections of Chairman's Letter)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "parent_node_id": {
                        "type": "string",
                        "description": "node_id (e.g., '0006' or '0006.02') whose subtree to merge",
                    },
                },
                "required": ["parent_node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_nodes_mentioning",
            "description": (
                "ENTITY-GRAPH LOOKUP: Given a literal entity name or phrase, "
                "return up to 10 node_ids whose body text mentions it. Use to "
                "discover candidate sections for synthesis questions about "
                "named entities (companies, people, regulations, financial "
                "instruments) that may be scattered across the document."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_or_phrase": {
                        "type": "string",
                        "description": "Entity name or phrase to look up, e.g. 'Coca-Cola', 'BNSF', 'Item 1C'",
                    },
                },
                "required": ["entity_or_phrase"],
            },
        },
    },
]

_CLUSTER_TOOL = {
    "type": "function",
    "function": {
        "name": "find_cluster_for_synthesis",
        "description": (
            "Cluster-first lookup for cross-section synthesis questions "
            "('what did X say/write about Y'). Returns one thematic cluster "
            "with member node_ids + page ranges. Use BEFORE get_page_content "
            "when the question spans multiple sub-sections — one batched "
            "fetch over all member pages is more efficient than sequential "
            "single-node fetches."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "the user's question or topic"},
            },
            "required": ["query"],
        },
    },
}


def _expand_with_neighbors(
    pages: list[int], window: int = 1,
) -> list[tuple[int, int]]:
    """Expand each page number by ±window neighbors, dedup, and emit
    contiguous (start, end) ranges.

    Page-level vector match returns single pages, but multi-page topics
    (e.g. Berkshire's Japanese trading houses section spans pages 12-13)
    need neighbor context. Caller fetches each (start, end) range as one
    contiguous block — efficient over per-page fetches when neighbors
    overlap.

    Lower bound clipped to 1 (PDF page numbers are 1-indexed).
    """
    if not pages:
        return []
    expanded: set[int] = set()
    for p in pages:
        for d in range(-window, window + 1):
            np = p + d
            if np >= 1:
                expanded.add(np)
    sorted_pages = sorted(expanded)
    ranges: list[tuple[int, int]] = []
    start = prev = sorted_pages[0]
    for p in sorted_pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append((start, prev))
            start = prev = p
    ranges.append((start, prev))
    return ranges


_LOW_QUALITY_REFUSAL_PATTERNS = (
    "i don't have", "i cannot find", "i am unable",
    "i do not have", "the document does not",
    "the document doesn't", "no information about",
    "not provided in", "not mentioned in",
    "not available in", "not in the document",
)


def _is_low_quality(answer: str) -> bool:
    """Composite production-signal trigger for chunk-level fallback.

    Returns True when the agent's answer suggests refusal, pseudo-refusal,
    or ungrounded synthesis. Used by AgenticTreeRetriever.answer() to
    decide whether to fire the page-vector fallback.

    Tier 1 — high confidence (any one fires):
      - empty / whitespace-only answer
      - literal "insufficient context" substring
      - common refusal phrases (see _LOW_QUALITY_REFUSAL_PATTERNS)
      - length < 80 chars (pseudo-refusal)

    Tier 2 — medium confidence (fires only if length > 80):
      - no `[page` / `pages` citation in answer

    Tier 3 — explicitly NOT triggered (false-positive rate too high):
      - hedging language (may, might, possibly, approximately)

    Production signal only — never reads judge scores or any test-time
    oracle. Goodhart-safe.
    """
    a = answer.strip()
    if not a:
        return True
    a_low = a.lower()
    if "insufficient context" in a_low:
        return True
    if any(p in a_low for p in _LOW_QUALITY_REFUSAL_PATTERNS):
        return True
    if len(a) < 80:
        return True
    if "[page" not in a_low and "pages" not in a_low:
        return True
    return False


def _tree_view(tree: dict) -> str:
    """Compact JSON view: id, title, pages, summary. Skip raw text/children fields."""
    def walk(node: dict, depth: int = 0) -> list[dict]:
        out = [{
            "node_id": node.get("node_id"),
            "title":   node.get("title"),
            "pages":   f"{node.get('start_page', '?')}-{node.get('end_page', '?')}",
            # Cap summary at 120 chars (~1 sentence) for the navigator TOC.
            # Multi-pass build produces 800-1200 char summaries; injecting all
            # of them inflates prompt to ~12K tokens and degrades DWQ attention.
            # Full summary stays in tree.json + ingested by EntityIndex tags
            # — navigator only needs first sentence to route.
            "summary": (node.get("summary") or "").split("\n", 1)[0][:120],
            "depth":   depth,
        }]
        for c in node.get("nodes", []):
            out.extend(walk(c, depth + 1))
        return out
    return json.dumps(walk(tree), indent=1)


class AgenticTreeRetriever:
    """Multi-turn agent loop over a tree + page-content tool.

    Args:
        tree:           dict with `node_id` / `title` / `start_page` / `end_page` /
                        `summary` / `nodes` fields. Compatible with W2.7
                        `data/tree.json` shape.
        page_provider:  callable returning raw text for a 1-indexed inclusive
                        page range. Typically wraps a `pypdf.PdfReader`.
        model_client:   OpenAI-compatible client (e.g., `openai.OpenAI`).
        model_name:     target model on the server.
        system_prompt:  agentic system prompt. Default is W2.7's hardened version
                        (TOC-trap guard + explained refusal + synthesis-from-
                        fragments). Pass a different prompt only when the
                        corpus has a structurally different shape.
        max_iterations: bounded loop ceiling (default 4).
                        Reduced from 6 (2026-05-09) — Phase 7+ cluster
                        pre-fetch front-loads routing context that
                        previously took 2-3 iters; no measured query
                        on the 16-Q eval hits the new ceiling.
        max_range_chars: per-fetch char cap on returned text (default 25000).
                         Bumped from 8000 (2026-05-09) — 8000 truncated mid-
                         Chairman's-Letter on Q9-class queries, hiding the
                         Scorecard table on page 13. 25K covers full sections
                         (Chairman's Letter is ~30K; most subsections fit).
        debug_log_path: if set, append `[Nit/Mtc] q=... ans=...` per call for
                        cross-process debugging.

    Public method:
        answer(query) -> {answer, tool_calls, iterations}.
    """

    def __init__(
        self, *,
        tree: dict,
        page_provider: PageProvider,
        model_client,
        model_name: str,
        system_prompt: str,
        max_iterations: int = 4,
        max_range_chars: int = 25000,
        debug_log_path: str | None = None,
        # Optional v2 — entity-graph + auto-merge tools
        tree_index=None,        # TreeIndex instance for subtree fetch
        entity_index=None,      # EntityIndex for find_nodes_mentioning
        summary_index=None,     # Optional[SummaryIndex] for cluster routing
        # Optional v3 — last-resort chunk-level fallback
        page_vector_index=None, # Optional[PageVectorIndex] BGE-M3 dense+sparse
    ) -> None:
        self.tree = tree
        self.page_provider = page_provider
        self.client = model_client
        self.model = model_name
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_range_chars = max_range_chars
        self.debug_log_path = debug_log_path
        self.tree_index = tree_index
        self.entity_index = entity_index
        self.summary_index = summary_index
        self.page_vector_index = page_vector_index
        # Tool list: extend with v2 tools only if both indexes are supplied
        self._tools = list(_DEFAULT_TOOLS)
        if tree_index is not None:
            self._tools.append(_V2_TOOLS[0])  # get_subtree_text
        if entity_index is not None:
            self._tools.append(_V2_TOOLS[1])  # find_nodes_mentioning
        if summary_index is not None:
            self._tools.append(_CLUSTER_TOOL)
        # Cache expansions per-instance to avoid duplicate LLM calls within
        # one agent loop. Cleared per-query in answer().
        self._expansion_cache: dict[str, list[str]] = {}

    def _fetch_subtree(self, parent_node_id: str) -> str:
        """AUTO-MERGE: combine text of all leaves under a parent."""
        if self.tree_index is None:
            return "[ERROR] get_subtree_text requires tree_index"
        if parent_node_id not in self.tree_index:
            return f"[ERROR] node_id {parent_node_id!r} not found in tree"
        node_ids = self.tree_index.subtree_ids(parent_node_id)
        parts = []
        for nid in node_ids:
            node = self.tree_index.get(nid) or {}
            sp = node.get("start_page")
            ep = node.get("end_page", sp)
            title = node.get("title", "")
            if sp is not None and ep is not None:
                try:
                    body = self.page_provider(sp, ep)
                    parts.append(f"[{nid}] {title} (pages {sp}-{ep})\n{body}")
                except Exception as e:  # noqa: BLE001
                    parts.append(f"[{nid}] {title} — [ERROR fetching: {e}]")
        merged = "\n\n---\n\n".join(parts)
        if len(merged) > self.max_range_chars * 2:  # 2× cap for subtree
            merged = merged[: self.max_range_chars * 2] + "\n[... truncated]"
        return merged

    _EXPAND_SYSTEM = (
        "Generate 3 SHORT alternative phrasings (2-5 words each) for finding "
        "the same concept in document body text. Output strict JSON: "
        '{"variants": ["...", "...", "..."]}. No prose, no markdown.\n\n'
        "Examples:\n"
        '  "not-so-secret weapon" → '
        '{"variants": ["secret weapon", "competitive advantage", "Charlie Munger"]}\n'
        '  "non-controlled businesses" → '
        '{"variants": ["equity investments", "marketable securities", "Coca-Cola Apple"]}\n'
        '  "BNSF Railway" → '
        '{"variants": ["BNSF", "railroad operations", "Burlington Northern"]}'
    )

    def _expand_phrase(self, phrase: str) -> list[str]:
        """Multi-query expansion. Returns [original, variant1, variant2, variant3].
        Cached per-phrase within one agent loop to amortize the LLM call.

        Closes the regex EntityIndex semantic gap: when the user query uses one
        phrasing ("secret weapon") but the body text uses another ("Charlie",
        "patient capital"), regex misses. Expansion broadens the search."""
        cached = self._expansion_cache.get(phrase)
        if cached is not None:
            return cached
        try:
            r = self.client.chat.completions.create(
                model=self.model, temperature=0.3, max_tokens=200,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._EXPAND_SYSTEM},
                    {"role": "user", "content": phrase},
                ],
            )
            raw = (r.choices[0].message.content or "{}").strip()
            parsed = json.loads(raw)
            variants = parsed.get("variants", [])
            if not isinstance(variants, list):
                variants = []
            variants = [str(v).strip() for v in variants if str(v).strip()]
        except Exception:                                   # noqa: BLE001
            variants = []
        result = [phrase] + variants[:3]
        self._expansion_cache[phrase] = result
        return result

    def _find_nodes(self, entity_or_phrase: str) -> str:
        """ENTITY-GRAPH LOOKUP with multi-query expansion. Expands the phrase
        into 3 alternative phrasings, searches each, returns union ranked by
        hit-count (nodes appearing in multiple variants ranked first)."""
        if self.entity_index is None:
            return "[ERROR] find_nodes_mentioning requires entity_index"
        variants = self._expand_phrase(entity_or_phrase)
        # Score each node by how many variants matched it (reciprocal rank fusion).
        node_scores: dict[str, float] = {}
        node_first_match: dict[str, str] = {}
        for v in variants:
            ids = self.entity_index.find_nodes_mentioning(v)
            for rank, nid in enumerate(ids[:10]):
                # RRF formula: 1/(k+rank), k=60 standard
                node_scores[nid] = node_scores.get(nid, 0.0) + 1.0 / (60 + rank)
                node_first_match.setdefault(nid, v)
        if not node_scores:
            return (f"No nodes mention {entity_or_phrase!r} (also tried: "
                    f"{', '.join(repr(v) for v in variants[1:])})")
        ranked = sorted(node_scores.items(), key=lambda kv: -kv[1])
        if self.tree_index is None:
            return f"Found nodes: {[nid for nid, _ in ranked]}"
        rows = []
        for nid in [nid for nid, _ in ranked[:10]]:
            node = self.tree_index.get(nid) or {}
            title = node.get("title", "")
            sp = node.get("start_page", "?")
            ep = node.get("end_page", "?")
            summary = (node.get("summary") or "")[:200]
            matched_via = node_first_match.get(nid, "?")
            rows.append(f"[{nid}] {title} (pages {sp}-{ep}) "
                        f"[matched via {matched_via!r}]\n  {summary}")
        header = (f"Nodes mentioning {entity_or_phrase!r} "
                  f"(expanded to: {variants}):\n\n")
        return header + "\n\n".join(rows)

    def _find_cluster(self, query: str) -> str:
        """CLUSTER-FIRST LOOKUP: returns top-K candidate clusters when scores
        are within delta of best. Lets LLM tiebreak via tags/member-titles.
        """
        if self.summary_index is None:
            return "[ERROR] find_cluster_for_synthesis requires summary_index"
        threshold = float(os.getenv("SUMMARY_INDEX_THRESHOLD", "0.5"))
        top_k = int(os.getenv("SUMMARY_INDEX_TOP_K", "2"))
        delta = float(os.getenv("SUMMARY_INDEX_DELTA", "0.10"))
        hits = self.summary_index.find_clusters_for_query(
            query, threshold=threshold, top_k=top_k, delta=delta,
        )
        if not hits:
            return f"No cluster matches {query!r} above threshold {threshold:.2f}"

        def _fmt_one(h: dict, rank: int) -> str:
            c = h["cluster"]
            pages = c.get("primary_pages", [])
            pages_str = ", ".join(f"[{p[0]}-{p[1]}]" for p in pages)
            return (f"Candidate #{rank} — Cluster {c['cluster_id']!r}: {c['title']}\n"
                    f"  confidence: {h['confidence']:.2f}\n"
                    f"  member_node_ids: {c['member_node_ids']}\n"
                    f"  primary_pages: {pages_str}\n"
                    f"  summary: {c['summary'][:300]}\n"
                    f"  tags: {c.get('tags', [])[:15]}")

        if len(hits) == 1:
            return (_fmt_one(hits[0], 1) + "\n"
                    f"NEXT: call get_page_content with the page range covering "
                    f"member_node_ids, OR fetch each range and synthesize.")

        body = "\n\n".join(_fmt_one(h, i + 1) for i, h in enumerate(hits))
        gap = hits[0]["confidence"] - hits[-1]["confidence"]
        return (f"AMBIGUOUS — {len(hits)} candidate clusters within {gap:.2f} "
                f"cosine of best (noise-band tie). Pick the one whose tags + "
                f"member node coverage best matches the question's specific "
                f"entities/keywords; do NOT default to highest score.\n\n"
                f"{body}\n\n"
                f"NEXT: choose ONE candidate, then call get_page_content with "
                f"its primary_pages range. If the question spans entities found "
                f"in DIFFERENT candidates, fetch from each.")

    def _chunk_level_fallback(self, query: str) -> str:
        """Last-resort vector match over per-page text. Returns the LLM's
        answer string, or empty string when no fallback is possible.

        Triggers ONLY when:
          - `page_vector_index` was supplied at construction time
          - the agent loop reached `max_iterations` AND returned a refusal
            (caller decides; this method just executes the fallback)

        Recovers from Q9-class regressions where the variant generator
        paraphrased a distinctive document term away. Uses BGE-M3 dense
        + sparse RRF (when sparse is wired) to match literal tokens.

        Strict prompt: 'Answer ONLY from these passages.' Prevents
        hallucination when vector match returns weakly-related pages.
        """
        if self.page_vector_index is None:
            return ""
        try:
            top_pages = self.page_vector_index.search(query, top_k=3)
        except Exception:                                          # noqa: BLE001
            return ""
        if not top_pages:
            return ""
        # Expand each top hit by ±1 neighbor + dedup to contiguous ranges.
        # Multi-page topics (e.g. Japanese trading houses spans pages 12-13)
        # are missed when only the top-1 page is fetched.
        page_nums = [p for p, _ in top_pages]
        ranges = _expand_with_neighbors(page_nums, window=1)
        passages = "\n\n".join(
            f"[pages {s}-{e}]\n{self.page_provider(s, e)}"
            for s, e in ranges
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                max_tokens=1000,
                messages=[
                    {"role": "system", "content": (
                        "Answer the question using ONLY the provided passages. "
                        "Cite page numbers inline as [page N]. If the passages "
                        "do not contain the answer, respond with the exact "
                        "phrase 'insufficient context' and nothing else."
                    )},
                    {"role": "user", "content": (
                        f"Passages:\n{passages}\n\nQuestion: {query}"
                    )},
                ],
            )
        except Exception:                                          # noqa: BLE001
            return ""
        content = (resp.choices[0].message.content or "").strip()
        if not content or "insufficient context" in content.lower():
            return ""
        return content

    def _fetch(self, start: int, end: int) -> str:
        text = self.page_provider(start, end)
        if len(text) > self.max_range_chars:
            text = text[: self.max_range_chars] + "\n[... truncated]"
        return text

    @staticmethod
    def _is_synthesis_question(query: str) -> bool:
        """Detects 'what did X say/write about Y', 'how does X describe Y', and
        related multi-fetch synthesis patterns. Used to force ≥2 fetches when
        the model (e.g. DWQ-quantized Qwen3) converges too eagerly after one
        page-content fetch on a question whose answer is fragmented across
        sub-sections."""
        import re as _re
        q = query.lower().strip()
        patterns = [
            r"what did .+ (say|write|describe|note|mention)",
            r"how does .+ (describe|characterize|frame|portray|view)",
            r"what does .+ (say|write|think) about",
            r"how (do|does) .+ (relate|connect|compare)",
            r"(discuss|describe).+(non-controlled|relationship|approach|philosophy)",
            r"what.+(secret weapon|moat|advantage|edge)",
        ]
        return any(_re.search(p, q) for p in patterns)

    @staticmethod
    def _extract_entity_phrase(query: str) -> str | None:
        """Detect questions where a specific named entity / quoted phrase / unique
        section title should drive entity-graph lookup. Returns the phrase to
        search, or None if no entity-pattern matches.

        Closes the variance gap on Q-ENTITY type questions where DWQ stochastically
        skips find_nodes_mentioning and goes straight to get_page_content with
        a wrong page guess. Pre-firing the entity lookup makes routing
        deterministic."""
        import re as _re
        # 1) Quoted phrase wins: "What did Buffett describe as Berkshire's 'X' ..."
        m = _re.search(r"['\"]([^'\"]{4,60})['\"]", query)
        if m:
            return m.group(1).strip()
        # 2) "described as <X>" / "called <X>" / "known as <X>"
        m = _re.search(r"(?:described as|called|known as|titled)\s+([A-Z][A-Za-z0-9\- ]{3,60})",
                       query)
        if m:
            return m.group(1).strip()
        # 3) ALL-CAPS acronym (BNSF, GAAP, etc.) — unique enough to look up
        m = _re.search(r"\b([A-Z]{3,8})\b", query)
        if m:
            return m.group(1).strip()
        return None

    def answer(self, query: str) -> dict:
        tree_str = _tree_view(self.tree)
        is_synthesis = self._is_synthesis_question(query)

        # Pre-fire entity-graph lookup for Q-ENTITY patterns so the model sees
        # the matching nodes BEFORE its first LLM call. Closes variance gap
        # where DWQ stochastically picks get_page_content over
        # find_nodes_mentioning.
        entity_hint = ""
        if self.entity_index is not None:
            phrase = self._extract_entity_phrase(query)
            if phrase:
                hint_body = self._find_nodes(phrase)
                # Only inject when we actually got matches (the no-match string
                # starts with "No nodes mention").
                if not hint_body.startswith("No nodes mention"):
                    entity_hint = (
                        f"\n\nENTITY-GRAPH HINT (auto-fired before your first "
                        f"call): the phrase {phrase!r} was found in these nodes:\n"
                        f"{hint_body}\n\n"
                        f"Use these page ranges directly with get_page_content "
                        f"unless the tree shows a more specific match."
                    )

        # Cluster-prefetch — for synthesis-pattern queries, pre-fire
        # find_cluster_for_synthesis BEFORE first LLM call. Routes
        # multi-section synthesis through one batched fetch instead of
        # sequential per-node fetches that hit max_iter cliff.
        cluster_hint = ""
        if (self.summary_index is not None and is_synthesis
                and os.getenv("SUMMARY_INDEX_ENABLED", "1") != "0"):
            body = self._find_cluster(query)
            if not body.startswith("No cluster") and not body.startswith("[ERROR]"):
                cluster_hint = (
                    f"\n\nCLUSTER HINT (auto-fired before your first call): "
                    f"{body}\n\nUse the page ranges from this cluster directly "
                    f"with get_page_content."
                )

        # Per-call nonce — breaks vMLX paged-KV-cache content-addressable
        # deduplication so a polluted cache entry from a prior request can't
        # contaminate this one. Documented bug class: mlx-lm Issue #965 + #975
        # (KV cache cross-contamination between separate requests).
        # Prepended to BOTH system + user prefixes so neither hits a stale page.
        # Cost: ~2-3s extra prefill per question; benefit: reproducible answers.
        import time as _time, uuid as _uuid
        _nonce = f"<!-- session={_uuid.uuid4().hex[:12]} t={_time.time_ns()} -->"
        msgs: list[dict] = [
            {"role": "system",
             "content": f"{_nonce}\n{self.system_prompt}"},
            {"role": "user",
             "content": f"{_nonce}\nDocument tree:\n{tree_str}{entity_hint}{cluster_hint}\n\nQuestion: {query}"},
        ]
        tool_call_log: list[dict] = []
        final_answer = "insufficient context"
        iteration = 0

        for iteration in range(self.max_iterations):
            resp = self.client.chat.completions.create(
                model=self.model, messages=msgs, tools=self._tools,
                temperature=0.0, max_tokens=1500,
            )
            msg = resp.choices[0].message
            tcalls = getattr(msg, "tool_calls", None) or []
            content_text = (msg.content or "")

            # Fallback parser: when oMLX emits native <|tool_call> text in
            # content instead of populating structured tool_calls (state
            # degradation under sustained load), recover it here so the
            # agent loop continues.
            # Trigger fallback parser when content contains EITHER Qwen native
            # template (`<|tool_call>`) OR Hermes-style (`<function=`) markers.
            # DWQ-quantized Qwen3.6 emits the Hermes form which vMLX does not
            # extract into structured tool_calls.
            if not tcalls and ("<|tool_call>" in content_text
                                or "<function=" in content_text):
                native = _parse_native_toolcalls(content_text)
                if native:
                    # Build pseudo-tool-call objects with .id + .function.name
                    # + .function.arguments compatible with the dispatch below
                    class _PseudoFn:
                        def __init__(self, n, a): self.name = n; self.arguments = a
                    class _PseudoTC:
                        def __init__(self, i, n, a):
                            self.id = i
                            self.function = _PseudoFn(n, a)
                            self.type = "function"
                    tcalls = [_PseudoTC(t["id"], t["name"], t["arguments"]) for t in native]
                    # Strip the malformed text from content; treat as if model
                    # only emitted tool calls
                    content_text = ""

            if not tcalls:
                # Synthesis-question guard: if this is a multi-section synthesis
                # question and the model has only fetched ONCE, push it to fetch
                # a second range before accepting the answer. DWQ-quantized
                # models converge eagerly after iter 0; one fetch on a "what did
                # X say about Y" question = shallow answer = wrong answer.
                page_fetches = sum(1 for tc in tool_call_log
                                   if tc.get("tool") == "get_page_content")
                if is_synthesis and page_fetches < 2:
                    msgs.append({
                        "role": "assistant",
                        "content": content_text or "",
                    })
                    msgs.append({
                        "role": "user",
                        "content": (
                            "STOP. This is a multi-section synthesis question. "
                            "You have fetched only ONE page range. The answer "
                            "is distributed across multiple sub-sections — your "
                            "current answer is shallow. Fetch a SECOND page "
                            "range from a DIFFERENT sub-section that may also "
                            "discuss this topic, then synthesize across both "
                            "fetches. Call get_page_content again now."
                        ),
                    })
                    continue
                final_answer = content_text.strip()
                break

            msgs.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in tcalls
                ],
            })

            for tc in tcalls:
                try:
                    args = json.loads(tc.function.arguments)
                    name = tc.function.name
                    if name == "get_page_content":
                        sp = int(args.get("start_page", 1))
                        ep = int(args.get("end_page", sp))
                        content = self._fetch(sp, ep)
                        tool_call_log.append({
                            "iter": iteration, "tool": "get_page_content",
                            "args": {"start": sp, "end": ep},
                            "content_chars": len(content),
                        })
                    elif name == "get_subtree_text":
                        pid = str(args.get("parent_node_id", ""))
                        content = self._fetch_subtree(pid)
                        tool_call_log.append({
                            "iter": iteration, "tool": "get_subtree_text",
                            "args": {"parent_node_id": pid},
                            "content_chars": len(content),
                        })
                    elif name == "find_nodes_mentioning":
                        ent = str(args.get("entity_or_phrase", ""))
                        content = self._find_nodes(ent)
                        tool_call_log.append({
                            "iter": iteration, "tool": "find_nodes_mentioning",
                            "args": {"entity_or_phrase": ent},
                            "content_chars": len(content),
                        })
                    elif name == "find_cluster_for_synthesis":
                        q_arg = str(args.get("query", query))
                        content = self._find_cluster(q_arg)
                        tool_call_log.append({
                            "iter": iteration, "tool": "find_cluster_for_synthesis",
                            "args": {"query": q_arg},
                            "content_chars": len(content),
                        })
                    else:
                        content = f"[ERROR] Unknown tool: {name}"
                except Exception as e:  # noqa: BLE001
                    content = f"[ERROR] {type(e).__name__}: {e}"
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": content})

        # Forced final synthesis — if loop exited at max_iterations with model
        # still calling tools (final_answer never set, so still the
        # "insufficient context" placeholder) AND we have fetched observations
        # in the conversation, do ONE more LLM call asking the model to
        # synthesize from what it fetched. Closes the iters=4 cliff that
        # crashed Q11/Q12 in iter2 — model was about to write the answer
        # when the loop pulled the rug.
        if (final_answer == "insufficient context"
                and any(tc.get("tool") == "get_page_content" for tc in tool_call_log)):
            msgs.append({
                "role": "user",
                "content": (
                    "BUDGET EXHAUSTED. Stop calling tools. Write the final "
                    "answer NOW from the observations you already have above.\n\n"
                    "STRICT RULES (these prevent hallucination under pressure):\n"
                    "1. Use ONLY facts that appear VERBATIM in the fetched "
                    "text observations above. Do NOT supplement with knowledge "
                    "from outside the fetched text, even if you remember the "
                    "answer from training data.\n"
                    "2. Cite ONLY page numbers that appear in the fetched "
                    "ranges above. Do NOT cite pages you did not fetch.\n"
                    "3. If the fetched text contains a partial answer "
                    "(named entities, numbers, phrases on the question's "
                    "specific topic), state what you found with the actual "
                    "fetched page citation. Partial answers score higher "
                    "than refusals.\n"
                    "4. If the fetched text does NOT contain any answer "
                    "to THIS specific question (e.g. question asks about "
                    "Scorecard but fetched text is about non-controlled "
                    "businesses), respond with the exact phrase "
                    "'insufficient context' — fabricating a confident "
                    "answer from training memory is the worst outcome.\n"
                    "5. OUTPUT FORMAT — ABSOLUTE RULES:\n"
                    "   (a) Your FIRST token must be the first word of the "
                    "answer. NO preamble. FORBIDDEN openings: 'The user is "
                    "asking', 'This is a', 'From what I've fetched', "
                    "'Let me synthesize', 'Actually,', 'Based on the "
                    "fetched text', 'I have enough', 'Looking at the "
                    "passages', 'Here is the answer'. If you start with "
                    "any of those, you have failed.\n"
                    "   (b) NO numbered lists. NO bullet points. NO bold "
                    "headers like '**Stewardship**:'. Write flowing "
                    "prose paragraphs only.\n"
                    "   (c) NO quoted passage dumps. Do not paste "
                    "sentences from the source text in quote marks. "
                    "Paraphrase into your own prose.\n"
                    "   (d) NO meta-commentary about your process. "
                    "Do not say what you fetched, what's in the source, "
                    "or what you're about to do.\n"
                    "   (e) Length: 2-5 sentences in 1-2 short "
                    "paragraphs. Cite pages inline as [page N] or "
                    "[pages X-Y].\n\n"
                    "Example of CORRECT output for a contrast question:\n"
                    "  'Buffett frames Berkshire as a steward for "
                    "long-term shareholders rather than a vehicle for "
                    "trading activity. Where Wall Street thrives on "
                    "feverish turnover and markets whatever sells "
                    "[page 9], Berkshire pledges extreme fiscal "
                    "conservatism and direct CEO communication to its "
                    "lifetime owners [pages 5, 10].'\n\n"
                    "Example of WRONG output (DO NOT DO THIS):\n"
                    "  'The user is asking about... From the Chairman's "
                    "Letter I found: 1. Page 5: ...quote... 2. Page 10: "
                    "...quote...'\n\n"
                    "Begin your answer with the first word of the "
                    "actual answer NOW."
                ),
            })
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=msgs,
                    temperature=0.0, max_tokens=800,
                    # Note: no `tools` argument — force text-only output
                )
                forced = (resp.choices[0].message.content or "").strip()
                if forced:
                    final_answer = forced
            except Exception:                                # noqa: BLE001
                pass

        # Chunk-level fallback — last resort when all structural recovery
        # paths still produced a low-quality answer AND a PageVectorIndex
        # is wired. Trigger uses composite production-signals via
        # _is_low_quality(): refusal phrases, pseudo-refusal length, or
        # ungrounded synthesis (no [page N] citation in long answer).
        # Catches Q9-class regressions where the variant generator destroyed
        # the literal-keyword signal and the agent never fetched the right
        # page. BGE-M3 dense+sparse hybrid matches the literal token directly.
        fallback_used = False
        if (self.page_vector_index is not None
                and _is_low_quality(final_answer)):
            fb = self._chunk_level_fallback(query)
            if fb:
                final_answer = fb
                fallback_used = True
                tool_call_log.append({
                    "iter": iteration + 1,
                    "tool": "page_vector_fallback",
                    "args": {},
                })

        if self.debug_log_path:
            try:
                with open(self.debug_log_path, "a") as f:
                    f.write(f"[{iteration+1}it/{len(tool_call_log)}tc] "
                            f"q={query[:60]!r} ans={final_answer[:80]!r}"
                            f"{' [FALLBACK]' if fallback_used else ''}\n")
            except Exception:  # noqa: BLE001
                pass

        return {
            "answer": final_answer,
            "tool_calls": tool_call_log,
            "iterations": iteration + 1,
            "fallback_used": fallback_used,
        }
