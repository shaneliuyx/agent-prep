"""EnsembleTreeRetriever — best-of-both v1 + v2 with LLM synthesis at the end.

Architecture:
  query
    ├─ Path A (v1, greedy nav): runs AgenticTreeRetriever with
    │                           AGENTIC_SYSTEM_TEMPLATE only
    └─ Path B (v2, entity-graph + auto-merge): runs AgenticTreeRetriever
                                with AGENTIC_SYSTEM_TEMPLATE_V2 +
                                tree_index + entity_index
  ↓
  Synthesis node: sees query + Path A answer + Path B answer →
                  LLM picks the better one (or combines if both contribute)

Why this beats either alone:
  - Path A excels at title-literal-match factoid (Q1/Q2 in W2.7 eval)
  - Path B excels at entity lookup + cross-section synthesis (Q4/Q5/Q6)
  - Synthesis node lets the LLM choose per-question

Cost: 2 retrieval passes + 1 synthesis call per question. Latency ~v1 + v2 + ~3s.
Worth it when each path's failure mode is non-overlapping with the other.

Reference: ensemble RAG / mixture-of-retrievers pattern. Closest published
analog is HyDE-then-vector + RAGAS comparison harnesses that pick best.
"""
from __future__ import annotations

from typing import Any

from .agentic import AgenticTreeRetriever, PageProvider
from .prompts import (
    AGENTIC_SYSTEM_TEMPLATE,
    AGENTIC_SYSTEM_TEMPLATE_V2,
)


ENSEMBLE_SYNTHESIS_PROMPT = """Two retrieval paths produced answers to the
same question. Pick the better answer, or combine them if both contribute
distinct evidence.

Question: {query}

Path A (greedy tree-walk navigator):
{a_answer}

Path B (entity-graph + auto-merge subtree fetcher):
{b_answer}

DECISION RULES:
- If one path says "insufficient context" or returns empty, and the other
  has a substantive answer with cited page numbers, pick the substantive one.
- If both have substantive answers and they AGREE on the key fact, return the
  one with the cleaner citation (prefer "[pages X-Y]" style inline citation).
- If both have substantive answers and they DISAGREE, pick the one with
  stronger evidence (specific numbers, exact entity names, explicit page
  citation).
- If both refuse and the question is genuinely out-of-document, return one
  refusal (preferring the one that explains *what the document is*).
- Do NOT add information that isn't in either path's answer.
- Preserve the inline [pages X-Y] citations from the chosen path.

Final answer:"""


class EnsembleTreeRetriever:
    """Runs v1 + v2 retrieval paths, synthesizes the final answer with one LLM call.

    Public method: `answer(query) -> {answer, v1_*, v2_*, synthesis_lat}`.

    Constructor takes the same args as `AgenticTreeRetriever` plus tree_index +
    entity_index (required — the v2 path needs them; v1 path uses only `tree`).
    """

    def __init__(
        self, *,
        tree: dict,
        page_provider: PageProvider,
        model_client,
        model_name: str,
        tree_index,                   # required
        entity_index,                 # required
        synthesis_model: str | None = None,  # default = model_name (single-model setup)
        max_iterations: int = 6,
        max_range_chars: int = 8000,
        debug_log_path: str | None = None,
        v1_system_prompt: str = AGENTIC_SYSTEM_TEMPLATE,
        v2_system_prompt: str = AGENTIC_SYSTEM_TEMPLATE_V2,
        synthesis_prompt: str = ENSEMBLE_SYNTHESIS_PROMPT,
    ) -> None:
        self.client = model_client
        self.model = model_name
        # Synthesis can route to a deeper model (e.g., 35B) while retrievers
        # use a faster model (e.g., 9B distill). Split set in run_one_variant.
        self.synthesis_model = synthesis_model or model_name
        self.synthesis_prompt = synthesis_prompt

        # Path A — greedy v1, no v2 tools
        self.v1 = AgenticTreeRetriever(
            tree=tree, page_provider=page_provider,
            model_client=model_client, model_name=model_name,
            system_prompt=v1_system_prompt,
            max_iterations=max_iterations,
            max_range_chars=max_range_chars,
            debug_log_path=debug_log_path,
        )

        # Path B — v2 with entity-graph + auto-merge
        self.v2 = AgenticTreeRetriever(
            tree=tree, page_provider=page_provider,
            model_client=model_client, model_name=model_name,
            system_prompt=v2_system_prompt,
            max_iterations=max_iterations,
            max_range_chars=max_range_chars,
            debug_log_path=debug_log_path,
            tree_index=tree_index,
            entity_index=entity_index,
        )

    def answer(self, query: str) -> dict[str, Any]:
        import time
        t0 = time.time()
        v1_out = self.v1.answer(query)
        t1 = time.time()
        v2_out = self.v2.answer(query)
        t2 = time.time()

        # Skip synthesis if both produced identical answers
        a, b = v1_out["answer"], v2_out["answer"]
        if a.strip() == b.strip() and a.strip():
            final = a
            synth_lat = 0.0
        else:
            prompt = self.synthesis_prompt.format(query=query, a_answer=a, b_answer=b)
            t_synth = time.time()
            r = self.client.chat.completions.create(
                model=self.synthesis_model, temperature=0.0, max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            final = (r.choices[0].message.content or "").strip()
            synth_lat = time.time() - t_synth

        return {
            "answer": final,
            "v1_answer": a,
            "v2_answer": b,
            "v1_iters": v1_out.get("iterations", 0),
            "v2_iters": v2_out.get("iterations", 0),
            "v1_tools": [tc["tool"] for tc in v1_out.get("tool_calls", [])],
            "v2_tools": [tc["tool"] for tc in v2_out.get("tool_calls", [])],
            "v1_lat": round(t1 - t0, 2),
            "v2_lat": round(t2 - t1, 2),
            "synth_lat": round(synth_lat, 2),
            "tool_calls": (v1_out.get("tool_calls", []) +
                           v2_out.get("tool_calls", [])),
            "iterations": v1_out.get("iterations", 0) + v2_out.get("iterations", 0),
        }
