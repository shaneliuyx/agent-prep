"""Production runbook MVP — atomisation router with 4 mechanical invariants.

Skips Layer 1 (declarative tags) — empirically unrealistic without pre-existing
metadata. Starts at Layer 1.5 (speaker-role auto-tag) + Layer 2 (schema-shape
classifier) and enforces the four invariants from W3.5.8 §5.3.4 + production
runbook:

  (a) ALWAYS preserve raw alongside derived
  (b) ENFORCE K_min >= 8 triples or DROP derived
  (c) NEVER position derived facts above raw
  (d) DEPLOY only after A/B beats raw-only baseline (DeploymentGate)

Cross-stage symmetry note: the same K_min guardrail catches both
write-time SUMMARIZE collapse (BCJ 16) and read-time constrained-atomise
collapse (BCJ 18) because the failure mechanism is identical.

This module is reference architecture, not full production code.
Production deployments need: per-tenant scoping, rate limiting, observability,
async writes, schema migrations, error budgets — none of which are MVP scope.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol


K_MIN_DEFAULT = 8


# ── Layer 1.5 — speaker-role auto-tag ──────────────────────────────

class SpeakerRole(str, Enum):
    USER = "user"            # → preferences extractor (typed schema, write-time)
    ASSISTANT = "assistant"  # → commitments extractor (audit log)
    SYSTEM = "system"        # → rules extractor (durable, write-time)
    TOOL = "tool"            # → typed outputs (structured schema, write-time)
    UNKNOWN = "unknown"      # → defer to Layer 2


# ── Layer 2 — schema-shape classifier (heuristic, no LLM) ──────────

class WorkloadShape(str, Enum):
    CONVERSATION = "conversation"    # heterogeneous queries → read-time atomise
    LOG_STREAM = "log_stream"        # uniform queries → write-time atomise + index
    STRUCTURED_RECORD = "structured" # ACID-eligible → typed schema, no atomise
    UNKNOWN = "unknown"              # → Layer 3 default (read-time)


@dataclass(frozen=True)
class ClassificationResult:
    shape: WorkloadShape
    confidence: float            # 0..1
    speaker_roles_present: tuple[SpeakerRole, ...]
    features: dict[str, float]   # diagnostic — what the classifier saw


# Regex patterns are deliberate, documented, and conservative.
# Conservative = false negatives OK (fall back to Layer 3 default);
# false positives NOT ok (would route data to wrong tier).
TURN_MARKER_RE = re.compile(
    r"^\s*(user|assistant|system|tool)\s*:", re.IGNORECASE | re.MULTILINE
)
JSON_KEY_RE = re.compile(r'"\w+"\s*:\s*[\["\d{tn]')   # crude JSON-key signature
TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?(Z|[+-]\d{2}:?\d{2})?)?\b"
)


def classify_workload(content: str) -> ClassificationResult:
    """Layer 2 schema-shape heuristic. Returns confidence-scored decision.

    Conservative — when in doubt, return UNKNOWN so the router falls
    back to Layer 3 (read-time atomise default). Misclassification to
    write-time is more expensive than misclassification to read-time
    because late-binding is reversible (re-atomise later); early-binding
    is not.
    """
    if not content or len(content) < 8:
        return ClassificationResult(
            WorkloadShape.UNKNOWN, 0.0, (), {"empty": 1.0}
        )

    # Feature extraction
    turn_markers = TURN_MARKER_RE.findall(content)
    json_keys = JSON_KEY_RE.findall(content)
    timestamps = TIMESTAMP_RE.findall(content)
    lines = content.splitlines()
    n_lines = max(len(lines), 1)
    avg_line_len = sum(len(l) for l in lines) / n_lines

    turn_density = len(turn_markers) / n_lines
    json_density = len(json_keys) / n_lines
    ts_density = len(timestamps) / n_lines

    features = {
        "turn_density": turn_density,
        "json_density": json_density,
        "ts_density": ts_density,
        "avg_line_len": avg_line_len,
        "n_lines": float(n_lines),
    }

    speakers = tuple({
        SpeakerRole(m.lower()) for m in turn_markers
        if m.lower() in SpeakerRole._value2member_map_
    })

    # Decision rules — deliberately simple, each rule documented.
    #
    # Rule 1: >= 2 DISTINCT speakers seen (user + assistant, or any pair)
    # → conversation. Strong signal because dialogue requires at least two
    # participants. Robust to multi-line message content (one marker per
    # turn but many lines per turn body) — counts speakers, not lines.
    # Previous version used turn_density >= 0.15 which broke on messages
    # whose content spans 10-50 lines per turn.
    if len(speakers) >= 2:
        return ClassificationResult(
            WorkloadShape.CONVERSATION, 0.9, speakers, features
        )
    # Single-speaker monologue with turn markers (e.g., assistant-only
    # synthetic data) — still conversational. Lower confidence.
    if len(speakers) == 1 and len(turn_markers) >= 1:
        return ClassificationResult(
            WorkloadShape.CONVERSATION, 0.8, speakers, features
        )

    # Rule 2: high JSON-key density AND high timestamp density AND
    # short uniform lines → log stream. Three signals must AGREE because
    # any one alone is too noisy (JSON could be config, timestamps could
    # be dialogue references, short lines could be poetry).
    if json_density >= 0.5 and ts_density >= 0.5 and avg_line_len < 200:
        return ClassificationResult(
            WorkloadShape.LOG_STREAM, 0.9, speakers, features
        )

    # Rule 3: pure JSON with few/no timestamps → structured record.
    # Caveat: this routes "user profile JSON" correctly but not
    # "JSON event log without timestamps" — for the latter, fall to UNKNOWN.
    if json_density >= 0.7 and ts_density < 0.2:
        return ClassificationResult(
            WorkloadShape.STRUCTURED_RECORD, 0.85, speakers, features
        )

    # Fall-through: UNKNOWN. Layer 3 will default to read-time atomise.
    return ClassificationResult(
        WorkloadShape.UNKNOWN, 0.4, speakers, features
    )


# ── Invariants (b) and (c) — SafeAtomiser ──────────────────────────

@dataclass(frozen=True)
class AtomiseResult:
    triples: list[str]
    dropped_below_k_min: bool   # True if extractor returned fewer than K_min
    n_emitted: int               # before drop decision
    k_min: int


class Atomiser(Protocol):
    """Anything that turns raw context into a list of triples."""
    def __call__(self, ctx: str) -> list[str]: ...


def safe_atomise(
    atomiser: Atomiser,
    ctx: str,
    k_min: int = K_MIN_DEFAULT,
) -> AtomiseResult:
    """Run extractor and enforce K_min volume floor.

    If extractor returns fewer than k_min triples for a non-trivial input,
    return empty triples — caller MUST fall back to raw-only. This is the
    mechanical guardrail for W3.5.8 §5.3.4: −30 to −35pt regression
    measured when this rule is violated.
    """
    raw_triples = atomiser(ctx)
    n = len(raw_triples)
    if n < k_min:
        return AtomiseResult(
            triples=[], dropped_below_k_min=True, n_emitted=n, k_min=k_min
        )
    return AtomiseResult(
        triples=raw_triples, dropped_below_k_min=False, n_emitted=n, k_min=k_min
    )


def build_composer_prompt(
    question: str,
    raw_ctx: str,
    triples: list[str],
) -> str:
    """Assemble the composer prompt with the position-discipline invariant:
    RAW context FIRST, derived facts LAST with neutral framing.

    Empirical evidence (§5.3.3 v5/v7): inverting this order contributed
    to a −30pt regression. Composers anchor on early-prompt structured
    content; raw context positioned below derived facts is functionally
    ignored even when present.
    """
    parts = [f"Question: {question}", "", "Original context:", raw_ctx]
    if triples:
        parts += ["", "Atomic facts (structured):", "\n".join(triples)]
    return "\n".join(parts)


# ── Invariant (d) — DeploymentGate (A/B accuracy comparison) ──────

@dataclass(frozen=True)
class GateResult:
    raw_only_acc: float
    raw_plus_facts_acc: float
    delta_pts: float
    threshold_pts: float
    ship: bool                   # True if extractor strictly improves
    n_eval: int


def deployment_gate(
    eval_items: list[dict],
    raw_pipeline: Callable[[dict], bool],     # returns True if answer correct
    raw_plus_facts_pipeline: Callable[[dict], bool],
    threshold_pts: float = 3.0,
) -> GateResult:
    """Mechanical A/B check: ship extractor only if it strictly improves
    composer accuracy by >= threshold pts vs raw-only baseline.

    Caller provides the two pipelines as closures over their own
    retrieval/composer stack — this function only does the A/B math.

    Production rule (W3.5.8 Production Runbook): if raw_plus_facts_acc
    <= raw_only_acc + threshold, REJECT the extractor — it's poisoning,
    not helping. This is the single highest-ROI mechanical gate; most
    production memory systems skip it and ship extractors that cost
    accuracy.
    """
    n = len(eval_items)
    if n == 0:
        return GateResult(0.0, 0.0, 0.0, threshold_pts, False, 0)

    raw_correct = sum(1 for q in eval_items if raw_pipeline(q))
    rpf_correct = sum(1 for q in eval_items if raw_plus_facts_pipeline(q))

    raw_acc = raw_correct / n
    rpf_acc = rpf_correct / n
    delta = (rpf_acc - raw_acc) * 100  # in percentage points

    return GateResult(
        raw_only_acc=raw_acc,
        raw_plus_facts_acc=rpf_acc,
        delta_pts=delta,
        threshold_pts=threshold_pts,
        ship=delta >= threshold_pts,
        n_eval=n,
    )


# ── Router orchestration (Layers 1.5 → 2 → 3) ──────────────────────

@dataclass(frozen=True)
class RoutingDecision:
    shape: WorkloadShape
    lifecycle: str               # "write_time" | "read_time"
    tier: str                    # "guild" | "qdrant" | "typed_schema"
    confidence: float
    speakers: tuple[SpeakerRole, ...]
    rationale: str


def route(content: str, declared_speaker: SpeakerRole | None = None) -> RoutingDecision:
    """Three-layer router (declarative tag layer SKIPPED — empirically
    unrealistic per user instruction).

    Layer 1.5: if declared_speaker is provided AND non-UNKNOWN, route by
        speaker role.
    Layer 2:   else, run schema-shape classifier; route by confident shape.
    Layer 3:   else, default to read-time atomise (conservative —
        late-binding is reversible; early-binding destroys raw signal).
    """
    # Layer 1.5 — speaker-role auto-tag
    if declared_speaker and declared_speaker is not SpeakerRole.UNKNOWN:
        if declared_speaker is SpeakerRole.USER:
            return RoutingDecision(
                WorkloadShape.STRUCTURED_RECORD, "write_time", "guild", 0.95,
                (declared_speaker,),
                "Layer 1.5: user-turn → preferences extractor (typed schema)",
            )
        if declared_speaker is SpeakerRole.SYSTEM:
            return RoutingDecision(
                WorkloadShape.STRUCTURED_RECORD, "write_time", "guild", 0.95,
                (declared_speaker,),
                "Layer 1.5: system-turn → rules extractor (durable)",
            )
        if declared_speaker is SpeakerRole.TOOL:
            return RoutingDecision(
                WorkloadShape.LOG_STREAM, "write_time", "typed_schema", 0.95,
                (declared_speaker,),
                "Layer 1.5: tool-output → typed schema (write-time)",
            )
        # ASSISTANT falls through to Layer 2 — assistant text is heterogeneous

    # Layer 2 — schema-shape classifier
    cls = classify_workload(content)
    if cls.confidence >= 0.85:
        if cls.shape is WorkloadShape.CONVERSATION:
            return RoutingDecision(
                cls.shape, "read_time", "qdrant", cls.confidence,
                cls.speaker_roles_present,
                f"Layer 2: schema-shape conversation (turn_density={cls.features.get('turn_density',0):.2f})",
            )
        if cls.shape is WorkloadShape.LOG_STREAM:
            return RoutingDecision(
                cls.shape, "write_time", "typed_schema", cls.confidence,
                cls.speaker_roles_present,
                f"Layer 2: schema-shape log_stream",
            )
        if cls.shape is WorkloadShape.STRUCTURED_RECORD:
            return RoutingDecision(
                cls.shape, "write_time", "guild", cls.confidence,
                cls.speaker_roles_present,
                f"Layer 2: schema-shape structured_record",
            )

    # Layer 3 — conservative default
    return RoutingDecision(
        WorkloadShape.UNKNOWN, "read_time", "qdrant", 0.5,
        cls.speaker_roles_present,
        "Layer 3: low-confidence classifier → conservative read-time default",
    )
