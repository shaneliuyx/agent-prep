"""W3.5.95 — ablation: does the LEARNING self-attribution leak come from the PROMPT
or the MODEL? (Validates BCJ Entry 1's claim that the filter is bounded by the
summarizer's JUDGMENT, not the code.)

2×2: {7B, 14B} × {current prompt, stronger prompt}. For each arm: seed 35 fresh
OBSERVABILITY rows (18 self / 8 environmental / 9 noise), extract, then count how
many KEPT facts are environmental (should be 0) and how many genuine self-patterns
survived (should stay ~4 — a too-aggressive filter that nukes real facts is also a
failure). Run `uv run python scripts/ablation_filter.py`.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")

import learning_extractor as le  # noqa: E402
import observability as obs  # noqa: E402
import seed_observability as seed  # noqa: E402

# Stronger prompt: explicit counterfactual test, environmental exemplars, a
# self-action-verb requirement, and a conservative "when unsure → false" default.
STRONGER_PROMPT = """You are a self-pattern extractor in an agent's memory pipeline. Input: OBSERVABILITY rows — the agent's own tool calls with outcomes. Output: typed facts about the AGENT'S OWN behavioral patterns. Data-processing task; do not describe yourself.

Each fact is a JSON object:
  {"type": one of [failure_pattern, success_pattern, tool_preference, recurring_mistake],
   "pattern_text": one sentence naming a CONTROLLABLE ACTION the agent took ("I keep choosing ...", "I forget to ..."),
   "confidence": 0.0-1.0,
   "is_self_caused": true|false}

THE SELF-ATTRIBUTION TEST (apply to every candidate before emitting):
  Counterfactual: would a DIFFERENT agent making BETTER choices STILL hit this outcome?
    - YES → the outcome is ENVIRONMENTAL, not the agent's doing → is_self_caused=false.
    - NO, it stems from THIS agent's tool choice / argument / ignored context → is_self_caused=true.

ENVIRONMENTAL (is_self_caused=false — these are NOT self-patterns, do not frame them as "I keep..."):
  - server-side errors: HTTP 500/502/503, "connection refused", "network was down", provider outage
  - infrastructure: database unreachable, disk full, rate-limited by the service
  Any caller would hit these regardless of skill. They are facts about the WORLD, not the agent.

SELF-CAUSED (is_self_caused=true):
  - choosing a tool that predictably fails on this input ("I keep running grep on huge repos and it times out")
  - malformed arguments, ignoring data already available, repeating a known-bad approach

HARD RULES:
  - pattern_text MUST name an action the agent controls (a choice/habit/mistake). If the sentence describes something that HAPPENED TO the agent, it is environmental → is_self_caused=false.
  - When uncertain, set is_self_caused=false. Poisoning self-memory with environmental noise is worse than missing one self-pattern.
  - Only emit a pattern that RECURS or is a clear actionable lesson. No one-fact-per-row, no trivia.

Output ONLY a JSON array of fact objects. No prose.

OBSERVABILITY ROWS:
{rows}
"""

# Keywords that mark a KEPT fact as actually-environmental (leaked). The two
# environmental seed templates are "database connection refused / network down"
# and "provider returned HTTP 500".
_ENV_MARKERS = ("500", "connection", "network", "refused", "provider", "http")


def _classify(conn) -> tuple[int, int, list[str]]:
    """Return (env_leaked, self_kept, all_kept_texts) for the current LEARNING table."""
    env_leaked = self_kept = 0
    texts = []
    for r in conn.execute("SELECT pattern_text FROM learning"):
        t = r["pattern_text"]; texts.append(t)
        if any(m in t.lower() for m in _ENV_MARKERS):
            env_leaked += 1
        else:
            self_kept += 1
    return env_leaked, self_kept, texts


REPS = 5  # the judgment is nondeterministic — measure leak FREQUENCY, not one run


def _run_arm(label: str, model: str, prompt: str | None, reps: int = REPS) -> None:
    leak_counts, self_counts, runs_with_leak = [], [], 0
    for _ in range(reps):
        seed.main()  # fresh 35 rows (DELETEs observability first)
        conn = obs.connect(str(seed.DB))
        conn.execute("DELETE FROM learning")  # isolate this run
        conn.commit()
        le.extract(conn, since_n=200, model=model, prompt_template=prompt)
        env_leaked, self_kept, _ = _classify(conn)
        leak_counts.append(env_leaked); self_counts.append(self_kept)
        runs_with_leak += (env_leaked > 0)
    total_leak = sum(leak_counts)
    print(f"\n=== {label} ===  (model={model.split('-MLX')[0]}, n={reps})")
    print(f"  runs that leaked ≥1 env fact: {runs_with_leak}/{reps}")
    print(f"  total env facts leaked:       {total_leak}  per-run {leak_counts}")
    print(f"  self-patterns kept (mean):    {sum(self_counts)/reps:.1f}  per-run {self_counts}")


def main() -> None:
    import os
    seven = os.getenv("MODEL_EXTRACTOR", "Qwen2.5-Coder-7B-Instruct-MLX-4bit")
    fourteen = os.getenv("MODEL_AGENT", "Qwen2.5-Coder-14B-Instruct-MLX-4bit")
    print(f"Ablation: self-attribution leak vs {{model}} × {{prompt}}, n={REPS} runs/arm")
    print("(env-leaked should be 0; self-patterns-kept should stay ~4 — over-aggressive")
    print(" filtering that nukes real self-patterns is also a failure)")
    _run_arm("A  7B  + current prompt (baseline)", seven, None)
    _run_arm("B  7B  + stronger prompt", seven, STRONGER_PROMPT)
    _run_arm("C  14B + current prompt", fourteen, None)
    _run_arm("D  14B + stronger prompt", fourteen, STRONGER_PROMPT)


if __name__ == "__main__":
    main()
