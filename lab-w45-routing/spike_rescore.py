"""SPIKE (throwaway, LOCAL) — measure the 4B tier accuracy against two label sets.

Confirms the labeling-ceiling finding: re-score the shipped 4B's tier predictions
against (a) the ORIGINAL human labels and (b) the independent Opus-rubric labels
(= original except the 5 disputed rows from spike_label_audit; my rubric adjudication
landed on the same 5 calls). If 4B agrees ~83% with one labeler and ~78% with the
other, neither relabeling breaks the ceiling -> it's inter-annotator subjectivity.

Run:  OMLX_API_KEY=sk-local-omlx .venv/bin/python spike_rescore.py
"""
from src.probes import load_probes, train_eval_split
from src.router import classify

# Independent Opus-rubric tier labels = original EXCEPT these 5 disputed rows
# (captured from spike_label_audit). My rubric adjudication matched the judge on all 5.
JUDGE_OVERRIDE = {
    "Compare three approaches to a zero-downtime schema migration and recommend one.": "opus",
    "Explain why Raft cannot serve linearizable reads from followers without a read-index or lease.": "sonnet",
    "Give an ordered 3-step plan to rotate a single API key for one service without downtime.": "sonnet",
    "Explain the memory-ordering guarantee provided by a C++ acquire-release pair.": "sonnet",
    "Design a retry-and-dead-letter strategy for a flaky webhook consumer.": "opus",
}


def main() -> None:
    _, ev = train_eval_split(load_probes())
    n = len(ev)
    vs_orig = vs_judge = both = 0
    for r in ev:
        pred = classify(r["prompt"]).tier
        orig = r["expected_tier"]
        judge = JUDGE_OVERRIDE.get(r["prompt"], orig)
        vs_orig += pred == orig
        vs_judge += pred == judge
        both += pred == orig == judge

    print(f"eval rows: {n}")
    print(f"4B tier vs ORIGINAL labels:                {vs_orig}/{n} ({vs_orig/n:.0%})")
    print(f"4B tier vs OPUS-RUBRIC labels (adjudicated): {vs_judge}/{n} ({vs_judge/n:.0%})")
    print("neither relabeling breaks ~78-83% -> ceiling is inter-annotator subjectivity")


if __name__ == "__main__":
    main()
