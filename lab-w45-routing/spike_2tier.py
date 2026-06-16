"""SPIKE (throwaway, LOCAL) — does collapsing tier to {haiku, heavy} clear the bar?

The label audit localized the ceiling to the sonnet<->opus boundary (78% inter-annotator
agreement). Proposal: merge sonnet+opus into one 'heavy' tier. This re-scores the EXISTING
4B predictions under the 2-tier mapping (lower bound — a classifier actually re-trained on
2 labels should do at least this well). If 2-tier tier accuracy jumps past ~0.90, the ceiling
was the taxonomy, not the model — and a 2-tier router is the workable solution.

Run:  OMLX_API_KEY=sk-local-omlx .venv/bin/python spike_2tier.py
"""
from src.probes import load_probes, train_eval_split
from src.router import classify


def merge(t: str) -> str:
    return "haiku" if t == "haiku" else "heavy"  # sonnet, opus -> heavy


def main() -> None:
    _, ev = train_eval_split(load_probes())
    n = len(ev)
    t3 = t2 = mode = 0
    cross = 0  # 4B errors that cross the haiku<->heavy line (merge can't fix these)
    for r in ev:
        v = classify(r["prompt"])
        et = r["expected_tier"]
        t3 += v.tier == et
        t2 += merge(v.tier) == merge(et)
        mode += v.mode == r["expected_mode"]
        if merge(v.tier) != merge(et):
            cross += 1

    print(f"eval rows: {n}")
    print(f"tier 3-way (haiku/sonnet/opus): {t3}/{n} ({t3/n:.2%})")
    print(f"tier 2-way (haiku/HEAVY):       {t2}/{n} ({t2/n:.2%})   <- proposal")
    print(f"per-mode (unchanged):           {mode}/{n} ({mode/n:.2%})")
    print(f"residual cross-line errors (merge can't fix): {cross}/{n}")
    print("targets: tier>=0.85, mode>=0.90")


if __name__ == "__main__":
    main()
