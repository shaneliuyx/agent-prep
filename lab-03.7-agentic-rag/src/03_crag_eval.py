"""§3.2 — run single-pass, structural RAG, and CRAG on OUT-OF-CORPUS queries.

The MS-MARCO corpus (2018-era passages) cannot answer recent / post-cutoff questions. The
prediction: the corpus-only arms (single-pass, structural) abstain or hallucinate, while CRAG's
evaluator scores corpus retrieval *low* and routes to web search - the regime §2.6 said the
corrective loop needs to earn its cost.

Writes observations/crag-out-of-corpus.md. Override the question set with DEV_SET=/path.jsonl.

    uv run python src/03_crag_eval.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from week3_pipeline import run_single_pass        # corpus-only baseline
from structural_rag import app as structural_app  # corpus-only, always retrieves (§2.5.1)
from crag_variant import crag_app                 # corpus + web fallback (Phase 3)

# 10 questions the 2018 MS-MARCO corpus cannot answer (post-cutoff / niche -> need the web)
OUT_OF_CORPUS = [
    "What is the most capable Claude model Anthropic released in 2026?",
    "Which team won the 2025 NBA Finals?",
    "What is the official release date of GPT-5?",
    "Who won the 2025 Nobel Prize in Physics?",
    "What AI regulation did the European Union pass in 2025?",
    "What is the newest Apple silicon chip announced in 2025?",
    "What is the latest stable version of Python released in 2026?",
    "Who is the CEO of OpenAI as of 2026?",
    "What was the headline feature of the iPhone announced in 2025?",
    "Which country hosted the 2025 G20 summit?",
]

_ABSTAIN = ("don't know", "do not know", "not in the", "no relevant", "cannot find",
            "unable to", "does not contain", "do not contain", "not contain information",
            "no information", "insufficient", "i'm not sure", "not provided", "do not have",
            "does not provide", "do not mention", "no mention", "rewrite loop", "recursion limit")


def answered(text: str) -> bool:
    t = (text or "").lower()
    return not any(p in t for p in _ABSTAIN)


def _structural(q: str) -> str:
    # On out-of-corpus queries the rewrite loop runs away (retrieval never grades relevant ->
    # rewrite -> retrieve -> ...), hitting the recursion limit. That IS the result: rewrite
    # cannot conjure data the corpus lacks. Count it as a (failed) abstain.
    from langgraph.errors import GraphRecursionError
    try:
        return structural_app.invoke({"messages": [("user", q)]},
                                     {"recursion_limit": 12})["messages"][-1].content
    except GraphRecursionError:
        return "(structural rewrite loop hit the recursion limit — no answer)"


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="CRAG vs single-pass vs structural on out-of-corpus")
    ap.add_argument("--dev-set", default=os.getenv("DEV_SET"))
    ap.add_argument("--show", action="store_true",
                    help="print each arm's FULL answer per question (inspect hallucinations)")
    args = ap.parse_args()
    src = args.dev_set
    questions = ([json.loads(l)["question"] for l in open(os.path.expanduser(src)) if l.strip()]
                 if src else OUT_OF_CORPUS)

    rows = []
    for q in questions:
        sp, _ = run_single_pass(q)
        st = _structural(q)
        cr = crag_app.invoke({"question": q})
        rows.append({"q": q, "single_pass": sp, "structural": st,
                     "crag": cr["answer"], "source": cr.get("source", "corpus"),
                     "score": cr.get("score", 0.0)})
        print(f"- {q[:58]:58} | sp:{'ANS ' if answered(sp) else 'abst'} "
              f"st:{'ANS ' if answered(st) else 'abst'} "
              f"crag:{cr.get('source','corpus'):6} (s={cr.get('score',0):.2f}) "
              f"{'ANS' if answered(cr['answer']) else 'abst'}")
        if args.show:
            print(f"    Q: {q}")
            print(f"    single-pass: {sp.strip()}")
            print(f"    structural : {st.strip()}")
            print(f"    CRAG ({cr.get('source','corpus')}): {cr['answer'].strip()}\n")

    web_routed = sum(1 for r in rows if r["source"] in ("web", "both"))
    crag_ans = sum(1 for r in rows if answered(r["crag"]))
    sp_ans = sum(1 for r in rows if answered(r["single_pass"]))
    st_ans = sum(1 for r in rows if answered(r["structural"]))
    n = len(rows)
    print(f"\nout-of-corpus ({n} q): CRAG routed to web {web_routed}/{n} | "
          f"answered  single-pass {sp_ans}/{n}  structural {st_ans}/{n}  CRAG {crag_ans}/{n}")

    obs = Path("observations")
    obs.mkdir(exist_ok=True)
    md = ["# CRAG on out-of-corpus queries (§3.2)", "",
          f"Question set: {'DEV_SET=' + src if src else 'built-in 10 post-cutoff questions'} (n={n}).",
          "",
          f"- CRAG routed to web (Incorrect/Ambiguous): **{web_routed}/{n}**",
          f"- Answered (non-abstaining): single-pass **{sp_ans}/{n}**, structural **{st_ans}/{n}**, "
          f"CRAG **{crag_ans}/{n}**", "",
          "| question | single-pass | structural | CRAG (source) |", "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['q']} | {'answered' if answered(r['single_pass']) else 'abstain'} "
                  f"| {'answered' if answered(r['structural']) else 'abstain'} "
                  f"| {'answered' if answered(r['crag']) else 'abstain'} "
                  f"via {r['source']} (s={r['score']:.2f}) |")
    (obs / "crag-out-of-corpus.md").write_text("\n".join(md) + "\n")
    print(f"wrote {obs / 'crag-out-of-corpus.md'}")


if __name__ == "__main__":
    main()
