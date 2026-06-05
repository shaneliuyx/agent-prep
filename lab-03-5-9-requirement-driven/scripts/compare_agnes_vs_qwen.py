"""Compare agnes-2.0-flash (Agnes AI API) vs Qwen2.5-Coder-14B-Instruct-MLX-4bit
(local oMLX) on general quality + latency.

Quality = pairwise LLM-judge (Sonnet via VibeProxy), judged in BOTH orders to
cancel position bias (a model only "wins" a prompt if it wins or ties both
orders). Latency = wall time per call (a warmup call per model is discarded so
oMLX cold-load doesn't skew the local model).

Run from the lab root WITH the agnes key in env:
    export ANGES_API_KEY=$(grep -m1 '^export ANGES_API_KEY=' ~/.zshrc | sed -E 's/^export ANGES_API_KEY=//; s/^["'"'"']//; s/["'"'"']$//')
    uv run python scripts/compare_agnes_vs_qwen.py
"""
from __future__ import annotations

import os
import pathlib
import statistics
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")

AGNES_BASE = "https://apihub.agnes-ai.com/v1"
AGNES_MODEL = "agnes-2.0-flash"
QWEN_BASE = os.getenv("OMLX_BASE_URL", "http://localhost:8000/v1")
QWEN_MODEL = "Qwen2.5-Coder-14B-Instruct-MLX-4bit"
JUDGE_BASE = os.getenv("LLM_BASE_URL", os.getenv("OMLX_BASE_URL"))
JUDGE_MODEL = os.getenv("MODEL_JUDGE", "claude-sonnet-4-6")

# How many timed calls per (model, prompt); latency = median, answer from the 1st.
REPEATS = int(os.getenv("REPEATS", "2"))

# Mixed general-capability prompts (lab-relevant extraction/temporal cases included).
PROMPTS: list[tuple[str, str]] = [
    ("reasoning", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
                  "than the ball. How much does the ball cost? Give just the amount."),
    ("reasoning2", "If 5 machines make 5 widgets in 5 minutes, how long do 100 "
                   "machines take to make 100 widgets? Answer with just the time."),
    ("code", "Write a Python function `dedupe_preserve_order(xs)` that removes "
             "duplicates from a list while keeping first-seen order. Code only."),
    ("code-edge", "Write a Python function `safe_div(a, b)` returning a/b, or None "
                  "if b is 0. Code only, no explanation."),
    ("summarization", "Summarize in exactly two sentences: Retrieval-augmented "
                      "generation grounds an LLM's answer in retrieved documents to "
                      "reduce hallucination, but its quality depends on retrieval "
                      "recall, chunking, and how the reader assembles the evidence; "
                      "weak retrieval or a fixed context window can still drop the "
                      "needle even when the fact was stored."),
    ("instruction-json", "List exactly three primary colors as a JSON array of "
                         "lowercase strings. Output only the JSON, nothing else."),
    ("structured", "Return a JSON object with keys name (string) and age (int) for: "
                   "'Maria is 34 years old'. Output only JSON."),
    ("classification", "Classify the sentiment as exactly one word — positive, "
                       "negative, or neutral: 'The battery dies in an hour, but the "
                       "screen is gorgeous.'"),
    ("extraction", "Extract atomic facts as a JSON array of short strings from this "
                   "message: '[USER] I returned the boots to Zara on Monday, picked "
                   "up a new pair, and dropped my navy blazer at the dry cleaner.'"),
    ("temporal", "I bought a Samsung phone in March and a Dell laptop in May. Which "
                 "did I get first? Answer with just the device."),
    ("counting", "How many times does the letter 'r' appear in 'strawberry'? Number only."),
    ("constraint", "Write a sentence about the ocean that does NOT contain the "
                   "letter 'e'. Output only the sentence."),
    ("math-word", "A shirt is $40 after a 20% discount. What was the original price? "
                  "Give just the amount."),
    ("format-convert", "Convert to ISO 8601 date (YYYY-MM-DD): 'March 3rd, 2024'. "
                       "Output only the date."),
    ("concise", "In one sentence, explain why fusing two retrievers can score below "
                "either one alone on a question the reader must reason over."),
    ("abstain", "Based only on this note, what is the user's blood type? Note: 'User "
                "likes hiking and lives in Denver.' Answer honestly."),
]

agnes = OpenAI(base_url=AGNES_BASE, api_key=os.environ["ANGES_API_KEY"])
qwen = OpenAI(base_url=QWEN_BASE, api_key=os.getenv("OMLX_API_KEY", "dummy"))
judge = OpenAI(base_url=JUDGE_BASE, api_key=os.getenv("LLM_API_KEY", os.getenv("OMLX_API_KEY", "dummy")))


def ask(client: OpenAI, model: str, prompt: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=600,
    )
    return (r.choices[0].message.content or "").strip(), time.perf_counter() - t0


def ask_timed(client: OpenAI, model: str, prompt: str) -> tuple[str, float]:
    """Call REPEATS times; return the FIRST answer + the MEDIAN latency (smooths
    network jitter for the API model and scheduler noise for the local one)."""
    ans, lats = "", []
    for i in range(REPEATS):
        a, dt = ask(client, model, prompt)
        if i == 0:
            ans = a
        lats.append(dt)
    return ans, statistics.median(lats)


def judge_once(prompt: str, ans_a: str, ans_b: str) -> str:
    """Return 'A', 'B', or 'TIE' — which answer is better."""
    jp = (f"You are a strict evaluator. QUESTION:\n{prompt}\n\n"
          f"ANSWER A:\n{ans_a}\n\nANSWER B:\n{ans_b}\n\n"
          "Which answer is better (more correct, follows the instruction, concise)? "
          "Reply with exactly one token: A, B, or TIE.")
    for attempt in range(4):
        try:
            r = judge.chat.completions.create(
                model=JUDGE_MODEL, messages=[{"role": "user", "content": jp}],
                temperature=0.0, max_tokens=4)
            v = (r.choices[0].message.content or "").strip().upper()
            if "A" in v and "B" not in v:
                return "A"
            if "B" in v and "A" not in v:
                return "B"
            return "TIE"
        except Exception:  # noqa: BLE001 — VibeProxy 503 cooldown; back off
            if attempt == 3:
                return "TIE"
            time.sleep(2 * (attempt + 1))
    return "TIE"


def judge_pair(prompt: str, agnes_ans: str, qwen_ans: str) -> str:
    """Order-swapped judging: agnes wins only if it wins/ties BOTH orders."""
    o1 = judge_once(prompt, agnes_ans, qwen_ans)   # A=agnes, B=qwen
    o2 = judge_once(prompt, qwen_ans, agnes_ans)   # A=qwen, B=agnes
    agnes_w = (o1 == "A") + (o2 == "B")
    qwen_w = (o1 == "B") + (o2 == "A")
    if agnes_w > qwen_w:
        return "agnes"
    if qwen_w > agnes_w:
        return "qwen"
    return "tie"


def main() -> None:
    print(f">>> agnes: {AGNES_MODEL} @ {AGNES_BASE}")
    print(f">>> qwen:  {QWEN_MODEL} @ {QWEN_BASE}")
    print(f">>> judge: {JUDGE_MODEL} @ {JUDGE_BASE}\n")

    # Warmup (discard) so oMLX cold-load doesn't skew qwen latency.
    print("warming up both models ...")
    for c, m in ((agnes, AGNES_MODEL), (qwen, QWEN_MODEL)):
        try:
            ask(c, m, "reply with: ok")
        except Exception as e:  # noqa: BLE001
            print(f"  WARN warmup failed for {m}: {e}")

    a_lat, q_lat = [], []
    wins = {"agnes": 0, "qwen": 0, "tie": 0}
    print(f"\n{'category':<16}{'agnes(s)':<10}{'qwen(s)':<10}{'winner'}")
    print("-" * 50)
    rows = []
    for cat, prompt in PROMPTS:
        aa, at = ask_timed(agnes, AGNES_MODEL, prompt)
        qa, qt = ask_timed(qwen, QWEN_MODEL, prompt)
        a_lat.append(at); q_lat.append(qt)
        w = judge_pair(prompt, aa, qa)
        wins[w] += 1
        rows.append((cat, prompt, aa, qa, at, qt, w))
        print(f"{cat:<16}{at:<10.2f}{qt:<10.2f}{w}")

    print("-" * 50)
    print(f"{'MEDIAN':<16}{statistics.median(a_lat):<10.2f}{statistics.median(q_lat):<10.2f}")
    print(f"\nquality (order-swapped judge): "
          f"agnes {wins['agnes']} / qwen {wins['qwen']} / tie {wins['tie']}  (n={len(PROMPTS)})")
    print(f"latency: agnes median {statistics.median(a_lat):.2f}s | "
          f"qwen median {statistics.median(q_lat):.2f}s")

    print("\n=== per-prompt answers (truncated) ===")
    for cat, prompt, aa, qa, at, qt, w in rows:
        print(f"\n[{cat}] winner={w}")
        print(f"  Q: {prompt[:80]}")
        print(f"  agnes ({at:.2f}s): {aa[:120]!r}")
        print(f"  qwen  ({qt:.2f}s): {qa[:120]!r}")


if __name__ == "__main__":
    main()
