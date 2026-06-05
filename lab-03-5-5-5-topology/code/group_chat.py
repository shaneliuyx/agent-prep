# code/group_chat.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal

from llm import chat


SelectorFlavor = Literal["round-robin", "llm-selected", "custom"]


@dataclass
class GroupAgent:
    name: str
    system_prompt: str

    def respond(self, pool: list[dict]) -> str:
        """One agent turn: read full pool, return a message."""
        transcript = "\n".join(f"{m['speaker']}: {m['content']}" for m in pool)
        prompt = f"CONVERSATION SO FAR:\n{transcript}\n\nYour turn ({self.name}):"
        return chat(prompt, system=self.system_prompt)


CODER = GroupAgent(
    name="coder",
    system_prompt=(
        "You are a Python coder. Propose code. Keep messages short (≤80 words). "
        "End with TERMINATE ONLY after AT LEAST 3 turns have happened AND "
        "reviewer + tester have both responded. Until then, write code and ask "
        "for review/tests but do NOT emit TERMINATE."
    ),
)
REVIEWER = GroupAgent(
    name="reviewer",
    system_prompt=(
        "You are a code reviewer. Critique the latest code for bugs / style / edge cases. "
        "≤80 words per message. Do NOT write code yourself; ask coder to fix."
    ),
)
TESTER = GroupAgent(
    name="tester",
    system_prompt=(
        "You are a tester. Propose ONE test case for the latest code. "
        "≤80 words per message. Don't repeat tests."
    ),
)
AGENTS = {"coder": CODER, "reviewer": REVIEWER, "tester": TESTER}

def select_round_robin(pool: list[dict], agents: list[GroupAgent]) -> GroupAgent:
    """Cyclic next-speaker; ignores context."""
    return agents[len(pool) % len(agents)]

def select_llm(pool: list[dict], agents: list[GroupAgent]) -> GroupAgent:
    """LLM reads pool, picks next agent name."""
    names = "/".join(a.name for a in agents)
    transcript = "\n".join(f"{m['speaker']}: {m['content']}" for m in pool[-5:])   # last 5
    prompt = (
        f"Conversation (last 5):\n{transcript}\n\n"
        f"Who should speak next? Pick ONE of: {names}. "
        f"Do NOT pick the speaker who just spoke unless required. "
        f"Reply with the single name only."
    )
    pick = chat(prompt).strip().lower()
    # Defensive: fall back to first agent if LLM returns garbage
    for a in agents:
        if a.name in pick:
            return a
    return agents[0]

def select_custom(pool: list[dict], agents: list[GroupAgent]) -> GroupAgent:
    """LLM-selected + RULE: tester always follows coder; reviewer always follows tester."""
    if not pool:
        return AGENTS["coder"]
    last = pool[-1]["speaker"]
    if last == "coder":
        return AGENTS["tester"]
    if last == "tester":
        return AGENTS["reviewer"]
    # Otherwise defer to LLM-selected
    return select_llm(pool, agents)


SELECTORS: dict[SelectorFlavor, Callable] = {
    "round-robin": select_round_robin,
    "llm-selected": select_llm,
    "custom": select_custom,
}

def group_chat_run(
    task: str,
    selector: SelectorFlavor = "round-robin",
    max_rounds: int = 9,
    min_rounds: int = 3,
) -> dict:
    """Run group chat until TERMINATE (after min_rounds) or max_rounds.

    `min_rounds` guards against premature-TERMINATE on commit-biased models
    (W3.5.5.5 BCJ Entry 10 — Qwen-Opus-distill emitted TERMINATE in round 1
    before any collaboration occurred). TERMINATE is only honored after
    round_n >= min_rounds; earlier TERMINATE tokens are ignored.
    """
    pool: list[dict] = [{"speaker": "user", "content": task}]
    agents = list(AGENTS.values())
    pick_fn = SELECTORS[selector]
    selector_calls = 0
    for round_n in range(1, max_rounds + 1):
        speaker = pick_fn(pool, agents)
        selector_calls += 1
        msg = speaker.respond(pool)
        pool.append({"speaker": speaker.name, "content": msg})
        if "TERMINATE" in msg.upper() and round_n >= min_rounds:
            return {
                "selector": selector,
                "rounds_used": round_n,
                "selector_calls": selector_calls,
                "terminated_by": "TERMINATE token",
                "pool": pool,
            }
    return {
        "selector": selector,
        "rounds_used": max_rounds,
        "selector_calls": selector_calls,
        "terminated_by": "max_rounds",
        "pool": pool,
    }


def print_group_chat(out: dict) -> None:
    """Pretty-print the group-chat discussion process with per-agent turns.

    Shared-memory note: every entry in `out['pool']` is visible to every agent
    on every turn (see GroupAgent.respond — full pool fed as transcript).
    The pool IS the shared memory; topology = blackboard pattern.
    """
    sel = out["selector"]
    print(f"\n{'=' * 70}")
    print(f"GROUP-CHAT (selector={sel}, rounds={out['rounds_used']}, "
          f"selector_calls={out['selector_calls']}, "
          f"terminated_by={out['terminated_by']})")
    print(f"{'=' * 70}")
    for i, msg in enumerate(out["pool"]):
        speaker = msg["speaker"]
        content = msg["content"]
        # Highlight terminator
        marker = " ← TERMINATE" if "TERMINATE" in content.upper() and speaker != "user" else ""
        print(f"\n[turn {i}] {speaker.upper()}{marker}")
        print(f"{'─' * 60}")
        for line in content.splitlines() or [""]:
            print(f"  {line}")
    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    for sel in ("round-robin", "llm-selected", "custom"):
        out = group_chat_run(
            "Write a Python function `is_palindrome(s: str) -> bool`. Reviewer + tester collaborate.",
            selector=sel,
        )
        print_group_chat(out)
