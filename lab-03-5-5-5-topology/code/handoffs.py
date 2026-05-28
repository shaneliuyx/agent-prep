# code/handoffs.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any

from llm import chat


@dataclass
class Agent:
    name: str
    system_prompt: str
    tools: list[Callable] = field(default_factory=list)

    def respond(self, user_msg: str, history: list[dict]) -> tuple[str, "Agent | None"]:
        """Run one agent turn. Returns (final_response, next_agent_or_None).
        If model emits a tool-call to a handoff fn, returns (None_text, that_agent).
        Otherwise returns (text_response, None) and the conversation ends."""
        tool_doc = "\n".join(f"- {t.__name__}(): {t.__doc__ or ''}" for t in self.tools)
        prompt = (
            f"USER MESSAGE: {user_msg}\n\n"
            f"AVAILABLE TOOLS:\n{tool_doc}\n\n"
            f"If you should hand off, reply with EXACTLY: HANDOFF: <tool_name>\n"
            f"Otherwise, reply with your final answer."
        )
        reply = chat(prompt, system=self.system_prompt).strip()
        if reply.upper().startswith("HANDOFF:"):
            # Defensive parse: handle both 'HANDOFF: transfer_to_X' (bare ID
            # per prompt contract) AND 'HANDOFF: transfer_to_X()' (function-
            # call format that models often emit despite the bare-ID
            # instruction). Sonnet 4.6 and gpt-oss-20b both observed emitting
            # the parens variant 2026-05-28; without this fix, the loop
            # silently stays at triage even though the model decided to
            # hand off. Same trap-class as W3.5.5.5 BCJ Entry 7 (models
            # emit format variants of explicit-format instructions).
            import re
            m = re.search(r"HANDOFF:\s*(\w+)", reply, re.IGNORECASE)
            if m:
                tool_name = m.group(1)
                for tool in self.tools:
                    if tool.__name__ == tool_name:
                        next_agent = tool()
                        return ("", next_agent)
        return (reply, None)


# Specialist agents
REFUND_AGENT = Agent(
    name="refund",
    system_prompt=(
        "You are a refund specialist. Process refunds. Be specific about "
        "refund amounts, timing, and policy."
    ),
)
SALES_AGENT = Agent(
    name="sales",
    system_prompt=(
        "You are a sales specialist. Help with pricing, plans, upgrades. "
        "Be helpful and specific."
    ),
)


# Handoff tools (return Agent objects)
def transfer_to_refunds() -> Agent:
    """Hand off to the refund specialist when user wants a refund or
    has a billing dispute."""
    return REFUND_AGENT

def transfer_to_sales() -> Agent:
    """Hand off to the sales specialist when user asks about plans,
    pricing, upgrades, or feature comparisons."""
    return SALES_AGENT


# Triage entry point with both handoff tools
TRIAGE_AGENT = Agent(
    name="triage",
    system_prompt=(
        "You are a triage agent. Route the user to the right specialist by "
        "calling a handoff tool. Don't try to handle specialist queries yourself."
    ),
    tools=[transfer_to_refunds, transfer_to_sales],
)

def swarm_run(user_msg: str, max_handoffs: int = 3) -> dict:
    """Run user_msg through the swarm; return final response + handoff trace."""
    active = TRIAGE_AGENT
    history: list[dict] = []
    handoff_trace: list[str] = [active.name]
    for _ in range(max_handoffs + 1):
        reply, next_agent = active.respond(user_msg, history)
        history.append({"agent": active.name, "reply": reply})
        if next_agent is None:
            return {
                "final_response": reply,
                "handoff_trace": handoff_trace,
                "handoff_count": len(handoff_trace) - 1,
                "history": history,
            }
        active = next_agent
        handoff_trace.append(active.name)
    return {
        "final_response": "(stopped: max_handoffs reached)",
        "handoff_trace": handoff_trace,
        "handoff_count": len(handoff_trace) - 1,
        "history": history,
    }


if __name__ == "__main__":
    import json
    test_msgs = [
        "I want a refund for my recent purchase.",
        "What's the difference between Pro and Enterprise plans?",
        "How do I cancel my subscription and get my money back?",
        "Can I upgrade to the Pro plan?",
        "My credit card was charged twice for the same order.",
    ]
    for msg in test_msgs:
        out = swarm_run(msg)
        print(f"\nUser: {msg}")
        print(f"Trace: {' → '.join(out['handoff_trace'])}")
        print(f"Final: {out['final_response'][:120]}")
