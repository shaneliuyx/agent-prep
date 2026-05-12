"""Non-interactive proof of cross-session memory. Three scripted sessions,
each a separate 'conversation'. At session 3, the agent must recall facts
from sessions 1 and 2 to answer correctly."""
import os, uuid
from openai import OpenAI
from dotenv import load_dotenv
from src.memory import recall, remember_turn, format_memory_block

load_dotenv()
omlx  = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
MODEL = os.getenv("MODEL_SONNET")
USER  = "demo_user_" + str(uuid.uuid4())[:8]


def turn(session_id: str, user_msg: str) -> str:
    mem = recall(USER, user_msg)
    system = "You are a helpful assistant."
    block = format_memory_block(mem)
    if block: system += "\n\n" + block
    resp = omlx.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        temperature=0.2, max_tokens=300,
    )
    reply = resp.choices[0].message.content
    remember_turn(USER, session_id, user_msg, reply)
    return reply


def main():
    # Session 1 — tell the agent about the user
    s1 = str(uuid.uuid4())
    print(">>> Session 1")
    print("you> I live in Taipei and I work as a cloud infrastructure engineer.")
    print("bot>", turn(s1, "I live in Taipei and I work as a cloud infrastructure engineer."), "\n")
    print("you> I'm vegan and I ride a bicycle to work every day.")
    print("bot>", turn(s1, "I'm vegan and I ride a bicycle to work every day."), "\n")

    # Session 2 — unrelated topic, more info
    s2 = str(uuid.uuid4())
    print(">>> Session 2 (new session)")
    print("you> I'm preparing for an agent-engineering interview next month.")
    print("bot>", turn(s2, "I'm preparing for an agent-engineering interview next month."), "\n")

    # Session 3 — the recall test
    s3 = str(uuid.uuid4())
    print(">>> Session 3 (new session — cross-session recall test)")
    question = "Recommend one restaurant near me for lunch and one activity for the weekend."
    print(f"you> {question}")
    answer = turn(s3, question)
    print("bot>", answer, "\n")

    # Heuristic check
    lower = answer.lower()
    hits = {
        "taipei": "taipei" in lower,
        "vegan":  "vegan" in lower or "plant" in lower,
        "cycling/biking":  "bike" in lower or "cycl" in lower,
    }
    print("Recall checks:", hits)
    print("PASS" if all(hits.values()) else "FAIL")


if __name__ == "__main__":
    main()
