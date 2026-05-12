"""REPL agent with cross-session memory. Each REPL invocation is a separate
session; memory carries across. Run it multiple times with --user alice to
watch memory accumulate."""
import os, sys, uuid, argparse
from openai import OpenAI
from dotenv import load_dotenv
from src.memory import recall, remember_turn, format_memory_block

load_dotenv()
omlx  = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
MODEL = os.getenv("MODEL_SONNET")


def chat_loop(user_id: str, session_id: str) -> None:
    history = []
    print(f"[session {session_id[:8]} for user '{user_id}']")
    print("Type 'exit' to quit. Type '/memory' to inspect current memory.\n")

    while True:
        user_msg = input("you> ").strip()
        if user_msg.lower() == "exit": break
        if user_msg == "/memory":
            mem = recall(user_id, "")
            print(format_memory_block(mem) or "(no memories yet)"); print()
            continue

        mem = recall(user_id, user_msg)
        memory_block = format_memory_block(mem)

        system = "You are a helpful assistant."
        if memory_block:
            system += "\n\n" + memory_block

        messages = [{"role": "system", "content": system}, *history,
                    {"role": "user", "content": user_msg}]
        resp = omlx.chat.completions.create(model=MODEL, messages=messages, temperature=0.4, max_tokens=400)
        assistant_msg = resp.choices[0].message.content
        print(f"bot> {assistant_msg}\n")

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})
        result = remember_turn(user_id, session_id, user_msg, assistant_msg)
        if result["semantic"] or result["episodic_count"]:
            tags = [f"{s['key']}={s['value']} [{s['status']}]" for s in result["semantic"]]
            print(f"  [remembered: {', '.join(tags) or '-'} | +{result['episodic_count']} episodic]\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="alice")
    args = ap.parse_args()
    chat_loop(args.user, str(uuid.uuid4()))
