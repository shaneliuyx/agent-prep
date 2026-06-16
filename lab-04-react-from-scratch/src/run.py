"""src/run.py — smoke-test entrypoint. Run one agent task and print the event log."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import src.tools   # must import before agent_run to register tools
from src.react import agent_run
from src.obs import query_run, run_summary

RUN_ID = "smoke_run_001"
TASK = "What is the square root of 144? Use the python_repl tool to verify."

print(f"Task: {TASK}\n")
answer = agent_run(TASK, obs=True, run_id=RUN_ID)
print(f"\nFinal answer: {answer}\n")

print("=== Event log ===")
for row in query_run(RUN_ID):
    print(row)

print("\n=== Run summary ===")
print(run_summary(RUN_ID))