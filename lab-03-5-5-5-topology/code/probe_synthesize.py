"""Diagnostic probe: send a heavy hierarchical-shape synthesize prompt
to oMLX and dump the raw JSON response. Reveals whether gpt-oss-20b's
output is in `content` (truncated/empty) or `reasoning_content` (the
W3.5.8 BCJ Entry 8 trap surface)."""
import json
import os
import httpx
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

# Mock the hierarchical TOP synthesize input: original question + 2 sub-answers
# each ~600-800 tokens (table-formatted, multi-paragraph)
HEAVY_PROMPT = """ORIGINAL QUESTION: Compare regulatory frameworks for AI across EU, US, and UK.

WORKER ANSWERS:

Worker 1 on 'EU AI Act components and risk classification':
The EU AI Act establishes a comprehensive regulatory framework with four risk categories: unacceptable, high, limited, and minimal. High-risk systems require conformity assessment, mandatory certification, data governance, and ongoing monitoring. Enforcement is via national supervisory authorities + the European AI Board, with fines up to €30M or 6% global turnover. Related EU regulations (DSA, DMA) complement the framework.

Worker 2 on 'US federal AI initiatives and sector rules':
US federal initiatives include the National AI Initiative Act 2020, Executive Order 14028 (AI Bill of Rights), AI Governance Act 2024, and AI Safety Act 2024. Sector-specific rules: FDA Digital Health Software Precertification for medical devices, OCC AI/ML guidance for banks, NHTSA ADS guidance for autonomous vehicles. FTC enforces consumer protection; NIST develops AI Risk Management Framework.

SYNTHESIZE."""

SYSTEM = """You are a research lead synthesizing 2 workers' answers into ONE coherent
answer to the original question. Cite which worker contributed which fact. Surface disagreement explicitly
if workers conflict — do NOT silently pick one side."""

url = (os.getenv("OMLX_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "http://localhost:8000/v1").rstrip("/") + "/chat/completions"
body = {
    "model": os.getenv("OMLX_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-oss-20b-MXFP4-Q8",
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": HEAVY_PROMPT},
    ],
    "temperature": 0.0,
    "max_tokens": 4096,
}
headers = {"Authorization": f"Bearer {os.getenv('OMLX_API_KEY') or os.getenv('OPENAI_API_KEY') or 'sk-local'}"}

print(f"POST {url}")
print(f"model: {body['model']}")
print(f"max_tokens: {body['max_tokens']}")
print(f"prompt size: ~{len(HEAVY_PROMPT)//4} tokens")
print()

r = httpx.post(url, json=body, headers=headers, timeout=300)
r.raise_for_status()
data = r.json()

print("=== TOP-LEVEL keys ===")
print(list(data.keys()))
print()
print("=== choices[0] keys ===")
choice = data["choices"][0]
print(list(choice.keys()))
print()
print("=== choices[0].message keys ===")
msg = choice.get("message", {})
print(list(msg.keys()))
print()
print(f"=== finish_reason ===")
print(choice.get("finish_reason"))
print()
print(f"=== content (len={len(msg.get('content') or '')}) ===")
content = msg.get("content")
if content:
    print(content[:500])
    if len(content) > 500:
        print(f"... [{len(content)-500} more chars]")
else:
    print(f"<{content!r}>")
print()
print(f"=== reasoning_content (len={len(msg.get('reasoning_content') or '')}) ===")
rc = msg.get("reasoning_content")
if rc:
    print(rc[:500])
    if len(rc) > 500:
        print(f"... [{len(rc)-500} more chars]")
else:
    print(f"<{rc!r}>")
print()
print(f"=== usage ===")
print(json.dumps(data.get("usage", {}), indent=2))
