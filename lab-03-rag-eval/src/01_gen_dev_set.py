"""Draft Q/A pairs from 100 random docs; human curation follows in Phase 1.3."""
import json
import os
import random
import re
from pathlib import Path

from openai import OpenAI


random.seed(7)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def extract_json(text: str) -> dict:
    """
    Parse either pure JSON or JSON embedded in a model response.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:300]}")

    return json.loads(match.group(0))


omlx = OpenAI(
    base_url=require_env("OMLX_BASE_URL"),
    api_key=require_env("OMLX_API_KEY"),
)

SONNET = require_env("MODEL_SONNET")

docs = [json.loads(l) for l in open("data/docs.jsonl")]
sample = random.sample(docs, min(100, len(docs)))

PROMPT = """Read the passage and write one concrete, factual question that the passage answers.

Return only valid JSON in this exact schema:
{{"question": "<one sentence>", "short_answer": "<20 words or fewer from the passage>"  }}

Passage:
{text}
"""

out = []

for i, d in enumerate(sample):
    try:
        r = omlx.chat.completions.create(
            model=SONNET,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT.format(text=d["text"]),
                }
            ],
            temperature=0.0,
            max_tokens=200,
            # Remove response_format because some OpenAI-compatible
            # gateways/models return content=None when it is unsupported.
            # response_format={"type": "json_object"},
        )

        msg = r.choices[0].message
        content = msg.content

        if content is None:
            print(f"  {i}: empty content")
            print(f"     raw message: {msg}")
            continue

        pair = extract_json(content)

        if "question" not in pair or "short_answer" not in pair:
            raise ValueError(f"Missing required keys: {pair}")

        out.append(
            {
                "source_doc_id": d["id"],
                "source_text": d["text"],
                "question": pair["question"],
                "short_answer": pair["short_answer"],
            }
        )

    except Exception as e:
        print(f"  {i}: skip ({type(e).__name__}: {e})")

    if i and i % 20 == 0:
        print(f"  {i}/100")

Path("data").mkdir(exist_ok=True)
Path("data/dev_candidates-test.jsonl").write_text(
    "\n".join(json.dumps(o, ensure_ascii=False) for o in out),
    encoding="utf-8",
)

print(f"wrote {len(out)} candidates")