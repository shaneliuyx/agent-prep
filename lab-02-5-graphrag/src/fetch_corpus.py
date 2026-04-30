# src/fetch_corpus.py
from datasets import load_dataset
from pathlib import Path
import json

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:200]")
out = [{"id": r["id"], "title": r["title"], "text": r["text"][:4000]} for r in ds]
Path("data/corpus.json").write_text(json.dumps(out, indent=2))
print(f"Wrote {len(out)} articles")