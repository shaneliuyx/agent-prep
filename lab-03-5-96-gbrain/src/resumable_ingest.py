"""W3.5.96 — LARGE-CORPUS ingest: per-file streaming + on-DISK staging + final
merge, so GBrain (the embedded store) only ever sees CANONICAL pages — each entity
is embedded exactly once. The scale variant of ingest_agent.py.

Earlier draft staged into GBrain itself (one put_page per file-variant). That made
the store embed every variant and then throw it away at merge — ~71% wasted
embedding on the 8-file run (46 staging embeds for 19 final pages). Embedding is
the throughput ceiling at scale, so staging must NOT touch the embedded store.

This version stages on disk (cheap, no embedding) and only writes finals to GBrain:

  driver, per file (resumable — skip files already staged on disk):
    pages = extract_file(file)                 # DRIVER-side LLM (no 30s sandbox)
    write pages -> ~/brain/.ingest_stage/<file>.json   # disk staging, NO embedding
  merge_from_disk(): group by entity across files, merge multi-file entities (one
    LLM call each) -> a list of CANONICAL pages
  agent: put_page each canonical page over MCP, in bounded batches  # embedded ONCE
  reconcile_graph(); query

The "agent uses GBrain as memory" lesson is intact: the agent still WRITES the
canonical pages and QUERIES them over MCP. Only the throwaway intermediate left
the embedded store.

Two checkpoints, both on disk, so resume re-does no expensive work:
  - EXTRACTION: a file with a stage JSON is skipped (no re-extract).
  - WRITES: ~/brain/.ingest_written.json records written canonical slugs, so a
    resumed run re-embeds ONLY un-written pages — "embed once" survives a crash.
Oversized pages (> BIG_PAGE_CHARS) are written driver-side (no 30s sandbox) since
one such page's single embed could approach the agent's per-step limit.

Run: python src/resumable_ingest.py   (re-run to resume)
Restart from scratch: rm -rf ~/brain/.ingest_stage ~/brain/.ingest_written.json
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess

from mcp import StdioServerParameters
from openai import OpenAI
from smolagents import CodeAgent, OpenAIServerModel, ToolCollection, tool

from ingest_agent import (
    NEEDED_TOOLS, QUERY_TASK, _EXTRACT_PROMPT, _GBRAIN, _server_env, reconcile_graph,
)

SOURCES = pathlib.Path(os.path.expanduser("~/brain/sources"))
STAGE_DIR = pathlib.Path(os.path.expanduser("~/brain/.ingest_stage"))   # disk staging
WRITTEN = pathlib.Path(os.path.expanduser("~/brain/.ingest_written.json"))  # write checkpoint
BATCH = 8                 # canonical pages per agent write call (bounded < 30s)
BIG_PAGE_CHARS = 8000     # bigger pages are written driver-side (one page's embed may near 30s)

_CURRENT: list = []       # the current write batch; the agent's tool reads this


def _files() -> list[tuple[str, str]]:
    """(stem, text) for every readable source file (skip dotted parts + binaries)."""
    out = []
    for f in sorted(SOURCES.rglob("*")):
        if not f.is_file() or any(part.startswith(".") for part in f.relative_to(SOURCES).parts):
            continue
        try:
            text = f.read_text()
        except UnicodeDecodeError:
            continue
        out.append((str(f.relative_to(SOURCES)).replace("/", "-").rsplit(".", 1)[0], text))
    return out


def _llm() -> OpenAI:
    return OpenAI(base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
                  api_key=os.getenv("LLM_API_KEY", "dummy"))


# ── write checkpoint (so a resumed run re-embeds only un-written pages) ──────
def _written_done() -> set[str]:
    if WRITTEN.exists():
        return set(json.loads(WRITTEN.read_text()).get("done", []))
    return set()


def _mark_written(slugs: list[str]) -> None:
    done = _written_done() | set(slugs)
    WRITTEN.write_text(json.dumps({"done": sorted(done)}))


def _gbrain_put(slug: str, content: str) -> None:
    """Driver-side write (no 30s sandbox) — used for oversized pages whose single
    embed could approach the agent's per-step limit."""
    subprocess.run([_GBRAIN, "put", slug], input=content, capture_output=True,
                   text=True, env=_server_env())


# ── per-file extraction → DISK staging (no embedding) ───────────────────────
def extract_file(text: str) -> list:
    """LLM-extract ONE file into pages (canonical base slugs — no DB namespacing
    needed; the disk filename records which file)."""
    resp = _llm().chat.completions.create(
        model=os.getenv("LLM_MODEL", "Qwen2.5-Coder-14B-Instruct-MLX-4bit"),
        messages=[{"role": "user", "content": _EXTRACT_PROMPT.replace("{raw}", text)}],
        temperature=0.0, max_tokens=4000, response_format={"type": "json_object"})
    data = json.loads(resp.choices[0].message.content or "{}")
    return [p for p in data.get("pages", []) if p.get("slug") and p.get("content")]


def stage_all() -> None:
    """Extract every not-yet-staged file to ~/brain/.ingest_stage/<stem>.json.
    Resumable: a file whose stage JSON already exists is skipped."""
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    for stem, text in _files():
        out = STAGE_DIR / f"{stem}.json"
        if out.exists():
            continue
        pages = extract_file(text)
        out.write_text(json.dumps(pages))
        print(f">>> staged {stem}: {len(pages)} pages (disk, no embedding)")


# ── merge across files (disk, no DB reads) ──────────────────────────────────
def _merge(base: str, contents: list[str]) -> str:
    """Merge per-file variants of one entity into a single page (one LLM call)."""
    joined = "\n\n--- VARIANT ---\n\n".join(contents)
    prompt = (
        f"These are {len(contents)} notes about the SAME entity ({base}), from "
        "different source files. Merge into ONE GBrain page with the exact two-layer "
        "shape (summary with [[dir/slug]] wikilinks, then a line that is exactly "
        "`---`, then `## Timeline` with the UNION of all timeline lines, deduplicated, "
        "chronological). Keep every distinct fact. Output ONLY the merged markdown.\n\n"
        f"{joined}"
    )
    resp = _llm().chat.completions.create(
        model=os.getenv("LLM_MODEL", "Qwen2.5-Coder-14B-Instruct-MLX-4bit"),
        messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=2000)
    return (resp.choices[0].message.content or contents[0]).strip()


def merge_from_disk() -> list:
    """Group staged pages by entity across all files; return CANONICAL pages.
    Single-file entities pass through; multi-file entities are merged once."""
    groups: dict[str, list[str]] = {}
    for jf in sorted(STAGE_DIR.glob("*.json")):
        for p in json.loads(jf.read_text()):
            groups.setdefault(p["slug"], []).append(p["content"])
    canonical, merged = [], 0
    for slug, contents in groups.items():
        content = contents[0] if len(contents) == 1 else _merge(slug, contents)
        if len(contents) > 1:
            merged += 1
        canonical.append({"slug": slug, "content": content})
    print(f">>> merge_from_disk: {len(canonical)} canonical ({merged} merged from >1 file)")
    return canonical


@tool
def current_pages() -> list:
    """Return the current batch of canonical pages ({slug, content}) for the agent
    to write. The driver sets this before each agent.run."""
    return _CURRENT


WRITE_TASK = """Write the current pages using ONLY the provided tools:
1. pages = current_pages()
2. for each page in pages: call put_page(slug=page["slug"], content=page["content"])
3. call final_answer with how many you wrote.
"""


def main() -> None:
    server = StdioServerParameters(command=_GBRAIN, args=["serve"], env=_server_env())
    model = OpenAIServerModel(
        model_id=os.getenv("LLM_MODEL", "Qwen2.5-Coder-14B-Instruct-MLX-4bit"),
        api_base=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("LLM_API_KEY", "dummy"))

    global _CURRENT
    # 1. EXTRACT every file to disk staging (resumable, no embedding).
    stage_all()
    # 2. MERGE across files → canonical pages (driver-side, no DB).
    canonical = merge_from_disk()

    with ToolCollection.from_mcp(server, trust_remote_code=True) as tc:
        mcp_tools = [t for t in tc.tools if t.name in NEEDED_TOOLS]
        agent = CodeAgent(
            tools=[current_pages, *mcp_tools], model=model, max_steps=3,
            use_structured_outputs_internally=True, verbosity_level=1)

        # 3. WRITE canonical pages to GBrain — embedded EXACTLY ONCE, and the
        # write checkpoint means a resumed run re-embeds only un-written pages.
        written = _written_done()
        pending = [p for p in canonical if p["slug"] not in written]
        print(f">>> {len(canonical)} canonical, {len(canonical) - len(pending)} "
              f"already written (resume), {len(pending)} to write")

        # Oversized pages → driver-side (a single big embed could near the 30s
        # sandbox limit); normal pages → agent, in bounded batches.
        big = [p for p in pending if len(p["content"]) > BIG_PAGE_CHARS]
        small = [p for p in pending if len(p["content"]) <= BIG_PAGE_CHARS]
        for p in big:
            _gbrain_put(p["slug"], p["content"])
            _mark_written([p["slug"]])
            print(f">>> big page driver-side: {p['slug']} ({len(p['content'])} chars)")
        for i in range(0, len(small), BATCH):
            batch = small[i:i + BATCH]
            _CURRENT = batch
            print(f">>> write batch {i // BATCH + 1} ({len(batch)} pages) -> "
                  + str(agent.run(WRITE_TASK)))
            _mark_written([p["slug"] for p in batch])   # checkpoint AFTER the batch lands

        # 4. RECONCILE links, then the agent queries its memory.
        print(">>> reconcile graph: " + reconcile_graph())
        answer = agent.run(QUERY_TASK)
        print("\n>>> agent final answer:\n" + str(answer))


if __name__ == "__main__":
    main()
