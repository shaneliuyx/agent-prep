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

A big file is split into deterministic chunks (`<file>#0`, `#1`, … by CHUNK_CHARS,
line-aligned), each its own staging unit — so "one file too big for the extract
context" is handled, and resume works at chunk granularity.

Each derived layer is a cache of the one above, rebuildable from it — so resume
re-does no expensive work, and losing a derived layer is recoverable, not fatal:
  source files (ground truth) → stage chunks → merged canonical → embedded
  - EXTRACTION: a file CHUNK with a stage JSON (`<file>#<idx>.json`) is skipped;
    a MISSING chunk is re-extracted from its source file (not a dead end).
  - MERGE: cached to ~/brain/.ingest_merged.json keyed by a stage fingerprint;
    unchanged staging → load cache, skip the (LLM-costly) re-merge.
  - WRITES: ~/brain/.ingest_written.json records written canonical slugs — but a
    slug is recorded only after its page is VERIFIED present in GBrain
    (verify-then-mark), so a silently-failed write stays un-checkpointed and is
    retried on resume. A resumed run re-embeds ONLY un-written pages — "embed
    once" survives a crash; the checkpoint can never claim a page that isn't there.
Oversized pages (> BIG_PAGE_CHARS) are written driver-side (no 30s sandbox) since
one such page's single embed could approach the agent's per-step limit.

Run: python src/resumable_ingest.py   (re-run to resume)
Restart: rm -rf ~/brain/.ingest_stage ~/brain/.ingest_merged.json ~/brain/.ingest_written.json
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
MERGED = pathlib.Path(os.path.expanduser("~/brain/.ingest_merged.json"))  # merge cache
WRITTEN = pathlib.Path(os.path.expanduser("~/brain/.ingest_written.json"))  # write checkpoint
BATCH = 8                 # canonical pages per agent write call (bounded < 30s)
BIG_PAGE_CHARS = 8000     # bigger pages are written driver-side (one page's embed may near 30s)
CHUNK_CHARS = 6000        # a file > this is split into <file>#0, #1, … (extract context budget)

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


def _verify_written(slugs: list[str]) -> list[str]:
    """Return the subset of `slugs` that ACTUALLY landed in GBrain. Existence in
    `pages` (deleted_at IS NULL) is the proof of a successful put_page (embed +
    upsert); only verified slugs get checkpointed, so a silently-failed write
    stays un-checkpointed → retried on resume. The invariant: a slug is in the
    write checkpoint IFF its page is really in the store."""
    if not slugs:
        return []
    cont = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                          capture_output=True, text=True).stdout
    name = next((n for n in cont.splitlines() if "gbrain-pg" in n), "gbrain-pg")
    arr = "{" + ",".join(slugs) + "}"   # slugs are kebab + '/', safe in a PG array literal
    sql = f"select slug from pages where deleted_at is null and slug = any('{arr}');"
    out = subprocess.run(["docker", "exec", "-i", name, "psql", "-U", "postgres",
                          "-d", "gbrain", "-tAc", sql], capture_output=True, text=True).stdout
    got = {s.strip() for s in out.splitlines() if s.strip()}
    return [s for s in slugs if s in got]


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


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split into ≤max_chars chunks on LINE boundaries (never mid-line). A file
    that fits returns one chunk; a big file returns several. Deterministic: the
    same input always yields the same chunks, so chunk N == the same bytes."""
    chunks, cur, n = [], [], 0
    for line in text.splitlines(keepends=True):
        if n + len(line) > max_chars and cur:
            chunks.append("".join(cur)); cur, n = [], 0
        cur.append(line); n += len(line)
    if cur:
        chunks.append("".join(cur))
    return chunks or [""]


def stage_all() -> None:
    """Extract every not-yet-staged file CHUNK to ~/brain/.ingest_stage/<stem>#<idx>.json.

    A file > CHUNK_CHARS is split into deterministic chunks (extract context budget);
    each chunk is its own staging unit. Resumable at CHUNK granularity for free —
    the existing 'skip if the JSON exists' check now skips already-staged chunks,
    so a crash mid-file re-extracts only the unfinished chunk(s). Cross-chunk
    entities are reunited later by merge_from_disk (same path as cross-file)."""
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    for stem, text in _files():
        chunks = _chunk_text(text, CHUNK_CHARS)
        for idx, chunk in enumerate(chunks):
            out = STAGE_DIR / f"{stem}#{idx}.json"
            if out.exists():
                continue
            pages = extract_file(chunk)
            out.write_text(json.dumps(pages))
            tag = f"{stem}#{idx}" + (f" of {len(chunks)}" if len(chunks) > 1 else "")
            print(f">>> staged {tag} ({len(chunk)} chars): {len(pages)} pages (disk, no embedding)")


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


def _stage_key() -> str:
    """Fingerprint of the staged chunks (name + mtime + size). If it's unchanged
    since the cached merge, the merge result is still valid — so resume can skip
    the (LLM-costly) re-merge. Any re-extracted chunk changes its mtime → miss."""
    parts = []
    for f in sorted(STAGE_DIR.glob("*.json")):
        st = f.stat()
        parts.append(f"{f.name}:{int(st.st_mtime)}:{st.st_size}")
    return "|".join(parts)


def merge_from_disk() -> list:
    """Group staged pages by entity across all files; return CANONICAL pages.
    Single-file entities pass through; multi-file entities are merged once (LLM).

    Cached: the merge (its per-entity LLM calls) is expensive, so the result is
    cached to MERGED keyed by the stage fingerprint. A resume whose staging is
    unchanged loads the cache and skips re-merging; if any chunk was re-extracted
    (fingerprint differs) it re-merges and re-caches."""
    key = _stage_key()
    if MERGED.exists():
        cached = json.loads(MERGED.read_text())
        if cached.get("key") == key:
            canonical = cached["canonical"]
            print(f">>> merge cache HIT: {len(canonical)} canonical (skipped re-merge)")
            return canonical

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
    MERGED.write_text(json.dumps({"key": key, "canonical": canonical}))
    print(f">>> merge_from_disk: {len(canonical)} canonical ({merged} merged from >1 file), cached")
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
            _mark_written(_verify_written([p["slug"]]))  # checkpoint ONLY if it landed
            print(f">>> big page driver-side: {p['slug']} ({len(p['content'])} chars)")
        for i in range(0, len(small), BATCH):
            batch = small[i:i + BATCH]
            _CURRENT = batch
            print(f">>> write batch {i // BATCH + 1} ({len(batch)} pages) -> "
                  + str(agent.run(WRITE_TASK)))
            # Verify-then-mark: checkpoint ONLY slugs confirmed present in GBrain.
            # A silently-failed put_page stays un-checkpointed → retried on resume.
            landed = _verify_written([p["slug"] for p in batch])
            _mark_written(landed)
            if len(landed) < len(batch):
                missing = sorted(set(p["slug"] for p in batch) - set(landed))
                print(f">>> WARNING: {len(missing)} page(s) did not land, left un-checkpointed "
                      f"(retry on resume): {missing}")

        # 4. RECONCILE links, then the agent queries its memory.
        print(">>> reconcile graph: " + reconcile_graph())
        answer = agent.run(QUERY_TASK)
        print("\n>>> agent final answer:\n" + str(answer))


if __name__ == "__main__":
    main()
