"""W3.5.96 — LARGE-CORPUS ingest: per-file streaming + per-file checkpoint +
final reconcile-merge. The scale variant of ingest_agent.py.

ingest_agent.py warms ONE extraction over ALL files concatenated — fine for a
handful of files, but at scale that single prompt blows the context window and is
un-resumable. This driver instead processes ONE FILE AT A TIME:

  driver: for each file NOT in the checkpoint:          # resumable
    pages = extract_file(file)        # extraction is DRIVER-side (no 30s sandbox),
                                      # and one small file always fits the context
    agent.run(WRITE_TASK)             # the AGENT writes this file's pages via
                                      # put_page over MCP (bounded → never hits 30s)
    mark_file_done(file)              # checkpoint after each file
  merge_pass()                        # consolidate entities seen across files
  reconcile_graph(); query

Cross-file dedup is DEFERRED to merge_pass: each file writes to a per-file staging
namespace (`staging/<file>/<entity>`) so files never overwrite each other; the
merge pass then groups variants by base entity, merges multi-file entities (one
LLM call each — the only place merge cost is paid), promotes to the canonical
slug, and drops the staging copies. Then reconcile wires the [[wikilinks]].

Run: python src/resumable_ingest.py   (re-run to resume; delete the checkpoint to restart)
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
from collections import defaultdict

from mcp import StdioServerParameters
from openai import OpenAI
from smolagents import CodeAgent, OpenAIServerModel, ToolCollection, tool

from ingest_agent import (
    NEEDED_TOOLS, QUERY_TASK, _EXTRACT_PROMPT, _GBRAIN, _server_env, reconcile_graph,
)

SOURCES = pathlib.Path(os.path.expanduser("~/brain/sources"))
CHECKPOINT = pathlib.Path(os.path.expanduser("~/brain/.ingest_files.json"))
STAGE = "staging"          # per-file slug namespace: staging/<file-stem>/<entity-slug>

_CURRENT: list = []        # the current file's staged pages; the agent's tool reads this


# ── files + checkpoint ──────────────────────────────────────────────────────
def _files() -> list[tuple[str, str]]:
    """(stem, text) for every readable source file. Stem is a slug-safe id."""
    out = []
    for f in sorted(SOURCES.rglob("*")):
        # skip non-files AND anything under a dotted dir (.DS_Store, .omc-state/…)
        if not f.is_file() or any(part.startswith(".") for part in f.relative_to(SOURCES).parts):
            continue
        try:
            text = f.read_text()
        except UnicodeDecodeError:
            continue
        stem = str(f.relative_to(SOURCES)).replace("/", "-").rsplit(".", 1)[0]
        out.append((stem, text))
    return out


def _done_files() -> set[str]:
    if CHECKPOINT.exists():
        return set(json.loads(CHECKPOINT.read_text()).get("done", []))
    return set()


def _mark_file(stem: str) -> None:
    done = _done_files() | {stem}
    CHECKPOINT.write_text(json.dumps({"done": sorted(done)}))


def _llm() -> OpenAI:
    return OpenAI(base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
                  api_key=os.getenv("LLM_API_KEY", "dummy"))


# ── per-file extraction (DRIVER-side: no 30s sandbox, one small file) ────────
def extract_file(stem: str, text: str) -> list:
    """LLM-extract ONE file into pages, slugs rewritten into this file's staging
    namespace so concurrent files never overwrite a shared entity."""
    resp = _llm().chat.completions.create(
        model=os.getenv("LLM_MODEL", "Qwen2.5-Coder-14B-Instruct-MLX-4bit"),
        messages=[{"role": "user", "content": _EXTRACT_PROMPT.replace("{raw}", text)}],
        temperature=0.0, max_tokens=4000, response_format={"type": "json_object"})
    data = json.loads(resp.choices[0].message.content or "{}")
    pages = []
    for p in data.get("pages", []):
        if p.get("slug") and p.get("content"):
            pages.append({"slug": f"{STAGE}/{stem}/{p['slug']}", "content": p["content"]})
    return pages


@tool
def current_pages() -> list:
    """Return the current file's staged pages ({slug, content}) for the agent to
    write. The driver sets these before each agent.run."""
    return _CURRENT


WRITE_TASK = """Write the current file's pages using ONLY the provided tools:
1. pages = current_pages()
2. for each page in pages: call put_page(slug=page["slug"], content=page["content"])
3. call final_answer with how many you wrote.
"""


# ── final reconcile-merge (DRIVER-side) ─────────────────────────────────────
def _base_slug(staged: str) -> str:
    """staging/<stem>/people/alice-chen -> people/alice-chen"""
    return staged.split("/", 2)[2]


def _gbrain_get(slug: str) -> str:
    body = subprocess.run([_GBRAIN, "get", slug], capture_output=True, text=True,
                          env=_server_env()).stdout
    return "\n".join(ln for ln in body.splitlines() if not ln.startswith(("Starting", "[gbrain")))


def _gbrain_put(slug: str, content: str) -> None:
    subprocess.run([_GBRAIN, "put", slug], input=content, capture_output=True,
                   text=True, env=_server_env())


def _gbrain_delete(slug: str) -> None:
    subprocess.run([_GBRAIN, "delete", slug], capture_output=True, text=True, env=_server_env())


def _merge(base: str, contents: list[str]) -> str:
    """Merge multiple per-file variants of one entity into a single page (one LLM
    call). Only invoked when an entity appears in >1 file."""
    joined = "\n\n--- VARIANT ---\n\n".join(contents)
    prompt = (
        f"These are {len(contents)} notes about the SAME entity ({base}), extracted "
        "from different source files. Merge them into ONE GBrain page with the exact "
        "two-layer shape (summary with [[dir/slug]] wikilinks, then a line that is "
        "exactly `---`, then `## Timeline` with the UNION of all timeline lines, "
        "deduplicated, chronological). Keep every distinct fact. Output ONLY the "
        f"merged markdown.\n\n{joined}"
    )
    resp = _llm().chat.completions.create(
        model=os.getenv("LLM_MODEL", "Qwen2.5-Coder-14B-Instruct-MLX-4bit"),
        messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=2000)
    return (resp.choices[0].message.content or contents[0]).strip()


def _list_staging() -> list[str]:
    """All staging slugs, via the DB (slugs are top-level, reliable to list)."""
    sql = "select slug from pages where deleted_at is null and slug like 'staging/%';"
    cont = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True,
                          text=True).stdout
    name = next((n for n in cont.splitlines() if "gbrain-pg" in n), "gbrain-pg")
    out = subprocess.run(["docker", "exec", "-i", name, "psql", "-U", "postgres",
                          "-d", "gbrain", "-tAc", sql], capture_output=True, text=True).stdout
    return [s.strip() for s in out.splitlines() if s.strip()]


def merge_pass() -> str:
    """Consolidate per-file staging pages into canonical entity pages."""
    groups: dict[str, list[str]] = defaultdict(list)
    for staged in _list_staging():
        groups[_base_slug(staged)].append(staged)

    merged, promoted = 0, 0
    for base, variants in groups.items():
        contents = [_gbrain_get(v) for v in variants]
        if len(contents) == 1:
            _gbrain_put(base, contents[0]); promoted += 1
        else:
            _gbrain_put(base, _merge(base, contents)); merged += 1
        for v in variants:
            _gbrain_delete(v)
    return f"merge_pass: {promoted} promoted, {merged} merged from {len(groups)} entities"


def main() -> None:
    server = StdioServerParameters(command=_GBRAIN, args=["serve"], env=_server_env())
    model = OpenAIServerModel(
        model_id=os.getenv("LLM_MODEL", "Qwen2.5-Coder-14B-Instruct-MLX-4bit"),
        api_base=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("LLM_API_KEY", "dummy"))

    global _CURRENT
    files = _files()
    done = _done_files()
    print(f">>> {len(files)} files; {len(done)} already done (checkpoint)")

    with ToolCollection.from_mcp(server, trust_remote_code=True) as tc:
        mcp_tools = [t for t in tc.tools if t.name in NEEDED_TOOLS]
        agent = CodeAgent(
            tools=[current_pages, *mcp_tools], model=model, max_steps=3,
            use_structured_outputs_internally=True, verbosity_level=1)

        # Per-file loop: extract (driver) → write (agent, bounded) → checkpoint.
        for stem, text in files:
            if stem in done:
                continue
            _CURRENT = extract_file(stem, text)       # driver-side, no sandbox limit
            print(f">>> {stem}: {len(_CURRENT)} pages -> " + str(agent.run(WRITE_TASK)))
            _mark_file(stem)                          # checkpoint after each file

        # Cross-file consolidation, then deterministic link reconciliation.
        print(">>> " + merge_pass())
        print(">>> reconcile graph: " + reconcile_graph())
        answer = agent.run(QUERY_TASK)
        print("\n>>> agent final answer:\n" + str(answer))


if __name__ == "__main__":
    main()
