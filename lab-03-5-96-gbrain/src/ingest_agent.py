"""W3.5.96 — a memory-augmented agent (smolagents) that uses GBrain as its memory
layer over MCP. The transferable pattern for FUTURE agent development.

Design = idiomatic smolagents: **thin agent, fat tools.** A small local model can't
reliably read files AND write a good extractor AND compose markdown in one code loop
(and the CodeAgent sandbox blocks `pathlib`/`json` anyway). So the hard work lives in
TOOLS; the agent just orchestrates:

  tools given to the agent:
    - read_sources()        local  — returns the raw text of ~/brain/sources/*
    - extract_pages(raw)    local  — LLM (oMLX) raw → structured GBrain pages (list)
    - put_page, query, ...  MCP    — GBrain, loaded via ToolCollection.from_mcp

  the agent's whole job (a few lines of code it writes itself):
    raw = read_sources(); pages = extract_pages(raw)
    for p in pages: put_page(slug=p['slug'], content=p['content'])
    answer = query(query="..."); final_answer(answer)

After the agent run, main() calls reconcile_graph() — a deterministic, zero-LLM
`gbrain extract links --source db` pass that materializes the [[wikilinks]] into
typed edges. This is infra, NOT an agent tool: put_page over MCP skips inline
auto-link (remote caller) and inline auto-link can't wire forward references
anyway, so the graph must be reconciled once the full corpus exists.

Brain = oMLX (no native tool-calls) → CodeAgent + use_structured_outputs_internally.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess

from dotenv import load_dotenv
from mcp import StdioServerParameters
from openai import OpenAI
from smolagents import CodeAgent, OpenAIServerModel, ToolCollection, tool

_ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

SOURCES = pathlib.Path(os.path.expanduser("~/brain/sources"))
_BUN_BIN = os.path.expanduser("~/.bun/bin")
_GBRAIN = os.getenv("GBRAIN_BIN", "gbrain")
if _GBRAIN == "gbrain":
    _GBRAIN = os.path.join(_BUN_BIN, "gbrain")

NEEDED_TOOLS = {"put_page", "query"}   # the MCP tools the agent calls

_EXTRACT_PROMPT = """Convert raw notes into GBrain pages. One page per entity.

Slug: path-qualified kebab-case — people/<name>, companies/<name>, deals/<name>, meetings/<name>.

content MUST follow this exact two-layer shape:

# <Title>

<one-paragraph summary. EVERY other entity you mention MUST be a path-qualified
wikilink [[dir/slug]], e.g. [[people/alice-chen]], [[companies/acme-ai]].>

---
## Timeline
- YYYY-MM-DD — <event, also using [[dir/slug]] wikilinks> (source: <raw filename>)

HARD RULES (a page that breaks these is WRONG):
- The separator between summary and Timeline is a line that is EXACTLY `---` (three hyphens). Never an HTML comment.
- EVERY mention of another entity is a [[dir/slug]] wikilink. A page with zero wikilinks is invalid.
- If you mention an entity, also emit its page, and link to it by the SAME slug.
- Deduplicate across docs (one page per entity). Use ONLY facts in the raw text.

Worked example of one page's content field:
"# Alice Chen\\n\\nFounder & CEO of [[companies/acme-ai]]; angel in [[companies/stripe]]; raising [[deals/acme-seed]] with [[people/sam-okafor]].\\n\\n---\\n## Timeline\\n- 2026-05-12 — dinner with [[people/sam-okafor]] re [[deals/acme-seed]] (source: sources/transcripts/dinner.txt)"

Output ONLY JSON: {"pages":[{"slug":"people/alice-chen","content":"..."}]}.

RAW:
{raw}
"""


@tool
def read_sources() -> str:
    """Read every raw file under ~/brain/sources/ and return their concatenated text,
    each prefixed with its relative path as a header."""
    parts = []
    for f in sorted(SOURCES.rglob("*")):
        # skip non-files AND anything under a dotted path part (.DS_Store, .omc-state/…)
        if not f.is_file() or any(part.startswith(".") for part in f.relative_to(SOURCES).parts):
            continue
        try:
            text = f.read_text()
        except UnicodeDecodeError:
            continue  # skip binary / non-UTF-8 files rather than crash the ingest
        parts.append(f"===== {f.relative_to(SOURCES.parent)} =====\n{text}")
    return "\n\n".join(parts)


_PAGES_CACHE: list | None = None


@tool
def extract_pages(raw: str) -> list:
    """Turn raw source text into structured GBrain pages via the local LLM.

    Cached: the ~60s oMLX extraction exceeds smolagents' 30s per-step sandbox
    timeout, and the agent re-runs its whole code block on each step. main()
    warms this cache ONCE before the agent loop (outside the sandbox, no
    timeout), so the agent's `extract_pages(raw)` call returns instantly and
    ingest finishes in one step. Single-corpus assumption: the cache ignores
    `raw` after the first compute.

    Args:
        raw: concatenated raw source text (from read_sources).
    """
    global _PAGES_CACHE
    if _PAGES_CACHE is not None:
        return _PAGES_CACHE
    client = OpenAI(base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
                    api_key=os.getenv("LLM_API_KEY", "dummy"))
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "Qwen2.5-Coder-14B-Instruct-MLX-4bit"),
        messages=[{"role": "user", "content": _EXTRACT_PROMPT.replace("{raw}", raw)}],
        temperature=0.0, max_tokens=4000, response_format={"type": "json_object"})
    data = json.loads(resp.choices[0].message.content or "{}")
    _PAGES_CACHE = [p for p in data.get("pages", []) if p.get("slug") and p.get("content")]
    return _PAGES_CACHE


def _server_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = _BUN_BIN + os.pathsep + env.get("PATH", "")
    for k in ("GBRAIN_DATABASE_URL", "OLLAMA_BASE_URL", "OLLAMA_API_KEY"):
        if (v := os.getenv(k)):
            env[k] = v
    return env


def reconcile_graph() -> str:
    """Deterministic post-ingest pass (zero LLM): materialize the `[[wikilinks]]`
    the agent wrote into typed graph edges. REQUIRED after an agent/MCP ingest,
    for two reasons baked into GBrain:
      1. `put_page` over MCP is a *remote* caller, so GBrain skips inline auto-link
         (operations.ts -> `skipped: 'remote'`); nothing wires on write.
      2. Even inline auto-link only wires targets that ALREADY exist (FK-safety),
         so the forward references a single-pass ingest creates would be dropped.
    `extract links --source db` reconciles the FINISHED corpus (all pages present),
    resolving every forward ref. Run it once, after all put_page writes."""
    out = subprocess.run(
        [_GBRAIN, "extract", "links", "--source", "db"],
        capture_output=True, text=True, env=_server_env(),
    )
    lines = [ln for ln in (out.stdout or out.stderr).splitlines() if ln.strip()]
    return lines[-1] if lines else "(no output)"


# Two phases on purpose: WRITE, then (infra reconcile), then READ. The query must
# run AFTER reconcile_graph() or it reads a graph whose edges aren't materialized.
INGEST_TASK = """Build the brain using ONLY the provided tools:
1. raw = read_sources()
2. pages = extract_pages(raw)
3. for each page in pages: call put_page(slug=page["slug"], content=page["content"])
4. return the number of pages written via final_answer.
"""
QUERY_TASK = """Answer using ONLY the query tool:
1. answer = query(query="Who is anchoring the acme-seed round and on what terms?")
2. return answer via final_answer.
"""


def main() -> None:
    server = StdioServerParameters(command=_GBRAIN, args=["serve"], env=_server_env())
    model = OpenAIServerModel(
        model_id=os.getenv("LLM_MODEL", "Qwen2.5-Coder-14B-Instruct-MLX-4bit"),
        api_base=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("LLM_API_KEY", "dummy"))

    with ToolCollection.from_mcp(server, trust_remote_code=True) as tc:
        mcp_tools = [t for t in tc.tools if t.name in NEEDED_TOOLS]
        print(f">>> GBrain MCP tools: {sorted(t.name for t in mcp_tools)}")
        agent = CodeAgent(
            tools=[read_sources, extract_pages, *mcp_tools],
            model=model, max_steps=6,
            use_structured_outputs_internally=True, verbosity_level=1)

        # 0. WARM the extraction cache OUTSIDE the agent sandbox. The oMLX
        # extraction is ~60s; smolagents kills any single step's code at 30s, so
        # if the agent triggered it inside its loop every step would time out and
        # re-extract. Running it once here (no sandbox) means the agent's
        # extract_pages(raw) call returns the cached result instantly.
        extract_pages(read_sources())

        # 1. WRITE — the agent ingests raw sources into GBrain pages (cache-fast).
        print(">>> ingest: " + str(agent.run(INGEST_TASK)))

        # 2. RECONCILE — deterministic, zero-LLM. NOT an agent tool (must not depend
        # on the LLM remembering). Materializes the [[wikilinks]] into typed edges
        # BEFORE the read, so the query sees the wired graph. See reconcile_graph().
        print(">>> reconcile graph: " + reconcile_graph())

        # 3. READ — query now runs over the reconciled graph.
        answer = agent.run(QUERY_TASK)
        print("\n>>> agent final answer:\n" + str(answer))


if __name__ == "__main__":
    main()
