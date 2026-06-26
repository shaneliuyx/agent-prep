# web_toolkit — structured web-research tools for custom agents

Four composable web tools with **structured (typed) results**, driven by local/self-hosted
backends — no required API keys. Built for custom agent loops (W4+ ReAct / tool-harness labs)
where a tool call must return a typed payload, not a free-form string.

Synthesized from [`Wade11s/pi-web-toolkit`](https://github.com/Wade11s/pi-web-toolkit)
(architecture: SearXNG + scrapling + agent-browser) and **absorbed the former `shared/web_search.py`**
(multi-backend precedence + on-disk reproducibility cache + cross-encoder rerank).

## Two contracts, one core (merge of the old `web_search.py`)

`web_toolkit` is now the single canonical web home. It exposes two search contracts over one
backend+cache core:

- **`web_search(...) -> list[SearchResult]`** — agent-facing, structured (title/url/snippet/engine/
  score), paginated + de-duped. Use inside an agent action-space.
- **`web_search_text(query, k) -> list[str]`** — the legacy RAG web-fallback contract (was
  `web_search.py:web_search`): single-page content strings, with the **original `web_cache_key`
  format preserved** so existing `.web_cache.json` pools replay identically (W3.7 CRAG
  reproducibility). The W3.7 labs import it as `web_search_text as web_search`.

The RAG primitives the W3.7 labs depend on — `rerank_results` (torch-free; reranker injected),
`cache_lookup`, `cache_store`, `web_cache_key`, `web_cache_enabled` — moved here unchanged. Use
`web_search_text`/`rerank_results` inside a RAG retriever; use `web_search`/`web_fetch`/`web_browse`
inside an agent's action space.

## Tools

| function | backend | returns | purpose |
|---|---|---|---|
| `web_search(query, results=10, language=None)` | SearXNG → Tavily → DuckDuckGo | `list[SearchResult]` | ranked results (title/url/snippet/engine/score), auto-paged ≤3 pages + URL de-dup + `suggestions` capture |
| `web_fetch(url, selector=None, stealthy=False, use_cache=True)` | scrapling CLI | `FetchResult` | one page → clean markdown; disk-cached; browser fetch, HTTP GET fallback; stealthy anti-bot |
| `web_batch_fetch(urls, max_concurrency=3)` | scrapling CLI | `BatchFetchResult` | 2–5 pages in parallel (bounded pool, order-preserved, per-page failure isolation) |
| `web_browse(url, actions, selector=None)` | agent-browser CLI | `BrowseResult` | click/fill/scroll/wait, then extract |

## Design decisions

- **Structured results.** Every tool returns a dataclass with `.to_dict()` for JSON tool-call
  payloads — agent code never parses free-form text. `BatchFetchResult` exposes
  `.succeeded/.failed/.ok_results()`.
- **CLI-driven backends.** `scrapling` and `agent-browser` are invoked as subprocesses, so the
  package needs only the CLIs on `PATH`, not their Python modules (env-independent). The
  invocations are pinned to the **installed** CLIs, not the reference repo's: scrapling here infers
  output format from the `.md` extension (no `--ai-targeted`), and `web_browse` drives a
  sequential one-command-per-session agent-browser (no `batch` subcommand).
- **Reproducible search.** Metasearch is non-deterministic (engines bot-block / rotate); results
  are disk-cached keyed by backend+config — same cache + key strategy as `web_search.py`.
- **Graceful degradation.** Per-page fetch failures return `ok=False` (never abort a batch);
  missing CLIs / optional deps raise one actionable error. Probe with `scrapling_available()` /
  `agent_browser_available()` before use.

## Use it

```python
import sys; sys.path.insert(0, "/Users/yuxinliu/code/agent-prep/shared")
from web_toolkit import web_search, web_fetch, web_batch_fetch, web_browse

hits  = web_search("python asyncio tutorial", results=5)         # -> list[SearchResult]
page  = web_fetch(hits[0].url, selector="article")               # -> FetchResult
batch = web_batch_fetch([h.url for h in hits[:3]])               # -> BatchFetchResult (parallel)
res   = web_browse("https://news.ycombinator.com",               # -> BrowseResult
                   [{"type": "scroll", "direction": "bottom"},
                    {"type": "wait", "ms": 800}], selector=".titleline")

from web_toolkit import web_search_text       # back-compat: list[str] like web_search.py
```

## Backends

- **web_search** → SearXNG. Reuse this repo's [`shared/searxng/`](../searxng/) docker-compose
  (free, local). `web_toolkit` defaults `SEARXNG_URL` to `http://localhost:8080`; the JSON API must
  be enabled (`search.formats: [html, json]`). Falls back to `TAVILY_API_KEY`, then `ddgs`.
- **web_fetch / web_batch_fetch** → `pip install "scrapling[all]" && scrapling install`.
- **web_browse** → `npm i -g agent-browser`.

## API reference

| symbol | signature | what |
|---|---|---|
| `web_search` | `(query, *, results=10, language=None, use_cache=True) -> list[SearchResult]` | ranked search across SearXNG→Tavily→DDG; after empty, see `web_toolkit.search.last_suggestions` |
| `web_search_text` | `(query, k=4) -> list[str]` | back-compat snippet strings (matches `web_search.py`'s shape) |
| `web_fetch` | `(url, *, selector=None, stealthy=False, timeout=60, use_cache=True) -> FetchResult` | single page → markdown; disk-cached by url+selector in `.web_cache.json` (same pool as search); per-page errors return `ok=False` |
| `web_batch_fetch` | `(urls, *, selector=None, stealthy=False, max_concurrency=3, timeout=60) -> BatchFetchResult` | bounded-parallel fetch, order-preserved |
| `web_browse` | `(url, actions=None, *, selector=None, headless=True, timeout=30) -> BrowseResult` | interactive session then extract |
| `scrapling_available` / `agent_browser_available` | `() -> bool` | capability probes |
| `SearchResult` / `FetchResult` / `BatchFetchResult` / `BrowseResult` / `BrowseAction` | dataclasses / TypedDict | typed results; `.to_dict()` for JSON |
| `SearchError` / `FetchError` / `BrowseError` | exceptions | backend unavailable / bad config |

Env: `SEARXNG_URL`, `SEARXNG_LANGUAGE` (default `en`), `SEARXNG_ENGINES`, `TAVILY_API_KEY`,
`WEB_CACHE` (1/0), `WEB_CACHE_PATH`, `SCRAPLING_BIN`, `AGENT_BROWSER_BIN`.

## Tests

```bash
PYTHONPATH=shared python -m pytest shared/web_toolkit/tests -q
```

Offline logic (command building, cache round-trip, result types) always runs; live tests
auto-skip when SearXNG / scrapling / agent-browser are unavailable. Verified **7 passed** with all
three backends present (SearXNG via `shared/searxng/`).
