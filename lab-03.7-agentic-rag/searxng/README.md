# SearXNG — free web-fallback backend for the agentic-RAG pipeline

`baseline_handrolled.web_search()` picks a backend by precedence:

1. **`SEARXNG_URL`** — this local metasearch (free, no key, best free-source ranking)
2. **`TAVILY_API_KEY`** — managed API (good, but ranks paywalled aggregators high)
3. **DuckDuckGo** — free, no key, weakest ranking

## Run it

```bash
docker compose -f searxng/docker-compose.yml up -d         # starts searxng-lab on :8080
# wait ~25s for first boot, then confirm the JSON API is live:
python -c "import urllib.request,json; \
print(len(json.load(urllib.request.urlopen('http://localhost:8080/search?q=test&format=json'))['results']))"

export SEARXNG_URL=http://localhost:8080                    # or add to .env / mcp-config.json env
```

Stop / remove: `docker compose -f searxng/docker-compose.yml down`.

## Why it exists

For some atomic figures the managed/free single engines fail in a way **swapping between them
can't fix**, because they share ranking signals. The canonical case (§3.3): asking for
**Berkshire Hathaway Energy's 2023 revenue**, both Tavily and DuckDuckGo rank a *paywalled*
Statista snippet (`**** billion`) above the free Wikipedia figure (`US$26.198 billion`), so
the synthesizer correctly refuses to fabricate and abstains. SearXNG aggregates Google +
Startpage and **reranks**, floating Wikipedia into the top results — so the comparison query
`Compare BNSF Railway and Berkshire Energy 2023 revenue` grounds **both** figures with `[#N]`
citations instead of half-abstaining.

That asymmetry — same query, opposite outcome on a metasearch vs. a single engine — is the
teaching point: two retrievers that share a backend aren't an independent confirmation of
"unreachable." Full diagnosis + fix history: the **Web-Fallback Decomposition** debug log in
the curriculum vault.

## Note on the committed `secret_key`

`settings.yml` ships a fixed `secret_key` because this instance is **local-only** (bound to
`localhost` by the compose port mapping). Rotate it with `openssl rand -hex 32` if you ever
expose the port beyond your machine.
