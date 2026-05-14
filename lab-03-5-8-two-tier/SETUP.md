# Lab 3.5.8 — Two-Tier Memory Setup

## 1.3 Bring up EverCore (semantic tier)

```bash
cd ~/code  # outside the lab repo
git clone https://github.com/EverMind-AI/EverOS.git
cd EverOS/methods/EverCore

# Copy env template and fill in OPENAI_API_KEY (or point at oMLX-compatible endpoint)
cp env.template .env
# edit .env: set OPENAI_API_KEY + OPENAI_API_BASE if using oMLX
```

### Start data services

`docker-compose.yaml` ships **data services only** (Mongo, Elasticsearch, Milvus, Redis). The EverCore app itself runs locally via `uv` — see [Start app](#start-app) below.

```bash
docker compose up -d
# Wait ~30s for containers to become healthy
docker ps --format '{{.Names}}\t{{.Status}}' | grep memsys
```

Expected: 6 containers `Up ... (healthy)`. If `memsys-milvus-etcd` shows `(unhealthy)`, see [Known issue: etcd healthcheck](#known-issue-etcd-healthcheck).

### Start app

EverCore listens on **`0.0.0.0:1995`** (configurable via `MEMSYS_PORT` / `--port`).

```bash
uv sync                       # first run only, ~30s
uv run web                    # foreground; Ctrl-C to stop
# or: uv run web --port 1995 --host 0.0.0.0
```

Verify in another terminal:

```bash
curl http://localhost:1995/health
# {"status": "ok"}
```

### Known issue: etcd healthcheck

Upstream `docker-compose.yaml` has a healthcheck/command port mismatch on `milvus-etcd`:

- `command:` listens on **2479** (`-listen-client-urls http://0.0.0.0:2479`)
- `healthcheck:` queries default **2379** → always fails → status flag stays `unhealthy`

etcd itself is fine — Milvus connects via `milvus-etcd:2479` and works. Patch (cosmetic, makes the flag green):

```yaml
# docker-compose.yaml, milvus-etcd service
healthcheck:
  test: ["CMD", "etcdctl", "--endpoints=http://127.0.0.1:2479", "endpoint", "health"]
  interval: 30s
  timeout: 20s
  retries: 3
```

Verify manually anytime without restart:

```bash
docker exec memsys-milvus-etcd etcdctl --endpoints=http://127.0.0.1:2479 endpoint health
# http://127.0.0.1:2479 is healthy: successfully committed proposal: took = ~1ms
curl -sf http://localhost:9091/healthz   # Milvus → OK
```

### Notes

- Upstream README/lab text says `docker compose up -d` then `curl localhost:1995/health` — that **does not work as written**: the app is not in compose. Use `uv run web` for the app.
- Entrypoint `web` is defined in `pyproject.toml` → `[project.scripts]` → `src.run:main`.
- Port override precedence: CLI `--port` > env `MEMSYS_PORT` > default `1995`.
