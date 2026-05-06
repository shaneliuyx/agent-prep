"""Wikidata QID resolver — entity-name → canonical QID via wbsearchentities API.

The v11 multi-hop ceiling at recall@2-hop = 0.21-0.34 was diagnosed (Bad-Case
Journal Entry 10) as caused by entity surface-form fragmentation: "Reid Hoffman"
and "Reid Garrett Hoffman" become two separate nodes, splitting edges across
both and breaking 2-hop chains that traverse through the person.

This module solves the fragmentation at extraction time (not post-hoc, not via
embedding clustering) by linking each entity mention to its canonical Wikidata
QID. Wikidata maintains a SHARED canonical knowledge graph: every entity has
exactly one QID, regardless of which surface form was used in source text.

  "Bill Gates"           → Q5284
  "William Henry Gates III" → Q5284
  "B. Gates"             → Q5284
  "Microsoft"            → Q2283
  "Apple Inc."           → Q312

Build-time impact:
  - First run: ~13K unique names × ~10ms API call ≈ 2 min added to build wall
  - Subsequent runs: cache-hit, instant
  - Cache persisted to data/wikidata_qid_cache.json across runs

Failure modes (graceful):
  - API timeout / network down  → return None, fall back to name-based MERGE
  - Entity not in Wikidata      → return None, fall back to name-based MERGE
  - JSON parse error            → return None, fall back to name-based MERGE
  - Top match wrong (rare)      → caller still creates ONE node per QID,
                                  just keyed to the wrong canonical entity.
                                  Rare-and-survivable: forward search via
                                  Q-disambiguation could pick top-K, but
                                  top-1 is the production-grade default.

Concurrency: Python dicts are thread-safe under GIL for `__contains__` reads;
the lock is only acquired during writes (cache populates) so the 6-worker
extraction pool sees minimal contention.

API: https://www.wikidata.org/w/api.php?action=wbsearchentities&...
Free, no auth, rate limit ~50 req/s anonymous. mwapi library is a thin wrapper
over this — using `requests` directly avoids one more dependency.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "agent-prep-lab-graphrag/0.1 (educational; https://github.com/anthropic-research)"


class QIDResolver:
    """Resolve entity-name strings to canonical Wikidata QIDs with disk-backed
    cache. Designed for thread-safe shared use across extraction workers.

    Usage:
        resolver = QIDResolver(Path("data/wikidata_qid_cache.json"))
        qid = resolver.resolve("Bill Gates")  # → "Q5284"
        qid = resolver.resolve("Reid Hoffman")  # → "Q1361415"
        unmappable = resolver.resolve("FictionalCharacterX")  # → None
        resolver.save()  # persist cache to disk
    """

    def __init__(
        self,
        cache_path: Path,
        timeout_s: float = 10.0,
        save_every_n_calls: int = 200,
    ):
        self.cache_path = Path(cache_path)
        self.timeout_s = timeout_s
        self.save_every_n_calls = save_every_n_calls

        self.cache: dict[str, str | None] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text())
                # JSON null → Python None already; no conversion needed
            except (json.JSONDecodeError, OSError):
                self.cache = {}

        self._lock = threading.Lock()
        self._writes_since_save = 0
        self.api_calls = 0
        self.cache_hits = 0
        self.api_errors = 0

        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def resolve(self, name: str) -> str | None:
        """Return canonical Wikidata QID for `name`, or None if unmappable.

        Cached on first call. Subsequent calls (same name) return cached
        result with no API hit, including cached negatives (None)."""
        if not name or not name.strip():
            return None
        # Lock-free read — Python dict __contains__/__getitem__ are
        # GIL-protected for individual atomic ops; we only need the lock
        # for compound check-then-set.
        if name in self.cache:
            self.cache_hits += 1
            return self.cache[name]

        raw = self._lookup_api(name)

        if raw == self._API_ERROR:
            # Transient error — return None to caller but do NOT persist to cache.
            # On next build run the name will be re-resolved rather than reading
            # a stale cached None (the v12 cache-poisoning bug).
            return None

        with self._lock:
            self.cache[name] = raw
            self.api_calls += 1
            self._writes_since_save += 1
            if self._writes_since_save >= self.save_every_n_calls:
                self._save_locked()
                self._writes_since_save = 0
        return raw

    def resolve_batch(
        self,
        names: list[str] | set[str],
        max_workers: int = 16,
    ) -> dict[str, str | None]:
        """Resolve many entity names concurrently. Returns {name: qid_or_None}.

        Cache-first: any name already in cache is served from RAM with no
        API call. Remaining names are fanned out across a thread pool
        (~16 concurrent HTTPS calls — well below Wikidata's anonymous
        ~50 req/s limit). Empty / whitespace names map to None.

        Why this exists: per-name serial calls in a 6-worker extraction
        pool produced ~10-30s of QID overhead per article; with parallel
        batching, an article's 80-150 unique uncached names resolve in
        ~5-10s total instead of 8-30s.

        The thread pool here is created per call rather than long-lived,
        because batch sizes vary (10-300 names) and `requests.Session` is
        already pooling the underlying TCP connections via the parent
        QIDResolver. Each worker thread does a single .get() and exits.
        """
        result: dict[str, str | None] = {}
        to_fetch: list[str] = []
        for name in names:
            if not name or not name.strip():
                result[name] = None
                continue
            if name in self.cache:
                self.cache_hits += 1
                result[name] = self.cache[name]
            else:
                to_fetch.append(name)

        if not to_fetch:
            return result

        # Concurrent API calls. Each worker invokes _lookup_api which is
        # session-thread-safe (requests.Session() is documented thread-safe
        # for separate get/post calls). Results collected in dict — Python
        # GIL serializes individual dict assignments, no lock needed for
        # writes to a dict by multiple threads as long as keys are unique
        # per write (they are — each thread writes its own name).
        raw_results: dict[str, str | None] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_name = {
                pool.submit(self._lookup_api, name): name for name in to_fetch
            }
            for fut in future_to_name:
                name = future_to_name[fut]
                try:
                    raw_results[name] = fut.result()
                except Exception:  # noqa: BLE001 — counted as miss
                    raw_results[name] = self._API_ERROR
                    self.api_errors += 1

        # Cache only non-error results. Errors are returned as None to the caller
        # but never persisted — prevents cache-poisoning (the v12 bug).
        cacheable = {n: v for n, v in raw_results.items() if v != self._API_ERROR}
        with self._lock:
            for name, qid in cacheable.items():
                self.cache[name] = qid
                self.api_calls += 1
                self._writes_since_save += 1
            if self._writes_since_save >= self.save_every_n_calls:
                self._save_locked()
                self._writes_since_save = 0

        # Callers receive None for errors (graceful fallback to name-based MERGE)
        api_results = {n: (None if v == self._API_ERROR else v)
                       for n, v in raw_results.items()}
        result.update(api_results)
        return result

    # Sentinel: transient API error — do NOT cache (distinct from None = genuine miss)
    _API_ERROR: str = "__API_ERROR__"

    def _lookup_api(self, name: str) -> str | None:
        """One Wikidata wbsearchentities call. Returns top match QID, None (genuine
        no-match), or _API_ERROR (transient — must not be cached as None)."""
        try:
            r = self.session.get(
                WIKIDATA_API,
                params={
                    "action":   "wbsearchentities",
                    "search":   name,
                    "language": "en",
                    "type":     "item",
                    "limit":    1,
                    "format":   "json",
                },
                timeout=self.timeout_s,
            )
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError):
            self.api_errors += 1
            return self._API_ERROR  # transient — caller must not cache this

        results = data.get("search", [])
        if not results:
            return None
        return results[0].get("id")

    def save(self) -> None:
        """Force-persist cache to disk."""
        with self._lock:
            self._save_locked()
            self._writes_since_save = 0

    def _save_locked(self) -> None:
        """Internal save. Caller must hold self._lock."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False))
            tmp.replace(self.cache_path)
        except OSError:
            # Cache write is best-effort — losing the cache means more API
            # calls on the next run, not data loss in the graph.
            pass

    def stats(self) -> dict:
        """Return resolver counters for end-of-build summary."""
        total_lookups = self.api_calls + self.cache_hits
        cache_hit_rate = self.cache_hits / total_lookups if total_lookups else 0.0
        mapped = sum(1 for v in self.cache.values() if v is not None)
        unmapped = sum(1 for v in self.cache.values() if v is None)
        return {
            "total_lookups":  total_lookups,
            "cache_hits":     self.cache_hits,
            "api_calls":      self.api_calls,
            "api_errors":     self.api_errors,
            "cache_hit_rate": cache_hit_rate,
            "cache_size":     len(self.cache),
            "names_mapped":   mapped,
            "names_unmapped": unmapped,
            "qid_coverage":   mapped / len(self.cache) if self.cache else 0.0,
        }
