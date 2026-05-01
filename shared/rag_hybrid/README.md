# rag_hybrid

Modular hybrid-RAG library extracted from `lab-02-rerank-compress` and `lab-02-5-graphrag`.

## Why

Both labs duplicate: chunker, BGE-M3 load, Qdrant schema, RRF, reranker call. Each duplicate
drifts (W2.5 had `ENCODE_BATCH=64` based on wrong reasoning; reranker `.half()` only in
`retrieve.py`; two different RRF impls). One canonical implementation kills the drift class.

## Status — refactor in progress

| Step | Module | Status |
|------|--------|--------|
| 0 | freeze parity baseline | done |
| 1 | `models.py` (specs) | pending |
| 2 | `fusion.py` + `rerank.py` | pending |
| 3 | `encoder.py` + `chunking.py` | pending |
| 4 | `ingest.py` | pending |
| 5 | `retrieve.py` | pending |
| 6 | `autoconfig.py` (system probe only) | pending |
| 7 | migrate W2 + W2.5 | pending |
| 8 | delete duplicates | pending |

Parity gate at every step: mechanical (point counts + sample vector signatures + result-file
hashes). Full eval rerun once at end.

## Public API (target)

```python
from rag_hybrid import Ingestor, Retriever, autoconfig

cfg = autoconfig.recommend(corpus_path="data/corpus.json")
Ingestor(cfg.ingest).run("data/corpus.json", collection="tech_corpus_hybrid")
r = Retriever(collection="tech_corpus_hybrid", cfg=cfg.retrieve)
result = r.search_with_rerank("Who founded Apple?", k=5)
```

## How labs import it

Labs add `sys.path.insert(0, "../shared")` then `from rag_hybrid import ...`. Pattern matches
the existing W2.5 → W2 cross-lab import. No package install needed.
