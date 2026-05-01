"""Ingestor — chunked corpus → encoded vectors → Qdrant upsert.

Replaces three scattered scripts (W2 dense ingest, W2 hybrid ingest, W2.5 dense
+ hybrid ingest) with one class. Caller supplies the final payloads; Ingestor
handles encoding (with the right encoder for the collection spec), batching,
upsert, and retry.

Why caller supplies payloads (not raw corpus):
  Each lab has its own metadata shape — W2.5 wants article_title, W2 has
  doc_id only, future labs may add date/author. Letting the caller produce
  the final payload list keeps Ingestor free of per-lab transforms. The
  ``chunk_corpus`` helper covers the common case.

Retry policy:
  Qdrant gets transiently slow during background HNSW indexing. Same retry
  helper as W2's 00_hybrid_ingest.py (exponential backoff, max attempts).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence, TYPE_CHECKING

from qdrant_client.models import PointStruct, SparseVector

from .encoder import DenseEncoder, HybridEncoder
from .models import CollectionSpec, HybridCollectionSpec
from .schema import ensure_collection

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


@dataclass(frozen=True)
class IngestConfig:
    """Tunables for the encode + upsert loop. autoconfig.probe_memory() will
    scale ``encode_batch`` with available memory; everything else is stable."""
    encode_batch: int = 64           # M1: 32; M5 Pro / 48GB: 128 (autoconfig will set)
    upsert_batch: int = 256          # qdrant body chunks — sized for the 32MB HTTP limit
    retry_attempts: int = 4          # exponential backoff on transient slowdowns
    progress_every: int = 10         # log every N encode batches


def chunk_corpus(
    corpus: Sequence[dict],
    *,
    chunker: Callable[[str], list[str]],
    text_field: str = "text",
    id_field: str = "id",
    extra_payload: tuple[str, ...] = (),
) -> list[dict]:
    """Expand a list of corpus items into chunk payloads.

    For each item, calls ``chunker(item[text_field])`` and emits one payload
    dict per chunk. Chunk doc_id follows ``{item_id}_{chunk_index}`` —
    matches the convention used by every existing ingest in the labs.

    ``extra_payload`` is a tuple of field names from the corpus item to copy
    onto every chunk's payload (e.g. ``("title",)`` so retrieval can show
    article titles).
    """
    out: list[dict] = []
    for item in corpus:
        text = item[text_field]
        item_id = item[id_field]
        for i, chunk in enumerate(chunker(text)):
            payload: dict = {"doc_id": f"{item_id}_{i}", "text": chunk}
            for k in extra_payload:
                if k in item:
                    payload[k] = item[k]
            out.append(payload)
    return out


class Ingestor:
    """Encodes payloads and upserts to Qdrant. Handles dense or hybrid based
    on the collection spec — caller must pass the matching encoder type
    (HybridEncoder for HybridCollectionSpec, DenseEncoder for CollectionSpec).

    Construct once, call .run() per (payload list, collection spec) pair.

    Usage:
        encoder = HybridEncoder(EncoderConfig(spec=BGE_M3))
        ing = Ingestor(qd=QdrantClient(...), encoder=encoder)
        n = ing.run(payloads, BGE_M3_HYBRID)
    """

    def __init__(
        self,
        qd: "QdrantClient",
        encoder: HybridEncoder | DenseEncoder,
        cfg: IngestConfig | None = None,
    ) -> None:
        self.qd = qd
        self.encoder = encoder
        self.cfg = cfg or IngestConfig()

    def run(
        self,
        payloads: list[dict],
        collection_spec: CollectionSpec | HybridCollectionSpec,
        *,
        recreate: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> int:
        """Encode every payload's 'text', upsert with payload metadata.

        Returns the number of points upserted. Recreates the collection by
        default (matches every existing ingest script). Pass a ``progress``
        callback to capture log lines; defaults to ``print``.
        """
        log = progress or print
        ensure_collection(self.qd, collection_spec, recreate=recreate)
        log(f"collection {collection_spec.name!r} ready")

        is_hybrid = isinstance(collection_spec, HybridCollectionSpec)
        if is_hybrid and not isinstance(self.encoder, HybridEncoder):
            raise TypeError(
                f"hybrid spec needs HybridEncoder; got {type(self.encoder).__name__}"
            )
        if not is_hybrid and not isinstance(self.encoder, DenseEncoder):
            # Dense collection with HybridEncoder also works (just wastes the sparse output)
            # but is almost always a misconfiguration — flag it.
            if not isinstance(self.encoder, (DenseEncoder, HybridEncoder)):
                raise TypeError(
                    f"dense spec needs DenseEncoder; got {type(self.encoder).__name__}"
                )

        texts = [p["text"] for p in payloads]
        log(f"encoding {len(texts)} payloads (batch={self.cfg.encode_batch})")
        t0 = time.time()
        points = self._encode_to_points(texts, payloads, collection_spec, log)
        log(f"encoded {len(points)} in {time.time() - t0:.1f}s")

        t0 = time.time()
        for i in range(0, len(points), self.cfg.upsert_batch):
            self._upsert_with_retry(
                collection_spec.name, points[i : i + self.cfg.upsert_batch]
            )
        log(f"upserted in {time.time() - t0:.1f}s")

        info = self.qd.get_collection(collection_spec.name)
        log(f"collection {collection_spec.name!r}: {info.points_count} points")
        return info.points_count

    def _encode_to_points(
        self,
        texts: list[str],
        payloads: list[dict],
        spec: CollectionSpec | HybridCollectionSpec,
        log: Callable[[str], None],
    ) -> list[PointStruct]:
        is_hybrid = isinstance(spec, HybridCollectionSpec)
        points: list[PointStruct] = []
        bs = self.cfg.encode_batch

        for batch_start in range(0, len(texts), bs):
            batch_texts = texts[batch_start : batch_start + bs]
            batch_payloads = payloads[batch_start : batch_start + bs]

            if is_hybrid:
                assert isinstance(self.encoder, HybridEncoder)
                out = self.encoder.encode(batch_texts, batch_size=bs, with_sparse=True)
                dense_vecs = out["dense_vecs"]
                sparse_dicts = out["lexical_weights"]
                for j, payload in enumerate(batch_payloads):
                    sparse = SparseVector(
                        indices=list(map(int, sparse_dicts[j].keys())),
                        values=list(map(float, sparse_dicts[j].values())),
                    )
                    points.append(
                        PointStruct(
                            id=batch_start + j,
                            vector={
                                spec.dense_vector_name: dense_vecs[j].tolist(),
                                spec.sparse_vector_name: sparse,
                            },
                            payload=payload,
                        )
                    )
            else:
                # Dense path — DenseEncoder returns ndarray, HybridEncoder returns dict
                if isinstance(self.encoder, HybridEncoder):
                    out = self.encoder.encode(batch_texts, batch_size=bs, with_sparse=False)
                    dense_vecs = out["dense_vecs"]
                else:
                    dense_vecs = self.encoder.encode(batch_texts, batch_size=bs)
                for j, payload in enumerate(batch_payloads):
                    points.append(
                        PointStruct(
                            id=batch_start + j,
                            vector=dense_vecs[j].tolist(),
                            payload=payload,
                        )
                    )

            batch_idx = batch_start // bs
            if batch_idx % self.cfg.progress_every == 0:
                done = batch_start + len(batch_texts)
                log(f"  encoded {done}/{len(texts)}")

        return points

    def _upsert_with_retry(self, collection: str, points: list[PointStruct]) -> None:
        """Retry on transient httpx.ReadTimeout — qdrant gets slow during
        background HNSW build. Exponential backoff."""
        attempts = self.cfg.retry_attempts
        for k in range(attempts):
            try:
                self.qd.upsert(collection, points)
                return
            except Exception as e:
                if k == attempts - 1:
                    raise
                wait_s = 2 ** k
                print(
                    f"  upsert retry {k + 1}/{attempts} after "
                    f"{type(e).__name__} — sleeping {wait_s}s"
                )
                time.sleep(wait_s)
