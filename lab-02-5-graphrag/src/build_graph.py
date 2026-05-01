"""Extract (entity, relationship, entity) triples from each article and
write them to Neo4j.

v10 — sentence-aware sliding-window extraction.

Earlier versions truncated each article to 3500 chars and asked the
LLM for 5-20 triples. This silently dropped ~80% of long Wikipedia
articles (Education, Personal Life, Awards sections sit past
char 5000) and capped GraphRAG hit rate on bio-event questions.

This version:
  - Splits each article into sentence-aware sliding windows
    (~3000 chars per window, ~500 char overlap), so the full article
    text reaches the LLM extractor.
  - Asks the LLM for 10-15 triples per window. Aggregate per-article
    triple count is now ~70-100 for long bios (~3-5× v9 density).
  - Logs per-article progress (window count, cumulative triples) so
    the operator can see what's happening without staring at a tqdm
    bar that doesn't tell the whole story.
  - Reports proxy metrics at end: total triples, unique predicates,
    avg triples/article, max triples/article. Lets the operator
    distinguish "extraction worked + retrieval bottlenecked" from
    "extraction broken upstream".

Build budget: ~40 min wall on M5 Pro / Gemma-4-26B / MAX_WORKERS=6
for 400 articles × ~7 windows each = ~2800 LLM extraction calls."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from tqdm import tqdm

# Sentence-aware chunker — canonical impl in shared/rag_hybrid/chunking.py.
# Local helpers were removed in phase 7 of the rag_hybrid refactor.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from rag_hybrid.chunking import sentence_window_chunks  # noqa: E402

# LLM-server max_concurrent_requests = 8. Keep workers strictly under that
# so query_graph.py / IDE autocomplete / ad-hoc queries don't queue behind
# the build. If the build runs in isolation, raising MAX_WORKERS to 8 gives
# the full 3-5× speedup; raising past 8 idles threads in server-side queue.
MAX_WORKERS = 6

# Sliding-window chunking parameters.
WINDOW_CHARS = 3000
WINDOW_OVERLAP_CHARS = 500
MIN_WINDOW_CHARS = 200  # tail windows shorter than this are dropped (low signal)

load_dotenv()
omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
MODEL = os.getenv("MODEL_SONNET")
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)

# Updated prompt: per-window cap 10-15, broader category guidance so the
# LLM emits biographical events ("dropped out of", "married", "donated to")
# alongside corporate relations ("founded", "acquired by"). The original
# example list (founded / acquired by / born in) biased the LLM toward
# affiliation/ownership predicates and silently skipped life-event facts.
EXTRACT_SYSTEM = """Extract entities and relationships from the text segment.
Output JSON only: {"triples": [{"subject": str, "relation": str, "object": str}, ...]}.

Rules:
- Use the exact surface form that appears in the text for subject/object.
- **Always emit triples in ACTIVE voice.** If the source text is passive ("Apple was acquired by NeXT"), invert subject/object so the agent is the subject: emit subject="NeXT", relation="acquired", object="Apple". Applies to all by-suffix passives ("X was founded by Y", "X was published by Y", "X was sold to Y"). For passives where the subject is genuinely the patient ("John was awarded the Nobel Prize", "John was named CEO"), keep subject=John — the relationship describes John, not the agent. Use linguistic judgment per triple, not a fixed list.
- Relations are 1-4-word verb phrases. Include BOTH:
  * Corporate / affiliation relations: "founded", "acquired by", "co-founded",
    "led", "merged with", "invested in", "joined", "left".
  * Biographical / life events: "dropped out of", "graduated from", "married",
    "divorced", "donated to", "was sued by", "testified before", "moved to",
    "served as", "studied at".
  * Education / employment: "attended", "earned a degree from", "worked at",
    "interned at", "served on the board of".
- Include 10-15 triples per text segment. Skip if the segment has no clear entities.
- Do not invent facts. Every triple must be supported by the segment text.
- A single segment may repeat facts that appeared in earlier segments — that's
  fine, MERGE will dedupe."""


def extract_triples(text: str) -> list[dict]:
    """Run one extraction call on a single text window."""
    resp = omlx.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user",   "content": text},
        ],
        temperature=0.1,
        max_tokens=1500,  # bumped from 1200 to fit 10-15 triples per window
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content).get("triples", [])
    except (json.JSONDecodeError, AttributeError, TypeError):
        return []


def write_triples_to_neo4j(tx, article_id: str, article_title: str, triples: list[dict]):
    """Each entity is a node, each triple creates a relationship.

    MERGE prevents duplicates across articles (e.g. 'Apple Inc.' in two articles
    resolves to the same node). Same triple from overlapping windows of the
    same article also dedupes via MERGE on (subject, predicate, object)."""
    for t in triples:
        s, r, o = t.get("subject"), t.get("relation"), t.get("object")
        if not (s and r and o):
            continue
        rel_type = re.sub(r'[^A-Z_]', '_', r.upper().replace(' ', '_'))[:40] or "RELATED_TO"
        tx.run(
            f"""
            MERGE (a:Entity {{name: $s}})
            MERGE (b:Entity {{name: $o}})
            MERGE (a)-[rel:{rel_type}]->(b)
            ON CREATE SET rel.source_article = $aid, rel.source_title = $title,
                          rel.raw_relation = $r
            """,
            s=s, o=o, aid=article_id, title=article_title, r=r,
        )


def _extract_one(article: dict) -> tuple[dict, list[dict], int, Exception | None]:
    """Worker — runs sliding-window extraction over one article.

    Returns (article, all_triples, n_windows, exc). Exception captured
    rather than raised so a single failure does not kill the pool."""
    try:
        windows = sentence_window_chunks(
            article["text"],
            target_chars=WINDOW_CHARS,
            overlap_chars=WINDOW_OVERLAP_CHARS,
            min_window_chars=MIN_WINDOW_CHARS,
        )
        all_triples: list[dict] = []
        for window in windows:
            triples = extract_triples(window)
            all_triples.extend(triples)
        return article, all_triples, len(windows), None
    except Exception as exc:  # noqa: BLE001 — we want all errors here
        return article, [], 0, exc


def main() -> None:
    corpus = json.loads(Path("data/corpus.json").read_text())
    t0 = time.time()
    total_triples = 0
    errors: list[tuple[str, Exception]] = []
    triples_per_article: list[int] = []
    windows_per_article: list[int] = []
    predicate_counts: Counter[str] = Counter()
    article_chars = [len(a.get("text", "")) for a in corpus]
    print(
        f"Build start: {len(corpus)} articles, "
        f"avg {sum(article_chars) // max(len(article_chars), 1)} chars/article, "
        f"max {max(article_chars) if article_chars else 0} chars, "
        f"window={WINDOW_CHARS}c overlap={WINDOW_OVERLAP_CHARS}c, "
        f"MAX_WORKERS={MAX_WORKERS}"
    )

    with driver.session() as session:
        # Clear previous runs — safe for a lab, not safe for production.
        session.run("MATCH (n) DETACH DELETE n")
        # Create the full-text index BEFORE extraction, not after. Earlier
        # versions did DROP-then-extract-then-CREATE; if extraction crashed
        # mid-loop the DROP had already executed but CREATE never ran,
        # leaving the graph queryable but un-searchable. The IF NOT EXISTS
        # form makes CREATE idempotent — safe to run on a fresh DB or when
        # rebuilding. Index over Entity.name (replaces v6's CONTAINS
        # substring matching that produced "meta" → "metal" false positives).
        session.run(
            "CREATE FULLTEXT INDEX entity_names IF NOT EXISTS "
            "FOR (n:Entity) ON EACH [n.name]"
        )

        # Threaded extraction across articles. Each worker walks all windows
        # of its article serially (no nested parallelism) so we keep the
        # inference server's 8 slots saturated by 6 articles in flight.
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(_extract_one, article) for article in corpus]
            done = 0
            for fut in tqdm(as_completed(futures), total=len(corpus), desc="articles"):
                article, triples, n_windows, exc = fut.result()
                done += 1
                if exc is not None:
                    errors.append((article["title"], exc))
                    tqdm.write(
                        f"  [{done}/{len(corpus)}] ERROR {article['title']!r}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                if triples:
                    session.execute_write(
                        write_triples_to_neo4j,
                        article["id"], article["title"], triples,
                    )
                total_triples += len(triples)
                triples_per_article.append(len(triples))
                windows_per_article.append(n_windows)
                predicate_counts.update(t.get("relation", "") for t in triples if t.get("relation"))
                # Per-article progress line — every article emits one log so
                # the operator sees progress and per-article triple density.
                # tqdm.write doesn't break the progress bar.
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta_s = (len(corpus) - done) / max(rate, 1e-6)
                tqdm.write(
                    f"  [{done:>3}/{len(corpus)}] "
                    f"chars={len(article.get('text', '')):>5} "
                    f"windows={n_windows:>2} "
                    f"triples={len(triples):>3} "
                    f"total_triples={total_triples:>5} "
                    f"rate={rate * 60:.1f}/min ETA={eta_s/60:.1f}min  "
                    f"{article['title']!r}"
                )

        # (CREATE FULLTEXT INDEX moved to BEFORE the extraction loop above —
        # see the comment there. The index is idempotent + always present
        # post-build regardless of whether extraction crashes mid-loop.)

    elapsed = time.time() - t0
    print()
    print("=" * 72)
    print("INGEST SUMMARY")
    print("=" * 72)
    print(f"Articles ingested:           {len(corpus)}")
    print(f"Total triples extracted:     {total_triples}")
    if triples_per_article:
        avg_t = sum(triples_per_article) / len(triples_per_article)
        avg_w = sum(windows_per_article) / len(windows_per_article)
        print(f"Triples per article (avg):   {avg_t:.1f}")
        print(f"Triples per article (max):   {max(triples_per_article)}")
        print(f"Windows per article (avg):   {avg_w:.1f}")
        print(f"Windows per article (max):   {max(windows_per_article)}")
    print(f"Unique relation predicates:  {len(predicate_counts)}")
    print(f"Top 10 predicates:           {predicate_counts.most_common(10)}")
    print(f"Wall time:                   {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"Extraction rate:             {total_triples / elapsed:.1f} triples/sec  (MAX_WORKERS={MAX_WORKERS})")
    print(f"Full-text index:             entity_names (over Entity.name)")
    if errors:
        print(f"\nExtraction errors ({len(errors)}):")
        for title, exc in errors[:10]:
            print(f"  {title!r}: {type(exc).__name__}: {exc}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
