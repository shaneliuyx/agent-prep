"""Extract (entity, relationship, entity) triples from each article
and write them to Neo4j. Ingestion is the expensive part of GraphRAG —
budget 8–12 minutes for 200 articles on local Gemma-4-26B."""
import os, json, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI
from neo4j import GraphDatabase
from tqdm import tqdm
from dotenv import load_dotenv

# LLM-server max_concurrent_requests = 8. Keep workers strictly under that
# so query_graph.py / IDE autocomplete / ad-hoc queries don't queue behind
# the build. If the build runs in isolation, raising MAX_WORKERS to 8 gives
# the full 3-5× speedup; raising past 8 idles threads in server-side queue.
MAX_WORKERS = 6

load_dotenv()
omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
MODEL = os.getenv("MODEL_SONNET")
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)

EXTRACT_SYSTEM = """Extract entities and relationships from the text.
Output JSON only: {"triples": [{"subject": str, "relation": str, "object": str}, ...]}.
Rules:
- Use the exact surface form that appears in the text for subject/object.
- Relations should be verb phrases, 1-4 words ("founded", "acquired by", "born in").
- Include 5-20 triples per article. Skip if the article has no clear entities.
- Do not invent facts. Every triple must be supported by the text."""


def extract_triples(text: str) -> list[dict]:
    resp = omlx.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user",   "content": text[:3500]},
        ],
        temperature=0.1, max_tokens=1200,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content).get("triples", [])
    except json.JSONDecodeError:
        return []


def write_triples_to_neo4j(tx, article_id: str, article_title: str, triples: list[dict]):
    """Each entity is a node, each triple creates a relationship.
    MERGE prevents duplicates across articles (e.g. 'Apple Inc.' in two articles
    resolves to the same node)."""
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


def _extract_one(article: dict) -> tuple[dict, list[dict], Exception | None]:
    """Worker function — runs extract_triples in a thread.

    Returns (article, triples, exc). Exception is captured rather than
    raised so a single failure doesn't kill the whole pool — bad article
    just contributes zero triples and we continue."""
    try:
        triples = extract_triples(article["text"])
        return article, triples, None
    except Exception as exc:  # noqa: BLE001 — we want all errors here
        return article, [], exc


def main():
    corpus = json.loads(Path("data/corpus.json").read_text())
    t0 = time.time()
    total_triples = 0
    errors: list[tuple[str, Exception]] = []

    with driver.session() as session:
        # Clear previous runs — safe for a lab, not safe for production.
        session.run("MATCH (n) DETACH DELETE n")
        # Drop the full-text index too so it gets rebuilt against fresh nodes.
        # IF EXISTS keeps the first-time-ever run from erroring.
        session.run("DROP INDEX entity_names IF EXISTS")

        # Threaded extraction: LLM calls run in parallel (LLM server caps at
        # MAX_CONCURRENT=8). Neo4j writes stay serial on the single session
        # — MERGE semantics are sensitive to interleaved concurrent writes
        # (deadlock potential on overlapping entity nodes) and the inference
        # server is the bottleneck anyway, not the database.
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(_extract_one, article) for article in corpus]
            for fut in tqdm(as_completed(futures), total=len(corpus)):
                article, triples, exc = fut.result()
                if exc is not None:
                    errors.append((article["title"], exc))
                    continue
                if triples:
                    session.execute_write(
                        write_triples_to_neo4j,
                        article["id"], article["title"], triples,
                    )
                total_triples += len(triples)

        # Full-text index over Entity.name. Replaces the CONTAINS-based fuzzy
        # match in query_graph.py, which produced false positives like
        # "meta" → "metal" / "mark" → "Denmark". Lucene tokenisation +
        # relevance scoring scopes matches to whole words and ranks them.
        session.run(
            "CREATE FULLTEXT INDEX entity_names IF NOT EXISTS "
            "FOR (n:Entity) ON EACH [n.name]"
        )

    elapsed = time.time() - t0
    print(f"\nIngested {len(corpus)} articles → {total_triples} triples in {elapsed:.0f}s")
    print(f"Average extraction rate: {total_triples / elapsed:.1f} triples/sec "
          f"(MAX_WORKERS={MAX_WORKERS})")
    print("Full-text index 'entity_names' created over Entity.name.")
    if errors:
        print(f"\nExtraction errors ({len(errors)}):")
        for title, exc in errors[:10]:
            print(f"  {title!r}: {type(exc).__name__}: {exc}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()