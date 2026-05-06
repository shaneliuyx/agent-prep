"""Pipeline variant: HyDE query rewriting before retrieval.

HyDE generates a short hypothetical answer used only as a retrieval query.
The final answer is still generated from retrieved context by 02_pipeline.py.
"""
import os
from openai import OpenAI
from src.script_wrap import load

pipeline = load("02_pipeline.py")
retrieve = pipeline.retrieve
rerank = pipeline.rerank
answer_from = pipeline.answer_from

omlx = OpenAI(base_url=os.getenv("OMLX_BASE_URL"), api_key=os.getenv("OMLX_API_KEY"))
SONNET = os.getenv("MODEL_SONNET")

# Final tested HyDE prompt, May 6.
# The original 3–5 sentence draft introduced extra vocabulary and slightly hurt ranking.
# A one-sentence draft reduced drift and improved context precision vs. long HyDE,
# but HyDE still did not beat the baseline overall on the current dev set.
ORIGINAL_HYDE_PROMPT = """Write a short factual paragraph (3–5 sentences) that would answer this question. If unsure, make a plausible draft — it's only used for retrieval, not shown to the user.

Question: {q}
Draft:"""

HYDE_PROMPT = """Write one concise factual sentence, fewer than 35 words, that would answer this question.
Use only the key entities, conditions, and relationships implied by the question.
Do not add examples, speculation, or extra background.
This sentence is only used for retrieval and will not be shown to the user.

Question: {q}
Retrieval sentence:"""


def hyde_rewrite(q):
    r = omlx.chat.completions.create(
        model=SONNET,
        temperature=0.0,
        max_tokens=80,
        messages=[{"role": "user", "content": HYDE_PROMPT.format(q=q)}],
    )
    return r.choices[0].message.content.strip()


def run_pipeline_hyde(q):
    hyp = hyde_rewrite(q)
    cands = retrieve(hyp, n=30)   # embed the hypothetical, not the query
    top = rerank(q, cands, k=5)   # rerank against the ORIGINAL query
    ans, _ = answer_from(q, top)
    return {
        "question": q,
        "answer": ans,
        "contexts": [h.payload["text"] for h in top],
        "context_ids": [h.payload["doc_id"] for h in top],
        "hypothetical": hyp,
    }
