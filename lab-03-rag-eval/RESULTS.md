# Lab 03 — RAG Evaluation Results

**Date:** 2026-05-06  
**Corpus:** local `docs.jsonl` indexed in Qdrant collection `bge_m3_hnsw`  
**Dev set:** regenerated harder 50-question set, preserving original candidate columns: `source_doc_id`, `source_text`, `question`, `short_answer`  
**Baseline pipeline:** BGE-M3 dense retrieval top-30 → BGE reranker top-5 → local oMLX synthesis → RAGAS evaluation

## Executive Summary

The final baseline is strong on retrieval and grounding but still has room to improve answer targeting.

The retriever/reranker is not the bottleneck: `context_precision = 0.9824` and `context_recall = 1.0000`. The generator is also very well grounded: `faithfulness = 0.9900`. The remaining weakness is answer formulation: `answer_relevancy = 0.7494`, which means some answers are supported by context but do not always map tightly enough to the exact question.

The prompt A/B showed that adding conditional/contrast guidance alone made the model too verbose and hurt both faithfulness and answer relevancy. Adding a strict one-sentence, fewer-than-35-words cap recovered faithfulness and produced the best result on the harder dev set.

## RAGAS Scores Across Prompt / Dev-Set Iterations

| Run | Dev set | Prompt variant | Faithfulness | Answer relevancy | Context precision | Context recall | Interpretation |
|---|---|---|---:|---:|---:|---:|---|
| 1 | Easier first curated set | Original concise prompt | 0.9800 | 0.8952 | 0.9726 | 1.0000 | Scores were very high; dev set likely too easy. |
| 2 | Harder regenerated set | Original concise prompt | 0.9800 | 0.7336 | 0.9824 | 1.0000 | Harder questions exposed synthesis weakness. |
| 3 | Harder regenerated set | Enhanced prompt, no strict length cap | 0.9400 | 0.6537 | 0.9824 | 1.0000 | Prompt became too verbose / explanatory. |
| 4 | Harder regenerated set | Enhanced prompt + one sentence under 35 words | 0.9900 | 0.7494 | 0.9824 | 1.0000 | Best on hand-rolled retrieval — adopted as v2. |
| **5** | Harder regenerated set | v2 + post `shared/rag_hybrid` migration (autoconfig'd encoder + fp16 reranker) | **1.0000** | **0.7297** | **0.9841** | 1.0000 | **Migration safe — all 4 metrics within ±0.02 of Run 4. Shipped state.** |

Run 5 deltas vs Run 4 (the §2.6 migration regression-test contract): faithfulness +0.0100 (improvement), answer_relevancy -0.0197 (within ±0.02 contract; well within ~0.03 LLM-judge variance on n=50), context_precision +0.0017 (within noise), context_recall flat. Migration is infrastructure-level (autoconfig probes device + memory tier; cross-encoder fp16-safe enabled where supported); no behavior change beyond LLM-judge noise floor.

## Final Baseline Decision

**Adopt baseline prompt v2:**

```text
Use ONLY the context below.
Answer the exact question asked.
If the question asks why, give the reason.
If it asks how two things differ, state the contrast.
If it asks under what condition, state the condition.
Keep the answer concise, but include enough detail to directly satisfy the question.
If the context does not contain the answer, say exactly: insufficient context.
Answer in one sentence of fewer than 35 words.
```

This version slightly improves over the original prompt on the harder dev set:

| Metric | Original prompt on harder dev set | Final prompt v2 | Delta |
|---|---:|---:|---:|
| Faithfulness | 0.9800 | 0.9900 | +0.0100 |
| Answer relevancy | 0.7336 | 0.7494 | +0.0158 |
| Context precision | 0.9824 | 0.9824 | 0.0000 |
| Context recall | 1.0000 | 1.0000 | 0.0000 |

The gain is modest, but it moves the only weak metric in the right direction without hurting retrieval or grounding.

## HyDE A/B Results

HyDE was tested twice on the same harder 50-question dev set. The baseline already had perfect context recall, so HyDE had no retrieval gap to close. The main question was whether a hypothetical answer could improve ranking quality without hurting answer targeting.

| Variant | Faithfulness | Answer relevancy | Context precision | Context recall | Decision |
|---|---:|---:|---:|---:|---|
| Baseline prompt v2 | 0.9900 | 0.7494 | 0.9824 | 1.0000 | Keep as default |
| HyDE, original 3–5 sentence draft | 0.9898 | 0.7286 | 0.9781 | 1.0000 | Reject |
| HyDE, one sentence under 35 words | 0.9900 | 0.7293 | 0.9851 | 1.0000 | Reject as default |

**Interpretation:** Shortening the HyDE draft improved context precision versus the original 3–5 sentence HyDE prompt (`0.9781 → 0.9851`), which confirms the long draft was adding retrieval drift. However, answer relevancy still stayed below the single-pass baseline (`0.7293` vs `0.7494`) while adding an extra LLM call per query.

**Decision:** Reject HyDE as the default retrieval strategy for this corpus/dev set. Keep it as an optional variant for future query clusters where baseline `context_recall` is low or vocabulary mismatch is visible.

**Implementation note:** The first HyDE eval run printed `=== HYDE ===` but still wrote to `results/ragas_baseline.json` and `results/ragas_baseline_debug.jsonl`. The HyDE eval should write to `results/ragas_hyde.json` and `results/ragas_hyde_debug.jsonl` to avoid overwriting baseline artifacts.

## Diagnostic Reading

The final pattern is:

```text
High context_recall + high context_precision + high faithfulness + medium answer_relevancy
= retrieval is working; generation is grounded; answer targeting is the main remaining issue.
```

The evaluation did what it should: the first easier dev set produced near-ceiling metrics, so we regenerated a harder dev set. That harder set lowered answer relevancy while leaving context recall and context precision high, showing that the retriever can find evidence but the answer prompt still needs to make the model respond more directly to the question.

## Debugging and Corrections Completed

### 1. Dev-set schema correction

Earlier output included extra fields such as `qid`. The curated dev set was corrected to preserve the original candidate format:

```text
source_doc_id, source_text, question, short_answer
```

This keeps the file compatible with the candidate generator and makes each question auditable against its source passage.

### 2. Harder dev-set regeneration

The first 50-question set was too easy. Many questions were direct definition, title, address, phone-number, or keyword lookup questions. The harder set keeps questions answerable from the source passage alone but prefers causal, conditional, contrastive, and relationship-style questions.

### 3. Protobuf dependency fix

RAGAS imports `instructor`, which can import `xai_sdk`; `xai_sdk` rejects protobuf 7.x. Pin protobuf below 7:

```bash
uv add "protobuf>=5.29.4,<7"
```

Venv-only repair option:

```bash
uv pip install "protobuf==6.33.0"
```

### 4. Numeric module import fix

Python cannot import `src.02_pipeline` using normal dotted import syntax because module names cannot start with a digit. The fix is `src/pipeline_wrap.py`, which loads `02_pipeline.py` by path and re-exports `run_pipeline`, `retrieve`, `rerank`, and `answer_from`.

### 5. `ModuleNotFoundError: No module named 'src'` fix

Running `python src/02b_ragas_eval.py` puts `src/` on `sys.path`, not the project root. The eval script now inserts the project root into `sys.path` before importing `src.pipeline_wrap`.

### 6. RAGAS API compatibility fix

The attempted modern path failed in this local setup:

- `ragas.metrics.collections` rejected `LangchainEmbeddingsWrapper`.
- RAGAS native `HuggingfaceEmbeddings` was abstract / not instantiable in the installed version.
- The reliable local path is legacy metric classes plus `LangchainEmbeddingsWrapper`.

The final eval script intentionally uses:

```python
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
```

Warnings are suppressed because this is the compatible local path for oMLX + local BGE-M3 embeddings.

### 7. Timeout and generation-count warnings

RAGAS sometimes reported:

```text
LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Exception raised in Job[86]: TimeoutError()
```

The generation-count warning is tolerable for local OpenAI-compatible backends that do not fully support `n=3`. The timeout was mitigated by adding:

```python
run_config=RunConfig(timeout=300, max_retries=3, max_workers=2)
```

This reduces concurrent judge pressure on the local model and gives slow calls room to finish.

## Bad-Case Journal

### Entry 1 — Dev set was too easy

**Symptom:** The first baseline scored `answer_relevancy = 0.8952`, `context_recall = 1.0000`, and `faithfulness = 0.9800`.

**Root cause:** Too many candidate questions were direct lookup or definition-style questions. These were answerable but not discriminative enough for retrieval or synthesis quality.

**Fix:** Regenerated / curated a harder 50-question dev set. Kept each question answerable from `source_text`, but preferred conditional, causal, contrastive, and relationship-style questions.

**Result:** On the harder set, the original prompt answer relevancy dropped from `0.8952` to `0.7336`, revealing a real synthesis-layer weakness.

### Entry 2 — More detailed prompt made answers worse

**Symptom:** Enhanced prompt without a strict length cap produced `faithfulness = 0.9400` and `answer_relevancy = 0.6537`, worse than the original prompt.

**Root cause:** The prompt encouraged explanation and multiple behavioral branches, which likely made answers longer and less focused. Longer answers create more chances for unsupported claims and make RAGAS reverse-question similarity worse.

**Fix:** Added a strict answer-shape constraint: one sentence and fewer than 35 words.

**Result:** Final prompt v2 improved to `faithfulness = 0.9900` and `answer_relevancy = 0.7494`.

### Entry 3 — RAGAS modern API conflict with local embeddings

**Symptom:** `ragas.metrics.collections` raised an error that collections metrics only support modern embeddings; native RAGAS HuggingFace embeddings then failed as an abstract class.

**Root cause:** Installed RAGAS version did not provide a clean fully-modern local HuggingFace embedding path for BGE-M3 on MPS.

**Fix:** Use the compatible legacy metric classes with `LangchainEmbeddingsWrapper`, suppressing deprecation warnings for this lab environment.

**Result:** Eval runs end-to-end with local oMLX and local BGE-M3.

### Entry 4 — Local judge timeout

**Symptom:** RAGAS evaluation timed out around the middle of the 200 metric jobs.

**Root cause:** RAGAS judge calls are expensive and concurrent; local oMLX can time out under too much parallelism.

**Fix:** Added `RunConfig(timeout=300, max_retries=3, max_workers=2)`.

**Result:** More stable local eval runs, with a clear knob to reduce `max_workers` to 1 if needed.

### Entry 5 — Numeric module import and project-root import issues

**Symptom:** `from src.02_pipeline import run_pipeline` caused `SyntaxError: invalid decimal literal`; later `from src.pipeline_wrap import run_pipeline` caused `ModuleNotFoundError: No module named 'src'` when running by path.

**Root cause:** Python module names cannot begin with digits, and running a script under `src/` does not automatically place the project root on `sys.path`.

**Fix:** Added `pipeline_wrap.py` and inserted project root into `sys.path` in `02b_ragas_eval.py`.

**Result:** `python src/02b_ragas_eval.py` works from the lab root.

### Entry 6 — HyDE added cost without improving the default pipeline

**Symptom:** HyDE produced `context_recall = 1.0000`, but answer relevancy stayed below the baseline. The original 3–5 sentence HyDE scored `answer_relevancy = 0.7286`; the shorter one-sentence HyDE improved ranking but still scored `answer_relevancy = 0.7293`, below the baseline `0.7494`.

**Root cause:** The baseline retriever already found the needed evidence for every question. HyDE therefore had no recall gap to close. The longer hypothetical answer introduced extra vocabulary that slightly hurt context precision; shortening the draft reduced that drift but still did not improve answer targeting.

**Fix:** Use a shorter HyDE retrieval sentence when testing HyDE, but reject HyDE as the default for this corpus/dev set. Keep it as an opt-in variant for future query clusters with low baseline recall or clear vocabulary mismatch.

**Result:** HyDE-short improved context precision over HyDE-long (`0.9781 → 0.9851`) but did not beat the baseline on answer relevancy.

**5-second sanity test:** If baseline `context_recall` is already `1.0000`, HyDE is unlikely to help unless it improves ranking or answer quality enough to offset the extra LLM call.

## Current Pipeline Code Notes

### `src/02_pipeline.py`

- Uses BGE-M3 dense retrieval with `n=30`.
- Uses BGE reranker top-5.
- Uses prompt v2 selected from A/B results.
- Uses `temperature=0.0` for deterministic, grounded synthesis.
- Uses `max_tokens=120`, enough for a one-sentence answer but lower than the earlier 200-token default.

### `src/02b_ragas_eval.py`

- Uses `pipeline_wrap.py` to avoid numeric-module import issues.
- Inserts project root into `sys.path` so the script works when invoked as `python src/02b_ragas_eval.py`.
- Uses compatible RAGAS legacy metric classes.
- Adds `RunConfig(timeout=300, max_retries=3, max_workers=2)`.
- Writes both aggregate scores and a debug JSONL file:
  - `results/ragas_baseline.json`
  - `results/ragas_baseline_debug.jsonl`

## Next Steps

1. Inspect `results/ragas_baseline_debug.jsonl` manually for low-answer-relevancy examples.
2. Tag each failure as one of:
   - answer too narrow
   - answer too broad
   - answer missing condition
   - answer not using contrast format
   - RAGAS judge noise
3. Only after manual inspection, decide whether to tune prompt again.
4. Run multi-query A/B on the same harder dev set before making final architecture decisions. HyDE has been tested and rejected as the default for this dev set.

## Results Table for `ARCHITECTURE.md`

| Variant | Faithfulness | Answer relevancy | Context precision | Context recall | p95 latency |
|---|---:|---:|---:|---:|---:|
| baseline prompt v2 | 0.9900 | 0.7494 | 0.9824 | 1.0000 | TBD |
| + HyDE, 3–5 sentence draft | 0.9898 | 0.7286 | 0.9781 | 1.0000 | TBD |
| + HyDE, 1 sentence draft | 0.9900 | 0.7293 | 0.9851 | 1.0000 | TBD |
| + multi-query fusion | TBD | TBD | TBD | TBD | TBD |

**Decision so far:** Adopt baseline prompt v2 as the single-pass baseline. Reject HyDE as the default because recall was already perfect and HyDE lowered answer relevancy while adding an extra LLM call. Multi-query fusion remains to be tested.
