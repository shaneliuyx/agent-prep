# Local MLX model comparison

Endpoint: oMLX `:8000` · perf trials: 3 · max_tokens(perf): 256
Reasoning set: 11 tasks (4 math, 3 logic, 4 code) · grading: programmatic

## Performance (higher tok/s = faster decode)

| Rank | Model | decode tok/s | TTFT (s) | total (s) | tok counted |
|------|-------|-------------:|---------:|----------:|:-----------:|
| 1 | `gpt-oss-20b-MXFP4-Q8` | 43.7 | 1.394 | 7.26 | server |
| 2 | `Qwen2.5-Coder-14B-Instruct-MLX-4bit` | 20.7 | 0.457 | 13.39 | server |
| 3 | `Codestral-22B-v0.1-mlx-nvfp4` | 15.6 | 0.401 | 17.16 | server |

## Reasoning (programmatic pass-rate)

| Rank | Model | overall | % | math | logic | code | med task (s) |
|------|-------|:-------:|--:|:----:|:-----:|:----:|-------------:|
| 1 | `gpt-oss-20b-MXFP4-Q8` | 11/11 | 100.0 | 4/4 | 3/3 | 4/4 | 8.49 |
| 2 | `Qwen2.5-Coder-14B-Instruct-MLX-4bit` | 7/11 | 63.6 | 3/4 | 1/3 | 3/4 | 6.83 |
| 3 | `Codestral-22B-v0.1-mlx-nvfp4` | 6/11 | 54.5 | 4/4 | 0/3 | 2/4 | 10.54 |

_Numbers come from this run only (model-compare-results/raw.json). tok/s 'est.' = endpoint omitted stream usage, counted as chars/4._
