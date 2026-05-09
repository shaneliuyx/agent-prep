#!/bin/bash
# Reset vMLX KV cache state to reduce cross-run variance.
#
# Two modes:
#   --soft (default) : send eviction-blast queries that flood KV cache
#                      with unrelated content, displacing prior eval residue.
#                      Fast (~5s), but only works if vMLX uses LRU eviction.
#   --hard           : kill per-model server processes; vMLX desktop app
#                      auto-respawns them (cold cache). Adds ~30-60s reload
#                      time but guarantees clean state.
#
# Usage:
#   ./scripts/reset_vmlx_cache.sh           # soft reset (default)
#   ./scripts/reset_vmlx_cache.sh --hard    # full process restart
set -euo pipefail

MODE="${1:---soft}"

if [[ "$MODE" == "--hard" ]]; then
  echo "[reset_vmlx_cache] HARD reset — killing per-model server processes"
  for pattern in "vmlx_engine.cli serve.*MLX-Qwen3.5-9B-GLM" \
                 "vmlx_engine.cli serve.*gemma-4-26B"; do
    pids=$(pgrep -f "$pattern" || true)
    if [[ -n "$pids" ]]; then
      echo "  kill -TERM $pids ($pattern)"
      kill -TERM $pids
    fi
  done

  # Wait for vMLX gateway to respawn + warm up
  echo "[reset_vmlx_cache] Waiting up to 90s for models to come back..."
  for i in $(seq 1 30); do
    if curl -sf http://localhost:8080/v1/models 2>/dev/null | grep -q "MLX-Qwen3.5-9B-GLM" && \
       curl -sf http://localhost:8080/v1/models 2>/dev/null | grep -q "gemma-4-26B"; then
      # Both models registered. Send warmup query to force load.
      python3 -c "
import os
from openai import OpenAI
c = OpenAI(base_url='http://localhost:8080/v1', api_key='nokey')
for m in ['models/MLX-Qwen3.5-9B-GLM5.1-Distill-v1-8bit', 'models/gemma-4-26B-A4B-it-heretic-4bit']:
    r = c.chat.completions.create(model=m, max_tokens=3, temperature=0.0,
                                   messages=[{'role':'user','content':'Reply OK only.'}])
    print(f'  warmed: {m} -> {(r.choices[0].message.content or \"\").strip()[:20]}')
" 2>/dev/null && break
    fi
    sleep 3
  done
  echo "[reset_vmlx_cache] HARD reset complete"
  exit 0
fi

# SOFT mode — eviction blast
echo "[reset_vmlx_cache] SOFT reset — flooding KV cache with eviction queries"
python3 << 'PY'
from openai import OpenAI
c = OpenAI(base_url='http://localhost:8080/v1', api_key='nokey')

# Long unrelated context fills cache blocks, evicting prior eval's prefix matches.
# vMLX paged cache uses block_size=64, max_cache_blocks=1000 for Gemma; flooding
# with 5 long queries (~3000 tokens each) displaces ~250 blocks of prior state.
EVICTION = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 200) + \
           "\n\nReply with the single word OK and nothing else."

for m in ['models/MLX-Qwen3.5-9B-GLM5.1-Distill-v1-8bit',
         'models/gemma-4-26B-A4B-it-heretic-4bit']:
    for i in range(3):
        try:
            c.chat.completions.create(
                model=m, max_tokens=3, temperature=0.0,
                messages=[{'role':'user','content': EVICTION}],
            )
        except Exception as e:
            print(f'  WARN {m}: {type(e).__name__}: {e}')
            break
    print(f'  evicted: {m}')
PY
echo "[reset_vmlx_cache] SOFT reset complete (~5-10s)"
