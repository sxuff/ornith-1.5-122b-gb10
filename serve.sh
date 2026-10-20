#!/usr/bin/env bash
# Ornith-1.5-122B Q6_K on one DGX Spark (GB10): llama.cpp + built-in MTP head.
set -euo pipefail
LLAMA=${LLAMA:-$HOME/src/llama.cpp}
MODEL=${MODEL:-$HOME/models/Ornith-1.5-122B-GGUF/Ornith-1.5-122B-Q6_K.gguf}
exec "$LLAMA/build/bin/llama-server" -m "$MODEL" \
  --alias ornith-1.5-122b --host "${HOST:-127.0.0.1}" --port "${PORT:-8002}" \
  -ngl 999 -fa on -c 262144 -np 1 \
  -b 4096 -ub 4096 -ctk f16 -ctv f16 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 512 \
  --load-mode none -fit off -cram 1024 -ctxcp 8 \
  --jinja --reasoning-format deepseek --metrics
