#!/usr/bin/env bash
# Ornith-1.5-122B Q6_K on one DGX Spark (GB10): llama.cpp + built-in MTP head.
# Defaults are the recipe. bench/bench.py overrides UB/BS/SPEC/NMAX/KV/CTX/EXTRA per arm.
set -euo pipefail
LLAMA=${LLAMA:-$HOME/src/llama.cpp}
MODEL=${MODEL:-$HOME/models/Ornith-1.5-122B-GGUF/Ornith-1.5-122B-Q6_K.gguf}
ARGS=(-m "$MODEL" --alias ornith-1.5-122b --host "${HOST:-127.0.0.1}" --port "${PORT:-8002}"
  -ngl 999 -fa on -c "${CTX:-262144}" -np 1
  -b "${BS:-4096}" -ub "${UB:-4096}" -ctk "${KV:-f16}" -ctv "${KV:-f16}"
  --load-mode none -fit off -cram 1024 -ctxcp 8
  --jinja --reasoning-format deepseek --metrics)
SPEC=${SPEC:-draft-mtp}
if [ "$SPEC" != none ]; then
  ARGS+=(--spec-type "$SPEC" --spec-draft-n-max "${NMAX:-3}" --spec-draft-p-min "${PMIN:-0.0}")
  [ -n "${EXTRA+x}" ] || EXTRA="--spec-draft-n-ubatch 512"
fi
exec "$LLAMA/build/bin/llama-server" "${ARGS[@]}" ${EXTRA:-}
