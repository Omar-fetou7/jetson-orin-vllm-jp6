#!/usr/bin/env bash
# Container entrypoint: serve the model found at $MODEL_DIR.
set -Eeuo pipefail

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR, e.g. /models/my-model}"
args=(
  "$MODEL_DIR"
  --served-model-name "${SERVED_NAME:-$(basename "$MODEL_DIR")}"
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
  --dtype "${DTYPE:-bfloat16}"
  --max-model-len "${MAX_MODEL_LEN:-8192}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.40}"
  --trust-remote-code
)
if [ "${NO_FA2:-0}" = "1" ]; then
  args+=(--attention-backend TRITON_ATTN --mm-encoder-attn-backend TORCH_SDPA)
fi
# Extra flags can be appended: docker compose run ... serve.sh --foo bar
exec vllm serve "${args[@]}" "$@"
