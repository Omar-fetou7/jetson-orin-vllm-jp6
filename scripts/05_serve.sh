#!/usr/bin/env bash
# Serve a local model directory through vLLM's OpenAI-compatible API.
#
#   ./scripts/05_serve.sh /path/to/model
#
# Download a model first, for example:
#   hf download <org>/<model> --local-dir ~/models/<model>
set -Eeuo pipefail

MODEL_PATH="${1:?usage: 05_serve.sh /path/to/model}"

# ---- settings -------------------------------------------------------------
WORKDIR="${WORKDIR:-$HOME/vllm-jp6}"
HOST="${HOST:-127.0.0.1}"          # use 0.0.0.0 only on a trusted network: no auth
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL_PATH")}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"      # max input + output tokens per request
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"  # sizes the KV cache
DTYPE="${DTYPE:-bfloat16}"
# 1 = skip FlashAttention 2 (use when FA2 lacks sm_87 kernels)
NO_FA2="${NO_FA2:-0}"
# 1 = disable torch.compile/CUDA Graphs (slower; only for debugging)
EAGER="${EAGER:-0}"
# ---------------------------------------------------------------------------

source "$WORKDIR/.venv/bin/activate"

args=(
  "$MODEL_PATH"
  --served-model-name "$SERVED_NAME"
  --host "$HOST" --port "$PORT"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --trust-remote-code
)
if [ "$NO_FA2" = "1" ]; then
  args+=(--attention-backend TRITON_ATTN --mm-encoder-attn-backend TORCH_SDPA)
fi
if [ "$EAGER" = "1" ]; then
  args+=(--enforce-eager)
fi

echo "vllm serve ${args[*]}"
exec vllm serve "${args[@]}"
