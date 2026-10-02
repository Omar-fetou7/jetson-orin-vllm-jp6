#!/usr/bin/env bash
# Create an isolated virtual environment with Jetson builds of PyTorch 2.11,
# torchvision and Triton for JetPack 6 / CUDA 12.6.
#
# The Jetson AI Lab package index publishes wheels compiled for Jetson (sm_87).
# Do NOT install the generic PyPI / download.pytorch.org aarch64 wheels: they
# detect the Orin but have no sm_87 kernels.
set -Eeuo pipefail

# ---- settings -------------------------------------------------------------
WORKDIR="${WORKDIR:-$HOME/vllm-jp6}"
JETSON_INDEX="${JETSON_INDEX:-https://pypi.jetson-ai-lab.io/jp6/cu126}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
TRITON_VERSION="${TRITON_VERSION:-3.6.0}"
# ---------------------------------------------------------------------------

mkdir -p "$WORKDIR"
python3 -m venv "$WORKDIR/.venv"
source "$WORKDIR/.venv/bin/activate"
python -m pip install --upgrade pip wheel setuptools

python -m pip install \
  --index-url "$JETSON_INDEX" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "triton==${TRITON_VERSION}"

python -m pip freeze > "$WORKDIR/pytorch-freeze.txt"

cat <<EOF

Installed into $WORKDIR/.venv
Next:
  source $WORKDIR/.venv/bin/activate
  python scripts/02_validate_pytorch.py

If 'import torch' fails with a missing libcudss library, install NVIDIA cuDSS
for CUDA 12 (see NVIDIA's cuDSS documentation) or point LD_LIBRARY_PATH at it.
Keep this environment untouched as your known-good checkpoint.
EOF
