#!/usr/bin/env bash
# Compile vLLM from source for Jetson Orin (sm_87) + CUDA 12.6, against the
# Jetson PyTorch installed by 01_setup_pytorch.sh.
#
# Runs for several hours. Start it inside tmux so a dropped SSH session does
# not kill the build:   tmux new -s vllm-build
set -Eeuo pipefail

# ---- settings -------------------------------------------------------------
WORKDIR="${WORKDIR:-$HOME/vllm-jp6}"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
# Pick the release or commit that supports your model. For example, the
# vLLM revision inside NVIDIA's Cosmos3 container was a 0.23.1 dev build.
VLLM_REF="${VLLM_REF:-v0.23.0}"
# Memory-safe parallelism for a 64 GB Orin; lower MAX_JOBS on 32 GB devices.
MAX_JOBS="${MAX_JOBS:-2}"
NVCC_THREADS="${NVCC_THREADS:-1}"
# ---------------------------------------------------------------------------

source "$WORKDIR/.venv/bin/activate"

# Build only for Orin and use the host CUDA 12.6 toolchain.
export TORCH_CUDA_ARCH_LIST="8.7"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS NVCC_THREADS
# Never download prebuilt (CUDA 13 / x86) binaries; compile everything here.
export VLLM_USE_PRECOMPILED=0
# Kernels written for Hopper/Blackwell do not apply to sm_87. Recent vLLM
# versions skip them for 8.7 anyway; these switches are harmless if unused.
export VLLM_USE_DEEP_GEMM=0

# vLLM needs a newer CMake than Ubuntu 22.04 ships. Install it into the venv
# only, without touching the system CMake.
python -m pip install "cmake>=3.26" ninja packaging setuptools-scm

mkdir -p "$WORKDIR/src" "$WORKDIR/wheels" "$WORKDIR/logs"
if [ ! -d "$WORKDIR/src/vllm" ]; then
  git clone "$VLLM_REPO" "$WORKDIR/src/vllm"
fi
cd "$WORKDIR/src/vllm"
git fetch --all --tags
git checkout "$VLLM_REF"

# Make vLLM use the PyTorch already installed instead of pinning its own.
python use_existing_torch.py
python -m pip install -r requirements/build.txt

LOG="$WORKDIR/logs/vllm-build-$(date +%Y%m%d-%H%M%S).log"
echo "Building vLLM $VLLM_REF for sm_87. Log: $LOG"
python -m pip wheel . --no-build-isolation --no-deps -w "$WORKDIR/wheels" -v 2>&1 | tee "$LOG"

ls -lh "$WORKDIR/wheels"/vllm-*.whl
sha256sum "$WORKDIR/wheels"/vllm-*.whl | tee "$WORKDIR/wheels/SHA256SUMS"

cat <<EOF

Install the wheel into a NEW environment cloned from the PyTorch checkpoint,
or into the same one if you prefer:
  python -m pip install $WORKDIR/wheels/vllm-*.whl
Then run scripts/04_check_cuda_arch.sh.
EOF
