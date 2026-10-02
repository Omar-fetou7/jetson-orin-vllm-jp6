#!/usr/bin/env bash
# Print everything that decides whether this guide applies to your Jetson.
set -uo pipefail

echo "== L4T / JetPack"
cat /etc/nv_tegra_release 2>/dev/null || echo "not a Jetson? /etc/nv_tegra_release missing"
dpkg-query -W -f='${Version}\n' nvidia-jetpack 2>/dev/null | sed 's/^/nvidia-jetpack /'

echo "== CUDA toolkit"
/usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -n 2 || echo "nvcc not found at /usr/local/cuda/bin"

echo "== Python"
python3 --version

echo "== Memory and disk"
free -h | head -n 2
df -h "$HOME" | tail -n 1

echo "== Power mode"
sudo -n nvpmodel -q 2>/dev/null || echo "(run 'sudo nvpmodel -q' to see the power mode)"

cat <<'EOF'

Expected for this guide:
  L4T R36.x (JetPack 6), CUDA 12.6, Python 3.10, AGX Orin (sm_87).
  At least 60 GB free disk: vLLM build tree + wheels + model weights.
EOF
