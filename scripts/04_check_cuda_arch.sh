#!/usr/bin/env bash
# List the GPU architectures compiled into every vLLM native library and
# check their CUDA linkage. Every library should contain sm_87 and link only
# against CUDA 12 libraries.
set -uo pipefail

SITE=$(python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
CUOBJDUMP=/usr/local/cuda/bin/cuobjdump
status=0

while IFS= read -r so; do
  archs=$("$CUOBJDUMP" --list-elf "$so" 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')
  [ -z "$archs" ] && continue
  if echo "$archs" | grep -q 'sm_87'; then mark="OK  "; else mark="MISS"; status=1; fi
  printf '%s %-60s %s\n' "$mark" "${so#$SITE/}" "$archs"
  if ldd "$so" | grep -q 'libcudart.so.13'; then
    echo "     ^ links against CUDA 13 -> will not run on JetPack 6"; status=1
  fi
  if ldd "$so" | grep -q 'not found'; then
    echo "     ^ unresolved libraries:"; ldd "$so" | grep 'not found'; status=1
  fi
done < <(find "$SITE" -name '*.so')

if [ "$status" -ne 0 ]; then
  cat <<'EOF'

Some libraries are missing sm_87 or have bad linkage.
If only the FlashAttention library (_vllm_fa2_C) shows sm_80, either serve with
the Triton/SDPA backends (see 05_serve.sh) or rebuild FA2 (docs/flash-attention.md).
EOF
fi
exit "$status"
