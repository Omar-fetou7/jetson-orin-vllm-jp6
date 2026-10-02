# FlashAttention 2 on Orin

vLLM vendors and compiles its own FlashAttention 2 extension (`vllm/vllm_flash_attn/_vllm_fa2_C*.so`). Depending on the vLLM revision, that sub-build may choose its own GPU architecture list and produce only `sm_80` code, even when `TORCH_CUDA_ARCH_LIST=8.7` is set for the rest of vLLM.

## Detect it

```bash
./scripts/04_check_cuda_arch.sh
# or directly:
/usr/local/cuda/bin/cuobjdump --list-elf \
  $(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')/vllm_flash_attn/_vllm_fa2_C*.so
```

If you see only `sm_80`, the extension relies on JIT compilation from PTX at best, and vLLM may still select FA2 by default on the Orin.

## Option A: serve without FA2

FA2 is an optimisation, not a requirement for most models. Pick other backends explicitly:

```bash
vllm serve <model> \
  --attention-backend TRITON_ATTN \
  --mm-encoder-attn-backend TORCH_SDPA     # vision encoder of multimodal models
```

`scripts/05_serve.sh` does this with `NO_FA2=1`. Check the server log to confirm which backends were selected.

## Option B: rebuild FA2 for sm_87

1. In your vLLM checkout, find where the vendored FlashAttention build sets its CUDA architectures. Search the CMake files for the FA2 target and its architecture list.
2. Make sure `8.7` is included (typically a one- or two-line change next to the existing `8.0` entry).
3. Rebuild. You can reuse the build tree, so only the FA2 target recompiles; on an AGX Orin this still took several hours.
4. Verify that `cuobjdump --list-elf` now lists `sm_87` cubins for `_vllm_fa2_C`, and that `ldd` shows only CUDA 12 libraries.

Keep the FA2-enabled build in a separate virtual environment until it is validated, so the working Triton/SDPA setup stays available as a fallback.

## Does it matter?

With FA2 plus CUDA Graphs (no `--enforce-eager`), single-request decode on a ~2B vision-language model rose from ~16 to ~30 tokens/s on an AGX Orin 64 GB. Most of that gain came from enabling compilation and CUDA Graphs, so measure both separately on your model.
