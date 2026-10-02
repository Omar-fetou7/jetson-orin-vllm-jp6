# Why the official vLLM container fails on JetPack 6

## The symptom

Running a recent official vLLM image on an AGX Orin with JetPack 6 stops before the model loads:

```text
nvidia-container-cli: requirement error: unsatisfied condition: cuda>=13.0
```

## Three different compatibility requirements

It helps to separate three things that are easy to conflate:

| Requirement | Decided by | Orin + JetPack 6 needs |
|---|---|---|
| GPU architecture | Which `sm_XX` kernels were compiled | `sm_87` |
| CUDA version | Which CUDA runtime the binaries link against, and the host **driver** | CUDA 12.x (12.6 on L4T R36.4) |
| Library ABI | Which PyTorch version vLLM's C++ extensions were built against | the exact PyTorch you run |

A container ships CUDA *user-space* libraries, but the kernel driver always comes from the host. JetPack 6's driver supports CUDA 12.6, so any image built on CUDA 13 fails, even if it contains `sm_87` kernels.

You can inspect an image without a GPU:

```bash
docker run --rm --runtime=runc --entrypoint python3 <image> -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch._C._cuda_getArchFlags())"
```

## Why not just pip install?

- **Public PyTorch aarch64 wheels** (download.pytorch.org, PyPI) target ARM servers (`sm_80`, `sm_90`, ...). On an Orin they report the GPU, then fail on the first kernel: `no kernel image is available for execution on the device`.
- **Recent vLLM requires a specific PyTorch** (0.23 needs 2.11) because its extensions use PyTorch's stable C++ ABI. Older Jetson PyTorch builds (2.5–2.10) cannot be used.
- **Prebuilt vLLM wheels** are built for x86 or for CUDA 13 on aarch64.

## What works

1. A PyTorch 2.11 wheel compiled for Jetson (`sm_87`, CUDA 12.6), from the Jetson AI Lab JP6 index.
2. vLLM compiled from source against that PyTorch with `TORCH_CUDA_ARCH_LIST=8.7`.
3. Optionally, FlashAttention 2 rebuilt for `sm_87` (see [flash-attention.md](flash-attention.md)).

Upgrading the Jetson to JetPack 7 (CUDA 13) is the officially supported alternative. It is often not an option when other software on the fleet is validated on JetPack 6.
