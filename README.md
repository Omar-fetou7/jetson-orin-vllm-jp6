# vLLM on Jetson AGX Orin with JetPack 6

Run recent vLLM releases (0.23+) on a **Jetson AGX Orin with JetPack 6 / CUDA 12.6**, where the official vLLM containers require CUDA 13 (JetPack 7) and the public PyTorch wheels lack Orin's GPU architecture.

This repository documents the working path end to end and gives you small, readable scripts for each step:

1. Check the system.
2. Install a PyTorch 2.11 build compiled for Orin (`sm_87`) and prove it runs on the GPU.
3. Compile vLLM from source for `sm_87` + CUDA 12.6.
4. Verify every native library really contains `sm_87` code.
5. Serve a model through vLLM's OpenAI-compatible API.
6. Measure latency, time to first token and throughput.
7. Package the result as a Docker image so nobody has to compile it again.

It was developed while deploying NVIDIA **Cosmos3-Edge** (a vision-language reasoner) on an AGX Orin 64 GB, but nothing here is specific to that model.

## Tested configuration

| Component | Version |
|---|---|
| Device | Jetson AGX Orin 64 GB Developer Kit |
| JetPack / L4T | 6.2.1 / R36.4.4 |
| CUDA (host) | 12.6 |
| Python | 3.10 |
| PyTorch | 2.11.0 (Jetson build, `sm_87`) |
| torchvision | 0.26.0 |
| Triton | 3.6.0 (Jetson aarch64 build) |
| vLLM | 0.23.x, compiled from source |

Other JetPack 6.x / R36.x releases on AGX Orin should behave the same. Orin NX and Orin Nano share `sm_87` but have far less memory.

## Why the official container fails

Short version (details in [docs/why-the-official-container-fails.md](docs/why-the-official-container-fails.md)):

- A container brings its own CUDA *libraries* but uses the host's *driver*. JetPack 6's driver supports CUDA 12.6, so a CUDA 13 image stops with `requirements not met: cuda>=13.0`.
- GPU architecture and CUDA version are separate requirements. An image can include `sm_87` kernels and still be unusable on JetPack 6 because those kernels are linked against CUDA 13.
- The public PyTorch 2.11 CUDA 12.8 wheel detects the Orin but fails on the first kernel with `no kernel image is available for execution on the device`, because it is not built for `sm_87`.
- Recent vLLM requires PyTorch 2.11 (its stable C++ ABI), so older Jetson PyTorch builds are not an option.

The fix: a Jetson-specific PyTorch 2.11 build plus vLLM compiled from source against it, for `sm_87` and CUDA 12.6 only.

## Quick start

```bash
git clone <this-repo> && cd jetson-orin-vllm-jp6

./scripts/00_check_system.sh                 # JetPack, CUDA, Python, disk, memory
./scripts/01_setup_pytorch.sh                # venv + Jetson PyTorch 2.11 / torchvision / Triton
source ~/vllm-jp6/.venv/bin/activate
python scripts/02_validate_pytorch.py        # must pass on the GPU, not just "cuda available"
./scripts/03_build_vllm.sh                   # 4–7 hours on an AGX Orin
./scripts/04_check_cuda_arch.sh              # every .so must list sm_87
./scripts/05_serve.sh /path/to/model         # OpenAI-compatible API on 127.0.0.1:8000
python tools/ask_image.py photo.jpg --prompt "What is happening?"
python tools/benchmark_concurrency.py --requests 40 --concurrency 8
```

Each script starts with a block of settings you can edit. Run them one at a time and read the output; each step proves something before the next one depends on it.

## Attention backends and FlashAttention 2

vLLM compiles its own FlashAttention 2 (FA2) extension. On some vLLM revisions the pinned FA2 source builds for `sm_80` only, which `04_check_cuda_arch.sh` will reveal. You have two options:

- **Run without FA2** (works immediately, somewhat slower): `--attention-backend TRITON_ATTN` for the language model and `--mm-encoder-attn-backend TORCH_SDPA` for vision encoders. `05_serve.sh` has a switch for this.
- **Rebuild FA2 for `sm_87`**: add `8.7` to the CUDA architecture list used by vLLM's FlashAttention build and rebuild. See [docs/flash-attention.md](docs/flash-attention.md).

## Performance notes

From an AGX Orin 64 GB serving a ~2B-parameter vision-language model in BF16 (details in [docs/performance.md](docs/performance.md)):

- Removing `--enforce-eager` (enabling `torch.compile` and CUDA Graphs) raised single-request decode from ~16 to ~30 tokens/s.
- 80 requests at concurrency 8 reached ~300 aggregate tokens/s.
- The real limit for video was **prefill**, about 900 input tokens/s for image-heavy prompts. Keep each request under ~2,000 input tokens if you send one per second. Resize frames (e.g. 640×352 ≈ 220 visual tokens) instead of sending full HD.
- `--gpu-memory-utilization` sizes the KV cache, not the model. 0.75 on a 64 GB Orin reserves ~46 GiB; most workloads need far less.

## Docker

Compile once, then ship an image so other Orins never rebuild. See [docker/README.md](docker/README.md). Keep model weights outside the image in a volume.

## Repository layout

```text
scripts/   step-by-step setup, build, verification and serving
tools/     OpenAI-compatible client with latency metrics; concurrency benchmark
docker/    Dockerfile + compose file for a reusable runtime image
docs/      background: the CUDA/driver mismatch, FlashAttention, performance tuning
```

## License

MIT. vLLM, PyTorch and model weights keep their own licenses.
