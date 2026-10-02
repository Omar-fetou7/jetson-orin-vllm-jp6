# Docker runtime

Compiling vLLM takes hours. Do it once, then build this small image around the resulting wheel and distribute the image through a container registry.

## Build

```bash
mkdir -p docker/wheels
cp ~/vllm-jp6/wheels/vllm-*.whl docker/wheels/
cd docker
docker build -t jetson-orin-vllm:jp6-cu126-sm87 .
```

## Download a model (once)

Model weights stay outside the image, in `docker/models/`:

```bash
MODEL_ID=<org>/<model> MODEL_REVISION=<commit> docker compose run --rm download-model
```

Gated models need `HF_TOKEN` in the environment and accepted terms on Hugging Face.

## Serve

```bash
MODEL_NAME=<model> docker compose up -d vllm
curl -s http://127.0.0.1:8000/v1/models
docker compose logs -f vllm
```

The server has no authentication. Keep `HOST=127.0.0.1` and reach it through an SSH tunnel or VPN unless the network is trusted.

## Share it

Push the image to your registry (for example GitLab or GitHub Container Registry):

```bash
docker tag jetson-orin-vllm:jp6-cu126-sm87 <registry>/<group>/jetson-orin-vllm:jp6-cu126-sm87
docker push <registry>/<group>/jetson-orin-vllm:jp6-cu126-sm87
```

Large pushes can be interrupted; rerunning `docker push` resumes and reuses finished layers. Expect roughly 7–8 GB compressed.

## Notes

- The image only works on Orin (`sm_87`) with a JetPack 6 / CUDA 12.x host driver.
- Disk: about 22 GB for the expanded image plus your model files.
- `GPU_MEMORY_UTILIZATION` sizes vLLM's KV cache. Start at 0.40 on a 64 GB Orin and raise it only if you need many concurrent long requests.
