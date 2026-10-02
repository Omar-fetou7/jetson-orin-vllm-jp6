# Reusable vLLM runtime for Jetson AGX Orin, JetPack 6 (L4T R36.4, CUDA 12.6).
#
# The expensive part (compiling vLLM) happens once, outside this file, with
# scripts/03_build_vllm.sh. This image only installs the resulting wheels, so
# it builds in minutes and other Orins can pull it instead of compiling.
#
# Before building, copy into docker/wheels/:
#   - your vllm-*.whl from scripts/03_build_vllm.sh
# PyTorch, torchvision and Triton are installed from the Jetson AI Lab index.

ARG BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE}

ARG JETSON_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TRITON_VERSION=3.6.0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    VENV=/opt/vllm-venv
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-venv python3-pip curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m venv ${VENV}
ENV PATH=${VENV}/bin:${PATH}

RUN pip install --upgrade pip && \
    pip install --index-url ${JETSON_INDEX} \
        torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} triton==${TRITON_VERSION}

COPY wheels/ /tmp/wheels/
RUN pip install /tmp/wheels/vllm-*.whl "huggingface_hub[cli]" && rm -rf /tmp/wheels

COPY serve.sh /usr/local/bin/serve.sh
RUN chmod +x /usr/local/bin/serve.sh

# Models are mounted at /models; nothing is downloaded at startup.
VOLUME ["/models"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
  CMD curl -fs http://127.0.0.1:${PORT:-8000}/health || exit 1
ENTRYPOINT ["/usr/local/bin/serve.sh"]
