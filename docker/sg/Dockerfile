# SG (SSM/gate split) images on the exact base of the original container `vllm-mamba-sg`:
#   vllm/vllm-openai:v0.24.0 == sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f
#   (image id 4091d5593f77, build commit ee0da84ab9e04ac7610e28580af62c365e898389, CUDA 13.0.2, torch 2.11.0+cu130)
#
# Target `sg-replica`: the installed package plus the ten files copied out of the original container's
#   site-packages (vllm_overlays/sg-2026-07-31). This is byte-for-byte what produced the Aug-Sep 2026 benchmarks.
# Target `sg-src`: toolchain only. The prebuilt vLLM package is REMOVED from the image; vLLM is built from the
#   checkout's own kernel sources inside the running container (docker/sg/build_vllm.sh) and installed editable from
#   the mounted git checkout /workspace/vllm_fork, so a `git pull` of the fork is what runs. Default target.
#
# Build (from the repo root, with the submodule initialised):
#   docker build --target sg-replica -t mmp:sg-replica -f docker/sg/Dockerfile .
#   docker build --target sg-src     -t mmp:sg-src     -f docker/sg/Dockerfile .
ARG BASE=vllm/vllm-openai:v0.24.0@sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f

FROM ${BASE} AS sg-replica
ENV PIP_ROOT_USER_ACTION=ignore
COPY vllm_overlays/sg-2026-07-31/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
RUN python3 -c "import vllm, vllm.distributed.mamba_sg_parallel_state as m, vllm.model_executor.layers.mamba.mamba_mixer_gate as g; print('sg overlay ok', vllm.__version__)"
RUN uv pip install --system tblib pytest pyyaml pandas
WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]

FROM ${BASE} AS sg-src
ENV PIP_ROOT_USER_ACTION=ignore \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git ccache numactl rsync curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*
# 1. Remove the prebuilt vLLM completely: no wheel, no compiled extensions, no CLI. Only the base's torch/CUDA stack stays.
RUN python3 -m pip uninstall -y vllm && rm -rf /usr/local/lib/python3.12/dist-packages/vllm* /vllm-workspace/dist \
    && python3 -c "import importlib.util as u; assert u.find_spec('vllm') is None, 'prebuilt vllm still importable'"
# 2. Build toolchain (pyproject [build-system] of vLLM v0.24.0) plus the Rust toolchain for the tool-parser frontend.
RUN python3 -m pip install "cmake>=3.26.1" ninja "packaging>=24.2" "setuptools>=77.0.3,<81" "setuptools-scm>=8" \
    "setuptools-rust>=1.9" wheel jinja2 regex tblib pytest pyyaml pandas numpy matplotlib safetensors "huggingface_hub[cli]"
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path \
    && chmod -R a+rwX /opt/rustup /opt/cargo && cargo --version
# 3. The container runs as the host user (NFS root squash); it must be able to register the editable install.
RUN chmod -R a+rwX /usr/local/lib/python3.12/dist-packages /usr/local/bin && mkdir -p /build && chmod 1777 /build
# 4. Build defaults for guppy's A6000s (sm_86). For H200 nodes use TORCH_CUDA_ARCH_LIST="8.6;9.0a".
ENV TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=32 NVCC_THREADS=2 VLLM_TARGET_DEVICE=cuda \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.24.0 CCACHE_DIR=/build/ccache CARGO_TARGET_DIR=/build/cargo-target
WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
