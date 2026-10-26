# SG (SSM/gate split) images on the exact base of the original container `vllm-mamba-sg`:
#   vllm/vllm-openai:v0.24.0 == sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f
#   (image id 4091d5593f77, build commit ee0da84ab9e04ac7610e28580af62c365e898389, CUDA 13.0.2, torch 2.11.0+cu130)
#
# Target `sg-replica`: the installed package plus the ten files copied out of the original container's
#   site-packages (vllm_overlays/sg-2026-07-31). This is byte-for-byte what produced the Aug-Sep 2026 benchmarks.
# Target `sg-dev`: an editable install of the fork checkout (git submodule ./vllm, branch mmp/sg) reusing the
#   image's compiled extensions, so edits to the fork run without a rebuild. Default target.
#
# Build (from the repo root, with the submodule initialised):
#   docker build --target sg-replica -t mmp:sg-replica -f docker/sg/Dockerfile .
#   docker build --target sg-dev     -t mmp:sg-dev     -f docker/sg/Dockerfile .
ARG BASE=vllm/vllm-openai:v0.24.0@sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f

FROM ${BASE} AS sg-replica
ENV PIP_ROOT_USER_ACTION=ignore
COPY vllm_overlays/sg-2026-07-31/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
RUN python3 -c "import vllm, vllm.distributed.mamba_sg_parallel_state as m, vllm.model_executor.layers.mamba.mamba_mixer_gate as g; print('sg overlay ok', vllm.__version__)"
RUN uv pip install --system tblib pytest pyyaml pandas
WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]

FROM ${BASE} AS sg-dev
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.24.0 \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.24.0 \
    VLLM_TARGET_DEVICE=empty \
    UV_LINK_MODE=copy \
    PIP_ROOT_USER_ACTION=ignore
RUN apt-get update && apt-get install -y --no-install-recommends git numactl rsync && rm -rf /var/lib/apt/lists/*
COPY vllm /opt/vllm
# the submodule's .git is a pointer into the parent repo and is meaningless inside the image
RUN rm -rf /opt/vllm/.git
RUN set -e; SP=/usr/local/lib/python3.12/dist-packages/vllm; \
    (cd "$SP" && find . -name "*.so" -print0 | xargs -0 -I{} cp --parents {} /opt/vllm/vllm/); \
    rm -rf /opt/vllm/vllm/vllm_flash_attn && cp -r "$SP/vllm_flash_attn" /opt/vllm/vllm/; \
    cd /opt/vllm && uv pip uninstall --system vllm; \
    uv pip install --system "setuptools>=77.0.3,<81" "setuptools-scm>=8" "setuptools-rust>=1.9" packaging wheel jinja2 regex; \
    cd /opt/vllm && uv pip install --system --no-build-isolation --no-deps -e .; \
    python3 -c "import os, vllm; p=os.path.dirname(vllm.__file__); assert p=='/opt/vllm/vllm', p; \
    assert os.path.exists(p+'/_C_stable_libtorch.abi3.so'); assert os.path.exists(p+'/vllm_flash_attn/_vllm_fa2_C.abi3.so'); \
    import vllm.distributed.mamba_sg_parallel_state; print('editable vllm', vllm.__version__, p)"
RUN uv pip install --system tblib pytest pyyaml pandas numpy matplotlib safetensors "huggingface_hub[cli]"
WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
