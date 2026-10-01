# Stage-1 image: vLLM v0.30.0 runtime + editable vLLM checkout + harness deps.
# The checkout defaults to the upstream tag; pass --build-arg VLLM_FORK=https://github.com/anuragdutt/vllm.git
# --build-arg VLLM_BRANCH=mmp/stage1 once the fork exists (or `git remote set-url` inside the container).
ARG VLLM_TAG=v0.30.0
FROM vllm/vllm-openai:${VLLM_TAG}
ARG VLLM_FORK=https://github.com/vllm-project/vllm.git
ARG VLLM_BRANCH=v0.30.0
ARG VLLM_WHEEL=https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0-cp38-abi3-manylinux_2_28_x86_64.whl
ENV VLLM_USE_PRECOMPILED=1 \
    VLLM_USE_PRECOMPILED_RUST=1 \
    VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_WHEEL} \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.30.0 \
    VLLM_TARGET_DEVICE=cuda \
    UV_LINK_MODE=copy \
    PIP_ROOT_USER_ACTION=ignore
RUN apt-get update && apt-get install -y --no-install-recommends git numactl rsync && rm -rf /var/lib/apt/lists/*
RUN git clone --branch ${VLLM_BRANCH} ${VLLM_FORK} /opt/vllm && cd /opt/vllm && git log -1 --oneline
RUN uv pip uninstall --system vllm \
 && uv pip install --system "setuptools>=77.0.3,<81" "setuptools-scm>=8" "setuptools-rust>=1.9" cmake ninja packaging wheel jinja2 regex \
 && cd /opt/vllm && uv pip install --system --no-build-isolation --no-deps -e . \
 && python3 -c "import vllm, vllm._C, vllm.vllm_flash_attn as fa; assert vllm.__file__.startswith('/opt/vllm'), vllm.__file__; print('vllm', vllm.__version__, 'FA2', fa.FA2_AVAILABLE)"
RUN uv pip install --system numpy pandas matplotlib pytest pyyaml rich safetensors "huggingface_hub[cli]"
WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
