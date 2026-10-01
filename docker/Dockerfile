# Stage-1 image: vLLM v0.30.0 runtime + editable vLLM checkout + harness deps.
# The compiled extensions (stable-libtorch _C ops, FlashAttention, MoE, allocator, rust parser) are copied
# from the image's own installed wheel into the editable tree, so they match this image's torch/CUDA exactly.
# The checkout defaults to the upstream tag; pass --build-arg VLLM_FORK=https://github.com/anuragdutt/vllm.git
# --build-arg VLLM_BRANCH=mmp/stage1 once the fork exists (or `git remote set-url` inside the container).
ARG VLLM_TAG=v0.30.0
FROM vllm/vllm-openai:${VLLM_TAG}
ARG VLLM_FORK=https://github.com/vllm-project/vllm.git
ARG VLLM_BRANCH=v0.30.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.30.0 \
    VLLM_TARGET_DEVICE=empty \
    UV_LINK_MODE=copy \
    PIP_ROOT_USER_ACTION=ignore
RUN apt-get update && apt-get install -y --no-install-recommends git numactl rsync && rm -rf /var/lib/apt/lists/*
RUN git clone --branch ${VLLM_BRANCH} ${VLLM_FORK} /opt/vllm && cd /opt/vllm && git log -1 --oneline
RUN set -e; SP=/usr/local/lib/python3.12/dist-packages/vllm; \
    cd "$SP" && find . -name "*.so" -print0 | xargs -0 -I{} cp --parents {} /opt/vllm/vllm/; \
    rm -rf /opt/vllm/vllm/vllm_flash_attn && cp -r "$SP/vllm_flash_attn" /opt/vllm/vllm/; \
    uv pip uninstall --system vllm; \
    uv pip install --system "setuptools>=77.0.3,<81" "setuptools-scm>=8" "setuptools-rust>=1.9" packaging wheel jinja2 regex; \
    cd /opt/vllm && uv pip install --system --no-build-isolation --no-deps -e .; \
    python3 -c "import os, vllm; p=os.path.dirname(vllm.__file__); assert p=='/opt/vllm/vllm', p; \
    assert os.path.exists(p+'/_C_stable_libtorch.abi3.so'); assert os.path.exists(p+'/vllm_flash_attn/_vllm_fa2_C.abi3.so'); \
    print('editable vllm', vllm.__version__, p)"
RUN uv pip install --system numpy pandas matplotlib pytest pyyaml rich safetensors "huggingface_hub[cli]"
WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
