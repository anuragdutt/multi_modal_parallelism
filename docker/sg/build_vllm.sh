#!/usr/bin/env bash
# Build vLLM from the mounted git checkout's own kernel sources and install it editable (run INSIDE mmp-sg):
#   docker exec mmp-sg bash /workspace/docker/sg/build_vllm.sh          # first build: 30-60 min
#   docker exec mmp-sg bash /workspace/docker/sg/build_vllm.sh          # after a kernel change: incremental (ccache + build dir)
# Python-only changes need no rebuild at all: the editable install imports straight from /workspace/vllm_fork.
set -euo pipefail
SRC=${SRC:-/workspace/vllm_fork}
BUILD=${BUILD:-/build}                      # host-local disk mounted by run.sh; survives container recreation
mkdir -p /workspace/logs "$BUILD/vllm_fork-build" "$BUILD/vllm_fork-deps" "$BUILD/ccache" "$BUILD/cargo-target"
LOG=${LOG:-/workspace/logs/vllm_build_$(date +%Y%m%d_%H%M).log}
cd "$SRC"; test -f setup.py || { echo "no vLLM checkout at $SRC (git submodule update --init vllm_fork)"; exit 1; }
# Heavy build output goes to local disk through symlinks; they are excluded locally, never committed.
[ -e build ] || ln -s "$BUILD/vllm_fork-build" build
[ -e .deps ] || ln -s "$BUILD/vllm_fork-deps" .deps
EXCL=$(git rev-parse --git-path info/exclude)
grep -qx 'build' "$EXCL" 2>/dev/null || printf 'build\n.deps\n' >> "$EXCL"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6} MAX_JOBS=${MAX_JOBS:-32} NVCC_THREADS=${NVCC_THREADS:-2} \
       VLLM_TARGET_DEVICE=cuda SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION:-0.24.0} \
       CCACHE_DIR="$BUILD/ccache" CARGO_TARGET_DIR="$BUILD/cargo-target"
echo "== building $(git log --oneline -1) | arch $TORCH_CUDA_ARCH_LIST | MAX_JOBS=$MAX_JOBS NVCC_THREADS=$NVCC_THREADS | log $LOG"
START=$(date +%s)
python3 -m pip install --no-build-isolation --no-deps -v -e . > "$LOG" 2>&1 || { echo "BUILD FAILED, see $LOG"; tail -40 "$LOG"; exit 1; }
echo "== build finished in $(( ($(date +%s) - START) / 60 )) min"
cd /tmp && python3 - <<'PY'
import os, vllm
p = os.path.dirname(vllm.__file__)
assert p == "/workspace/vllm_fork/vllm", p
so = sorted(f for f in os.listdir(p) if f.endswith(".so"))
fa = os.path.exists(os.path.join(p, "vllm_flash_attn", "_vllm_fa2_C.abi3.so"))
from vllm import _custom_ops  # noqa: F401  (loads the compiled ops)
print("editable vllm from", p, "| extensions:", so, "| flash-attn fa2:", fa)
PY
echo "== OK: vLLM imports from the git checkout with freshly compiled extensions"
