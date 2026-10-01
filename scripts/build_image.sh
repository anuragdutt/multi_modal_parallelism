#!/usr/bin/env bash
# Build the stage-1 image. Pass VLLM_FORK / VLLM_BRANCH to use the fork once it exists.
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -t mmp:stage1 \
  --build-arg VLLM_FORK="${VLLM_FORK:-https://github.com/vllm-project/vllm.git}" \
  --build-arg VLLM_BRANCH="${VLLM_BRANCH:-v0.30.0}" \
  -f docker/Dockerfile docker/
