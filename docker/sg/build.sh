#!/usr/bin/env bash
# Build the SG images on guppy from the repo root (sg-replica: frozen; sg-src: toolchain, vLLM compiled later inside the container).
set -euo pipefail
cd "$(dirname "$0")/../.."
test -f vllm_fork/setup.py || { echo "submodule ./vllm is empty: run git submodule update --init vllm_fork"; exit 1; }
docker build --target sg-replica -t mmp:sg-replica -f docker/sg/Dockerfile .
docker build --target sg-src     -t mmp:sg-src     -f docker/sg/Dockerfile .
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep '^mmp:sg'
