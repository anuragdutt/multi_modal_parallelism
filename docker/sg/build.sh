#!/usr/bin/env bash
# Build the SG images on guppy from the repo root. Requires the submodule: git submodule update --init vllm
set -euo pipefail
cd "$(dirname "$0")/../.."
test -f vllm_fork/setup.py || { echo "submodule ./vllm is empty: run git submodule update --init vllm_fork"; exit 1; }
docker build --target sg-replica -t mmp:sg-replica -f docker/sg/Dockerfile .
docker build --target sg-dev     -t mmp:sg-dev     -f docker/sg/Dockerfile .
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' | grep '^mmp:sg'
