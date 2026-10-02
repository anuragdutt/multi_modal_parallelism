#!/usr/bin/env bash
# Inside the container: Nemotron 3 Nano per-layer-type correctness and timing (graph variant), then composite.
set -o pipefail
cd /workspace
D=$(ls -d /hf/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/*/ | head -1); D=${D%/}
export MMP_MODEL_DIR=$D; TAG=nemotron3_nano
mkdir -p results/$TAG/raw
echo "== $TAG correctness $(date)"
for cfg in "mamba tp4" "attention tp4" "attention dp4" "moe tp4" "moe ep4"; do
  set -- $cfg
  torchrun --standalone --nnodes=1 --nproc_per_node=4 -m harness.run --template seq --layer-type $1 --mode $2 --batch 8 --ctx 2048 --n-warm 3 --n-iter 10 --check \
    --out results/$TAG/raw/dev.csv --correctness-out results/$TAG/raw/correctness.csv 2>&1 | grep -E "^\[check\]|^\[rank0\].*Error|Error:" | head -3
done
echo "== $TAG timing $(date)"
for cfg in "mamba tp4" "attention tp4" "attention dp4" "moe tp4" "moe ep4"; do
  set -- $cfg
  for B in 1 4 16 64; do for c in 512 8192 32768; do
    torchrun --standalone --nnodes=1 --nproc_per_node=4 -m harness.run --template seq --layer-type $1 --mode $2 --batch $B --ctx $c --variant graph --n-warm 20 --n-iter 100 --repeats 3 \
      --out results/$TAG/raw/sweep_full.csv --tag $1 2>&1 | grep -E "^\[time\]|Error:" | head -2
  done; done
done
echo "== $TAG done $(date)"
