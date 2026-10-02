#!/usr/bin/env bash
# Inside the container: Falcon-40B layer-0 correctness + sweep + analysis.
set -o pipefail
cd /workspace
D=$(ls -d /hf/hub/models--tiiuae--falcon-40b/snapshots/*/ | head -1); D=${D%/}
export MMP_MODEL_DIR=$D; TAG=falcon40b
mkdir -p results/$TAG/raw
echo "== $TAG correctness $(date)"
for cfg in "tp4 1" "tp4 4" "tp4_streams 4" "split13 4" "split22 4"; do
  set -- $cfg; mode=$1; np=$2
  torchrun --standalone --nnodes=1 --nproc_per_node=$np -m harness.run --template pr --mode $mode --batch 4 --ctx 2048 --n-warm 3 --n-iter 10 --check \
    --out results/$TAG/raw/dev.csv --correctness-out results/$TAG/raw/correctness.csv 2>&1 | grep -E "^\[check\]|^\[rank0\].*Error|Error:" | head -3
done
echo "== $TAG sweep $(date)"
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m harness.sweep --template pr --config configs/sweep_falcon40b.yaml --out results/$TAG/raw/sweep_full.csv --correctness-out results/$TAG/raw/correctness.csv 2>&1 | grep -E "^\[sweep\]|FAILED" | tail -3
echo "== $TAG analysis $(date)"
python3 -m analysis.make_all --raw results/$TAG/raw --out results/$TAG 2>&1 | head -24
echo "== $TAG done $(date)"
