#!/usr/bin/env bash
# Inside the container: correctness checks + family sweep + analysis for one model snapshot.
# Usage: run_model_sweep.sh <model_dir> <tag>
set -o pipefail
cd /workspace
export MMP_MODEL_DIR=$1; TAG=$2
mkdir -p results/$TAG/raw
echo "== $TAG correctness $(date)"
for cfg in "tp1 --nproc 1" "tp4 --nproc 4" "tp4_fused --nproc 4" "tp4_streams --nproc 4" "split22 --nproc 4"; do
  set -- $cfg; mode=$1; np=$3
  torchrun --standalone --nnodes=1 --nproc_per_node=$np -m harness.run --mode $mode --batch 4 --ctx 2048 --n-warm 3 --n-iter 10 --check \
    --out results/$TAG/raw/dev.csv --correctness-out results/$TAG/raw/correctness.csv 2>&1 | grep -E "^\[check\]|Error" | head -3
done
echo "== $TAG sweep $(date)"
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m harness.sweep --config configs/sweep_family.yaml --out results/$TAG/raw/sweep_full.csv --correctness-out results/$TAG/raw/correctness.csv 2>&1 | grep -E "^\[sweep\]|FAILED" | tail -3
echo "== $TAG analysis $(date)"
python3 -m analysis.make_all --raw results/$TAG/raw --out results/$TAG 2>&1 | head -24
echo "== $TAG done $(date)"
