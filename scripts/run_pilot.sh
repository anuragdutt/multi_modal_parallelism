#!/usr/bin/env bash
set -o pipefail
cd /workspace
export MMP_MODEL_DIR=/hf/hub/models--tiiuae--Falcon-H1-7B-Instruct/snapshots/41e72f27effbab80cd45b6e884688452253a3686
echo "== microbench $(date)"
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m harness.microbench --out results/raw/microbench.csv
echo "== pilot sweep $(date)"
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m harness.sweep --config configs/sweep_pilot.yaml --out results/raw/sweep_pilot.csv
echo "== analysis $(date)"
python3 -m analysis.make_all --raw results/raw --out results/pilot
echo "== done $(date)"
