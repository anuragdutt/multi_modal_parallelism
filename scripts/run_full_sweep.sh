#!/usr/bin/env bash
cd /workspace
export MMP_MODEL_DIR=/hf/hub/models--tiiuae--Falcon-H1-7B-Instruct/snapshots/41e72f27effbab80cd45b6e884688452253a3686
echo "== GPU check $(date)"; nvidia-smi --query-gpu=index,utilization.gpu,power.draw,memory.used --format=csv
echo "== full sweep $(date)"
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m harness.sweep --config configs/sweep_stage1.yaml --out results/raw/sweep_full.csv
echo "== analysis $(date)"
python3 -m analysis.make_all --raw results/raw --out results/full
echo "== done $(date)"
