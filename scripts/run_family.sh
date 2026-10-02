#!/usr/bin/env bash
cd /workspace
echo "== microbench (graph variants) $(date)"
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m harness.microbench --out results/raw/microbench.csv 2>&1 | grep microbench
echo "== synth re-score $(date)"
for m in falcon_h1_7b falcon_h1_3b falcon_h1_34b; do python3 -m harness.synth --model configs/$m.yaml --template falcon_h1 --world 4; done
python3 -m harness.synth --model configs/falcon_40b.yaml --template parallel_residual --world 4
bash scripts/run_model_sweep.sh /hf/hub/models--tiiuae--Falcon-H1-34B-Instruct/snapshots/6e6890e26cba1f58b1e6ee70d654ddf054f9fce8 h1_34b
S3=$(ls -d /hf/hub/models--tiiuae--Falcon-H1-3B-Instruct/snapshots/*/ | head -1)
bash scripts/run_model_sweep.sh ${S3%/} h1_3b
echo "== family done $(date)"
