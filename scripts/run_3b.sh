#!/usr/bin/env bash
cd /workspace
rm -rf results/h1_3b
bash scripts/run_model_sweep.sh /hf/hub/models--tiiuae--Falcon-H1-3B-Instruct/snapshots/01087ec4c132d7f186908716b3530ea187f062a1 h1_3b
echo "== tp2 reference points $(date)"
for m in 3b 7b; do
  if [ $m = 3b ]; then D=/hf/hub/models--tiiuae--Falcon-H1-3B-Instruct/snapshots/01087ec4c132d7f186908716b3530ea187f062a1; else D=/hf/hub/models--tiiuae--Falcon-H1-7B-Instruct/snapshots/41e72f27effbab80cd45b6e884688452253a3686; fi
  for c in 512 8192 32768; do
    MMP_MODEL_DIR=$D torchrun --standalone --nnodes=1 --nproc_per_node=2 -m harness.run --mode tp4 --batch 16 --ctx $c --variant graph --n-warm 20 --n-iter 100 --repeats 3 --tag tp2_$m --out results/tp2_reference.csv 2>&1 | grep "^\[time\]"
  done
done
echo "== 3b chain done $(date)"
