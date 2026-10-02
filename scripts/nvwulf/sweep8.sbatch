#!/bin/bash
#SBATCH --job-name=mmp-sweep8
#SBATCH --partition=h200x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
# Falcon-H1-34B (and 7B) layer-0 family sweep at 8 ranks: tp8 (homogeneous), fused, streams, and split44.
set -o pipefail
export MMP_HOME=${MMP_HOME:-$HOME/mmp}
source "$MMP_HOME/venv/bin/activate"
export HF_HOME="$MMP_HOME/hf" NCCL_P2P_LEVEL=NVL TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$MMP_HOME/repo"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
for M in Falcon-H1-34B-Instruct Falcon-H1-7B-Instruct; do
  D=$(ls -d "$HF_HOME"/hub/models--tiiuae--$M/snapshots/*/ | head -1); D=${D%/}
  TAG=nvwulf8_${M%-Instruct}
  mkdir -p results/$TAG/raw
  export MMP_MODEL_DIR=$D
  echo "== $TAG microbench $(date)"
  torchrun --standalone --nnodes=1 --nproc_per_node=8 -m harness.microbench --quick --out results/$TAG/raw/microbench.csv
  echo "== $TAG correctness $(date)"
  for mode in tp4 tp4_fused tp4_streams split44; do
    torchrun --standalone --nnodes=1 --nproc_per_node=8 -m harness.run --mode $mode --batch 4 --ctx 2048 --n-warm 3 --n-iter 10 --check \
      --out results/$TAG/raw/dev.csv --correctness-out results/$TAG/raw/correctness.csv --bw results/$TAG/raw/microbench.csv 2>&1 | grep -E "^\[check\]|Error:" | head -2
  done
  echo "== $TAG sweep $(date)"
  torchrun --standalone --nnodes=1 --nproc_per_node=8 -m harness.sweep --config configs/sweep_family8.yaml --out results/$TAG/raw/sweep_full.csv \
    --correctness-out results/$TAG/raw/correctness.csv --bw results/$TAG/raw/microbench.csv 2>&1 | grep -E "^\[sweep\]|FAILED" | tail -3
  python -m analysis.make_all --raw results/$TAG/raw --out results/$TAG | head -24
done
echo "== done $(date)"
