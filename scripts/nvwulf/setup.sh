#!/usr/bin/env bash
# One-time setup on an NVwulf login node (run interactively after Duo login). Creates a venv with vLLM v0.30.0
# wheels (no Docker/Apptainer needed: the harness imports only vLLM's kernels), clones the repo, and fetches
# the Falcon-H1-34B layer-0 shard into $MMP_HOME/hf.
set -euo pipefail
export MMP_HOME=${MMP_HOME:-$HOME/mmp}
mkdir -p "$MMP_HOME" && cd "$MMP_HOME"
module load python/3.12 2>/dev/null || true
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install "vllm==0.30.0" numpy pandas matplotlib pytest pyyaml rich safetensors huggingface_hub
git clone https://github.com/anuragdutt/multi_modal_parallelism.git repo || (cd repo && git pull)
export HF_HOME="$MMP_HOME/hf"
PY=python bash repo/scripts/download_layer0.sh tiiuae/Falcon-H1-34B-Instruct model.layers.0
PY=python bash repo/scripts/download_layer0.sh tiiuae/Falcon-H1-7B-Instruct model.layers.0
echo "setup done in $MMP_HOME"
