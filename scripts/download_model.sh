#!/usr/bin/env bash
# Download Falcon-H1-7B-Instruct into the shared HF cache (host side, vllm-env has huggingface_hub).
set -euo pipefail
MODEL=${MODEL:-tiiuae/Falcon-H1-7B-Instruct}
PY=${PY:-/home/adutt/anaconda3/envs/vllm-env/bin/python}
"$PY" - "$MODEL" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], max_workers=8)
print("downloaded to", p)
PYEOF
