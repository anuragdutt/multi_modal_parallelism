#!/usr/bin/env bash
# Download only what the layer-level harness needs: config, tokenizer-free, index, and the shard(s) holding
# layer 0 (plus the modeling code for trust_remote_code models). Usage: download_layer0.sh <repo_id> [layer_prefix]
set -euo pipefail
REPO=$1
PREFIX=${2:-}
PY=${PY:-/home/adutt/anaconda3/envs/vllm-env/bin/python}
"$PY" - "$REPO" "$PREFIX" <<'PYEOF'
import json, sys
from huggingface_hub import hf_hub_download, list_repo_files
repo, prefix = sys.argv[1], sys.argv[2]
files = list_repo_files(repo)
need = [f for f in files if f in ("config.json", "model.safetensors.index.json", "generation_config.json") or f.endswith(".py")]
for f in need:
    print("get", f, hf_hub_download(repo, f))
if "model.safetensors.index.json" in files:
    idx = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))["weight_map"]
    if not prefix:
        cands = sorted({k.rsplit(".", 2)[0] for k in idx if ".0." in k})
        print("layer-0-like prefixes:", cands[:5])
        prefix = next((c for c in cands if c.endswith(".0")), cands[0])
    shards = sorted({v for k, v in idx.items() if k.startswith(prefix + ".")})
    print("prefix", prefix, "-> shards", shards)
    for s in shards:
        print("get", s, hf_hub_download(repo, s))
else:
    single = [f for f in files if f.endswith(".safetensors")]
    for s in single:
        print("get", s, hf_hub_download(repo, s))
print("done", repo)
PYEOF
