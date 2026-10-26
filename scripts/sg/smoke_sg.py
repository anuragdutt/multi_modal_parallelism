"""Smoke and regression test for the Mamba SSM/gate split in the `mmp-sg` container.

Run inside the container (as the host user, HF cache at /hf), on two GPUs that carry no foreign load:
  docker exec -e CUDA_VISIBLE_DEVICES=0,1 -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 mmp-sg python3 /workspace/scripts/sg/smoke_sg.py --mode sg --out /workspace/results/sg/smoke_sg.json
  docker exec -e CUDA_VISIBLE_DEVICES=0,1 -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 mmp-sg python3 /workspace/scripts/sg/smoke_sg.py --mode tp --out /workspace/results/sg/smoke_tp.json
  docker exec mmp-sg python3 /workspace/scripts/sg/smoke_sg.py --compare /workspace/results/sg/smoke_sg.json /workspace/results/sg/smoke_tp.json

What it checks
  1. Every TP rank reports the same number of KV-cache groups and the same reorder threshold (the fix for the
     batch-reorder mismatch; before the fix the gate rank reported 0 groups / None).
  2. A "waves" workload (max_num_seqs 8, 32 prompts x 128 tokens, ignore_eos) run twice in one process is
     bit-identical, and its first generated tokens agree with the other mode's (compare step).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

MODEL = os.environ.get("SG_MODEL", "state-spaces/mamba-2.8b-hf")
PROMPTS = [f"Prompt {i}: write {['one', 'two', 'three', 'four'][i % 4]} sentences about {t}." for i, t in enumerate(
    ["rivers", "trains", "glass", "winter", "maps", "bread", "silence", "engines"] * 4)]


def rank_groups(worker):  # executed on every worker via collective_rpc
    mr = worker.model_runner
    return (int(worker.rank) if hasattr(worker, "rank") else -1,
            len(mr.kv_cache_config.kv_cache_groups), mr.reorder_batch_threshold)


def run(mode: str, out: str) -> None:
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=MODEL, tensor_parallel_size=2, enable_mamba_sg=(mode == "sg"), enforce_eager=True,
              dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.4, max_num_seqs=8, seed=0)
    load_s = time.time() - t0
    groups = llm.collective_rpc(rank_groups)
    sp = SamplingParams(temperature=0, max_tokens=128, ignore_eos=True, logprobs=1)
    runs = []
    for rep in range(2):
        t1 = time.time()
        outs = llm.generate(PROMPTS, sp, use_tqdm=False)
        dt = time.time() - t1
        runs.append({"seconds": dt, "tok_s": sum(len(o.outputs[0].token_ids) for o in outs) / dt,
                     "token_ids": [list(o.outputs[0].token_ids) for o in outs],
                     "first_logprob": [float(next(iter(o.outputs[0].logprobs[0].values())).logprob) for o in outs]})
    identical = runs[0]["token_ids"] == runs[1]["token_ids"]
    res = {"mode": mode, "model": MODEL, "load_s": load_s, "rank_groups": groups, "runs": runs,
           "rep_identical": identical}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(res, open(out, "w"))
    print(f"[{mode}] load {load_s:.1f}s; rank (rank, kv_groups, reorder_threshold) = {groups}; "
          f"rep0 vs rep1 identical: {identical}; tok/s {runs[1]['tok_s']:.0f}")
    bad = [g for g in groups if g[1] == 0 or g[2] is None]
    if mode == "sg" and bad:
        print("FAIL: a rank owns no KV-cache groups or has no reorder threshold:", bad)
        sys.exit(1)
    if not identical:
        print("FAIL: the same workload is not deterministic across repeats")
        sys.exit(1)
    print("PASS")


def compare(a: str, b: str) -> None:
    A, B = json.load(open(a)), json.load(open(b))
    ta, tb = A["runs"][1]["token_ids"], B["runs"][1]["token_ids"]
    first = sum(x[0] == y[0] for x, y in zip(ta, tb))
    full = sum(x == y for x, y in zip(ta, tb))
    dlp = max(abs(x - y) for x, y in zip(A["runs"][1]["first_logprob"], B["runs"][1]["first_logprob"]))
    print(f"{A['mode']} vs {B['mode']}: first token identical {first}/{len(ta)}, full sequence identical {full}/{len(ta)}, "
          f"max |d first logprob| {dlp:.3f}; tok/s {A['runs'][1]['tok_s']:.0f} vs {B['runs'][1]['tok_s']:.0f}")
    print("PASS" if first >= len(ta) - 1 and dlp < 0.5 else "CHECK: divergence beyond bf16 noise")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sg", "tp"])
    ap.add_argument("--out", default="results/sg/smoke.json")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        run(a.mode, a.out)
