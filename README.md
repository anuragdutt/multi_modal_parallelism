# multi_modal_parallelism

Architecture-aware parallelism for SSM-based and hybrid LLMs across multiple GPUs.

This repository holds the experiments behind a thesis chapter on *elastic mixer parallelism*: a
branch-split tensor-parallel layout for parallel hybrids such as Falcon-H1 (attention mixer and Mamba-2
mixer summed in every layer), balanced per step by "swing" MLP shards instead of a static split. It is the
third step in a line of work that began with tensor parallelism for Mamba-1 (arXiv 2602.21144) and the
SSM/gate branch split in vLLM.

## Status

Planning. No experiment has been run yet. The full stage-1 plan, including the hypotheses, the verified
environment and model facts, the exact sharding math, the Docker recipe, the harness design, the sweep,
the byte model and the go/no-go criteria, is in
[`doc/stage1_elastic_mixer_parallelism_plan.md`](doc/stage1_elastic_mixer_parallelism_plan.md).

## What is being tested

1. Under plain tensor parallelism with more ranks than KV heads, every replica rank reads the whole
   replicated KV head each decode step, so replication wastes bandwidth, not just memory.
2. A branch split (attention ranks vs SSM ranks) with a single fused reduction removes that waste and cuts
   vLLM's four collectives per Falcon-H1 layer to two full-group collectives plus one two-rank collective.
3. Swing MLP shards absorb the context-dependent imbalance between the branches at zero data-movement
   cost, which is the piece TII's mixer parallelism lacked and the reason it reversed at large batch.

Primary model: Falcon-H1-7B on 4 GPUs. Generalization set: Falcon-H1-34B, Hymba-1.5B, Qwen3.5-35B-A3B or
Nemotron 3 Nano, Falcon-40B or Command-R.

## Layout (planned)

```
doc/        plans and write-ups
docker/     Dockerfile, run script and env for the mmp-dev container (vLLM v0.30.0, editable fork)
scripts/    backup, fork, build, download, smoke test, sweep and analysis drivers
configs/    model dimensions and sweep definitions
harness/    4-rank layer harness on vLLM's own kernels and collectives
analysis/   figures and the go/no-go summary
tests/      CPU unit tests for sharding, swing bookkeeping and the byte model; one 4-GPU correctness test
results/    raw CSVs (ignored), figures and summaries (committed)
```

## Build and run

See sections 7, 13 and 14 of the stage-1 plan. In short: back up the old container, fork vLLM at tag
v0.30.0, build `mmp:stage1`, start `mmp-dev`, download Falcon-H1-7B, run the smoke test, the microbench,
the sweep and the analysis.

## Hardware

guppy.fsl.cs.sunysb.edu: 4x RTX A6000 (48 GB), PCIe Gen4 only, driver 550 running CUDA 13.0 containers
through the compatibility layer. Larger runs move to NVwulf H200 nodes.
