# multi_modal_parallelism

Architecture-aware parallelism for state-space, Gated DeltaNet and hybrid LLMs across multiple GPUs. This repository
is the one-stop shop for the project: the documents, the vLLM fork (as a submodule), the containers, the layer harness,
the synthesizer and every result.

## Map

| where | what |
|---|---|
| [`docs/PLAN.md`](docs/PLAN.md) | the current plan: ontology graph, rules R1-R8, synthesizer v2, model set, experiments E0-E9, predictions, gates, schedule |
| [`docs/NOTES.md`](docs/NOTES.md) | exhaustive notes on the SSM/gate split deep dive: code map, container facts, benchmark audit, the reorder bug, reproduction |
| [`docs/generalization.md`](docs/generalization.md) | design note: why per-model tricks do not generalize and how the synthesizer is meant to |
| [`docs/blog/README.md`](docs/blog/README.md) | stage-1 results write-up (Falcon-H1 branch split vs tensor parallelism) with figures; `docs/blog/make_figs.py` rebuilds them |
| [`docs/stage1_elastic_mixer_parallelism_plan.md`](docs/stage1_elastic_mixer_parallelism_plan.md) | the stage-1 plan with its execution log and findings (sections 21-24) |
| `vllm/` | git submodule: fork `anuragdutt/vllm`, branch `mmp/sg` (vLLM v0.24.0 + the SSM/gate split as it ran + the batch-reorder fix); later `mmp/stage2` for the elastic attention work |
| `vllm_overlays/sg-2026-07-31/` | the exact files copied from the original container's installed vLLM (with checksums and provenance): ground truth of what produced the SG benchmarks |
| `docker/` | two images: `mmp:stage1` (v0.30.0 harness, container `mmp-dev`) and `mmp:sg-{replica,dev}` (v0.24.0 SG, container `mmp-sg`); see `docker/README.md` |
| `harness/` | 4-rank layer harness on vLLM kernels (`run.py`, `sweep.py`, `synth.py` recipe v1, model templates); `harness/synth2/` arrives per the plan |
| `configs/` | model dims (`falcon_h1_*`, `falcon_40b`, `nemotron3_nano`, `qwen35_35b_a3b`) and sweep grids |
| `scripts/` | build, download, smoke test, sweep chains per model, NVwulf sbatch |
| `analysis/` | loaders, figures, summaries, sequential composite |
| `results/<model>/` | figures and summaries (raw CSVs are gitignored and live on the GPU host) |

## Quick start

```bash
git clone --recurse-submodules git@github.com:anuragdutt/multi_modal_parallelism.git
# harness image (stage 1 / synthesizer experiments), on the GPU host:
scripts/build_image.sh && docker/run.sh && docker exec -it mmp-dev bash
# SG image (gate-split close-out on vLLM v0.24.0):
docker/sg/build.sh && docker/sg/run.sh && docker exec -it mmp-sg bash
```

## Status (2026-10-26)
Stage 1 measured (branch split up to 1.38x per layer on Falcon-H1-7B, recipe predictions right in sign on every cell,
Falcon-40B negative confirmed). SSM/gate split understood and closed: the only clean gain is Jamba +3 to +7%, and a
batch-reorder mismatch on gate ranks corrupted the pure-Mamba benchmarks (fix in the fork, re-measurement pending).
Next: synthesizer v2 and the Qwen3.5 elastic attention layouts per `docs/PLAN.md`.

## Hardware
guppy.fsl.cs.sunysb.edu: 4x RTX A6000 (48 GB), PCIe Gen4 only, driver 550, shared. Larger runs: NVwulf H200 nodes
(allocation pending).
