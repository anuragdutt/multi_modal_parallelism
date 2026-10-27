# Containers

| image | base | purpose | files | container |
|---|---|---|---|---|
| `mmp:stage1` | `vllm/vllm-openai:v0.30.0` + editable upstream v0.30.0 clone at `/opt/vllm` | the layer harness (`harness/`), stage-1 and synthesizer experiments | `docker/Dockerfile`, `docker/run.sh`, `docker/env.list`, `scripts/build_image.sh` | `mmp-dev` |
| `mmp:sg-replica` | `vllm/vllm-openai:v0.24.0` (digest-pinned) + `vllm_overlays/sg-2026-07-31` | byte-exact replica of the original `vllm-mamba-sg` environment that produced the Aug-Sep 2026 SG benchmarks | `docker/sg/Dockerfile` (target `sg-replica`) | - |
| `mmp:sg-src` | same base with the prebuilt vLLM removed, plus cmake/ninja/ccache/Rust; vLLM is compiled from the mounted checkout `./vllm_fork` (fork branch `mmp/sg`) by `docker/sg/build_vllm.sh` and installed editable | all vLLM-side development: edit, commit, push, pull, run, no rebuild for Python; incremental rebuild for kernels | `docker/sg/Dockerfile` (target `sg-src`), `docker/sg/build.sh`, `docker/sg/run.sh`, `docker/sg/build_vllm.sh` | `mmp-sg` (host port 8001; build dir on host `/tmp/mmp-build-<user>`) |

The workflow is in `docs/dev_workflow.md`. Saved images go to `docker/images/` (gitignored): `docker save mmp:sg-dev | gzip > docker/images/mmp-sg-dev-<date>.tar.gz`.
The original container `vllm-mamba-sg` (host port 8000, the user's own) is left untouched by every script here.
