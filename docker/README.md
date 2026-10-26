# Containers

| image | base | purpose | files | container |
|---|---|---|---|---|
| `mmp:stage1` | `vllm/vllm-openai:v0.30.0` + editable upstream v0.30.0 clone at `/opt/vllm` | the layer harness (`harness/`), stage-1 and synthesizer experiments | `docker/Dockerfile`, `docker/run.sh`, `docker/env.list`, `scripts/build_image.sh` | `mmp-dev` |
| `mmp:sg-replica` | `vllm/vllm-openai:v0.24.0` (digest-pinned) + `vllm_overlays/sg-2026-07-31` | byte-exact replica of the original `vllm-mamba-sg` environment that produced the Aug-Sep 2026 SG benchmarks | `docker/sg/Dockerfile` (target `sg-replica`) | - |
| `mmp:sg-dev` | same base + editable install of the `./vllm_fork` submodule (fork branch `mmp/sg`) | SG close-out (bug fix, re-measurement) and any vLLM-side development on v0.24.0 | `docker/sg/Dockerfile` (target `sg-dev`), `docker/sg/build.sh`, `docker/sg/run.sh` | `mmp-sg` (host port 8001) |

Saved images go to `docker/images/` (gitignored): `docker save mmp:sg-dev | gzip > docker/images/mmp-sg-dev-<date>.tar.gz`.
The original container `vllm-mamba-sg` (host port 8000, the user's own) is left untouched by every script here.
