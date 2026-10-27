# Development workflow for the vLLM fork (one checkout, built from source, editable)

The vLLM code that runs in `mmp-sg` is the git checkout `vllm_fork/` of this repository (submodule of
`anuragdutt/vllm`, branch `mmp/sg`), mounted into the container at `/workspace/vllm_fork` and installed editable there.
Its compiled extensions are built inside the container from the checkout's own kernel sources; the image carries no
prebuilt vLLM at all.

## First time on a machine
```bash
git clone --recurse-submodules git@github.com:anuragdutt/multi_modal_parallelism.git && cd multi_modal_parallelism
docker/sg/build.sh                         # images mmp:sg-replica and mmp:sg-src (toolchain only), ~10 min
docker/sg/run.sh                           # container mmp-sg; prints "vLLM is not built yet" the first time
docker exec mmp-sg bash /workspace/docker/sg/build_vllm.sh      # 30-60 min: compiles for sm_86, installs editable
docker exec -e CUDA_VISIBLE_DEVICES=0,1 -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 mmp-sg \
  python3 /workspace/scripts/sg/smoke_sg.py --mode sg --out /workspace/results/sg/smoke_sg.json   # regression gate
```

## Every change
1. Edit files under `vllm_fork/` (on the host or inside the container; same files).
2. Python change: it is live immediately. Kernel change (`csrc/`, CMake): `docker exec mmp-sg bash /workspace/docker/sg/build_vllm.sh` (incremental).
3. Commit and push the fork, then pin it in this repo:
   ```bash
   git -C vllm_fork add -A && git -C vllm_fork commit -m "..." && git -C vllm_fork push origin mmp/sg
   git add vllm_fork && git commit -m "vllm_fork: <what changed>" && git push
   ```
   Every vLLM change is pushed immediately (project rule).
4. Re-run the smoke test before relying on a change.

## Pulling someone else's change
```bash
git pull && git submodule update --init --remote vllm_fork   # or: git -C vllm_fork pull origin mmp/sg
docker/sg/run.sh                                            # container already up: nothing to do for Python changes
docker exec mmp-sg bash /workspace/docker/sg/build_vllm.sh  # only if csrc/ or CMake files changed
```

## Where things are inside the container
| path | what |
|---|---|
| `/workspace` | this repository (bind mount of the host checkout) |
| `/workspace/vllm_fork` | the vLLM checkout; `vllm/*.so` and `vllm/_version.py` are build products, ignored by git |
| `/build` | host-local disk (`/tmp/mmp-build-<user>`): cmake build dir, FetchContent deps (cutlass, vllm-flash-attn), ccache, cargo target; `vllm_fork/build` and `vllm_fork/.deps` are symlinks into it, excluded through `.git/info/exclude` |
| `/hf` | the host HF cache (`HF_HOME`) |
| `/workspace/.container_home` | `HOME` inside the container (Triton cache, pip user site) |

## Build knobs (environment of `build_vllm.sh`)
`TORCH_CUDA_ARCH_LIST` (default `8.6` for the A6000s; `8.6;9.0a` to also cover H200), `MAX_JOBS` (32), `NVCC_THREADS` (2),
`SETUPTOOLS_SCM_PRETEND_VERSION` (0.24.0; the shallow submodule has no tags). The build downloads CUTLASS and
vllm-flash-attn sources at the commits pinned by vLLM's CMake files, so it needs network access the first time.

## Switching to another base version later
A branch on a newer upstream (e.g. for the Qwen3.5 work) needs the matching base image for torch/CUDA: change `BASE`
in `docker/sg/Dockerfile`, point the submodule at the new branch, rebuild the image and run `build_vllm.sh`. The
mechanism is identical.

## Build record (guppy, 2026-10-27)
Image `mmp:sg-src` (36.4 GB: base v0.24.0 with the prebuilt vLLM removed, CUDA 13.0 dev libraries, cmake/ninja/ccache,
Rust stable). First from-source build of `mmp/sg` at `558de61`: 19 minutes with `MAX_JOBS=32 NVCC_THREADS=2
TORCH_CUDA_ARCH_LIST=8.6` (336 build steps; `_C_stable_libtorch`, `_moe_C_stable_libtorch`, `cumem_allocator`,
`spinloop`, `_rust_tool_parser`, `vllm_flash_attn/_vllm_fa2_C` and `_vllm_fa3_C`). `_qutlass_C` is not built for sm_86
and vLLM logs a harmless warning about it. Build products on the host: `/tmp/mmp-build-adutt` (deps 3.3 GB, cargo 2 GB,
ccache 0.5 GB). The first attempt failed at CMake configure with `CUDA_nvrtc_LIBRARY NOTFOUND` because the runtime base
image has no NVRTC dev library; `cuda-nvrtc-dev-13-0` + `cuda-libraries-dev-13-0` fixed it.
Smoke test on this build (`results/sg/smoke_{sg,tp}_src.json`): SG rank check (1, 1) on both ranks, SG repeats
bit-identical, SG vs TP first token 32/32, max |d logp0| 0.107. TP repeats differed on 15/32 sequences after token 8 to
110 with first-token agreement 32/32 and max |d logp0| 0.161: batch-composition noise, which is why the smoke test's
repeat criterion is noise-bounded rather than exact.
