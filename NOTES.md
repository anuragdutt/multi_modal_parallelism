# NOTES.md — SSM/gate split deep dive, benchmark audit, and container verification

Session: 2026-10-25 to 2026-10-26 (Claude Code on etracker1, operating on guppy over SSH). Earlier parts of the
same conversation (2026-10-05): the stage-1 write-up in `blog/`. This file is written for a reader with no access to
the conversation. Provenance tags: `[verified]` = I ran the command or read the code in this session;
`[from-agent: X]` = reported by subagent X and not re-checked; `[inferred]` = my reasoning; `[unknown]` = open.

Subagents referenced: **Paper-era** (paper-era TP/quantized all-reduce code), **Prototypes** (standalone SG
prototypes), **Audit** (vLLM AIPerf campaign audit; first attempt died on a usage limit, relaunched and completed),
**Tester** (docker exec tests inside `vllm-mamba-sg`), **Impl** (vLLM implementation reader; died on a usage limit
before reporting, so I read the diffs myself, see section 5).

## 0. TL;DR

- **What was investigated.** The team's "SSM/gate split" (SG) for Mamba-1: whole GPUs are assigned to the gate path
  (z) or the SSM path (x -> conv -> scan) of every Mamba block, instead of channel-sharding both paths on every GPU as
  tensor parallelism does. Three generations exist in `guppy:~/masarani` (2-GPU HF prototypes, Mar 2026; roles x TP
  on 4 GPUs, May-Jul 2026; vLLM `--enable-mamba-sg`, Aug-Sep 2026) plus an AIPerf benchmark campaign `[verified]`.
- **Mechanism (vLLM 2+2).** Ranks 0,1 = gate, 2,3 = SSM. A gate rank computes only its z-half shard of `in_proj`
  (I/2 x H), P2P-sends it to its paired SSM rank and receives the full layer output back. An SSM rank holds the x
  shard, conv1d, x_proj, dt_proj, A, D, its `out_proj` column shard and the recurrent state; per layer it does two
  2-rank all-reduces inside the SSM group plus the P2P exchange. TP does two 4-rank all-reduces. `[verified]`
- **The implementation is numerically correct on ordinary batches** (first-token top-1 agreement 32/32 vs regular
  TP at TP=2, logprob deltas within bf16 noise) `[from-agent: Tester]`.
- **It corrupts every resident request on any step where the v1 runner reorders the batch.** Gate ranks own zero
  KV-cache groups, so `GPUModelRunner._may_reorder_batch` returns early on them while SSM ranks (Mamba1 backend,
  `reorder_batch_threshold = 1`) move decodes first; the two ranks then exchange rows of different tokens. Confirmed
  by per-rank instrumentation; the code path is `[verified]` in the installed copy (section 6, F2).
- **That bug is the "output-length anomaly" of the August-September benchmarks.** Under SG, Mamba-2.8B and
  Falcon-Mamba-7B emit early EOS 2-3x more often and 10-60% non-canonical token sequences at concurrency >= 8; Jamba is
  clean because its gate ranks still own attention KV groups and reorder identically `[from-agent: Audit]`.
- **The headline chart of 2026-10-20 (`request_throughput_benefit.png`, +18 to +24% for Falcon-Mamba) is an artifact**
  of shorter outputs from a corrupting server; corrected to TP's output length it is +0.3 to +8%, from invalid runs.
  Defensible numbers: Jamba 2+2 = -3.5% (c=1) to +7.1% (c=128) output tokens/s with 6-14% lower TTFT; the 1+3 split
  loses 14-40% everywhere `[from-agent: Audit]`.
- **Why SG cannot win much even when correct:** the gate path is one GEMM, so gate ranks idle while SSM ranks run conv,
  scan and `out_proj` over twice the channels of a TP-4 rank and read ~37% more weight bytes per step; the split only
  buys fewer, smaller collectives `[inferred from verified shard math]`. At TP=2 the Tester measured SG 11-14% slower
  than regular TP on a 32-prompt batch `[from-agent: Tester]`.
- **Code provenance mismatch.** The code that actually runs in the container is a patched copy inside
  `/usr/local/lib/python3.12/dist-packages/vllm` (files dated 2026-07-31, equal roles only); the `/vllm` checkout is
  not importable (`libcudart.so.12` missing), its HEAD is the first SG commit with an inconsistent WIP diff, and the
  Falcon/Jamba commit and the 1+3 code live only in `stash@{0}` and a backup branch `[verified]`.
- **Changes made this session:** none to any code or container. Added `blog/` (commit `c7a8044`, 2026-10-05) and
  this `NOTES.md` to the repo; scratch files under `/tmp/sg_tests` (container) and `/home/adutt/sg_tests` (host).
- **State:** SG is partial: correct and runnable only when no batch reorder is required; broken under continuous
  arrivals; equal-size roles only in the running code; TP=4 and 1+3 could not be re-tested (all four GPUs are held by
  the user's own idle SG server since 2026-10-23).

## 1. Goal and scope

**The task, as given (2026-10-25/26):** "I would really like you to understand how the ssm/gate split works and see if
we can do a better job of doing it. I want you to first do a deep dive in masarani folder in guppy to understand the
efforts in this domain... reading existing implementation to look at the proposed ideal. Let us learn and try to gain
context first." Then: "docker exec into the vllm-mamba-sg docker and then test the implementation changes that are
made in vllm", and finally this NOTES.md.

**Context before that (same conversation, 2026-10-05):** a blog-style write-up of the stage-1 "elastic mixer
parallelism" experiments on Falcon-H1 was committed (`blog/README.md`, `blog/make_figs.py`, `blog/figs/*`, commit
`c7a8044`). On 2026-10-25 the user asked to pivot: start from sequential hybrids (Nemotron 3 Nano, Qwen3.5 GDN) and
derive their parallelism; the Nemotron architecture was reconstructed from the HF config (see section 6, F8). Then the
user asked what the SG split results were, which led to this deep dive.

**In scope:** everything under `guppy:~/masarani` and `guppy:~/backups/vllm-mamba-sg-2026-10-01`; the SG code and its
history; the AIPerf campaign; functional tests inside the existing container; this write-up.

**Explicitly out of scope / constraints:**
- Read-only on guppy except: `docker start`/`docker exec` of `vllm-mamba-sg`, scratch files under `/home/adutt/sg_tests`
  (host) and `/tmp/sg_tests` (container). No `docker commit/rm/build/run`, no package installs, no git state changes in
  any checkout, no touching other people's GPU processes. Guppy is a shared machine (standing rule from the user).
- Never place cluster-access config on guppy; never handle the user's cluster password (standing rules).
- Every repo change is pushed to GitHub immediately (standing rule).
- GPU use limited to GPUs without foreign processes and `gpu_memory_utilization <= 0.45` (my rule to the Tester).

**Assumptions made:** the user's own `vllm serve ... --enable-mamba-sg` process inside the container (idle, since
2026-10-23) must not be killed; tests therefore ran at TP=2 on GPUs 0 and 1 only (where that server's gate ranks hold
just 2 GB each).

## 2. Environment snapshot (for reproduction)

### Operator host (where Claude Code ran)
- etracker1 (`/home/pace`), Linux 6.8.0-138-generic, python 3.12 (miniconda). GitHub access as `anuragdutt` through
  `~/.ssh/github_ed25519` (`ssh -T git@github.com` -> "Hi anuragdutt!") `[verified]`. `ssh guppy` works with key auth
  (alias from `~/.ssh/config`; user `adutt`) `[verified]`.
- Scratch used this session (session-temporary, not durable): `/tmp/pace-mmp` (clone of the repo),
  `/tmp/masarani-ro/` (read-only copies and parsed data pulled from guppy by the subagents: `commit1.show`,
  `commit2.show`, `committed.diff`, `uncommitted.patch`, `runs.json`, `server_metrics.json`, `tables.md`,
  `jsonl_summary.txt`, `image_config_full.json`, `tar_extract/*`, `nbdump_*.txt`, `scripts/*.py`, per-directory copies).

### GPU host: guppy.fsl.cs.sunysb.edu `[verified]`
- Ubuntu 24.04.5 LTS, kernel 6.8.0-142-generic, Docker 29.8.1 (build 4a63305).
- 4x NVIDIA RTX A6000, 46068 MiB each, driver 550.163.01, host `nvidia-smi` reports CUDA 12.4; PCIe only, no NVLink
  (GPU pairs show "SYS" in `nvidia-smi topo -m`, from the stage-1 plan and the Prototypes agent).
- Home is NFS with root squash (`/home/adutt` mode 700): the container's bind mount of the HF cache is unreadable
  from inside the container as root `[from-agent: Tester]`.

### Repositories
| repo / tree | path | remote | branch / HEAD | dirty? |
|---|---|---|---|---|
| multi_modal_parallelism (this repo) | etracker1 `/tmp/pace-mmp`; also `guppy:~/multi_modal_parallelism` (behind) | `git@github.com:anuragdutt/multi_modal_parallelism.git` | `main` @ `c7a8044` before this file | clean before NOTES.md `[verified]` |
| masarani top level | `guppy:~/masarani` | `hazemmasarani/mamba-tp` `[from-agent: Prototypes]` | commits "dtype", "quantize" (Oct 2025) | untracked work dirs `[from-agent]` |
| masarani vLLM clone | `guppy:~/masarani/vllm` | `https://github.com/hazemmasarani/vllm.git` | `main` @ `3034c8d38` (upstream) | clean; **contains no SG code** `[verified]` |
| SG backup tree | `guppy:~/backups/vllm-mamba-sg-2026-10-01/vllm` | (copy of the container tree taken 2026-10-01) | `4c0c73684` falcon-Jamba <- `2195e2a4b` mamba ssm-Gate TP <- `3034c8d38` | 9 modified + untracked `tests/config/test_mamba_sg_parallel.py` = the 1+3 patch (`uncommitted.patch`, 579 lines) `[verified]` |
| container checkout | `vllm-mamba-sg:/vllm` | origin = hazemmasarani/vllm `[from-agent: Tester]` | `main` @ `2195e2a4b`; branch `backup/pre-rollback-20261023` = `4c0c73684`; `stash@{0}: On main: pre-rollback-2026-10-23` (the 1+3 code + test file) | 2 modified files (WIP, see section 4) `[verified]` |
| Other SG git repos | `~/masarani/{model-distribution,HydraMamba,mamba-ssm-gate,falcon-mamba-ssm-gate,mamba-sg-tp,mamba-tp,mamba2-tp}` | various (e.g. `github.com/hazemmasarani/mamba2-tp`) | see section 3.2 | `[from-agent: Prototypes]` |

### Container `vllm-mamba-sg` `[verified via docker inspect / docker exec]`
- Id `db558a35677a636d8dc00ceeba46146fb218f3260f2b9dd4eb52a4bf69a6390f`; created 2026-07-21T01:14:07Z; running since
  2026-09-25T00:35:33Z (status `running` at session end).
- Image `vllm/vllm-openai:latest` = `sha256:4091d5593f77189cafa013299e927ee5cb979e7c38330cbd5e30ea54c8fea4fd`, created
  2026-06-29, 20.36 GB; labels `ai.vllm.image.tag=vllm/vllm-openai:v0.24.0`, `ai.vllm.build.commit=ee0da84ab9e04ac7610e28580af62c365e898389`.
- Entrypoint `["/bin/bash"]`, Cmd null, WorkingDir `/vllm-workspace`, `IpcMode=host`, `ShmSize=64 MiB`, network bridge,
  port `8000/tcp -> host 8000`, all GPUs (`DeviceRequests Count=-1 Capabilities=[[gpu]]`).
- Mounts: `/home/adutt/.cache/huggingface -> /root/.cache/huggingface` (rw bind; unreadable inside, see above).
- Env (from image; redacted; no secrets in the container config itself): `CUDA_VERSION=13.0.2`,
  `NV_CUDA_CUDART_VERSION=13.0.96-1`, `VLLM_ENABLE_CUDA_COMPATIBILITY=0`, `TORCH_CUDA_ARCH_LIST=7.5 8.0 8.6 8.9 9.0 10.0 12.0`,
  `VLLM_BUILD_COMMIT=ee0da84ab9e04ac7610e28580af62c365e898389`, `VLLM_IMAGE_TAG=vllm/vllm-openai:v0.24.0`, `UV_*`,
  `NVIDIA_VISIBLE_DEVICES=all`. The user's running server process additionally has `HF_HOME=/tmp/vllm-hf-cache` and
  `HF_TOKEN=<REDACTED>` in its environ `[from-agent: Tester]`.
- Related images on the host: `vllm-mamba-sg-backup:2026-08-06` (`9cea7436782e`, 23 GB; the image saved as
  `~/masarani/vllm-mamba-sg-backup-2026-08-06.tar`), `vllm-mamba-sg:local` (`b4a877ea3b57`, 43.9 GB, provenance
  `[unknown]`), `vllm/vllm-openai:v0.30.0` and `mmp:stage1*` (the stage-1 harness, container `mmp-dev`, unrelated).
- Other containers: `mmp-dev` (Up), `sglang-dev` (Exited), `vllm-mamba` (Exited, 2026-07-09; predecessor `[inferred]`).

### Toolchain inside the container `[verified]`
- Ubuntu 22.04.5 LTS, Python 3.12.13, nvcc `cuda_13.0.r13.0`, torch `2.11.0+cu130` (`torch.version.cuda` 13.0),
  torchvision 0.26.0+cu130, vllm 0.24.0 (`vllm.__file__ = /usr/local/lib/python3.12/dist-packages/vllm/__init__.py`),
  flashinfer-python 0.6.12, transformers 5.12.1, triton 3.6.0, numpy 2.2.6, pydantic 2.13.4; `tblib` not installed.
- `torch.cuda.device_count() == 4` inside.
- Model cache actually used by the running server: `/tmp/vllm-hf-cache/hub`, containing only
  `state-spaces/mamba-2.8b-hf` (11 GB, fp32) `[from-agent: Tester]`. Falcon-Mamba-7B and Jamba-v0.1 exist only in the
  host NFS cache (`/home/adutt/.cache/huggingface`, 14 GB / 5 MB metadata) `[from-agent: Tester]`.

### Processes found running inside (not started by this session) `[verified]`
`/usr/bin/python3 /usr/local/bin/vllm serve state-spaces/mamba-2.8b-hf --tensor-parallel-size 4 --enable-mamba-sg --device-ids 0,1,2,3 --dtype float16 --enforce-eager --port 8000`
(container PID 165288, elapsed 3 d 04 h at 2026-10-26 ~06:20 UTC, i.e. started ~2026-10-23 01:30 UTC; idle; GPU memory
0/1: 1965 MiB each (gate ranks), 2/3: 42813 MiB each (SSM ranks)). Left untouched.

### From a clean machine to the session's starting state
```bash
# on a machine with GitHub access as anuragdutt
git clone git@github.com:anuragdutt/multi_modal_parallelism.git && cd multi_modal_parallelism   # HEAD c7a8044 or later
# guppy access: ssh alias "guppy" -> guppy.fsl.cs.sunysb.edu, user adutt, key auth (see ~/.ssh/config on etracker1)
ssh guppy 'ls ~/masarani ~/backups/vllm-mamba-sg-2026-10-01'
ssh guppy 'docker ps -a; docker inspect vllm-mamba-sg --format "{{.State.Status}}"'     # expect: running
# the SG diffs, without docker:
ssh guppy 'cd ~/backups/vllm-mamba-sg-2026-10-01 && git -C vllm show 2195e2a4b > /dev/null && cat uncommitted.patch | head'
```

## 3. Codebase map

### 3.1 `guppy:~/masarani` (153 GB) — annotated tree with dates `[verified: top-level listing; roles from agents]`
```
~/masarani/
├── .git/                              top-level repo (remote hazemmasarani/mamba-tp; commits "dtype","quantize" Oct 2025)   [from-agent: Prototypes]
├── parallelism-practice/  (Sep 2025)  toy DP/TP exercises                                                                   [from-agent: Paper-era]
├── gpu-multiproccessing/  (Oct 2025)  NCCL all-reduce micro-benchmarks per dtype; int8/fp8 attempts never reduced in int8    [from-agent: Paper-era]
├── mamba-dtype/, quantize/, quantize-per-layer/, quantize-reduce-exp/, quantize_input/, ntk-accuracy/ (Oct 2025-Jan 2026)
│                                      the paper-era TP "server" line (arXiv 2602.21144): channel-sharded Mamba-1, 2 all-reduces/layer,
│                                      fp16-cast "quantized" all-reduce, per-op timing; ntk-accuracy = fp32 vs fp16 study (50 GB)        [from-agent: Paper-era]
├── TP_Mamba/ (Mar 2026), falcon_mamba_tp/, tp-experiment/ (Apr 2026)   TP=2 latency sweeps; HF FalconMamba copy (no TP); roles x TP sketch
├── model-distribution/ (Feb-Mar 2026, git)   first SG split (model4: both mixers in one file; model6/7: separate HF checkpoints
│                                      HMasarani/mamba-ssm + HMasarani/mamba-gate)                                                         [from-agent: Prototypes]
├── HydraMamba/ (Mar 2026, git)        2-GPU SG with CUDA-event timing; README: "decoupled Mamba ... separating SSM and gating pathways"
├── create_falcon_ssm_gate/, falcon-mamba-ssm-gate/ (Mar 2026)   same split on tiiuae/falcon-mamba-7b (hmasarani/falcon-mamba-{ssm,gate})
├── SG_TP_compare/ (Mar-Apr 2026)      tp_latency.log, ssm_gate_latency*.log, SSM_Gate_TP_Compare.ipynb (2-GPU SG vs TP-2 table)
├── mamba-ssm-gate/ (Jun 2026, git)    2-GPU SG + 4-GPU TP/HP baselines, prefill+cached-decode benchmark
├── mamba-tp/ (Jun 2026, git, 11 GB)   clean HF-based TP-4 reference with MambaCache (prefill+decode), mamba_tp_architecture.png
├── mamba-sg-tp/ (May-Jul 2026, git, 21 GB)   roles x TP on 4 GPUs: variants model/{scatter_gather,all_to_all_single,send_recv}, run_1/2/3.py,
│                                      run_1_cache.py, results/{first_cycle.csv,avg_cache_cycles.csv,logits_diff.log,comp.ipynb}
├── mamba2-tp/ (Jul 2026, git, 28 GB)  plain TP-2 for Mamba-Codestral-7B; NO split attempted                                               [verified: grep]
├── vllm/ (Aug 2026, git, 345 MB)      clean upstream clone 3034c8d38 of hazemmasarani/vllm; NO SG code                                    [verified]
├── docker-vllm/ (empty), speed-bench/ (throughput_1k.jsonl 7.3 MB, stackselect_1k.json, textsort_1k.json), artifacts/ (one AIPerf smoke run)
├── vllm-mamba-benchmarking/, vllm-falcon-benchmarking/, vllm-jamba-benchmarking/ (Aug 2026)   AIPerf results: regular-tp/ and ssm-gate-tp/
├── vllm-{mamba,falcon,jamba}-benchmarking-1-3/ (Sep 2026)   the 1+3 arm; their regular-tp/ logs are byte copies of the August ones        [verified: md5]
├── vllm-falcon-benchmarking copy/     synthetic ISL512/OSL128 ignore_eos series for Falcon-Mamba (TP vs SG 2+2)
├── comparision.ipynb (Sep 25)         cross-model notebook (tokens/s/GPU vs tokens/s/user, Pareto)                                         [verified: parsed]
├── plot_request_throughput_benefit.py + request_throughput_benefit.{png,pdf} (Oct 20)   request/s benefit chart, Falcon-Mamba and Jamba, c16-128  [verified: read]
└── vllm-mamba-sg-backup-2026-08-06.tar (23 GB)   docker save of image vllm-mamba-sg-backup:2026-08-06
```
Backup of the container tree: `~/backups/vllm-mamba-sg-2026-10-01/{head.txt,status.txt,local-commits.bundle,uncommitted.patch,untracked.tgz,SHA256SUMS,vllm/}` `[verified]`.

### 3.2 The SG implementation in vLLM (commit `2195e2a4b`, 2026-08-09, "mamba ssm-Gate TP") `[verified: read the full diff]`
Line numbers below are new-file lines of that commit (computed from the diff hunks); the installed copy in the
container is an older July variant whose lines differ by a few (e.g. `forward_impl` at 103, `send` at 107, `broadcast` at 108 in
`mamba_mixer_gate.py`) `[verified: grep in container]`.

| file | what it adds |
|---|---|
| `vllm/config/parallel.py:165` | `ParallelConfig.enable_mamba_sg: bool = Field(default=False)` |
| `vllm/engine/arg_utils.py:475`, `:1016-1019`, `:2103` | `EngineArgs.enable_mamba_sg`; CLI flag `--enable-mamba-sg`; passed into `ParallelConfig` |
| `vllm/distributed/mamba_sg_parallel_state.py` (116 lines) | `MambaSGRole` enum (`:12`); `validate_mamba_sg_sizes` (`:17`, requires gate+ssm == tp and, at `:40-41`, gate == ssm); `get_mamba_sg_role` (`:44`): `tp_rank < gate_parallel_size -> GATE else SSM` (`:55-56`); `get_ssm_model_parallel_rank` returns `tp_rank - gate_parallel_size` (`:116`) |
| `vllm/distributed/parallel_state.py:1374-1388` | globals `_MAMBA_SG_BRANCH`, `_MAMBA_SG_PAIR` and getters `get_mamba_sg_branch_group`, `get_mamba_sg_pair_group` |
| `vllm/distributed/parallel_state.py:1838-1864` (inside `initialize_model_parallel`) | if enabled: requires even TP >= 2; `branch_size = tp // 2`; for each TP group `gate_ranks = tp_ranks[:branch_size]`, `ssm_ranks = tp_ranks[branch_size:]`; `init_model_parallel_group(..., group_name="mamba_sg_branch")` over `[gate_ranks, ssm_ranks]` and `group_name="mamba_sg_pair"` over `[[gate_i, ssm_i]]` |
| `vllm/distributed/parallel_state.py:2101-2112` | `destroy_model_parallel` tears both groups down |
| `vllm/model_executor/layers/mamba/mamba_mixer_gate.py` (143 lines) | `MambaMixerGate(PluggableLayer)` registered as `"mamba_mixerGate"` (`:25-26`); `in_proj = nn.Linear(H, I/branch)` (`:65`) with a custom `weight_loader` that copies rows `[I + branch_rank*local, +local)` of the fused `[2I, H]` checkpoint tensor, i.e. the **z half** (`:78-88`); `forward` dispatches to the custom op `torch.ops.vllm.mamba_mixer_gate` (`:102-105`); `forward_impl` (`:107-112`): `gate = in_proj(h).transpose(-2,-1).contiguous()`; `pair_group.send(gate, dst=1)`; `pair_group.broadcast(output, src=1)`; registered with `direct_register_custom_op(op_name="mamba_mixer_gate", mutates_args=["output"])` (`:138-143`) |
| `vllm/model_executor/layers/mamba/mamba_mixer_ssm.py` (580 lines) | `MambaMixerSSM(MambaBase, PluggableLayer)` registered as `"mamba_mixerSSM"` (`:46-47`); branch-local shards built with plain `nn.Linear` (comment `:103-105`: "intentionally do not use vLLM's global TP linear layers, because gate ranks execute a different collective sequence"): `conv1d` (`:106`), `in_proj` x-half (`:110`, loader rows `[branch_rank*local, +local)` at `:188`), `x_proj` with input dim local (`:113`, loader `narrow(1, offset, local)` at `:198`), `dt_proj` (`:118`), `A` (`:121`, loader `-exp(A_log)` at `:210`), `D` (`:126`), `out_proj` input-column shard (`:127`, loader at `:201`); `_row_parallel_output` (`:226-229`): `F.linear` then `branch_group.all_reduce`; `_ssm_transform` (`:231-250`): `branch_group.all_reduce(self.x_proj(x))` then split dt/B/C, Falcon RMS norms, `dt_proj` without bias; `forward_impl` (`:272-503`): see call chain below; `get_state_shape` uses `tp_world_size=self.branch_size` (`:514-520`) |
| `vllm/model_executor/models/mamba.py:60-83` | `MambaDecoderLayer(..., enable_mamba_sg, mamba_sg_role)` picks `MambaMixer` / `MambaMixerGate` / `MambaMixerSSM` |
| `vllm/model_executor/models/mamba.py:126-160` | `MambaModel` reads `parallel_config.enable_mamba_sg`, sets `gate_size = ssm_size = tp // 2`, `self.mamba_sg_role = get_mamba_sg_role(...)` |
| `vllm/model_executor/models/mamba.py:212-233` | custom `load_weights`: renames `A_log -> A`, and under SG silently skips checkpoint tensors the role does not own (`:223-227`); `hf_to_vllm_mapper` removed |
| `vllm/model_executor/models/mamba.py:305-309` | `get_mamba_state_shape_from_config` uses `tp // 2` under SG |
| `vllm/v1/core/kv_cache_utils.py:1807-1813` | `generate_scheduler_kv_cache_config`: initialise the scheduler from a **cache-owning** worker config ("Role-parallel workers can be cacheless") |
| `vllm/v1/core/kv_cache_utils.py:2023-2028` | `_project_kv_cache_groups_to_worker`: a worker with no layers in a group gets **no group at all** ("Do not leave an empty group behind") |
| `vllm/v1/core/kv_cache_utils.py:2167-2170` | `get_kv_cache_configs`: `min_num_blocks` over cache-owning configs only |

Commit `4c0c73684` (2026-08-27, "falcon-Jamba") `[verified: read]`: `_custom_ops.topk_softmax` shim for a 7-argument
extension build; `ParallelConfig.__post_init__` even-TP check; `MambaMixerGate` now inherits `MambaBase` and gains
`get_state_dtype`, `get_state_shape`, `mamba_type` and `get_kv_cache_spec` (returns the Mamba spec **only when
`hf_config.model_type == "jamba"`**, else `None`: "advertising the paired Mamba spec keeps Jamba's scheduler block tables
rank-symmetric"); `layer_name = prefix`; `models/jamba.py` gets the same mixer selection, skips `.mamba.` weights the
role does not own, and halves `tp_world_size` in `get_mamba_state_shape_from_config`.

Uncommitted 1+3 patch (`~/backups/.../uncommitted.patch`, also `stash@{0}` in the container) `[verified: read]`:
`ParallelConfig.ssm_tensor_parallel_size` / `gate_tensor_parallel_size` (+ CLI `--ssm-tensor-parallel-size`,
`--gate-tensor-parallel-size`; `__post_init__` derives the missing one and validates the sum); drops the pair group and
the equal-size check; both mixers use `ceil(I / role_size)` local widths with zero-padded weight loaders; the gate forward
becomes `for gate_rank in range(branch_size): tp_group.broadcast(shard, src=gate_rank)` then
`tp_group.broadcast(output, src=self.branch_size)`; the SSM forward gathers every gate shard by the same broadcasts,
concatenates, pads to `local_I * branch_size` and re-slices its own `[branch_rank*local_I, +local_I)` rows; state shapes
are padded accordingly; `tests/config/test_mamba_sg_parallel.py` (role assignment for (2,2), (3,1), (1,3)) and a
`test_mamba_sg_parallel_size_args` case in `tests/engine/test_arg_utils.py`.

### 3.3 Control and data flow of one forward step under SG 2+2 (pure Mamba model) `[verified from the diff; runtime order from the Tester]`
Shapes: H = hidden, I = `intermediate_size` (2H), T = tokens in the step (decodes first, then prefill tokens, padded).
1. Engine core schedules a step; every TP worker (ranks 0-3) receives the same `SchedulerOutput` and builds its persistent
   input batch (`vllm/v1/worker/gpu_model_runner.py`). `_may_reorder_batch` (installed copy `:1065-1087`): returns
   immediately if `len(self.kv_cache_config.kv_cache_groups) == 0`, otherwise calls
   `reorder_batch_to_split_decodes_and_prefills(self.input_batch, scheduler_output, decode_threshold=self.reorder_batch_threshold)`.
   The threshold comes from `calculate_reorder_batch_threshold` (min over attention groups; `None` when there are no
   groups); the Mamba backends set `reorder_batch_threshold = 1` (`vllm/v1/attention/backends/mamba_attn.py:81`, `:162`).
   **Gate ranks have no groups** (section 3.2, `kv_cache_utils.py:2023-2028`) so they skip this; **SSM ranks reorder**.
2. Embedding (`VocabParallelEmbedding`, TP all-reduce over all 4 ranks) and the layer loop in `MambaModel.forward`.
3. Per decoder layer, `MambaDecoderLayer.forward`: `hidden_states, residual = input_layernorm(h, residual)`, then
   `self.mixer(hidden_states, output)`; the residual add is in the layer, identical on every rank.
4. Gate rank (`mamba_mixer_gate.py:107-112`): `gate [I/2, T] = in_proj_z(hidden)`, `pair_group.send(gate, dst=1)`,
   then blocks in `pair_group.broadcast(output, src=1)` until the SSM partner has produced the layer output.
5. SSM rank (`mamba_mixer_ssm.py:272-503`): `hidden_states_BC [I/2, T] = in_proj_x(hidden)` (`:321`);
   `gate = pair_group.recv(size, dtype, src=0)` (`:322-324`); profile run shortcut if `attn_metadata is None` (`:330-339`);
   `gate_d, gate_p = split(gate[..., :num_actual_tokens], [num_decode_tokens, num_prefill_tokens])` (`:356-360`);
   prefill: `causal_conv1d_fn` (`:394`), `_ssm_transform` (x_proj + 2-rank all-reduce of `[T, dt_rank+2N]`, `:239`),
   `selective_scan_fn(..., gate_p, ...)` (`:417-437`, the kernel applies `silu(z)`); decode: `causal_conv1d_update`
   (`:455`), `_ssm_transform`, `selective_state_update(..., z=gate_d.T, ...)` (`:474-488`) updating `ssm_state` in place
   via `state_batch_indices`; concat decode+prefill outputs; `out = _row_parallel_output(projected)` (`:501`: `out_proj`
   shard then 2-rank all-reduce of `[T, H]`); `output[:num_actual_tokens] = out` (`:502`); `pair_group.broadcast(output, src=1)` (`:503`).
6. Both ranks now hold the identical `[T, H]` mixer output; residual add; next layer. After the last layer: final norm,
   `ParallelLMHead` (vocab-parallel, all-gather of logits across all 4 ranks), sampling on the driver (rank 0 = a gate rank).
Collectives per layer: SSM ranks 2 x two-rank all-reduce + 1 recv + 1 broadcast; gate ranks 1 send + 1 broadcast; TP
baseline (`vllm/model_executor/layers/mamba/mamba_mixer.py`: `in_proj = MergedColumnParallelLinear`, `conv1d` and
`dt_proj = ColumnParallelLinear`, `x_proj` and `out_proj = RowParallelLinear`) = 2 x four-rank all-reduce `[verified: grep of the local copy]`.

### 3.4 Config resolution `[verified]`
`vllm serve ... --enable-mamba-sg` -> `EngineArgs.enable_mamba_sg` -> `ParallelConfig(enable_mamba_sg=...)` ->
`initialize_model_parallel(parallel_config)` builds the two extra NCCL groups -> `MambaModel.__init__` computes the role
from the TP rank -> `MambaDecoderLayer` instantiates the role's mixer class. The role split is purely positional
(first half of each TP group = gate). The running server also passes `--device-ids 0,1,2,3 --dtype float16
--enforce-eager`; the Aug benchmarks used `--tensor_parallel_size 4 --enable_mamba_sg --enforce-eager` with vLLM defaults
otherwise (`gpu_memory_utilization 0.92`, `max_model_len 2048` for the Mamba configs) `[from-agent: Audit, from the
container's bash history and server_metrics exports]`.

### 3.5 Conventions worth knowing `[verified from the diff]`
- `PluggableLayer.register("mamba_mixerGate")` / `("mamba_mixerSSM")`: vLLM's layer registry; the mixers register
  themselves in `compilation_config.static_forward_context` under their layer name and run through
  `direct_register_custom_op` wrappers (`torch.ops.vllm.mamba_mixer_gate/_ssm`) so `torch.compile` sees opaque ops with a
  `fake_impl`; collectives happen inside the op.
- Weight loading is by per-parameter `weight_loader` attributes (`set_weight_attrs`), slicing the fused HF `in_proj`
  `[2I, H]` tensor by role and branch rank; `A_log -> A` rename happens in `MambaModel.load_weights`.
- State ownership: only `MambaMixerSSM` subclasses `MambaBase` in commit 1, so only SSM ranks register Mamba cache
  layers; commit 2 makes the gate mixer a `MambaBase` too but still advertises no spec for pure-Mamba models.
- The profile run (`attn_metadata is None`) must execute the same collective sequence on both roles, hence the
  `_row_parallel_output` + broadcast shortcut in the SSM mixer.

### 3.6 Benchmark harness layout `[from-agent: Audit, spot-checked by me: md5 of the copied TP logs]`
- Client: NVIDIA AIPerf 0.11.0 from conda env `mamba-ssm` on the host, run by `<repo>/<arm>/scripts/benchmark.sh`;
  result dirs `<repo>/{regular-tp,ssm-gate-tp}/results/<workload>_c<N>/aiperf.log` (+ `profile_export*.json(l)`,
  `server_metrics_export.json` under `artifacts/`).
- SPEED-Bench series: `aiperf profile --model <id> --tokenizer <id> --url http://localhost:8000 --endpoint-type completions
  --endpoint /v1/completions --streaming --custom-dataset-type speed_bench_throughput_1k --input-file
  /home/adutt/masarani/speed-bench/throughput_1k.jsonl --osl 128 --concurrency {1,2,4,8,16,32,64,128} --request-count 1536
  --warmup-request-count 20 --extra-inputs temperature:0` (greedy, max_tokens 128, **no ignore_eos**).
- Synthetic series (Mamba TP/SG2+2, Falcon "copy"): `--synthetic-input-tokens-mean 512 --synthetic-input-tokens-stddev 0
  --output-tokens-mean 128 --output-tokens-stddev 0 --request-count 500 ... --extra-inputs ignore_eos:true`.
- `comparision.ipynb` and `plot_request_throughput_benefit.py` parse the box-drawn AIPerf tables from `aiperf.log`.

### 3.7 This repo (`multi_modal_parallelism`) in one paragraph `[verified]`
`harness/` = 4-rank layer harness on vLLM v0.30.0 kernels (Falcon-H1 branch split experiments), `configs/`, `scripts/`,
`analysis/`, `results/<model>/{figs,summary}`, `doc/stage1_elastic_mixer_parallelism_plan.md` (plan + findings sections
21-24), `blog/README.md` + `blog/make_figs.py` + `blog/figs/` (the stage-1 write-up, commit `c7a8044`), `docker/`
(image `mmp:stage1`, container `mmp-dev`). It is unrelated to the SG code except as the successor line of work.

## 4. Container internals

### 4.1 How the container relates to the repos `[verified unless tagged]`
- The container was created from the stock `vllm/vllm-openai:latest` (= v0.24.0) image with `/bin/bash` as entrypoint.
  Nothing from `~/masarani` is mounted; the only bind mount is the HF cache, which is unreadable inside. The vLLM source
  checkout `/vllm` was **cloned inside the container's writable layer** (84,823 of the 285,631 `docker diff` entries are
  under `/vllm`), so it is container-only state.
- **What actually executes is not `/vllm`.** `vllm.__file__` is the pip-installed package under
  `/usr/local/lib/python3.12/dist-packages/vllm`. The SG files were patched into that installed package by hand:
  `distributed/mamba_sg_parallel_state.py` (mtime 2026-07-31 04:18), `model_executor/layers/mamba/mamba_mixer_gate.py`
  (04:19), `mamba_mixer_ssm.py` (04:30), `mamba_mixer_ssm_old.py` (2026-07-23 03:53), `v1/core/kv_cache_utils.py` (04:19),
  `model_executor/models/mamba.py` (04:26); `models/jamba.py` is the stock image file `[from-agent: Tester]`. This copy
  predates both git commits (Aug 9 / Aug 27); it implements the pair send/recv + broadcast variant with equal role sizes only
  (`validate_mamba_sg_sizes` raises "Mamba SG currently requires equal gate and SSM groups"; the 1+3 fields are commented
  out in its `config/parallel.py:124-126`) `[from-agent: Tester]`. Differences to `/vllm` HEAD, in diff lines:
  mamba_sg_parallel_state 29, parallel_state 67, mamba_mixer_gate 12, mamba_mixer_ssm 47, models/mamba 49,
  config/parallel 19, engine/arg_utils 122, kv_cache_utils 156 `[from-agent: Tester]`.
- `/vllm` **cannot be imported**: `import vllm._C_stable_libtorch` fails with `libcudart.so.12: cannot open shared object
  file` because its compiled extensions were built for CUDA 12 while the image ships CUDA 13.0.2 `[from-agent: Tester]`.
  Commit `4c0c73684`'s `_custom_ops.topk_softmax` shim ("Keep source checkouts compatible with extension builds...")
  shows the source tree was at some point run against the image's prebuilt extensions `[inferred]`.
- So the exact code that produced the Aug 11 - Sep 21 benchmark numbers cannot be pinned to a commit `[unknown]`;
  the strongest evidence is the installed copy's mtimes (2026-07-31) plus the Aug-6 image tar containing the same files
  `[from-agent: Audit]`.

### 4.2 Git state of `/vllm` inside the container `[verified via docker exec]`
```
 M vllm/distributed/parallel_state.py
 M vllm/model_executor/layers/mamba/mamba_mixer_gate.py
2195e2a4b mamba ssm-Gate TP            <- HEAD, branch main
3034c8d38 [CI][PD] Add optional/nightly DSv4 Disaggregated eval (#42310)
stash@{0}: On main: pre-rollback-2026-10-23
branches: main, backup/pre-rollback-20261023 (= 4c0c73684 falcon-Jamba), origin/main
```
Reading `[inferred from the names and dates]`: on 2026-10-23 the user rolled the checkout back from `4c0c73684` + the 1+3
working tree to the first commit, keeping the old state as a branch and a stash, then started the idle TP=4 SG server.
The current working-tree diff (16+/3-) adds `GroupCoordinator.all_to_all_single(...)` and replaces the gate forward's
`send`+`broadcast` with `pair_group.all_to_all_single(gate, output, ...)` while `mamba_mixer_ssm.py` still does
`recv`+`broadcast`: an inconsistent work-in-progress that would deadlock or mismatch if run `[from-agent: Tester]`.
Note the discrepancy with the Oct-1 backup tree, which has `4c0c73684` as HEAD and the 1+3 patch applied: the backup
is the pre-rollback state `[verified: both trees inspected]`.

### 4.3 `docker diff vllm-mamba-sg` (filtered) `[verified]`
Total entries 285,631 (the writable layer is large: a clone, caches, vscode-server, codex plugins per the Aug-6 tar
listing). Relevant lines:
```
A /root/.bash_history                 (interactive shells; contains the vllm serve launch history used by the Audit)
A /vllm/...                           (84,823 entries: the source checkout, including stale tests/config/__pycache__/*.pyc)
A /tmp/sg_tests, /tmp/sg_tests/{results,logs}/*, sg_run.py, sg_diag.py, sg_lp.py, sg_selfcheck.py, sg_compare.py,
  sg_anom.py, sg_diag_analyze.py, run_step3.sh, run_step4.sh, run_step4b.sh, run_step5.sh, run_step5b.sh,
  run_chain2.sh, run_chain3.sh, stash_test_arg_utils.py, stash_test_mamba_sg_parallel.py   (this session's tests, 15 MB)
/tmp/vllm-hf-cache/...                (HF_HOME of the running server: state-spaces/mamba-2.8b-hf, 11 GB)   [from-agent: Tester]
/usr/local/lib/python3.12/dist-packages/vllm/...   (the hand-patched SG files listed in 4.1; mtimes verified, diff lines not captured in my 60-line window)
```
No file under `/vllm` was created or modified by this session (the Tester ran pytest with `PYTHONDONTWRITEBYTECODE=1
-p no:cacheprovider` and checked for files newer than the session start) `[from-agent: Tester]`.

## 5. Exploration log

Order of work in this session (2026-10-25/26). Agents ran read-only on guppy via `ssh guppy` and copied small files to
`/tmp/masarani-ro` on etracker1. The vLLM-implementation reader is the only thread I did myself after its agent failed.

### 5.0 Pre-work in the same conversation (2026-10-05, 10-25)
- Blog assets for the stage-1 Falcon-H1 experiments: `blog/README.md`, `blog/make_figs.py`, `blog/figs/*`, README
  status update; commit `c7a8044`, pushed `[verified: git log]`.
- Explanation of the recipe heuristics in `harness/synth.py` (R1 replication cap, R2 summed branches, R3 phase balance) and
  of the exact `split22` layout; reconstruction of the Nemotron 3 Nano architecture from
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` `config.json` and `modeling_nemotron_h.py` (fetched from the HF hub) and
  the repo's `configs/nemotron3_nano.yaml` (section 6, F8) `[verified]`.
- Reminder of the SG results from `~/masarani/comparision.ipynb` (copied and parsed) `[verified]`.

### 5.1 Map of `~/masarani` (me) `[verified]`
`ls`, `du`, `find` for git repos, READMEs, notebooks; backup dir listing; `git ls-remote` of the GitHub fork (no SG
branch names). Found: `~/masarani/vllm` is clean upstream; the SG tree is the backup; the project is a two-year trail
(section 3.1). SSH connection resets occurred when four agents opened sessions concurrently (sshd `MaxStartups`), so later
agents were told to batch commands.

### 5.2 Agent "Impl" (vLLM implementation) — failed, replaced by my own reading
Asked for a line-level description of the SG implementation. It died on the API usage limit after 9 tool calls, having
copied `commit1.show`, `commit2.show`, `committed.diff` and `uncommitted.patch` to `/tmp/masarani-ro`. I read all four
in full (`[verified]`); the results are sections 3.2-3.5 and findings F1, F2, F7.

### 5.3 Agent "Prototypes" (standalone SG prototypes) — completed
Reported the dated timeline (section 3.1), the mechanics of each prototype, the intended design, results and an assessment.
Key points (all `[from-agent: Prototypes]` unless marked):
- First split 2026-02-14 (`model-distribution/model4/two_devices_modeling.py`): `MambaMixerSSM`/`MambaMixerGate` classes in
  one file, `mp.spawn` inside `MambaModel.forward`; results never returned to the caller (`hidden_states = inputs_embeds`
  after the spawn) — mechanics only.
- March 2026 (`model6/7`, `HydraMamba`, `mamba-ssm-gate`, `falcon-mamba-ssm-gate`): two separate HF checkpoints
  (`HMasarani/mamba-ssm`, `HMasarani/mamba-gate`; `hmasarani/falcon-mamba-{ssm,gate}`), one process per GPU; **both roles
  run `out_proj`, RMSNorm, residual, embeddings and lm_head**; the exchange per layer is two blocking NCCL broadcasts of
  `[B, 5120, L]` fp32 (`gate` from rank 0, `scan_output` from rank 1) — I re-read
  `HydraMamba/mamba_ssm_gate/mamba_gate_modeling.py:198-231` and confirm `[verified]`. SSM state only on the SSM rank.
  Inputs are random *integer* tensors fed as `inputs_embeds`.
- `tp-experiment/test2.py` (Apr 2026): sketch of two TP groups whose leaders exchange tensors by send/recv then broadcast
  in-group (not what was built later).
- `mamba-sg-tp` (May-Jul 2026): roles x TP on 4 GPUs. Role checkpoints sharded over the role's ranks
  (`run_1_cache.py:76-96`), `local_group` per role, a `reshard_strategy` mapping channel overlaps between a gate rank and
  each SSM rank (`run_1_cache.py:48-74`); final variant `model/all_to_all_single/`: x_proj partials all-reduced in the SSM
  group, the cross-role exchange as one `dist.all_to_all_single` over the world with zero-sized splits for same-role
  ranks (I re-read `mamba_ssm_modeling.py:359-392`, `[verified]`), out_proj partials all-reduced within the role group.
  `send_recv` variant never `.wait()`s its `isend/irecv` handles (abandoned); `scatter_gather` is hard-wired to 3 SSM ranks.
- Results: 2-GPU SG vs TP-2 (HF Python scan, fp32, `SG_TP_compare/SSM_Gate_TP_Compare.ipynb` cell 9): 1.22-1.83x at
  batch <= 4, 0.41-0.98x at batch >= 8. Where the time goes (`HydraMamba/ssm_gate_latency.log`): communication 2-7% of the
  forward; the gate GPU's time equals the SSM GPU's because it blocks in the broadcast; the SSM GPU runs the full 5120-channel
  scan. 4-GPU no-cache: 2+2 all_to_all 1.20-1.27x only at tiny shapes, 0.72-1.05x otherwise; HP (two TP-2 replicas)
  1.2-1.4x at large batch. 4-GPU cached decode: SG 2+2 0.85-0.99x of clean TP-4 (launch-bound, ~55-63 ms/token for every
  batch 1-64). Correctness: 2+2 vs HF original max abs logit diff 4.7e-5..2.0e-4 (`mamba-sg-tp/results/logits_diff.log`).
- `mamba2-tp`: plain TP-2 for Mamba-Codestral-7B, gated-RMSNorm variance all-reduced, prefill only, no split (`[verified:
  grep]`).
- Its assessment: the SSM role is the critical path, the gate role is slack; the split buys fewer/cheaper collectives, not
  less work; unexplored: send only z and fuse `silu(z)*y` (which the vLLM port then did), overlap the gate transfer with
  the scan, give the gate GPU real work, Mamba-2 version.

### 5.4 Agent "Paper-era" (TP for Mamba-1, quantized all-reduce) — completed
Reported (all `[from-agent: Paper-era]` unless marked): channel-split TP (`quantize/mamba/weight_split.py:14-35`,
`mamba-tp/run.py:28-46`): `in_proj` column-parallel (x and z halves separately), conv1d per-channel, `x_proj` row-parallel,
`dt_proj` column, `A_log/D` per-channel, `out_proj` row-parallel; two `torch.distributed.all_reduce` per layer
(`mamba-tp/model/modeling_mamba.py:306-307` for `[B,L,dt_rank+2N]`, `:599-600` for `[B,L,H]`); residual added on rank 0
only; pure-PyTorch sequential scan (no CUDA kernels); `MambaCache` holds per-rank channel shards of conv/SSM state
(`modeling_mamba.py:110-147`), never communicated; the "quantized" all-reduce is an fp32->fp16 cast per tensor
(`quantize/mamba/mixer.py:135-140`), no scales; int8/fp8 never actually reduced; measured all-reduce 87 vs 143 us (SSM
params) and 572 vs 1676 us (hidden) at TP=4 batch 4 x 64 — latency-bound. **Bug** `[verified: I read the lines]`:
`ntk-accuracy/mamba/mixer.py:129` and `:451` call `dist.all_reduce(x.to(torch.float16), group=grp)` on a temporary, so
the reduced result is discarded and every "float16" accuracy run effectively had no collectives; the fp16 accuracy study
(top-1 agreement 98.81%) is therefore invalid. The serving harness spawned one process per GPU per request (5-20 s of
NCCL/CUDA init per call). `mamba2-tp` TP-2 prefill matches the baseline for the 1x32 case (max |diff| 2.1e-5). No
single-GPU baseline or Falcon/Zamba/A100 code exists in the tree; the paper's numbers for those were not produced here.

### 5.5 Agent "Audit" (benchmark campaign) — first run died on the usage limit, relaunched with its parsed data, completed
Reported (section 6, F3; details `[from-agent: Audit]`): server/client configurations, the per-run metric tables for all
series, the output-length anomaly quantified from per-request exports, the invalidity of the Oct-20 chart, missing
telemetry, which cells are trustworthy, and a re-measurement protocol. I verified: the 1+3 directories' `regular-tp` logs
are byte copies of the August ones (`md5sum`, Falcon c1/c128, Jamba c1/c128, Mamba c1 identical) `[verified]`; the
container bash history contains the recorded launch commands (`/tmp/masarani-ro/tar_extract/history_and_jsonl_stats.txt`
lines 4-146) `[verified: read the extract]`.

### 5.6 Agent "Tester" (docker exec tests) — completed; report in section 6, F2 and F5
Instructed: container `vllm-mamba-sg` only, free GPUs only, `gpu_memory_utilization <= 0.45`, no installs, no git state
changes, scratch only under `/home/adutt/sg_tests` and `/tmp/sg_tests`, clean up. Mid-run I sent it the reorder
hypothesis (from the Audit's "corrupts only with several residents" pattern plus commit 2's Jamba-only cache-spec comment)
and asked for mixed prefill/decode batches with small `max_num_seqs` and a read of `_may_reorder_batch`. It confirmed the
mechanism with per-rank instrumentation.

### 5.7 Dead ends and ruled-out hypotheses
- **Client/server truncation** (max_tokens, aborts, errors): ruled out by the Audit — `max_tokens` is 128 in every
  export, no cancelled/aborted/error requests, no preemptions; the shorter outputs are early EOS plus non-canonical tokens
  `[from-agent: Audit]`.
- **KV/state block-count or state-index mismatch between scheduler and SSM ranks** (the Audit's and my first hypothesis):
  not supported — the Tester's instrumentation found corruption exactly and only on steps where the SSM rank reordered and
  the gate rank did not, and runs without such steps were bit-identical on repeat `[from-agent: Tester]`.
- **SG non-determinism as an explanation of divergence**: ruled out — back-to-back identical SG runs are 32/32 identical;
  the apparent run-to-run differences come from `LLM.generate` submission racing the engine loop (batch composition
  differs, bf16 results are batch-composition dependent), which affects TP1/TP2 equally `[from-agent: Tester]`.
- **"SG wins at high concurrency on Falcon-Mamba"**: ruled out — the gain is the output-length deficit `[from-agent: Audit]`.
- **The SG code is in `~/masarani/vllm` or on a GitHub fork branch**: ruled out — clean upstream; no SG branch names
  (`git ls-remote --heads`) `[verified]`.
- **Running the stash's unit tests against the installed vLLM**: 13 failures because the installed copy lacks the
  `ssm_tensor_parallel_size` fields (expected) `[from-agent: Tester]`.
- **TP=4 and 1+3 re-tests**: not possible (GPUs 2/3 hold the user's SSM ranks; 1+3 code only in the stash) `[verified]`.

### 5.8 Known losses
- The two failed agents' partial work exists only as files in `/tmp/masarani-ro`; their reasoning is lost.
- I have only final reports from the four successful agents, not their transcripts; numbers tagged `[from-agent]` were
  not re-derived except the spot checks listed.
- The container's `/root/.bash_history` and the Aug-6 tar were read by the Audit; I did not re-read the tar.
- The stage-1 experiments (2026-10-01/05) are documented in `doc/stage1_elastic_mixer_parallelism_plan.md` and
  `blog/README.md`; this file does not repeat them.

## 6. Findings and explanations

### F1. What the SG split is, exactly (vLLM version) `[verified: diff of 2195e2a4b and 4c0c73684]`
**Claim.** Under `--enable-mamba-sg` with TP = 2k, the k gate ranks hold only the z-half of `in_proj` (I/k x H each),
the k SSM ranks hold the x-half of `in_proj`, conv1d, x_proj, dt_proj, A, D, an `out_proj` column shard and the whole
recurrent state (sharded k ways); each layer costs the SSM ranks two k-rank all-reduces plus one P2P receive and one
pair broadcast, and the gate ranks one P2P send and one pair broadcast; TP costs two 2k-rank all-reduces.
**Evidence.** `mamba_mixer_gate.py:65,78-88,107-112`; `mamba_mixer_ssm.py:106-127,188-210,226-250,321-324,501-503`;
`parallel_state.py:1838-1864` (section 3.2-3.3).
**Explanation.** Mamba-1's gate enters only as `y * silu(z)` after the scan, and vLLM's scan kernels take `z` as an
argument (`selective_scan_fn(..., gate_p, ...)`, `selective_state_update(..., z=...)`), so the z projection can be
computed anywhere and shipped as `[I/k, T]`. The design sends raw z (no SiLU) and returns the finished layer output,
unlike the prototypes which exchanged both directions and duplicated `out_proj`. The gate rank is otherwise idle and
holds no state; the SSM rank's `out_proj` partial is reduced inside the SSM group only.
**Implications (per-rank byte math for Mamba-2.8B, I = 5120, H = 2560, bf16, 64 layers) `[inferred]`:** a TP-4 rank reads
per layer `in_proj` 2I/4 x H = 13.1 MB + `out_proj` H x I/4 = 6.6 MB + x_proj/dt_proj/conv ~1 MB = ~20.7 MB; an SG 2+2 SSM
rank reads `in_proj` I/2 x H = 13.1 MB + `out_proj` H x I/2 = 13.1 MB + ~1.8 MB = ~28 MB (+37%); a gate rank reads 13.1 MB
and then waits. The recurrent state (5120 x 16 x 2 B = 160 KB per layer per sequence) is small next to weights for
Mamba-1, so decode is weight-bandwidth bound and the SSM rank is strictly slower than a TP-4 rank unless collective
latency dominates (tiny batch). This is why the measured wins are confined to small batch x sequence (prototypes) or are
within a few percent (vLLM, Jamba), and why 1+3 loses more: the single gate rank must produce and ship the whole
z projection alone (an I x H GEMM and I x T of transfer per layer) while three SSM ranks each read I/3 of the SSM-side
weights, so the gate rank is the bottleneck and its broadcasts add volume; all measured 1+3 arms lost 14-40%.

### F2. Root cause of the output corruption: batch reorder on SSM ranks only `[code path verified; runtime evidence from-agent: Tester]`
**Claim.** Whenever a step's persistent batch needs the decode-first reorder, SSM ranks permute their rows and gate ranks do
not, so the P2P'd `gate` rows, the broadcast `output` rows and the TP all-reduced embedding/lm_head rows pair up tokens of
different requests; every resident request is corrupted in that step.
**Evidence.**
- Installed `vllm/v1/worker/gpu_model_runner.py:1065-1087` `_may_reorder_batch`: `if len(self.kv_cache_config.kv_cache_groups) == 0: return`
  then `if self.reorder_batch_threshold is not None: reorder_batch_to_split_decodes_and_prefills(...)`; `calculate_reorder_batch_threshold`
  sets `None` when there are no attention groups; `vllm/v1/attention/backends/mamba_attn.py:81` `reorder_batch_threshold: int = 1`
  and `:162` `self._init_reorder_batch_threshold(1, ...)` `[verified: read in the container]`.
- The SG patch makes gate workers group-less: `kv_cache_utils.py:2023-2028` ("Role-parallel workers such as Mamba SG gate
  ranks can own no cache layers. Do not leave an empty group behind") and the gate mixer advertises no cache spec for
  pure-Mamba models (`get_kv_cache_spec` returns `None` unless `model_type == "jamba"`, commit `4c0c73684`; the installed
  copy has no `get_kv_cache_spec` at all) `[verified]`.
- Tester instrumentation (`collective_rpc` monkeypatches of `_may_reorder_batch` and layer-0/last-layer mixer inputs,
  `VLLM_ALLOW_INSECURE_SERIALIZATION=1`, TP=2, `max_num_seqs=8`): gate rank `n_kv_groups=0, reorder_threshold=None`; SSM
  rank `n_kv_groups=1 (64 layers), reorder_threshold=1`. In two of three identical "waves" SG runs the SSM rank reordered
  at steps 17, 18, 34, 35 (e.g. step 17 gate order `[39, 40(prefill), 34, 35, 36, 37, 38]` vs SSM `[39, 38, 34, 35, 36, 37, 40]`)
  and exactly those requests diverged (request 40 first-token |dlogp| 0.96, requests 34-37 at the token generated in step
  17, request 48 1.19); the third run needed no reorder and was 24/24 identical; regular TP reordered on both ranks and
  showed only bf16 noise (max 0.18) `[from-agent: Tester]`.
- Benchmark signature: corruption appears only at concurrency >= 8 (continuous arrivals put a new prefill into a freed
  slot ahead of decoding requests), grows with concurrency, is absent for Jamba `[from-agent: Audit]`.
**Explanation.** vLLM v1 keeps a persistent `InputBatch`; the Mamba backends ask for decodes-before-prefills ordering and
the runner performs it by swapping request rows (`reorder_batch_to_split_decodes_and_prefills` in
`vllm/v1/attention/backends/utils.py`). The decision is per worker and keyed on that worker's KV-cache groups. The SG
patch deliberately gives gate workers no groups (to make cache sizing work for cacheless ranks), which also disables
their reorder. Jamba's gate ranks still own attention KV groups (and after commit 2 also advertise the Mamba spec), so
they reorder identically; Mamba-2.8B and Falcon-Mamba have no attention, hence the model-dependent symptom. Since
logits are vocab-parallel across all four ranks and the driver (rank 0) is a gate rank, the corruption is global:
cross-rank fingerprints were equal because the TP all-reduces force both ranks to the same garbled values
`[from-agent: Tester]`.
**Implications.** Every SG number taken at concurrency >= 8 on a pure-Mamba model is from a server producing wrong text
(F3). Fix options `[inferred]`: (a) let gate ranks advertise the paired Mamba cache spec on all models (as commit 2 does for
Jamba) so they get a group and the same threshold, accepting that they then allocate a (small, unused) state buffer;
(b) move the reorder decision to the scheduler output so all workers apply the same permutation; (c) have the SSM rank
send its row permutation with the broadcast. (a) is a two-line change in `MambaMixerGate.get_kv_cache_spec` and should be
tried first; it must be validated with the Tester's mixed-admission test (`max_num_seqs` 8, waves of finishing requests).

### F3. What the AIPerf campaign actually shows `[from-agent: Audit; md5 and history spot-checked by me]`
**Claims.**
- One TP baseline per model (Aug 2 Mamba, Aug 19 Falcon, Aug 25 Jamba); the `-1-3` repos' `regular-tp` logs are byte copies
  `[verified: md5]`; Mamba has no TP/SG pair on SPEED-Bench (its TP and SG 2+2 are the synthetic ISL-512 `ignore_eos` series).
- Under SG on pure-Mamba models at c >= 8: output length -9 to -17%, server `finished_reason=stop` share 2-3x TP's
  (Falcon 27-38% vs 12-13%), 10-60% of requests return text that does not re-tokenize to the streamed token count; even with
  `ignore_eos` 5-30% of outputs are non-canonical vs 1-4% for TP. Jamba: no anomaly in any arm.
- The Oct-20 chart `request_throughput_benefit.png` (`plot_request_throughput_benefit.py`, `[verified: read]`) compares
  requests/s only: Falcon-Mamba +17.8/+19.4/+23.9/+22.7% at c16/32/64/128, Jamba +2.6/+5.0/+5.8/+7.1%. Corrected to TP's
  output length, Falcon is +0.4/+0.3/+2.7/+7.9% (from corrupted runs); Jamba's bars stand (output tok/s +2.9/+4.8/+6.0/+7.1%,
  ITL 2-7% better, TTFT 7-11% better).
- Defensible SG 2+2 vs TP deltas: Jamba -3.5/-3.2/-0.6/-0.4% output tok/s at c1/2/4/8, +2.9/+4.8/+6.0/+7.1% at c16-128, TTFT
  6-14% lower throughout; Falcon-Mamba at c1-4: 2-5% slower per token, TTFT 13-19% lower; Mamba synthetic 2+2: -8..-4% at
  c1-16, +0.3/+0.8% at c32/64. SG 1+3: -14% (c1) to -40% (c128) on Jamba; worse elsewhere.
- Settings evidenced by `server_metrics_export.json`: `gpu_memory_utilization 0.92` (default), `max_model_len 2048` for
  the Mamba configs (4 x HTTP 400 "maximum context length is 2048 tokens" per SPEED-Bench run, TP and SG alike), Jamba
  262144; `enable_prefix_caching False`; no preemptions; `--enforce-eager` recorded for the Mamba launches and inferred for
  the rest (ITL at c=1 42-47 ms is launch-bound).
- No GPU telemetry was ever captured (DCGM exporter unreachable; all `gpu_telemetry_export.jsonl` empty); the Falcon SG
  2+2 AIPerf artifacts were overwritten by the later TP run (only `aiperf.log` survives).
**Implications.** The only clean positive result of the whole campaign is Jamba 2+2 at c >= 16 (+3 to +7% tokens/s,
TTFT -6 to -14%), where Mamba layers are a fraction of the model. Everything claimed for Falcon-Mamba at c >= 8 must be
withdrawn until F2 is fixed.

### F4. Functional test results in the container `[from-agent: Tester]`
Config: `state-spaces/mamba-2.8b-hf` bf16, `enforce_eager=True`, `max_model_len=1024`, `gpu_memory_utilization=0.45`,
`max_num_seqs=64`, seed 0, GPUs 0,1, `HF_HOME=/tmp/vllm-hf-cache`, `HF_HUB_OFFLINE=1`; `LLM(..., enable_mamba_sg=True)`
accepted (installed `EngineArgs` field at `arg_utils.py:471`). 8 or 32 fixed prompts (6-500 words), greedy, `logprobs=1`.
- Loads: TP1 6.4 s, TP2 8.7 s, TP2+SG 10.6 s. All runs exit 0, no hangs. Gate rank ~3 GB on GPU 0, SSM rank ~20.7 GB on GPU 1.
- SG vs regular TP2 (same TP size, the comparison that matters): 8 prompts x 64 tokens: first token 8/8 identical, max
  |d logp0| 0.0285 (median 0.018), 1/8 sequences fully identical; 32 prompts x 128: first token 32/32, max 0.194.
  Noise floor from same-config repeats with different batch composition: first token 31/32, max 0.19-0.24. Back-to-back
  identical runs are 32/32 identical in every config. Conclusion: no SG-specific deviation on reorder-free steps.
- Output length: 32 prompts at once, `max_tokens=128`, `ignore_eos=False` -> all 128/'length' in TP1, TP2 and SG (this base
  model emits no EOS on these prompts); `ignore_eos=True` -> all 128. Mixed admission with `max_num_seqs=8` and per-request
  `max_tokens` 8-128: SG vs TP2 first token 32/32, max 0.172; two SG processes 32/32 identical. "Waves" (`max_num_seqs=8`,
  32 x 128 tokens, 8 finish together): one of three identical SG processes corrupted (8/32 identical, divergence at the last
  token of the finishing requests and token 0 of the admitted ones, max |d logp0| 1.20; e.g. prompt 6 last token `187 @ -3.97`
  vs `6150 @ -0.035`) — the F2 signature.
- Falcon-Mamba-7B bf16 TP=2 (weights streamed into `/tmp/sg_tests/hfcache`, deleted afterwards): SG vs TP2 8 prompts first
  token 8/8, max 0.30; 32 prompts 31/32, max 0.356; Falcon's own noise floor is higher (TP1 vs TP1 repeat 28/32, max 0.62).
- Throughput (32 prompts x 128, `ignore_eos`, warm): Mamba-2.8B TP1 623, TP2 641, TP2+SG 572 tok/s (-11% vs TP2);
  Falcon-7B TP1 437, TP2 497, TP2+SG 426 (-14%). Batch sweeps Mamba TP1/TP2/SG: bs1 28.5/24.2/26.4, bs4 107/92.9/100.5,
  bs8 201/181/189, bs16 358/350/344 tok/s.
- Not run: CPU unit tests (`tests/conftest.py` needs `tblib`, not installed; the SG test file exists only in `stash@{0}`;
  the stash's tests fail 13/13 against the installed vLLM because it lacks the 1+3 fields), TP=4, 1+3.

### F5. The three generations and why the gains stayed small `[from-agent: Prototypes, Paper-era; my synthesis inferred]`
- Gen 1 (Mar 2026, 2 GPUs): both roles duplicate `out_proj`/norm/residual/lm_head; exchange both tensors by broadcast;
  SSM GPU runs the full-width scan; gate GPU idle. Wins 1.2-1.8x only at tiny batch x seq against the instrumented
  paper-era TP server; loses at batch >= 8.
- Gen 2 (May-Jul 2026, 4 GPUs): roles x TP, all_to_all reshard; 2+2 within +/-5% of TP-4 except tiny shapes; decode
  0.85-0.99x of TP-4. The best 4-GPU configuration at large batch in their own data was two TP-2 replicas (HP).
- Gen 3 (Aug 2026, vLLM): send only z, SSM owns `out_proj` and state; benchmark gains <= 8% (Jamba) and otherwise invalid.
- Common cause: the gate path is one GEMM; removing it from the SSM ranks saves little, while concentrating conv/scan/
  out_proj on half the ranks doubles their per-rank work. The lever the data points to is fewer, cheaper collectives and
  better kernel occupancy at small batch, not the role split itself.

### F6. Paper-era lessons that carry over `[from-agent: Paper-era; the fp16 bug verified]`
Per-layer messages are tiny (768 B + 10 KB per token for Mamba-2.8B) and all-reduce cost is latency-bound (~1-2 GB/s
effective on PCIe); halving bytes with fp16 cut 143 -> 87 us on the small tensor only; the sequential-scan prototypes were
75-95% Python scan; the fp16 accuracy study is invalid (discarded reduce) and int8/fp8 reduction never ran; Mamba-2/GDN
remove the x_proj all-reduce structurally (B, C, dt come from the column-parallel `in_proj`, heads stay aligned with
their groups), leaving one `out_proj` reduce plus a 4-byte-per-token gated-norm reduce.

### F7. Provenance and state of the SG code `[verified]`
Running code = hand-patched site-packages copy (July 31); committed code = two commits in the Oct-1 backup tree and on
`backup/pre-rollback-20261023`; 1+3 code = uncommitted patch / `stash@{0}` only; `/vllm` HEAD rolled back to commit 1 with
an inconsistent all_to_all WIP; `/vllm` not importable in this image; GitHub fork `hazemmasarani/vllm` main = upstream
(no SG branch found). Reproducing the benchmarks from source would first require a CUDA-13 build of the checkout or
re-applying the diffs to the installed package.

### F8. Nemotron 3 Nano architecture (context for the planned pivot) `[verified from HF config and modeling code]`
`NemotronHForCausalLM`, hidden 2688, 52 pre-norm blocks with one mixer each,
`hybrid_override_pattern = MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME` (23 Mamba-2 "M", 23 MoE "E", 6 GQA
attention "*"), no dense MLP, attention without positional encoding (no rotary path in `modeling_nemotron_h.py`),
32 q heads x 128 with 2 KV heads (KV 1 KiB/token/layer, 6 KiB/token total), Mamba-2 with 64 heads x 64 dims, 8 groups,
d_state 128 (`in_proj [10304, 2688]`, SSM state 2 MiB fp32 per layer per sequence), MoE with 128 routed experts
(`up [1856, 2688]`, `down [2688, 1856]`, relu^2) + 1 shared expert (3712), sigmoid routing with bias correction, top-6,
`routed_scaling_factor 2.5`; ~31.6B total / ~3.6B active parameters from the shapes (model card: 30B / 3.5B). Per decode
step the routed expert weights dominate (up to ~56 GB summed over ranks at batch 64), KV only matters at long context
(6 KiB x B x c), Mamba state read+write is 94 MiB x B. The repo's `configs/nemotron3_nano.yaml` encodes these dims.

## 7. Changes made

No source code, configuration, container state or git state was modified anywhere in this session. The changes are
documentation in this repo, scratch files on guppy created by the Tester, and memory notes on etracker1.

### 7.1 This repo (`multi_modal_parallelism`)
**(a) `blog/` and README status (commit `c7a8044`, 2026-10-05, pushed) `[verified: git log, git ls-files]`**
- `blog/README.md` (250 lines): technical write-up of the stage-1 Falcon-H1 experiments (architecture, measurement
  stack, 7B result, swing-shard negative result, recipe, family results, byte-model accuracy, next steps, reproduce).
- `blog/make_figs.py` (466 lines): rebuilds `blog/figs/` from `results/*/summary/summary.md` and
  `results/tp2_reference.csv`: PNGs `split22_speedup_heatmaps.png`, `recipe_pred_vs_meas.png`,
  `phase_balance_no_collectives.png`, `tp2_vs_split22.png`; SVG schematics `layer_dataflow.svg`, `rank_layouts.svg`,
  `barrier_timeline.svg`, `recipe_pipeline.svg`, `harness_stack.svg` (hand-written SVG via a small helper class).
- `README.md`: "Status" paragraph replaced (was "Planning. No experiment has been run yet."), layout list gains `blog/`,
  intro sentence notes that swing MLP shards were rejected.
- Why: the user asked for blog-style assets of the stage-1 experiments. Tested by viewing every rendered figure
  (PNG directly, SVG rasterised with cairosvg in a scratch venv). Revert: `git revert c7a8044`.
```diff
 README.md | status + layout (minimal edit)
 blog/README.md | 250 ++++  (new)
 blog/make_figs.py | 466 ++++ (new)
 blog/figs/{5 svg, 4 png} (new)
```
**(b) `NOTES.md` (this file), commit on 2026-10-26.** Revert: `git rm NOTES.md`.

`git diff --stat` for this session's two commits: see `git show --stat c7a8044` and the NOTES.md commit; the working
tree is otherwise clean `[verified: git status]`.

### 7.2 Outside the repo (would be lost if the container were recreated or /tmp cleaned)
| location | what | size | created by | keep? |
|---|---|---|---|---|
| container `vllm-mamba-sg:/tmp/sg_tests/` | test scripts (`sg_run.py`, `sg_diag.py`, `sg_lp.py`, `sg_selfcheck.py`, `sg_compare.py`, `sg_anom.py`, `sg_diag_analyze.py`), driver shells (`run_step3.sh`, `run_step4.sh`, `run_step4b.sh`, `run_step5.sh`, `run_step5b.sh`, `run_chain2.sh`, `run_chain3.sh`), stash test copies, `results/*.json` (23 files), `logs/*.log` | 15 MB | Tester | copy to the repo if the tests are to be reused (`docker cp vllm-mamba-sg:/tmp/sg_tests ...`) |
| host `guppy:/home/adutt/sg_tests/` | the same scripts plus `container_copy/{results,logs}` | small | Tester | yes (durable) |
| container `/tmp/sg_tests/hfcache` | Falcon-Mamba-7B weights streamed in for step 5 | 14 GB | Tester | **deleted** by the Tester `[from-agent: Tester]` |
| container `/root/.bash_history` | appended by interactive `docker exec` shells `[inferred]` | - | - | - |
| etracker1 `/tmp/masarani-ro/`, `/tmp/pace-mmp/`, `/tmp/nemotron_*.{json,md,py}`, `/tmp/comparision.ipynb`, `/tmp/runs_table.txt` | read-only copies, parsed data, this clone | ~2 GB | agents / me | session temp; nothing unique except the parsed tables (`tables.md`, `runs.json`, `server_metrics.json`), which can be regenerated with `/tmp/masarani-ro/scripts/*.py` |
| etracker1 `~/.claude/projects/-home-pace/memory/ssm-gate-split-deep-dive.md` (+ index line in `MEMORY.md`) | Claude Code memory note with the SG facts | 4 KB | me | yes |

Container-only changes that are **not** from this session but matter: the hand-patched SG files in
`/usr/local/lib/python3.12/dist-packages/vllm` (section 4.1), the `/vllm` checkout with its stash and backup branch,
`/tmp/vllm-hf-cache` (11 GB model cache). None of these exist outside the container except the Oct-1 backup tree (git
objects only; no site-packages copy) and the Aug-6 image tar `[verified: backup dir listing; tar contents from-agent: Audit]`.

## 8. Reproduction guide

Times are from this session (shared box, eager mode). Steps 1-4 need only ssh; 5-8 need `docker exec` on the running
container and two free GPUs.

1. **Get the code and the diffs (1 min).**
   ```bash
   git clone git@github.com:anuragdutt/multi_modal_parallelism.git && cd multi_modal_parallelism && cat NOTES.md
   ssh guppy 'cd ~/backups/vllm-mamba-sg-2026-10-01 && cat head.txt status.txt && git -C vllm log --oneline -3'
   # expected: 4c0c7368...; 9 " M" files + "?? tests/config/test_mamba_sg_parallel.py"; falcon-Jamba / mamba ssm-Gate TP / 3034c8d38
   ssh guppy 'git -C ~/backups/vllm-mamba-sg-2026-10-01/vllm show 2195e2a4b --stat && cat ~/backups/vllm-mamba-sg-2026-10-01/uncommitted.patch | wc -l'   # 8 files; 579 lines
   ```
2. **Confirm the campaign facts (2 min).**
   ```bash
   ssh guppy 'for m in falcon jamba; do md5sum ~/masarani/vllm-$m-benchmarking/regular-tp/results/speed_bench_throughput_1k_osl128_c1/aiperf.log ~/masarani/vllm-$m-benchmarking-1-3/regular-tp/results/speed_bench_throughput_1k_osl128_c1/aiperf.log; done'
   # expected: identical hashes per model (the 1+3 repos reuse the August TP logs)
   ssh guppy 'grep -h "Output Sequence Length\|Request Throughput" ~/masarani/vllm-falcon-benchmarking/{regular-tp,ssm-gate-tp}/results/speed_bench_throughput_1k_osl128_c32/aiperf.log'
   # expected: OSL avg ~119.6 (TP) vs ~100.4 (SG); request throughput 2.32 vs 2.77 req/s
   scp guppy:~/masarani/comparision.ipynb /tmp/ && python3 - <<'EOF'
   import json; nb=json.load(open('/tmp/comparision.ipynb')); print(len(nb['cells']), 'cells')   # 12
   EOF
   ```
3. **Reproduce the per-run tables (5 min, optional).** Re-run the Audit's parsers from `/tmp/masarani-ro/scripts/` after
   `scp`-ing the `results/*/aiperf.log` and `artifacts/*/server_metrics_export.json` trees, or read `tables.md`.
4. **Inspect the container and the code that runs (1 min).**
   ```bash
   ssh guppy 'docker inspect vllm-mamba-sg --format "{{.State.Status}} {{.Image}}"; docker exec vllm-mamba-sg bash -c "python3 -c \"import vllm;print(vllm.__version__, vllm.__file__)\"; cd /vllm && git status --short && git log --oneline -1 && git stash list"'
   # expected: running sha256:4091d559...; 0.24.0 /usr/local/lib/python3.12/dist-packages/vllm/__init__.py; 2 modified files; 2195e2a4b; stash@{0}: On main: pre-rollback-2026-10-23
   ssh guppy 'docker exec vllm-mamba-sg bash -c "cd /vllm && python3 -c \"from vllm import LLM\""'   # expected: ImportError libcudart.so.12 (checkout not importable)
   ssh guppy 'docker exec vllm-mamba-sg sed -n 1065,1087p /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py'   # _may_reorder_batch early return
   ```
5. **Check GPU availability (10 s).** `ssh guppy 'nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv'`.
   Expect the user's SG server on all four GPUs (gate ranks ~2 GB on 0/1, SSM ranks ~42.8 GB on 2/3). Use GPUs 0,1 with
   `gpu_memory_utilization<=0.45`; TP=4 is impossible while it runs.
6. **Correctness, SG vs TP at TP=2 (3 x ~2 min).** The Tester's `sg_run.py` is in `guppy:/home/adutt/sg_tests/`
   (and `/tmp/sg_tests` in the container). Its core is equivalent to:
   ```python
   from vllm import LLM, SamplingParams
   llm = LLM(model="state-spaces/mamba-2.8b-hf", tensor_parallel_size=2, enable_mamba_sg=True,   # False for regular TP
             enforce_eager=True, dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.45, max_num_seqs=64, seed=0)
   outs = llm.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=64, logprobs=1))
   ```
   run as `docker exec -i -e CUDA_VISIBLE_DEVICES=0,1 -e HF_HOME=/tmp/vllm-hf-cache -e HF_HUB_OFFLINE=1 vllm-mamba-sg python3 - < sg_run.py`
   once per config, then compare token ids and first-token logprobs. Expected: first-token top-1 identical on 8/8 and
   32/32 prompts, max |d logp0| <= 0.03 (8 prompts) / ~0.2 (32 prompts, batch-composition noise), loads 6-11 s.
7. **Reproduce the corruption (3 x ~3 min).** `max_num_seqs=8`, 32 prompts x `max_tokens=128`, `ignore_eos=True`
   ("waves": 8 finish together, 8 admitted). Run SG three times and compare: expect at least one run to diverge at the
   last token of finishing requests and token 0 of admitted requests with |d logp0| up to ~1.2, while regular TP2 shows
   only noise (< 0.2). To see the mechanism, monkeypatch `GPUModelRunner._may_reorder_batch` via `llm.collective_rpc`
   (needs `VLLM_ALLOW_INSECURE_SERIALIZATION=1`) and log `self.input_batch.req_ids` per rank: the SSM rank's order changes
   on the admitting steps, the gate rank's does not (`sg_diag.py`).
8. **Throughput sanity (2 x ~1 min).** Same 32 prompts, `max_tokens=128`, `ignore_eos=True`, `max_num_seqs=64`, warm
   second run: expect ~640 tok/s (TP2) vs ~570 tok/s (SG) on Mamba-2.8B.
9. **Rebuild the stage-1 figures (10 s, optional).** `python3 -m venv v && v/bin/pip install matplotlib pandas numpy && v/bin/python blog/make_figs.py`.

## 9. Commands reference

All remote commands are `ssh -o BatchMode=yes guppy '<cmd>'` from etracker1 (one session at a time; four concurrent
sessions triggered `kex_exchange_identification: Connection reset by peer`).

**Mapping the folder**
- `cd ~/masarani && ls -la && du -sh * | sort -h` — top level and sizes.
- `find . -maxdepth 3 -name .git -prune -print` — which subdirectories are git repos.
- `ls -la ~/backups/vllm-mamba-sg-2026-10-01` — the SG backup (head.txt, status.txt, bundle, patch, tree).
- `git ls-remote --heads https://github.com/hazemmasarani/vllm.git | grep -iE "sg|mamba|gate"` — no SG branches on the fork.
- `git -C ~/backups/vllm-mamba-sg-2026-10-01 bundle list-heads local-commits.bundle` — commits in the bundle.

**Reading the implementation**
- `git -C ~/backups/vllm-mamba-sg-2026-10-01/vllm show 2195e2a4b` / `show 4c0c73684` — the two SG commits (copies: `/tmp/masarani-ro/commit{1,2}.show`).
- `cat ~/backups/vllm-mamba-sg-2026-10-01/uncommitted.patch` — the 1+3 patch.
- `grep -n "ParallelLinear\|all_reduce" vllm/model_executor/layers/mamba/mamba_mixer.py` — TP baseline collectives.
- `docker exec vllm-mamba-sg sed -n 1065,1087p /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py` — `_may_reorder_batch`.
- `docker exec vllm-mamba-sg grep -n reorder_batch_threshold /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mamba_attn.py` — threshold = 1.
- `docker exec vllm-mamba-sg grep -n "Role-parallel\|cache_owning" /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py` — the cacheless-worker patch in the installed copy.

**Container state**
- `docker ps -a --format "{{.Names}}\t{{.Image}}\t{{.ID}}\t{{.Status}}"`
- `docker inspect vllm-mamba-sg --format "{{.Id}} {{.Image}} {{.State.Status}} {{json .Mounts}} {{json .HostConfig.PortBindings}} {{json .Config.Entrypoint}}"`
- `docker inspect vllm-mamba-sg --format "{{json .Config.Env}}"` — image env (no secrets; redact anyway).
- `docker image inspect <image-id> --format "{{.RepoTags}} {{.Created}} {{json .Config.Labels}}"`
- `docker exec vllm-mamba-sg bash -c "python3 -c 'import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__, vllm.__file__)'; pip list | grep -iE '^(vllm|torch|flashinfer-python|transformers|triton) '"`
- `docker exec vllm-mamba-sg bash -c "cd /vllm && git status --short && git log --oneline -3 && git stash list && git branch -a"`
- `timeout 900 docker diff vllm-mamba-sg | grep -E "^[ACD] /(tmp/sg_tests|vllm/vllm/model_executor/layers/mamba)" | head` — filtered filesystem diff (full diff: 285,631 lines, ~1-2 min).
- `docker exec vllm-mamba-sg ps -eo pid,etime,args | grep "vllm serve"` and `nvidia-smi --query-gpu=index,memory.used --format=csv` — who holds the GPUs.

**Benchmark data**
- `scp guppy:~/masarani/comparision.ipynb /tmp/` then parse `cells[4]` outputs (`display_data` text/plain) for the per-run table.
- `md5sum ~/masarani/vllm-{falcon,jamba,mamba}-benchmarking{,-1-3}/regular-tp/results/*c1/aiperf.log` — copies check.
- `grep -h "CLI Command" ~/masarani/vllm-*-benchmarking*/*/results/*/aiperf.log | sort -u` — the AIPerf command lines.
- `python3 /tmp/masarani-ro/scripts/parse_runs.py`, `parse_server_metrics.py`, `build_tables.py` — the Audit's parsers (inputs must be re-copied).

**Tests (Tester; scripts in `guppy:/home/adutt/sg_tests/` and `vllm-mamba-sg:/tmp/sg_tests/`)**
- `docker exec -i -e CUDA_VISIBLE_DEVICES=0,1 -e HF_HOME=/tmp/vllm-hf-cache -e HF_HUB_OFFLINE=1 vllm-mamba-sg python3 - < /home/adutt/sg_tests/sg_run.py` — one config run (args via env/argv as in the driver shells `run_step3.sh` etc.).
- `docker exec -i ... vllm-mamba-sg python3 - < sg_diag.py` — per-rank reorder/fingerprint instrumentation (needs `VLLM_ALLOW_INSECURE_SERIALIZATION=1`).
- `cd /vllm && PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider --noconftest --import-mode=importlib tests/engine/test_arg_utils.py -k mamba_sg` — what was attempted (96 deselected; conftest needs `tblib`).

**This repo**
- `python3 -m venv venv && venv/bin/pip install matplotlib pandas numpy cairosvg` then `venv/bin/python blog/make_figs.py` — rebuild figures (cairosvg only to rasterise SVGs for checking).
- `git log --oneline -5`, `git status --short`, `git push origin HEAD:main`.

## 10. Open questions and next steps

**Unresolved questions**
1. Does giving gate ranks a Mamba cache spec on pure-Mamba models (make `MambaMixerGate.get_kv_cache_spec` return the
   spec unconditionally, as commit `4c0c73684` does for Jamba) fully fix F2? Investigate by applying that change to a
   fresh copy of the installed package (not `/vllm`), re-running the Tester's waves test (`max_num_seqs=8`, three SG runs)
   and checking `reorder_threshold` on the gate rank; then confirm the AIPerf OSL statistics match TP at c=32.
2. Which exact source produced each benchmark series? `[unknown]` — the installed copy is dated 2026-07-31 but Jamba
   support (commit 2, Aug 27) needs a `jamba.py` change the installed copy does not have `[from-agent: Tester]`; either
   the Jamba runs used a different tree or the Tester's file comparison missed it. Check the Aug-6 tar's site-packages
   `jamba.py` and the Sep-21 Jamba 1+3 server logs if they exist.
3. What is image `vllm-mamba-sg:local` (43.9 GB, `b4a877ea3b57`)? `[unknown]` — `docker history` it.
4. Is the `--dtype float16` of the running server the dtype used in August (vLLM `auto` -> fp16 for the fp32 Mamba
   config, bf16 for Falcon/Jamba)? `[unknown]`; it matters for the Mamba synthetic numbers.
5. Why did the one-of-three waves run corrupt while two did not? Scheduler timing (which slot a finished request frees)
   `[inferred]`; make the test deterministic by submitting through `AsyncLLM` with controlled arrival times.
6. TP=4 2+2 and 1+3 functional behaviour under the fix: untested (GPUs held).

**Known bugs / suspicious code**
- F2 (reorder mismatch) — the only correctness bug found; fatal for serving.
- `/vllm` working-tree WIP (`all_to_all_single` on the gate side only) is inconsistent with the SSM side; do not run.
- The 1+3 patch pads `I` to a multiple of the SSM role size (5120 -> 5121 for Mamba-2.8B, 8192 -> 8193 for Falcon) and
  re-slices gathered gate shards; its correctness was never checked at TP=4 (`[from-agent: Audit, Tester]`).
- Stash unit tests cannot run in this image (`tblib` missing in `tests/conftest.py`; the SG tests exist only in the stash).
- Benchmarks: `max_model_len` 2048 rejected 1921-token prompts (4 x HTTP 400 per run); no `ignore_eos`; no telemetry;
  overwritten artifacts; TP baselines re-used across months.
- Paper-era: `ntk-accuracy/mamba/mixer.py:129,451` discard the fp16 all-reduce result.

**What I would do next, in order**
1. Fix F2 with option (a) in a scratch copy, validate with the waves test, then re-measure SG 2+2 vs TP on Mamba-2.8B
   and Falcon-Mamba with `ignore_eos:true`, fixed `max_tokens` 128 and 512, explicit `--max-model-len 4096`,
   `--max-num-seqs`, dtype, both eager and CUDA-graph modes, interleaved trials, one artifact dir per run, and a DCGM
   exporter so per-GPU utilisation of gate vs SSM ranks is recorded. Expect SG to be within a few percent of TP and
   slower at low concurrency (F1 byte math).
2. Decide whether SG survives as a layout at all. The data says no for pure Mamba-1 (gate ranks idle, SSM ranks read more
   bytes). The idea survives as a *balancing move* inside a mixer phase — the gate projection is the one piece of Mamba
   work that does not depend on the scan, so idle ranks can compute it and ship it off the critical path — which is how it
   re-enters the Falcon-H1 branch-split design (`doc/stage1_elastic_mixer_parallelism_plan.md` section 22).
3. For the planned pivot to sequential hybrids (Nemotron 3 Nano, Qwen3.5), extend `harness/synth.py`'s
   `sequential_hybrid` template with data-parallel Mamba layers (no collectives, replicated weights, state bytes equal to
   TP) and mixed groups, re-score with graph curves, and measure the attention-DP and Mamba-DP cells the stage-1 sweep
   never ran (section 6, F8 for the byte budget).

## 11. Glossary

- **SG / SSM-gate split / mixer parallelism (Mamba-1):** assigning whole GPUs to the gate (z) path or the SSM (x, conv,
  scan) path of a Mamba-1 block instead of channel-sharding both on every GPU. "2+2" = 2 gate + 2 SSM ranks on 4 GPUs;
  "1+3" = 1 gate + 3 SSM.
- **gate rank / SSM rank:** roles under SG; `MambaSGRole.GATE` is the first half of each TP group.
- **branch group / pair group:** NCCL groups `mamba_sg_branch` (all gate ranks; all SSM ranks) and `mamba_sg_pair`
  (gate_i with ssm_i) created in `initialize_model_parallel`.
- **x / z:** the two halves of Mamba-1's fused `in_proj` output: x feeds conv1d + selective scan, z is the gate
  (`y * silu(z)`).
- **x_proj / dt_proj:** Mamba-1 projections producing dt, B, C from x (row-parallel under TP, hence an all-reduce).
- **TP / regular TP / tp4 / tp2:** vLLM tensor parallelism over 4 or 2 ranks.
- **HP:** the prototypes' "hybrid parallel" baseline: two TP-2 replicas each serving half the batch.
- **Persistent batch / InputBatch / reorder:** vLLM v1's per-worker request table; `reorder_batch_to_split_decodes_and_prefills`
  moves decode requests before prefills when a backend sets `reorder_batch_threshold`.
- **KV-cache groups:** vLLM v1's grouping of cache-owning layers per worker; Mamba state lives in them; a worker with no
  groups skips reordering.
- **AIPerf:** NVIDIA's load generator (0.11.0 here); **ISL/OSL** input/output sequence length; **ITL/TPOT** inter-token
  latency; **TTFT** time to first token; **c / concurrency** simultaneous requests.
- **SPEED-Bench throughput_1k:** nvidia/SPEED-Bench ~1K-token prompt split (1536 records) used for Falcon-Mamba and Jamba.
- **OSL anomaly:** shorter/non-canonical outputs under SG at c >= 8; caused by F2.
- **Stage 1 / elastic mixer parallelism / split22 / swing:** this repo's Falcon-H1 branch-split experiments (see
  `blog/README.md`), unrelated to SG except as the successor idea.
- **Recipe (`harness/synth.py`):** rules R1 (replication cap), R2 (summed branches share one all-reduce), R3 (phase balance)
  that rank rank-group layouts from a model's config.
- **guppy / etracker1:** the shared 4x A6000 host and the operator's own machine.
- **mmp-dev / mmp:stage1:** the stage-1 harness container and image (unrelated to `vllm-mamba-sg`).
