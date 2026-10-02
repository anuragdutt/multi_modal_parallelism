# Stage 1 plan: elastic mixer parallelism on Falcon-H1

Status: plan only. Nothing below has been executed. Implementation starts only on explicit go-ahead.

Author: Anurag Dutt (with Claude as planning assistant). Date: 2026-10-01.

---

## 0. Purpose

Falcon-H1 is a *parallel* hybrid language model: every decoder layer computes an attention mixer and a
Mamba-2 mixer on the same normalized input, combines them as a weighted sum, adds the residual, and then
runs an MLP with its own residual. TII's technical report introduces "mixer parallelism" (MP): the
tensor-parallel (TP) group is split into an attention group and an SSM group, each branch runs on its
own ranks, and one all-reduce sums the outputs. TII reports that MP accelerates inference only at small
batch and short generation, and that the advantage "diminishes and reverses for larger batches and longer
generation sequences". They did not explain the reversal or fix it.

This document plans a one-week, layer-level experiment that tests our explanation of the reversal and our
fix for it, on guppy (4x RTX A6000, PCIe-only), with the exact kernels and collectives that vLLM uses. The
outcome decides whether "elastic mixer parallelism" becomes a chapter of the thesis on architecture-aware
parallelism for SSM-based LLMs.

---

## 1. Hypotheses and the theory behind them

Notation: P ranks in the TP group; A attention ranks and M SSM ranks under MP (A + M = P); B decode
sequences per step; c context length; T_attn, T_ssm, T_mlp the branch and MLP times of one layer on one
GPU if that GPU did all of the work.

**Why plain MP reverses.** With the MLP evenly sharded, the mixer phase of a layer takes
max(T_attn / A, T_ssm / M) under MP but (T_attn + T_ssm) / P under TP. Once kernels saturate memory
bandwidth, max over a split can never beat the sum over the whole group; equality needs A : M to equal
T_attn : T_ssm. T_attn grows with c (KV reads) while T_ssm is fixed (state size does not depend on c), so
any static split is wrong for most of a batch's life. Below saturation MP wins because each rank runs
fewer, larger kernels at better occupancy, which is the regime TII measured.

**H1 (replicated KV costs bandwidth, not just memory).** Under TP with P > number of KV heads, each KV head
is replicated on P / n_kv ranks and every replica rank reads the whole head each decode step. Falcon-H1-7B
has 2 KV heads, so at P = 4 the aggregate KV read traffic is 2x what the model needs.

**H2 (branch split with one reduction).** Ranks 0 and 1 form the attention group, each owning one KV head
and 6 of the 12 query heads; ranks 2 and 3 form the SSM group, each owning 12 of the 24 Mamba heads. Each
rank produces a row-parallel partial of its branch, scaled by the branch multiplier, and a single 4-rank
all-reduce sums everything. Aggregate KV reads halve. vLLM today pays four collectives per layer in decode
(attention `o_proj` all-reduce, Mamba gated-norm sum-of-squares all-reduce, Mamba `out_proj` all-reduce,
MLP `down_proj` all-reduce), so a fused single mixer reduction is a free baseline improvement on its own.

**H3 (swing MLP shards absorb the imbalance at zero migration cost).** Each rank holds its base MLP slice
plus a replicated "swing" slice of the MLP. Per step, swing columns are assigned to the ranks with slack:
attention ranks are slower at long context, SSM ranks at short context. Reassignment moves no data, because
the MLP input is already replicated after the mixer all-reduce and `down_proj` partials are summed by the
MLP all-reduce regardless of who computed them. This is the piece TII's MP lacked.

**Quantitative prediction (per rank, per layer, Falcon-H1-7B, P = 4, bf16).** A TP rank reads 79.4 MB of
weights + 0.512 KB x B x c of KV + 0.786 MB x B of SSM state (read + write). A split attention rank reads
67.6 MB + the same KV; a split SSM rank reads 88.2 MB + 1.573 MB x B of state and no KV. The attention rank
therefore reads the *same* KV bytes as a TP rank; the per-rank gain comes entirely from moving MLP columns
onto the KV-free SSM ranks, bounded by the swing fraction S:

| B, c | TP rank MB | split22 attn / SSM rank MB | balanced max MB | predicted max-rank ratio |
|---|---|---|---|---|
| 1, 512 | 80.5 | 67.9 / 89.8 | ~78.9 (S >= 0.2) | ~1.0 (tie) |
| 16, 8192 | 159.1 | 134.7 / 113.4 | ~124 (S ~ 0.2) | ~1.28 |
| 64, 8192 | 398 | 336 / 189 | 313 at S = 0.4; 279 at S = 1.0 | 1.27 / 1.42 |
| 64, 32768 | 1203 | 1141 / 189 | ~1085 even at S = 1.0 | ~1.1 (attention-bound) |

So the expected regime is: tie at short context, 1.2 to 1.4x at medium context and moderate batch, and a
fade at very long context where the attention branch alone exceeds everything else and would need more
ranks (position-split attention inside the attention group; that is a later stage). Collective counts per
layer: TP 4 four-rank; fused TP 3 four-rank; split22 2 four-rank + 1 two-rank. On a PCIe-only box each
four-rank all-reduce costs on the order of 50 to 100 us at decode sizes, so with 44 layers the collective
count alone is worth several milliseconds per step at small batch. The experiment separates the two effects
(bandwidth vs collective count) by including the fused-TP baseline.

**Falsifiable corollary.** Falcon-H1-34B has 4 KV heads, so at P = 4 it should *tie*, and win only at
P = 8 (NVwulf H200 nodes, later).

---

## 2. Decisions taken (2026-10-01)

- Two-stage approach. Stage 1 (this plan): a 4-rank PyTorch layer harness that imports vLLM's own kernels
  and uses vLLM's own distributed groups, so TP, branch split and swing are compared on identical kernels
  with per-op timing. Stage 2 (later, separate plan): wire the winning mode into the vLLM fork, reusing the
  role-group machinery from the earlier SSM/gate split work, and measure end to end with AIPerf.
- Generalization set: Falcon-H1-7B (primary), Falcon-H1-34B (4 KV heads, falsifiable prediction),
  Hymba-1.5B (NVIDIA's parallel-head hybrid), Qwen3.5-35B-A3B or Nemotron 3 Nano (sequential hybrids served
  data-parallel; swing as slack-driven replicated experts), Falcon-40B or Command-R (parallel-residual
  Transformers: attention and MLP as summed branches). Stage 1 runs only the 7B; the harness is designed
  so the others drop in (section 15).
- vLLM base: fresh fork `anuragdutt/vllm`, branch `mmp/stage1` from tag `v0.30.0` (2026-09-21), editable
  install inside the matching `vllm/vllm-openai:v0.30.0` image using the precompiled release wheel.
- Scope: stage 1 only, about one week, then regroup.

---

## 3. Verified facts

### 3.1 Host: guppy.fsl.cs.sunysb.edu

- 4x RTX A6000, 46,068 MiB each. PCIe Gen4; `nvidia-smi topo -m` reports SYS for every pair; P2P reads
  work; NVLink inactive. AMD EPYC 7543 (64 threads), 251 GB RAM.
- Driver 550.163.01 (CUDA 12.4 natively). CUDA 13.0 containers run through the compatibility libraries
  shipped in the vLLM images; verified on the existing v0.24.0 container.
- Docker 29.8.1, NVIDIA Container Toolkit 1.20.1, runtime available via `--gpus`.
- Disk: 175 GB free on `/` (holds `/var/lib/docker`); HF cache at `~/.cache/huggingface` on NFS with
  1.4 TB free; no Falcon-H1 snapshot cached (7B download is 15.2 GB in 4 shards).
- Existing container `vllm-mamba-sg` (vllm 0.24.0, torch 2.11+cu130, CUDA 13.0.2 image, 289 GB writable
  layer) holds the earlier SSM/gate split work in `/vllm`: two local commits on top of upstream `3034c8d38`
  (`2195e2a4b mamba ssm-Gate TP`, `4c0c73684 falcon-Jamba`), 10 modified tracked files and 1 untracked test
  (`tests/config/test_mamba_sg_parallel.py`), all uncommitted. It must not be stopped or removed; its
  contents are backed up as step 0.
- GPU 0 was occupied on 2026-10-01 by a foreign process (`~/anaconda3/envs/infop/bin/python`, 1.9 GB,
  100% util). Timing runs require all four GPUs idle; the harness refuses to time when
  `nvidia-smi --query-compute-apps` lists foreign processes unless `--allow-shared-gpu` is given
  (development only).
- Upstream vLLM release tags: ..., v0.28.0, v0.29.0, v0.30.0. Pre-releases v0.30.1rc0 and v0.31.0rc1-3
  exist. The `latest-cu129` image also exists (CUDA 12.9, needs no compat layer). Avoid cu134 nightlies.
- `~/multi_modal_parallelism` is this repository (remote `git@github.com:anuragdutt/multi_modal_parallelism.git`,
  identity `anuragdutt <anurag2709@gmail.com>`, SSH to GitHub authenticates as anuragdutt).

### 3.2 Falcon-H1 configurations (bf16, from the HF `config.json` files; parameter totals verified)

| | 3B | 7B and H1R-7B | 34B |
|---|---|---|---|
| hidden / layers / intermediate | 2560 / 32 / 6144 | 3072 / 44 / 12288 | 5120 / 72 / 21504 |
| q heads / kv heads, head_dim | 10 / 2, 128 | 12 / 2, 128 | 20 / 4, 128 |
| mamba heads x d_head (d_ssm) | 32 x 128 (4096) | 24 x 128 (3072) | 32 x 128 (4096) |
| d_state / n_groups / d_conv / chunk | 256 / 1 / 4 / 128 | 256 / 1 / 4 / 256 (H1R: 128) | 256 / 2 / 4 / 128 |
| per-layer weights: attention / Mamba / MLP | 15.7 / 65.8 / 94.4 MB | 22.0 / 60.0 / 226.5 MB | 62.9 / 136.7 / 660.6 MB |
| KV per token per layer; all layers | 1 KiB; 32 KiB | 1 KiB; 44 KiB | 2 KiB; 144 KiB |
| SSM state per sequence per layer; all layers | 2.0 MiB; 64.8 MiB | 1.5 MiB; 66.9 MiB | 2.0 MiB; 146 MiB |
| conv state per sequence per layer | 27 KiB | 21 KiB | 30 KiB |
| whole model | 6.3 GB | 15.2 GB | 67.3 GB |
| attention_in / attention_out / key multiplier | 1.0 / 0.15 / 0.0442 | 1.0 / 0.1042 / 0.0307 | 1.0 / 0.0375 / 0.0110 |
| ssm_in / ssm_out multiplier | 0.5 / 0.0884 | 0.4167 / 0.1179 | 0.25 / 0.0884 |
| ssm_multipliers [z, x, B, C, dt] | [0.354, 0.25, 0.177, 0.5, 0.354] | same | same |
| mlp multipliers [gate, down] | [0.354, 0.0781] | [0.295, 0.0326] | [0.177, 0.0112] |
| embedding / lm_head multiplier | 5.657 / 0.0156 | 5.657 / 0.0130 | 5.657 / 0.0078 |

All checkpoints: `attn_layer_indices` null (every layer has both mixers), `mamba_rms_norm=true`,
`mamba_norm_before_gate=false`, `mamba_conv_bias=true`, `mamba_proj_bias=false`, untied `lm_head`.

### 3.3 vLLM code facts (clone at `guppy:~/masarani/vllm`, upstream main of 2026-07-15; tag v0.30.0 checked on GitHub)

- `vllm/model_executor/models/falcon_h1.py` lines 371-411: forward = `input_layernorm` ->
  `self_attn(h * attention_in_multiplier)` and `mamba(h * ssm_in_multiplier)` ->
  `attn * attn_out_multiplier + ssm * ssm_out_multiplier` -> `+ residual` -> `pre_ff_layernorm` -> MLP ->
  `+ residual`. Both `o_proj` and Mamba `out_proj` are `RowParallelLinear` with the default
  `reduce_results=True`, so each performs its own all-reduce. `k` is scaled by `key_multiplier`
  (line 294). The `mup_vector` (lines 142-196) scales the `in_proj` output blocks `[z | x | B | C | dt]`
  with indices divided by `tp_size`.
- `vllm/model_executor/layers/mamba/mamba_mixer2.py`: heads sharded by TP (line 285); assert
  `n_groups % tp == 0 or n_groups == 1` (line 289); with `n_groups=1` the B and C projections are
  replicated on every rank (`duplicate_groups`). `Mixer2RMSNormGated` (lines 65-168): gate applied before
  the norm; with TP > 1 and `n_groups=1` it all-reduces a `[T, 1]` fp32 sum of squares and divides by
  `tp_size * local_width`; the fused Triton `rms_norm_gated` is used only at TP = 1. Decode kernels:
  `ops/causal_conv1d.causal_conv1d_update` then `ops/mamba_ssm.selective_state_update`; prefill kernels:
  `causal_conv1d_fn` then `ops/ssd_combined.mamba_chunk_scan_combined_varlen`. The whole conv+SSM step is
  the custom op `vllm::mamba_mixer2`, a CUDA-graph splitting op.
- Attention: vLLM's bundled FlashAttention-2 (`vllm.vllm_flash_attn`); FA3 is Hopper-only. The only
  entry point is `flash_attn_varlen_func(...)` with a *paged* KV cache (`block_table`, `seqused_k`);
  there is no `flash_attn_with_kvcache`. KV layout in v0.30.0: `(num_blocks, 2, block_size, num_kv_heads,
  head_dim)`; writes through `vllm._custom_ops.reshape_and_cache_flash`.
- Collectives: `init_distributed_environment` + `initialize_model_parallel(tensor_model_parallel_size=4)`
  build the TP `GroupCoordinator`; `init_model_parallel_group(rank_lists, local_rank, backend)` builds
  sub-groups (must be called on all ranks); `group.all_reduce` goes through `torch.ops.vllm.all_reduce`.
  vLLM's custom P2P all-reduce is disabled for groups of more than two PCIe-only GPUs, so on guppy the
  4-rank TP group uses pynccl and only the 2-rank branch groups can use the custom all-reduce. This is what
  the engine itself does at TP = 4 on this box, so it is the faithful baseline; the harness logs the path.
  `graph_capture()` enters only the TP and PP groups; sub-groups need their own capture context.
- Precedent for "sum partials, then one all-reduce": `vllm/model_executor/models/falcon.py` (original
  Falcon `parallel_attn`) uses `RowParallelLinear(reduce_results=False)` and a single
  `tensor_model_parallel_all_reduce`.
- Decode collectives per Falcon-H1 layer at TP > 1: `o_proj` all-reduce `[T, H]`, gated-norm all-reduce
  `[T, 1]` fp32, `out_proj` all-reduce `[T, H]`, MLP `down_proj` all-reduce `[T, H]`: four for 1.5B, 3B,
  7B and H1R-7B; three for 0.5B (no RMS norm) and 34B (`n_groups=2`, local group norm). Per decode step
  for the 7B: 4 x 44 + 2 (embedding all-reduce, logits gather) = 178.
- v0.30.0 pins `torch==2.13.0`; release asset `vllm-0.30.0-cp38-abi3-manylinux_2_28_x86_64.whl` is the
  CUDA 13.0 build (a `+cu129` wheel is separate); tag commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
  The precompiled editable path also needs `VLLM_USE_PRECOMPILED_RUST=1` and `setuptools-rust`.
- vLLM's max TP for Falcon-H1: 7B is 4 (12 attention heads), 3B is 2 (10 heads), 34B is 2 (the
  `n_groups % tp` assert). The harness owns its sharding, so 34B at 4 ranks is a parameter change there.
- Reusable machinery from the earlier SSM/gate split fork (stage 2 only): `ParallelConfig.enable_mamba_sg`
  and `--enable-mamba-sg`; role assignment in `mamba_sg_parallel_state.py`; `_MAMBA_SG_BRANCH` and
  `_MAMBA_SG_PAIR` groups in `parallel_state.py`; role-dependent layer construction and weight loading;
  KV-cache-spec patches for cache-less ranks. Mamba-1-specific pieces (x_proj/dt_proj, selective scan,
  the send/recv pair protocol) are not reused.

### 3.4 Prior work that bounds the novelty claim

- Falcon-H1 report, arXiv 2507.22448, section 3.3.2: mixer parallelism (naive and interleaved), 1.43x
  training speedup, inference gains at small batch that reverse at large batch; vLLM fork published.
- Decode context parallelism (vLLM blog 2026-08-07) removes KV replication by position-splitting KV at
  the cost of an extra merge collective per attention layer; hybrids listed as future work.
- OpWeave, arXiv 2609.14237: operator-level disaggregation, evaluates Qwen3-Next, notes stage-balance
  bubbles for hybrids; a pipelined, multi-stage approach rather than a synchronous within-layer split.
- Component-aware self-speculative decoding, arXiv 2605.01106: parallel hybrids (Falcon-H1) admit
  SSM-branch drafting with 68% acceptance; orthogonal to this work.
- Our own: arXiv 2602.21144 (TP for Mamba-1 with state cache and quantized all-reduce) and the vLLM
  SSM/gate split (branch split of the Mamba-1 block; gains of at most 8% at concurrency 128 and losses at
  low concurrency because the gate branch is trivially cheap). The swing mechanism explains and fixes that
  imbalance too (section 15).

---

## 4. Experiment design

### 4.1 Unit of measurement

One Falcon-H1-7B decoder layer (layer 0, real weights from the HF safetensors; `--random-weights`
available), one decode step for B sequences each at context c, on 4 ranks, repeated. Per-layer numbers
multiply by 44 for the model; the embedding/lm_head terms are identical across modes and excluded.

### 4.2 Modes

| mode | ranks | attention | Mamba | collectives per layer |
|---|---|---|---|---|
| `tp1` | 1 | all heads | all heads | none (correctness anchor; also the launch-overhead reference) |
| `tp4` | 4 | 3 q heads + replicated KV head per rank (head 0 on ranks 0,1; head 1 on ranks 2,3) | 6 heads per rank, B/C replicated | 4 four-rank: `ar.attn`, `ar.norm` `[T,1]`, `ar.ssm`, `ar.mlp` (vLLM-faithful) |
| `tp4_fused` | 4 | same | same | 3 four-rank: `ar.norm`, `ar.mixer` on `attn_out * oA + ssm_out * oS`, `ar.mlp` |
| `tp4_streams` | 4 | same | same | as fused; the two branches run on two CUDA streams per rank (the baseline TII never ran) |
| `split22` | 4 | ranks 0,1: 6 q heads + 1 KV head each | ranks 2,3: 12 heads each | 1 two-rank `ar.norm` (SSM pair) + 2 four-rank (`ar.mixer`, `ar.mlp`) |
| `split22_swing{S}` | 4 | as split22 | as split22 | as split22; MLP base + swing slice with per-step cuts |

Swing fractions S in {0.1, 0.2, 0.3, 0.4} plus S = 1.0 as the ceiling point. For each (B, c) the swing
mode is run with (a) cuts chosen by the byte model, (b) neutral cuts (must reproduce plain TP), and (c) a
grid over the symmetric cut family, to find the empirical optimum and the regret of the model's choice.

### 4.3 Sweep

B in {1, 4, 16, 64}; c in {512, 2048, 8192, 32768}; variants eager (per-op breakdown) and CUDA graph
(headline numbers); KV block size 16 (vLLM-faithful) with a 256-block contiguous-like comparison at one
cell; 3 repeats interleaved across modes. Roughly 60 to 90 configurations x 16 cells x 2 variants x 3
repeats, 3 to 4 hours unattended.

### 4.4 Metrics

Per rank: CUDA-event time per op (fixed vocabulary, section 7.3), per-step time, wall time per step.
Reported: median over 100 timed steps after 20 warm-up steps; `step_max` = max over ranks; imbalance =
max over ranks of compute time (excluding collectives) divided by the mean; collective share; predicted vs
measured per op and per step; max abs / mean rel error and cosine of each mode's output against an fp32
pure-torch reference and against `tp4`.

### 4.5 Go / no-go

Go if, in the CUDA-graph variant, `split22_swing` with the best S <= 0.4 and model-chosen cuts is at least
1.2x faster than `tp4` at c in {8192, 32768} for B in {16, 64}, within 5% of `tp4` at c = 512 for all B,
and the byte model predicts per-op and per-step times within 15% median relative error.

No-go if no split22 variant ever beats `tp4_fused`: then only the collective-count effect is real and
stage 2 is re-scoped to the fused reduction.

---

## 5. Repository layout

```
multi_modal_parallelism/
├── README.md
├── .gitignore
├── pyproject.toml                  # harness + analysis as an installable package
├── doc/
│   └── stage1_elastic_mixer_parallelism_plan.md   # this document
├── docker/
│   ├── Dockerfile                  # FROM vllm/vllm-openai:v0.30.0 + editable fork + harness deps
│   ├── run.sh                      # creates/starts container mmp-dev with mounts
│   └── env.list                    # NCCL / vLLM env vars used for every run
├── scripts/
│   ├── backup_old_container.sh     # step 0, idempotent
│   ├── fork_vllm.sh                # branch mmp/stage1 from v0.30.0 in anuragdutt/vllm
│   ├── build_image.sh
│   ├── download_model.sh           # tiiuae/Falcon-H1-7B-Instruct into the NFS HF cache
│   ├── smoke_test.sh               # inside the container: imports, 4-rank groups, kernel calls
│   ├── run_microbench.sh
│   ├── run_sweep.sh                # full stage-1 sweep (eager + graph) -> results/raw/
│   ├── run_analysis.sh
│   └── reproduce.sh                # build -> download -> smoke -> microbench -> sweep -> analysis
├── configs/
│   ├── falcon_h1_7b.yaml           # dims and multipliers (fallback for --random-weights)
│   ├── sweep_stage1.yaml           # modes, S values, batches, contexts, variants, repeats
│   └── microbench.yaml
├── harness/
│   ├── config.py                   # FalconH1Dims, RunSpec
│   ├── dist.py                     # vLLM group setup (TP group + branch pair groups), capture contexts
│   ├── sharding.py                 # exact per-rank index math for every mode (pure python, unit-tested)
│   ├── weights.py                  # safetensors loader, random weights, shard-to-device
│   ├── kernels.py                  # the ONLY file that imports vLLM kernels and custom ops
│   ├── layer.py                    # Branch protocol, AttentionBranch, SsmBranch, SwingMLP, ParallelHybridLayer
│   ├── state.py                    # KV cache, SSM state, conv state allocation and deterministic fill
│   ├── timing.py                   # CUDA-event op timer, wall clock, CUDA-graph capture and replay
│   ├── correctness.py              # compare modes vs fp32 reference and vs tp4
│   ├── reference.py                # pure-torch fp32 full-layer decode step (no vLLM)
│   ├── model.py                    # byte model, bandwidth tables, swing cut chooser
│   ├── microbench.py               # achieved-bandwidth curves per kernel class, collective latency tables
│   ├── csvio.py                    # CSV schemas and writers
│   ├── run.py                      # torchrun entry: one (mode, S, cuts, B, c, variant)
│   └── sweep.py                    # torchrun entry: in-process sweep from YAML
├── analysis/
│   ├── load.py
│   ├── plot_layer_time.py
│   ├── plot_imbalance.py
│   ├── plot_collectives.py
│   ├── plot_pred_vs_meas.py
│   ├── plot_bytes_model.py
│   ├── summarize.py                # go/no-go table -> results/summary/summary.md
│   └── make_all.py
├── tests/
│   ├── test_sharding.py            # partition/cover invariants; mup_vector equals vLLM's formula
│   ├── test_swing.py               # cut bookkeeping; neutral cuts equal plain TP
│   ├── test_byte_model.py          # closed-form byte counts
│   └── test_correctness_4gpu.py    # launched via torchrun (marked gpu4)
└── results/
    ├── raw/        (git-ignored)   # *.csv from runs, microbench, correctness
    ├── figs/       (committed)
    └── summary/    (committed)
```

`.gitignore` covers caches, build artifacts, model weights, `results/raw/`, logs, profiler reports,
docker tarballs and local env files.

---

## 6. Step 0: back up the old container

`scripts/backup_old_container.sh`, run on the host as adutt, idempotent:

```bash
set -euo pipefail
B=/home/adutt/backups/vllm-mamba-sg-$(date +%F); mkdir -p "$B"
docker exec vllm-mamba-sg bash -c 'cd /vllm && git status --short > /tmp/status.txt \
  && git diff > /tmp/uncommitted.patch \
  && git bundle create /tmp/local-commits.bundle 3034c8d38..HEAD \
  && tar czf /tmp/untracked.tgz $(git ls-files --others --exclude-standard) \
  && git rev-parse HEAD > /tmp/head.txt'
for f in status.txt uncommitted.patch local-commits.bundle untracked.tgz head.txt; do
  docker cp vllm-mamba-sg:/tmp/$f "$B/"
done
docker cp vllm-mamba-sg:/vllm "$B/vllm"             # full tree, belt and braces (about 14 GB)
docker commit vllm-mamba-sg vllm-mamba-sg-backup:$(date +%F)
sha256sum "$B"/*.patch "$B"/*.bundle "$B"/*.tgz > "$B/SHA256SUMS"
```

Verify with `git bundle verify "$B/local-commits.bundle"` and by counting the 10 modified files in the
patch plus the untracked test in the tarball. Never `docker stop` or `docker rm` `vllm-mamba-sg`.

---

## 7. Docker

### 7.1 Fork

`scripts/fork_vllm.sh`: fork `vllm-project/vllm` to `anuragdutt/vllm` in the GitHub UI (no `gh` CLI on
guppy), then

```bash
git clone https://github.com/anuragdutt/vllm.git /home/adutt/vllm-fork     # full clone, tags needed
cd /home/adutt/vllm-fork && git fetch origin --tags
git checkout -b mmp/stage1 v0.30.0 && git push -u origin mmp/stage1
```

Stage 1 makes no source changes on that branch; it exists so the image has a stable ref and so stage 2
cherry-picks land somewhere.

### 7.2 Image

`docker/Dockerfile`:

```dockerfile
ARG VLLM_TAG=v0.30.0
FROM vllm/vllm-openai:${VLLM_TAG}
ARG VLLM_FORK=https://github.com/anuragdutt/vllm.git
ARG VLLM_BRANCH=mmp/stage1
ARG VLLM_WHEEL=https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0-cp38-abi3-manylinux_2_28_x86_64.whl
ENV VLLM_USE_PRECOMPILED=1 VLLM_USE_PRECOMPILED_RUST=1 VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_WHEEL} \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.30.0 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends git numactl && rm -rf /var/lib/apt/lists/*
RUN git clone --branch ${VLLM_BRANCH} ${VLLM_FORK} /opt/vllm && cd /opt/vllm && git log -1 --oneline
RUN uv pip uninstall --system vllm \
 && uv pip install --system "setuptools>=77.0.3,<81" "setuptools-scm>=8" "setuptools-rust>=1.9" cmake ninja packaging wheel jinja2 \
 && cd /opt/vllm && uv pip install --system --no-build-isolation --no-deps -e . \
 && python3 -c "import vllm, vllm._C, vllm.vllm_flash_attn as fa; assert vllm.__file__.startswith('/opt/vllm'); print(vllm.__version__, fa.FA2_AVAILABLE)"
RUN uv pip install --system numpy pandas matplotlib pytest pyyaml rich safetensors "huggingface_hub[cli]"
WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
```

Rationale: the image already contains the exact torch (2.13.0+cu130) the release wheel was built against;
`--no-deps` keeps uv away from torch; `--no-build-isolation` lets setup.py see that torch; the precompiled
path only extracts the compiled `.so` files from the pinned wheel and skips cmake and cargo.

Fallback if the editable install fails to import (`ImportError` on `vllm._C` or `_vllm_fa2_C`): build
the fork's own `docker/Dockerfile` from source (official devel base, 1 to 2 hours on 64 threads):

```bash
cd /home/adutt/vllm-fork && DOCKER_BUILDKIT=1 docker build --target vllm-openai \
  --build-arg max_jobs=16 --build-arg nvcc_threads=2 --build-arg torch_cuda_arch_list="8.6" \
  -t local/vllm-openai:v0.30.0-src -f docker/Dockerfile .
```

then base the harness image on `local/vllm-openai:v0.30.0-src` and drop the editable-install step. If the
CUDA 13.0 compatibility layer ever regresses on the 550 driver, rebase on `vllm/vllm-openai:v0.30.0-cu129`
with the `+cu129` wheel URL (one-line change).

### 7.3 Container

`docker/run.sh`:

```bash
docker run -d --name mmp-dev --gpus all --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 --env-file docker/env.list \
  -v /home/adutt/multi_modal_parallelism:/workspace \
  -v /home/adutt/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface -w /workspace mmp:stage1 -c "sleep infinity"
docker exec -it mmp-dev bash
```

`docker/env.list`: `NCCL_P2P_LEVEL=SYS` (force P2P across the SYS topology; validate once with
`NCCL_DEBUG=INFO`, expecting "via P2P/direct pointer"), `VLLM_LOGGING_LEVEL=WARNING`,
`TOKENIZERS_PARALLELISM=false`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `PYTHONUNBUFFERED=1`.

Build: `docker build -t mmp:stage1 -f docker/Dockerfile docker/`.

Smoke test (`scripts/smoke_test.sh`, inside the container): import checks (`vllm.__file__` under
`/opt/vllm`, `FA2_AVAILABLE`, `selective_state_update`, `causal_conv1d_update`, `ops.rms_norm`,
`ops.silu_and_mul`, `ops.rotary_embedding`, `ops.reshape_and_cache_flash`), then
`torchrun --standalone --nproc_per_node=4 -m harness.dist --selftest`, which initializes the groups, runs
one `[64, 3072]` bf16 all-reduce on the TP group and one `[64, 1]` fp32 all-reduce on each pair group, and
prints `describe_comm()` for every group (expected: TP group pynccl with custom all-reduce disabled; pair
groups custom all-reduce enabled).

Model: `scripts/download_model.sh` runs `hf download tiiuae/Falcon-H1-7B-Instruct` into the cache
(15.2 GB; `--include config.json model.safetensors.index.json model-00001-of-00004.safetensors` is enough
for layer 0 if time is short). All of layer 0's tensors live in shard 1.

---

## 8. Harness modules

### 8.1 `harness/config.py`

```python
@dataclass(frozen=True)
class FalconH1Dims:
    hidden: int; intermediate: int; n_q: int; n_kv: int; head_dim: int
    n_mamba_heads: int; mamba_head_dim: int; d_ssm: int; d_state: int; n_groups: int; d_conv: int; chunk: int
    rms_eps: float; rope_theta: float; max_pos: int
    attn_in: float; attn_out: float; key_mult: float; ssm_in: float; ssm_out: float
    ssm_mults: tuple[float, float, float, float, float]; mlp_mults: tuple[float, float]
    @classmethod
    def from_hf(cls, model_dir: Path) -> "FalconH1Dims"       # reads config.json
    @classmethod
    def from_yaml(cls, path: Path) -> "FalconH1Dims"          # configs/falcon_h1_7b.yaml

Mode = Literal["tp1", "tp4", "tp4_fused", "tp4_streams", "split22"]

@dataclass(frozen=True)
class RunSpec:
    mode: Mode; swing: float = 0.0; cuts: tuple[int, ...] | None = None   # None -> byte-model choice
    batch: int = 1; ctx: int = 512; variant: Literal["eager", "graph"] = "eager"; kv_block: int = 16
    n_warm: int = 20; n_iter: int = 100; repeats: int = 3; layer_idx: int = 0; seed: int = 0
```

### 8.2 `harness/dist.py`

```python
@dataclass
class Groups:
    rank: int; local_rank: int; world_size: int
    tp: GroupCoordinator            # all 4 ranks, from initialize_model_parallel(tensor_model_parallel_size=4)
    pair: GroupCoordinator          # init_model_parallel_group([[0,1],[2,3]], local_rank, "nccl", group_name="branch_pair")
    attn_ranks: tuple[int, ...]; ssm_ranks: tuple[int, ...]
    device: torch.device

def init_groups(mode: Mode) -> Groups       # set_device; init_distributed_environment(...,"env://",...,"nccl");
                                            # initialize_model_parallel(4); pair group created on ALL ranks in every mode
def describe_comm(g: GroupCoordinator) -> dict   # ranks, custom-AR enabled and max size, pynccl, NCCL version
@contextmanager
def capture_context(groups: Groups): ...    # with graph_capture(device) as ctx: with groups.pair.graph_capture(ctx): yield ctx
def barrier_sync(groups: Groups) -> None    # cuda.synchronize(); dist.barrier()
def destroy() -> None
```

All 4-rank collectives go through `get_tp_group()`; the world group has no device communicator.
`tp1` runs with `--nproc_per_node=1`, where every all-reduce is a no-op.

### 8.3 `harness/sharding.py` (pure python, single source of truth)

```python
@dataclass(frozen=True)
class AttnShard:  q_heads: range; kv_head: int; q_rows: slice; k_rows: slice; v_rows: slice; o_cols: slice; group_rank: int; group_size: int
@dataclass(frozen=True)
class SsmShard:   heads: range; z_rows: slice; x_rows: slice; b_rows: slice; c_rows: slice; dt_rows: slice
                  conv_rows: tuple[slice, slice, slice]; out_cols: slice; norm_cols: slice; group_rank: int; group_size: int
@dataclass(frozen=True)
class MlpShard:   base_cols: slice; fixed_cols: slice; w: int; swing_pieces: tuple[slice, slice, slice, slice]; cuts: tuple[int, ...]
@dataclass(frozen=True)
class RankPlan:   mode: Mode; rank: int; attn: AttnShard | None; ssm: SsmShard | None; mlp: MlpShard
                  norm_group: Literal["tp", "pair", "none"]; combine: Literal["allreduce_each", "sum_then_allreduce", "none"]

def plan_for(mode, rank, world, dims, swing, cuts) -> RankPlan
def swing_width(S: float, cols_per_rank: int = 3072, granule: int = 64) -> int    # w = granule * round(S * cols / granule)
def neutral_cuts(w: int) -> tuple[int, ...]                                        # (0, w, 2w, 3w, 4w)
def symmetric_cuts(w: int, n_attn: int) -> tuple[int, ...]                         # (0, n_a, 2n_a, 2n_a + (2w - n_a), 4w)
def build_mup_vector(dims, group_size: int) -> torch.Tensor                        # fp32 [1, vec_len], identical across the group
```

### 8.4 `harness/weights.py`

```python
def load_layer_from_safetensors(model_dir: Path, layer_idx: int) -> dict[str, torch.Tensor]   # index.json + safe_open; CPU bf16, A_log/D/dt_bias fp32
def random_layer(dims, seed) -> dict[str, torch.Tensor]        # same keys/shapes; N(0, 0.02) linears; A_log = log U(1,16); dt_bias = softplus^-1 U(1e-3, 0.1); D = 1; norms = 1
@dataclass
class RankWeights: ...   # device tensors: wqkv, wo, win, conv_w, conv_b, A (= -exp(A_log)), D, dt_bias, norm_w, wout,
                         # mlp_gate_up_fixed, mlp_down_fixed, mlp_gate_up_swing, mlp_down_swing_T, in_norm, ff_norm, mup_vector
def shard_to_device(full, plan, dims, device) -> RankWeights
def resident_bytes(rw: RankWeights) -> dict[str, int]
```

Weight names in the checkpoint: `model.layers.N.self_attn.{q,k,v,o}_proj.weight`,
`model.layers.N.mamba.{in_proj.weight, conv1d.weight, conv1d.bias, A_log, D, dt_bias, norm.weight, out_proj.weight}`,
`model.layers.N.feed_forward.{gate,up,down}_proj.weight`, `model.layers.N.{input_layernorm,pre_ff_layernorm}.weight`.

### 8.5 `harness/kernels.py` (sole vLLM kernel boundary; version-pinned imports with clear errors)

```python
def rms_norm(x, w, eps, out)                               # vllm._custom_ops.rms_norm
def silu_and_mul(x, out)                                   # ops.silu_and_mul
def rope_cos_sin_cache(dims, device)                       # RotaryEmbedding(128, 128, max_pos, rope_theta, is_neox_style=True, bf16).cos_sin_cache
def apply_rope(positions, q, k, cos_sin_cache)             # ops.rotary_embedding(positions, q, k, 128, cache, True), in place
def alloc_kv(num_blocks, block, nkv, hd, device)           # FlashAttentionBackend.get_kv_cache_shape -> kv, kv.unbind(1)
def write_kv(k, v, key_cache, value_cache, slot_mapping, k_scale, v_scale)   # ops.reshape_and_cache_flash(..., "auto", ...)
def attn_decode(q, key_cache, value_cache, cu_q, seqused_k, max_k, block_table, scale, out)   # flash_attn_varlen_func(max_seqlen_q=1, causal=True, fa_version=2, num_splits=0)
def conv_update(xBC, conv_state_T, w, b, idx)              # causal_conv1d_update(x, conv_state, weight, bias, "silu", conv_state_indices=idx)
def ssm_update(state, x, dt, A, B, C, D, dt_bias, idx, out)   # selective_state_update(..., dt_softplus=True, state_batch_indices=idx, dst_state_batch_indices=idx, out=out)
def gated_rmsnorm(y, z, w, eps, group)                     # group None -> rms_norm_gated(norm_before_gate=False); else native path with group.all_reduce on [T,1] fp32
def gemm(x, w); def gemm_T(x, wT)                          # F.linear / matmul wrappers so the timer can name them
```

### 8.6 `harness/state.py`

```python
@dataclass
class DecodeCtx:  positions: Tensor[B]; cu_q: Tensor[B+1]; seqused_k: Tensor[B]; block_table: Tensor[B, nblk]
                  slot_mapping: Tensor[B] int64; state_idx: Tensor[B] int32; max_k: int
def make_decode_ctx(B, c, kv_block, device) -> DecodeCtx
def alloc_states(plan, dims, B, c, kv_block, device) -> dict   # key/value cache (attention ranks), conv_state [B+1, 3, conv_rows] bf16 (SD layout), ssm_state [B+1, heads, 128, 256] bf16
def fill_states(states, plan, seed) -> None                    # deterministic per (sequence, GLOBAL head, position): every mode's shard holds identical content
```

`make_decode_ctx`: `nblk = ceil((c + 1) / kv_block)`; `block_table[i, j] = i * nblk + j`;
`seqused_k[i] = c + 1`; `positions[i] = c`; `slot_mapping[i] = block_table[i, c // kv_block] * kv_block + c % kv_block`;
`cu_q = arange(B + 1)`; `state_idx = arange(B)` int32. The KV cache has `B * nblk + 1` blocks. The input
`h ~ N(0, 1) [B, 3072]` bf16 comes from a CPU generator with `seed`, identical on all ranks.

### 8.7 `harness/layer.py`

```python
class Branch(Protocol):
    name: str
    def decode(self, x_norm, ctx, timer) -> Tensor           # returns partial [T, H], already scaled by its out multiplier
    def step_bytes(self, B, c) -> dict[str, int]              # for the byte model
class AttentionBranch(Branch)   # qkv gemm -> k *= key_mult -> rope -> write_kv -> attn_decode -> o gemm -> * attn_out
class SsmBranch(Branch)         # in gemm -> * mup -> split(z | xBC | dt) -> conv_update -> split(x | B | C) -> ssm_update -> gated_rmsnorm(group) -> out gemm -> * ssm_out
class SwingMLP:
    def set_cuts(self, cuts) -> None                           # changes two python ints per rank; no data movement
    def forward(self, y, timer) -> Tensor                      # fixed chain + (if assigned) swing chain; partial = (fixed + swing) * down_mult; gate multiplier applied to the gate half before the activation
class ParallelHybridLayer:
    def __init__(self, plan, weights, groups, dims, spec, states, ctx)
    def step(self, h, timer) -> Tensor                         # mode dispatch; writes into a static output buffer
    def capture(self, h_static) -> torch.cuda.CUDAGraph
```

Step order per mode (T = B decode tokens; `rms` = `ops.rms_norm`):

- `tp1`: `x = rms(h)`; `A = attn(x * attn_in)`; `S = ssm(x * ssm_in)` with the fused gated norm;
  `h = h + A + S`; `y = rms(h)`; `h = h + mlp(y)`. No collectives.
- `tp4` (vLLM-faithful order): `x = rms(h)`; `oA = o_gemm(...)`; `A = tp.all_reduce(oA) * attn_out`
  [AR 1]; SSM norm sum-of-squares `tp.all_reduce` [AR 2]; `oS = out_gemm(...)`;
  `S = tp.all_reduce(oS) * ssm_out` [AR 3]; `h = h + A + S`; MLP; `tp.all_reduce(down_partial)` [AR 4];
  residual.
- `tp4_fused`: as `tp4` but `mix = tp.all_reduce(oA * attn_out + oS * ssm_out)` [1 AR]; norm AR and
  MLP AR unchanged; 3 collectives.
- `tp4_streams`: `tp4_fused` with `s_attn` and `s_ssm` streams forked after `x = rms(h)` (event wait), the
  norm AR issued on `s_ssm`, join by events, then the fused sum and mixer AR on the main stream. The order
  on the TP communicator is identical on all ranks (norm AR before mixer AR), so no deadlock.
- `split22`: attention ranks run `AttentionBranch` only (6 q heads, 1 KV head); SSM ranks run `SsmBranch`
  only with `gated_rmsnorm(group=pair)` [pair AR]; every rank then does `mix = tp.all_reduce(partial)`
  [4-rank AR]; `h = h + mix`; `y = rms(h)`; MLP (base or swing); `tp.all_reduce(down_partial)`; residual.
- `split22_swing{S}`: as `split22` with `SwingMLP.set_cuts(cuts)`.

### 8.8 `harness/timing.py`

```python
class OpTimer:   # CUDA events with enable_timing on the stream where the op runs; disabled in the graph variant
    def begin(self, name, stream=None); def end(self, name, stream=None); def new_step(self); def summarize(self) -> dict
def time_eager(layer, h, timer, n_warm, n_iter, groups) -> (step_ms_list, op_stats, wall_ms_per_step)
def time_graph(layer, h, n_warm, n_iter, groups) -> (step_ms_list, wall_ms_per_step)   # capture under dist.capture_context; replay with events around each replay
```

Op vocabulary (used by analysis): `norm_in, attn.qkv, attn.rope, attn.fa, attn.oproj, ssm.inproj,
ssm.conv, ssm.ssu, ssm.norm_local, ar.norm, ssm.norm_post, ssm.outproj, ar.attn, ar.ssm, ar.mixer,
residual, norm_ff, mlp.gateup, mlp.act, mlp.down, mlp.gateup_swing, mlp.act_swing, mlp.down_swing, ar.mlp,
step`.

### 8.9 `harness/correctness.py` and `harness/reference.py`

`reference.decode_step(full_weights, dims, h, kv_full, conv_full, ssm_full, c)` is a pure-torch fp32
implementation on rank 0: attention over the full cache with RoPE; conv shift; `dt = softplus(dt + dt_bias)`,
`dA = exp(dt * A)`, `state = state * dA + dt * x (outer) B`, `y = state . C + D * x`; gated RMSNorm with
`y * silu(z)` before the norm; MLP. `correctness.compare(out, ref)` returns max abs, max rel, mean rel and
cosine. Pass criteria: vs fp32 reference `mean_rel <= 2e-3`, `max_abs <= 2e-2 * max|ref|`,
`cos >= 0.9999`; vs `tp4` `mean_rel <= 1e-3`. All ranks must hold bitwise-identical outputs after the final
all-reduce (gathered on rank 0).

### 8.10 `harness/model.py`

```python
@dataclass
class BwTables: gemv: Curve; attn: dict[int, Curve]; ssu: Curve; conv: Curve; ar: dict[tuple[int, str], Curve]   # from results/raw/microbench.csv
def bytes_per_op(plan, dims, B, c) -> dict[str, int]        # weights read (only ACTIVE MLP columns), kv read, state read+write, conv state, activations
def predict(plan, dims, B, c, bw, variant) -> dict[str, float]   # per-op ms + launch overhead (eager ~4 us/kernel, graph ~1 us) + collectives; streams: max(attn, ssm)
def predict_step(dims, mode, S, cuts, B, c, bw, variant) -> (step_ms, per_rank_ms)   # max over ranks
def choose_cuts(dims, B, c, S, bw, variant) -> tuple[int, ...]    # 1-D search over symmetric_cuts(w, n_a), n_a in range(0, 2w + 1, 128); minimize max-rank predicted time
def total_hbm_bytes(...) -> int                              # summed over ranks (H1 replication waste)
```

### 8.11 `harness/microbench.py`

`torchrun --nproc_per_node=4 -m harness.microbench --out results/raw/microbench.csv` measures, per rank,
median of 50 after 10 warm-up, eager and graph: (a) streaming read bandwidth (`torch.sum` over 1 GB bf16)
for the roofline; (b) `gemv` for `[T, 3072] x W^T` with W row counts {384, 768, 1280, 1536, 2054, 3072,
3596, 6144, 6680, 12288} and T in {1, 4, 16, 64}; (c) `attn_decode` for B in {1, 4, 16, 64}, c in {512,
2048, 8192, 32768}, q/kv head counts in {3/1, 6/1, 12/2}, kv_block in {16, 256}; (d) `ssm_update` for B in
{1, 4, 16, 64, 256}, heads in {6, 12, 24}; (e) `conv_update` for the same B; (f) all-reduce on the TP
group `[T, 3072]` bf16 and `[T, 1]` fp32, and on the pair group for the same shapes, T in {1, 4, 16, 64},
eager and captured. Columns: `kernel, params_json, bytes, median_ms, p10_ms, p90_ms, GBps, rank, variant`.

### 8.12 `harness/run.py` and `harness/sweep.py`

```
python -m harness.run --mode split22 --swing 0.2 --cuts model|neutral|"0,256,512,1024,1280" \
   --batch 16 --ctx 8192 --variant graph --kv-block 16 \
   --model-dir /root/.cache/huggingface/hub/models--tiiuae--Falcon-H1-7B-Instruct/snapshots/<sha> \
   [--random-weights] --layer-idx 0 --n-warm 20 --n-iter 100 --repeats 3 --check --out results/raw/dev.csv --tag dev
```

`harness.sweep --config configs/sweep_stage1.yaml --out results/raw/sweep_<date>.csv` iterates in process
(weights loaded once per mode; states re-allocated per (B, c); for swing it loops S then cuts, with
model-chosen, neutral and the symmetric grid), interleaving repeats across modes
(`for repeat: for mode: for (B, c): ...`) to defeat clock and thermal drift. Rank 0 gathers all ranks'
statistics with `gather_object` and appends CSV rows.

CSV schema (`results/raw/*.csv`): `run_id, ts, image_tag, vllm_commit, mode, S, w, cuts, cuts_src, batch,
ctx, kv_block, variant, repeat, rank, op, n, median_ms, p10_ms, p90_ms, mean_ms, bytes_pred, ms_pred,
comm_tp, comm_pair, sm_clock_mhz, temp_c`. Step rows: `op=step` per rank, `op=step_max` with `rank=-1`
(max over ranks of per-rank medians), `op=step_wall` (wall per step, max over ranks).
`results/raw/correctness.csv`: `mode, S, cuts, batch, ctx, ref, max_abs, max_rel, mean_rel, cos, pass`.

---

## 9. Exact sharding math (Falcon-H1-7B: H = 3072, I = 12288, nq = 12, nkv = 2, d = 128, nh = 24, p = 128, N = 256, G = 1)

Full-layer tensors: `q_proj [1536, 3072]`, `k_proj [256, 3072]`, `v_proj [256, 3072]`, `o_proj [3072, 1536]`,
`in_proj [6680, 3072]` with rows `[z 0:3072 | x 3072:6144 | B 6144:6400 | C 6400:6656 | dt 6656:6680]`,
`conv1d.weight [3584, 1, 4]` with rows `[x 0:3072 | B 3072:3328 | C 3328:3584]` (used as `[rows, 4]`),
`A_log / D / dt_bias [24]`, `norm.weight [3072]`, `out_proj [3072, 3072]`, `gate_proj / up_proj [12288, 3072]`,
`down_proj [3072, 12288]`. `A = -exp(A_log.float())` (vLLM's `a_weight_loader`). Mamba head `j` owns
`z` and `x` columns `[128j, 128j + 128)`, `dt[j]`, `A[j]`, `D[j]`, `dt_bias[j]`, `norm.weight[128j : 128j + 128]`
and `out_proj` input columns `[128j, 128j + 128)`. GQA: q head `i` uses KV head `i // 6`.

### 9.1 `tp4` (rank r in 0..3)

Reproduces `QKVParallelLinear` with `num_kv_head_replicas = 2` and `MambaMixer2`'s sharded loader with
`duplicate_groups=True`.

- Attention: q heads `[3r, 3r + 3)`; KV head `r // 2`. Replication map: head 0 on ranks {0, 1}, head 1 on
  ranks {2, 3}; both replicas compute k and v for the head, write it to their own cache and read all of it
  every step. `q_rows = [384r, 384r + 384)`, `k_rows = v_rows = [128 (r // 2), 128 (r // 2) + 128)`,
  `o_cols = [384r, 384r + 384)`. Per-rank attention weights 5.5 MB.
- Mamba: heads `[6r, 6r + 6)`; `z_rows = [768r, 768r + 768)`, `x_rows = 3072 + [768r, 768r + 768)`,
  `b_rows = [6144, 6400)` and `c_rows = [6400, 6656)` replicated on every rank, `dt_rows = 6656 + [6r, 6r + 6)`;
  per-rank `in_proj [2054, 3072]` ordered `[z 768 | x 768 | B 256 | C 256 | dt 6]`;
  `conv_rows = ([768r, 768r + 768), [3072, 3328), [3328, 3584))` giving `[1280, 4]`; `A, D, dt_bias[6r : 6r + 6]`;
  `norm_cols = out_cols = [768r, 768r + 768)`, `out_proj [3072, 768]`. Per-rank Mamba weights 17.3 MB
  (15.0 + 2.3 MB of replicated B and C).
- `mup_vector` (length 2054, matches `_init_mup_vector` with `tp_size = 4`): `[0 : 768) *= 0.35355`,
  `[768 : 1536) *= 0.25`, `[1536 : 1792) *= 0.17678`, `[1792 : 2048) *= 0.5`, `[2048 : 2054) *= 0.35355`.
- MLP base columns `[3072r, 3072r + 3072)` (gate and up rows; down columns); 56.6 MB per rank.
- States per rank: KV `[num_blocks, 2, 16, 1, 128]` (512 B per token per sequence), conv `[B + 1, 3, 1280]`,
  SSM `[B + 1, 6, 128, 256]` bf16 (384 KiB per sequence, read and written each step).
- Gated-norm all-reduce group: TP (4 ranks), `[T, 1]` fp32; variance = `global_sum / (4 * 768)`.
- Collectives: `ar.attn [T, 3072]`, `ar.norm [T, 1]`, `ar.ssm [T, 3072]`, `ar.mlp [T, 3072]`.

### 9.2 `tp4_fused` and `tp4_streams`

Identical shards and groups; `ar.attn` and `ar.ssm` are replaced by `ar.mixer` on
`attn_out * oA + ssm_out * oS` (3 collectives).

### 9.3 `split22`

- Attention group = ranks {0, 1}, `a = rank`: q heads `[6a, 6a + 6)`, KV head `a` (no replication);
  `q_rows = [768a, 768a + 768)`, `k_rows = v_rows = [128a, 128a + 128)`, `o_cols = [768a, 768a + 768)`,
  `o_proj [3072, 768]`. 11.0 MB per attention rank. KV 512 B per token per sequence; no SSM state.
- SSM group = ranks {2, 3}, `s = rank - 2`: heads `[12s, 12s + 12)`; `z_rows = [1536s, 1536s + 1536)`,
  `x_rows = 3072 + [1536s, 1536s + 1536)`, `b_rows = [6144, 6400)`, `c_rows = [6400, 6656)` (replicated on
  the two SSM ranks), `dt_rows = 6656 + [12s, 12s + 12)`; per-rank `in_proj [3596, 3072]` ordered
  `[z 1536 | x 1536 | B 256 | C 256 | dt 12]`; `conv_rows = ([1536s, 1536s + 1536), [3072, 3328), [3328, 3584))`
  giving `[2048, 4]`; `A, D, dt_bias[12s : 12s + 12]`; `norm_cols = out_cols = [1536s, 1536s + 1536)`;
  `out_proj [3072, 1536]`. 31.6 MB per SSM rank. Conv state `[B + 1, 3, 2048]`, SSM state
  `[B + 1, 12, 128, 256]` (768 KiB per sequence); no KV.
- `mup_vector` (length 3596): `[0 : 1536) *= 0.35355`, `[1536 : 3072) *= 0.25`, `[3072 : 3328) *= 0.17678`,
  `[3328 : 3584) *= 0.5`, `[3584 : 3596) *= 0.35355`. General rule in `build_mup_vector(dims, G)`:
  `[z d_ssm / G | x d_ssm / G | B N | C N | dt nh / G]`; for `n_groups = 1` the B and C blocks are always the
  full N per rank, matching vLLM's extended-groups trick.
- Gated-norm all-reduce group: pair group {2, 3} only; variance = `global_sum / (2 * 1536)`. Attention
  ranks never participate.
- MLP base columns `[3072r, 3072r + 3072)` on all four ranks (56.6 MB each).
- Branch scaling: attention ranks feed `x * 1.0` and return `oA * 0.104167`; SSM ranks feed `x * 0.416667`
  and return `oS * 0.117851`; `ar.mixer` sums the partials over 4 ranks, then `ar.mlp`. The issue order per
  communicator is identical on all members (pair: norm; TP: mixer then MLP), so there is no deadlock risk.

### 9.4 Swing bookkeeping (`split22_swing{S}`)

- `w = 64 * round(S * 3072 / 64)`: S = 0.1 -> 320, 0.2 -> 640, 0.3 -> 896, 0.4 -> 1216, 1.0 -> 3072
  (effective S = 0.104, 0.208, 0.292, 0.396, 1.0).
- Swing piece of rank r: `W_r = [3072 (r + 1) - w, 3072 (r + 1))` (the last `w` columns of its base slice);
  fixed part `F_r = [3072r, 3072 (r + 1) - w)`. Global swing space `W = W_0 | W_1 | W_2 | W_3`, `|W| = 4w`.
- Resident on every rank: `F_r` (gate and up rows, down columns) plus all of `W`
  (`gate_up_swing [2 * 4w, 3072]` ordered `[gate W_0..W_3 | up W_0..W_3]`, `down_swing_T [4w, 3072]`, that
  is, `down` stored transposed so an assignment is a contiguous row slice). Resident MLP bytes per rank
  `= (3072 + 3w) * 18432 B` (S = 0.4: 123.9 MB; S = 1.0: 226.5 MB).
- Assignment = cut vector `cuts = (0 = c_0 <= c_1 <= c_2 <= c_3 <= c_4 = 4w)`; rank r computes
  `F_r` union `W[c_r : c_{r+1}]`. Neutral `(0, w, 2w, 3w, 4w)` reproduces plain TP exactly (unit test).
  Symmetric family `symmetric_cuts(w, n_a) = (0, n_a, 2n_a, 2n_a + (2w - n_a), 4w)`, where `n_a` is the
  number of swing columns per attention rank, grid step 128 (so every slice is a multiple of 64 columns).
  Changing cuts changes two integers per rank; no tensor is moved or copied.
- Per rank the MLP becomes two GEMM chains (fixed and swing); the extra launches (about 3 kernels) are a
  real cost of elasticity and are kept in the measurement (the byte model adds them as launch overhead).

### 9.5 34B later (not this week)

`n_groups = 2, nh = 32`: rank r owns heads `[8r, 8r + 8)` and group `r // 2`, so B and C rows for that
group only (`[6144 + 256 (r // 2), ...)`), with the gated norm per group, again a pair group. `sharding.py`
takes `n_groups` as input, so this is a parameter change, not new code.

---

## 10. Byte model

Per rank, per layer, per decode step:

- `weights = attn_shard + mamba_shard + MLP(F_r + assigned swing)` bytes.
- `kv = nkv_r * 512 B * B * c` read, plus `B * 512 B` written.
- `state = 2 * B * heads_r * 64 KiB` (SSM read + write) `+ 2 * B * 3 * conv_rows_r * 2 B`.
- `t_r = sum over classes of bytes / BW_class(bytes) + n_kernels * launch_overhead + sum of ar(group, msg)`;
  `tp4_streams` uses `max(attn_chain, ssm_chain)` for the branch phase.
- Reference totals (MB) to check the implementation against: `tp4` rank `79.4 + 0.000512 B c + 0.786 B`;
  `split22` attention rank `67.6 + 0.000512 B c`; SSM rank `88.2 + 1.573 B`. The resulting predictions are
  the table in section 1.

The achieved-bandwidth curves come from `harness/microbench.py`, because at small batch the Mamba decode
kernel and the attention decode kernel run far below peak bandwidth (published GDN decode numbers on H100:
about 2% of peak at batch 1, 29% at 16, 70% at 64, 79% at 256), and the per-kernel size differs between TP
and split modes. That curve is the whole reason occupancy effects appear at small batch.

---

## 11. Measurement methodology

- Alignment: `barrier_sync()` immediately before each timed block; 20 untimed steps (Triton autotune, JIT,
  NCCL warm-up), then 100 timed steps back to back with no host sync between them.
- Eager: per-op CUDA events on the op's stream and per-step events; medians over the 100 steps;
  `step_max` = max over ranks of the median step; wall time = elapsed / n_iter after a final sync.
- Graph: capture one full step (collectives included) under `dist.capture_context()` on the capture
  stream after two warm steps on that stream; replay 20 + 100 times with events around each replay.
  Headline numbers come from graph mode; eager provides the op breakdown.
- Repeats: 3, interleaved across modes; the final value is the median of per-repeat medians; CoV across
  repeats must be below 5% or the cell is re-run and flagged.
- Collective accounting: per-op events around an all-reduce include cross-rank wait (skew), which is the
  real cost on that rank; pure collective cost comes from the microbench with aligned ranks. Imbalance
  metric: max over ranks of compute time (excluding `ar.*`) divided by the mean.
- Environment logging per run: SM clock and temperature from `nvidia-smi`, `describe_comm()` per group,
  NCCL version, image tag, vLLM commit (`git -C /opt/vllm rev-parse HEAD`), and the list of foreign GPU
  processes, which must be empty.

---

## 12. Analysis outputs

- `plot_layer_time.py` -> `results/figs/layer_time_vs_ctx_B{1,4,16,64}.png`: x = context (log2),
  y = layer step ms (graph variant solid, eager dashed), one line per mode including swing with the best S
  under model-chosen cuts and under empirical cuts.
- `plot_imbalance.py` -> `imbalance.png`: max / mean per-rank compute for `split22` vs swing (model and
  empirical) as a (B, c) heat map.
- `plot_collectives.py` -> `collective_share.png`: stacked bars of compute vs `ar.*` per mode at
  B in {1, 16}, c in {512, 8192, 32768}.
- `plot_pred_vs_meas.py` -> `pred_vs_meas.png`: scatter per op and per step with `y = x`, plus a histogram
  of relative error; prints median and p90 absolute relative error.
- `plot_bytes_model.py` -> `bytes_per_rank.png`: predicted max-rank bytes per mode vs context (the H1/H3
  theory) with measured step time on a secondary axis.
- `summarize.py` -> `results/summary/summary.md`: table of `tp4 / mode` speedups per (B, c) in graph mode,
  best S and cuts per cell, regret of model-chosen cuts vs the empirical optimum, correctness table, and
  the go/no-go verdict.

---

## 13. Commands

```bash
# host (guppy, as adutt)
bash scripts/backup_old_container.sh && bash scripts/fork_vllm.sh && bash scripts/build_image.sh && bash docker/run.sh
bash scripts/download_model.sh

# inside mmp-dev (docker exec -it mmp-dev bash)
pip install -e /workspace && bash scripts/smoke_test.sh
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m harness.microbench --out results/raw/microbench.csv
torchrun --standalone --nnodes=1 --nproc_per_node=1 -m harness.run --mode tp1 --batch 4 --ctx 2048 --check --out results/raw/dev.csv
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m harness.run --mode tp4 --batch 4 --ctx 2048 --variant eager --check --out results/raw/dev.csv
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m harness.run --mode split22 --swing 0.2 --cuts model --batch 16 --ctx 8192 --variant graph --check --out results/raw/dev.csv
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m harness.sweep --config configs/sweep_stage1.yaml --out results/raw/sweep_$(date +%F).csv
python -m analysis.make_all --raw results/raw --out results

# one command (host)
bash scripts/reproduce.sh
```

---

## 14. Schedule (7 days, 6 to 8 hours per day)

| Day | Work | Estimate |
|---|---|---|
| 1 | Backup (0.5 h); fork and branch (0.5 h); Dockerfile, build, `run.sh`, `env.list` (2 h including pull and build); smoke test with 4-rank groups and `describe_comm` (1.5 h); model download in the background; `config.py`, `csvio.py` (1 h) | 6 h |
| 2 | `sharding.py` and `tests/test_sharding.py` (2 h); `weights.py` (1.5 h); `kernels.py`, `state.py` (2 h); `reference.py`, `tp1` mode, correctness vs fp32 (2.5 h) | 8 h |
| 3 | `tp4` (1.5 h); `tp4_fused`, `tp4_streams` (1.5 h); `split22` with pair-group norm (2 h); `timing.py` eager and graph capture (2 h); correctness for all modes (1 h) | 8 h |
| 4 | `SwingMLP`, cuts, `test_swing.py` (2.5 h); `microbench.py` (2 h); `model.py`, `choose_cuts`, `test_byte_model.py` (2.5 h); `sweep.py` and YAML (1 h) | 8 h |
| 5 | Pilot sweep at B = 16, c in {512, 8192}, all modes, inspection (2 h); fix outliers (2 h); launch the full sweep (about 4 h unattended); draft analysis scripts meanwhile (3 h) | 7 h |
| 6 | Analysis run, figures, `summary.md`, variance re-runs (4 h); README results and roadmap (2 h) | 6 h |
| 7 | Buffer: kv_block = 256 comparison, optional prefill step (`causal_conv1d_fn` + `mamba_chunk_scan_combined_varlen`), write-up | 4 to 6 h |

---

## 15. Risks and mitigations

1. Precompiled wheel or torch mismatch: image and wheel are both the v0.30.0 CUDA 13.0 artifacts; the smoke
   test imports `vllm._C` and `_vllm_fa2_C` and runs a real FA2 and `selective_state_update` call. Fallback:
   the from-source image (section 7.2), 1 to 2 hours.
2. Driver 550 with CUDA 13 compat: proven with the 0.24.0 container; the first smoke test re-proves it. If
   it regresses, rebase on the cu129 image (one-line change).
3. Custom all-reduce disabled for the 4-rank group on PCIe: expected and faithful; log the path; keep NCCL
   environment identical across modes; validate `NCCL_P2P_LEVEL=SYS` once with `NCCL_DEBUG=INFO`.
4. CUDA-graph capture of collectives: use vLLM's own `graph_capture()` plus the pair group's capture
   context; capture on the capture stream; all ranks capture and exit together; no `.item()` inside; warm
   before capture. If capture fails or hangs for a mode, run that mode eager-only, document it in
   `summary.md`, and use eager minus measured launch overhead for go/no-go.
5. FA2 paged vs contiguous KV bandwidth: quantify with kv_block in {16, 256} in the microbench and at
   (B = 16, c = 8192) in the sweep; the byte model fits attention bandwidth per block size; block 16 is the
   reported number.
6. PCIe all-reduce variance and thermal drift: interleaved repeats, medians, a CoV gate, clock and
   temperature logging, no foreign GPU processes (coordinate the GPU 0 user before day 5).
7. Methodology: per-op events include cross-rank waits by design; pure collective costs come from the
   microbench; graph mode gives headline numbers; eager launch overhead is reported separately so fused and
   streams savings are not conflated with launch savings.
8. Numerics: fused sums change summation order; tolerances in section 8.9; `tp1` and the fp32 reference
   catch sharding bugs; the pair-norm variance divisor (`2 * 1536` vs `4 * 768`) and the mup-vector
   offsets are unit-tested against vLLM's formula.
9. setuptools-scm and Rust in the editable install: `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM` and
   `VLLM_USE_PRECOMPILED_RUST=1` set in the Dockerfile; full clone, no `--depth`.
10. Memory: worst case per rank at B = 64, c = 32768 is KV 1 GiB + swing (S = 1.0) 226 MB + states 50 MB,
    far below 48 GB.
11. HF download: public repository; on a 401, accept the Falcon license on HF and set `HF_TOKEN` in
    `docker/env.list` (never committed).
12. Disk: 175 GB free covers one more image; `docker system prune` would free about 136 GB but needs
    explicit approval because it touches other images.

---

## 16. Verification checklist

1. `scripts/backup_old_container.sh` produced the patch, bundle and tarball with matching `SHA256SUMS`;
   `git bundle verify` passes; `docker ps` still shows `vllm-mamba-sg` up.
2. `git log` on `anuragdutt/multi_modal_parallelism` shows the commits pushed; `anuragdutt/vllm` has
   branch `mmp/stage1` at `ced6857a`.
3. `docker exec mmp-dev python3 -c "import vllm; print(vllm.__file__)"` prints
   `/opt/vllm/vllm/__init__.py`; `torch.__version__ == 2.13.0+cu130`; the smoke test passes;
   `describe_comm` shows TP: pynccl, pair: custom all-reduce enabled.
4. `pytest tests/test_sharding.py tests/test_swing.py tests/test_byte_model.py` pass on CPU: shards
   partition q heads, Mamba heads and MLP columns exactly once per group; the KV replication map is
   `{0: (0, 1), 1: (2, 3)}`; mup vectors equal vLLM's formula for G in {1, 2, 4}; neutral cuts equal base
   slices; byte totals match section 10 within rounding.
5. Correctness with real weights at (B = 4, c = 2048) and (B = 64, c = 512): every mode, all S values and
   both cut sources pass the tolerances vs the fp32 reference and vs `tp4`; all ranks hold identical
   outputs; rows in `results/raw/correctness.csv`.
6. Timing sanity: `tp1` at B = 1, c = 512 is about 308 MB divided by achieved bandwidth (roughly 0.5 ms);
   `tp4` is about 80 MB / BW plus 4 collectives; graph step is below eager step; per-op sums are within 10%
   of the step time in eager; `ar.*` on the pair group is cheaper than on the TP group; CoV below 5%.
7. Byte model: predicted vs measured per-op median absolute relative error at most 15% in graph mode
   across the sweep; per step at most 15%; model-chosen cuts within 10% of the empirical optimum in at
   least 80% of (B, c) cells.
8. Figures produced for all four batch sizes; `summary.md` complete with the go/no-go verdict.

---

## 17. Generalization roadmap (how the harness extends beyond Falcon-H1)

The layer abstraction is a `Branch` (methods `shard(plan)`, `decode(x, ctx)`, `prefill(x, ctx)`,
`step_bytes(B, c)`, `resident_bytes()`), a `Combine` op (`sum_then_allreduce(partials, scales, group)` with
an optional per-branch pre-norm that itself needs only a `[T, 1]` group all-reduce), and an `Elastic`
component (a replicated slice of some column-parallel / row-parallel pair plus a per-step cut vector).

- Falcon-H1-34B: `n_groups = 2`, group-aware B and C rows and pair norm groups in `sharding.py`; bypasses
  vLLM's `n_groups % tp` assert. Prediction: tie at 4 ranks, win at 8 (H200 nodes).
- Hymba-1.5B: two branches, sliding-window attention heads (`window_size` in `attn_decode`, KV bytes
  `min(c, window)`) and SSM heads; combine with per-branch norm then mean. Demonstrates architecture
  generality and the small-batch regime at 2 ranks.
- Falcon-40B / Command-R: branches are attention and MLP (no SSM); the split puts attention ranks against
  MLP ranks and `SwingMLP` is reused unchanged. Shows the branch split is a property of parallel blocks.
- Sequential hybrids served data-parallel (Qwen3.5-35B-A3B, Nemotron 3 Nano): each layer type is a
  `Branch` executed sequentially on every replica; `Elastic` reassigns replicated experts (tokens, not
  columns) to the replica with slack, and the byte model gains expert-weight and routing terms. This is the
  bridge to the thesis target models.
- The earlier SSM/gate split for Mamba-1 (Jamba) is the degenerate case where one branch is trivially
  cheap; swing assignment of the MLP or expert share to the gate ranks is the cheap experiment that turns
  that negative result into motivation for the swing mechanism.

---

## 18. Stage 2 outline (separate plan, after go)

Implement `--enable-mixer-parallel` in the fork on branch `mmp/stage1`: branch groups built in
`initialize_model_parallel` (reusing the role-group machinery), role-dependent Falcon-H1 layer
construction and weight loading, `o_proj` and `out_proj` with `reduce_results=False` and one
`tensor_model_parallel_all_reduce`, gated norm on the branch group, KV and Mamba cache specs per role
(cache-less ranks handled as in the SSM/gate fork), a `SwingMLP` module with fixed-shape swing GEMMs
(padding or index tensors so CUDA graphs and `torch.compile` guards stay valid), and a per-step controller
fed by CUDA-event timings. Evaluate end to end with AIPerf on Falcon-H1-7B at TP = 4 on guppy and 34B at
TP = 8 on NVwulf against vLLM TP with CUDA graphs, a DCP-equivalent built from the cost model if no
implementation exists for the model, and TII's vLLM fork.

---

## 19. Items that need explicit permission before execution

- Running `scripts/backup_old_container.sh` (writes about 14 GB under `/home/adutt/backups` and commits a
  backup image).
- Creating the `anuragdutt/vllm` fork (GitHub UI) and pushing branch `mmp/stage1`.
- Pulling `vllm/vllm-openai:v0.30.0` (about 10 GB) and building `mmp:stage1`.
- Downloading `tiiuae/Falcon-H1-7B-Instruct` (15.2 GB) into the NFS HF cache.
- Any `docker system prune`.
- Timing runs, which need the GPU 0 user to be done.

---

## 21. Execution log and deviations (2026-10-01)

Facts discovered while building the infrastructure, all now reflected in the code:

- vLLM v0.30.0 ships its core ops as `vllm/_C_stable_libtorch.abi3.so` loaded through `torch.ops`; there is no
  `vllm._C` module. FlashAttention can only be imported with a GPU present, so the Docker build verifies by file
  presence and the GPU checks live in `scripts/smoke_test.sh`. The image no longer ships its wheel, so the editable
  tree receives the compiled extensions copied from the installed package (Dockerfile step 3) and the Python
  package is installed with `VLLM_TARGET_DEVICE=empty`.
- `initialize_model_parallel` and vLLM CustomOps require an engine config context in v0.30.0; the harness enters a
  default `VllmConfig` for the process lifetime (`harness/dist.py: ensure_vllm_config`) and calls the compiled
  `silu_and_mul` op directly.
- The FlashAttention backend stores the paged cache as `(blocks, kv_heads, block_size, 2*head_dim)` and derives
  strided key and value views; `harness/kernels.py` mirrors that layout.
- vLLM's Mamba kernels treat state line 0 as the null block (`null_block_id`), so harness state indices run 1..B
  and buffers have B+1 lines.
- On this box vLLM's custom all-reduce is disabled for both the 4-rank and the 2-rank groups; all collectives go
  through pynccl (NCCL 2.30.7). The harness logs this per group.
- The NFS home squashes root, so the container runs as the host user (`docker/run.sh`), with `HOME` inside the repo
  for Triton caches and the HF cache mounted at `/hf`.
- The backup of the old container dropped the `docker commit` step (its writable layer is 289 GB); the patch,
  bundle, untracked tarball and a full copy of `/vllm` are in `/home/adutt/backups/vllm-mamba-sg-2026-10-01`.
- Model: `tiiuae/Falcon-H1-7B-Instruct` snapshot `41e72f27`; `rope_theta` is 1e11.
- Correctness: every mode (tp1, tp4, tp4_fused, tp4_streams, split22, split22 with swing S in {0.2, 1.0}, neutral
  and grid cuts) matches the fp32 reference with mean relative error 0.7 to 0.9 percent and cosine 0.99997, and
  ranks hold bitwise-identical outputs. Per-stage errors are 2 to 3e-3, so the end-to-end tolerance is 1.5e-2.
- CUDA-graph capture works for all modes with the collectives inside. At batch 4, context 2048 the captured step is
  0.31 ms versus 1.9 ms eager, so eager is launch-bound and the graph variant carries the headline numbers; the byte
  model prefers graph-captured kernel curves.

## 22. Pilot findings (2026-10-01, batch 16, graph variant, 4x A6000)

| context | tp4 | tp4_fused | tp4_streams | split22 (no swing) |
|---|---|---|---|---|
| 512 | 0.366 ms | 1.10x | 1.17x | 1.12x |
| 8192 | 0.467 ms | 1.07x | 1.18x | 1.37x |

Per-rank compute with collectives removed (diagnostic `--no-collectives`, graph variant, batch 16): tp4 ranks
0.265 ms at 512 and 0.368 ms at 8192; split22 attention ranks 0.174 / 0.278 ms, SSM ranks 0.262 / 0.262 ms. So the
branch split is SSM-bound at short context and attention-bound at long context exactly as the byte model says,
and the split wins because each rank carries one branch plus the MLP instead of both branches plus the MLP, on
top of one fewer four-rank collective.

**Swing MLP shards do not work in a synchronous layer.** With collectives removed, moving all swing columns to
the attention ranks balances compute at context 512 (0.233 ms versus 0.263 ms neutral). With collectives present
the mixer all-reduce is a barrier between the branch phase and the MLP phase, so the step is the sum of
per-phase maxima: branch max + all-reduce + MLP max + all-reduce. Columns moved across that barrier only raise the
MLP maximum. Both swing implementations confirmed it: the two-chain version cost extra small-GEMM overhead, the
contiguous-pool version is overhead-free and still never beats the neutral split (section 22 table in
`results/pilot/summary`). Hypothesis H3 as written is therefore rejected for single-micro-batch execution; it
could only pay with two interleaved micro-batches, where the attention ranks fill their branch-phase slack with
the other micro-batch's work.

**What balancing must look like instead: move work within the mixer phase.** The attention ranks are idle while
the SSM ranks finish the recurrence, so the balancing unit has to be a piece of the SSM branch that does not need
the recurrence output, and symmetrically at long context a piece of the attention branch that does not need the
SSM output:
- Short context (SSM-bound): attention ranks compute the SSM gate projection z (1536 of the 3596 `in_proj` rows
  per SSM rank, 9.4 MB of weights) and, if needed, the B/C/dt rows, and send the results point-to-point to their
  SSM partner. z is consumed only at the gated norm, late in the branch, so the transfer (batch x 1536 x 2 bytes)
  is off the critical path. This is the earlier SSM/gate split reborn as a balancing move rather than a static
  layout.
- Long context (attention-bound): SSM ranks hold the second half of their partner's KV positions and compute a
  partial attention over them; the partner merges the two partials with the log-sum-exp trick (the DCP merge,
  inside a pair). The transfer is one partial output plus one LSE per head.
The split point then is "how many gate rows" at short context and "how many KV positions" at long context, both
chosen per step from the same slack measurement, with no state migration.

## 23. The recipe: architecture-aware parallelism synthesis (2026-10-02)

`harness/synth.py` turns a model's layer structure into ranked rank-group assignments. Three rules, each one a
measured effect of the 7B sweep:

- R1, replication: an operator's rank group never exceeds its unit count (KV heads, SSM heads or groups,
  experts). Beyond it the data is replicated and every replica reads it each step.
- R2, summed branches: operators that read the same input and are added into the residual can live on disjoint
  groups with one all-reduce of partial sums; internal reductions such as the gated norm run on the branch group.
- R3, phase balance: every all-reduce is a barrier, so the step is the sum over phases of the slowest group's
  time plus collectives. Groups are sized to balance the branches inside a phase; work can be moved only
  within a phase.

The scorer uses the microbench bandwidth curves, which encode kernel occupancy versus bytes, and the measured
all-reduce costs per group size. Homogeneous tensor parallelism is the candidate where every group is the world.

Predictions before measurement (graph variant, 4 ranks, speedup over tp4; eager-curve scoring, to be re-scored
with graph-captured curves):

| model | layout | B1 c512 | B16 c512 | B16 c8192 | B64 c8192 | B64 c32768 |
|---|---|---|---|---|---|---|
| Falcon-H1-7B | attention 2 + mamba 2 | 1.35x | 1.29x | 1.44x | 1.31x | 1.12x |
| Falcon-H1-7B measured | same | 1.11x | 1.11x | 1.38x | 1.31x | 1.12x |
| Falcon-H1-3B | attention 2 + mamba 2 | 1.39x | 1.25x | 1.47x | 1.37x | 1.14x |
| Falcon-H1-34B | attention 2 + mamba 2 | 1.18x | 1.12x | 1.09x | 0.91x | 0.66x |
| Falcon-H1-34B at 8 ranks | attention 4 + mamba 4 | 1.20x | 1.23x | 1.37x | 1.29x | 1.11x |
| Falcon-40B | attention 1 + mlp 3 | 1.08x | 1.08x | 0.85x | 0.54x | 0.34x |
| Falcon-40B | attention 2 + mlp 2 | 0.73x | 0.74x | 0.81x | 1.02x | 0.68x |

Reading: the 3B should behave like the 7B; the 34B at 4 ranks should win at short context and lose at long
context because its 4 KV heads make tp4 replication-free while the split doubles each attention rank's KV read,
and at 8 ranks it should win everywhere; Falcon-40B has no winning split at 4 ranks because its MLP dominates and
is unbounded, so the recipe recommends tensor parallelism with the fused reduction and stream overlap. For
sequential hybrids with 2 KV heads (Nemotron 3 Nano, Qwen3.5) the same rules emit data-parallel attention beside
tensor-parallel SSM layers and expert parallelism, which is the configuration engines reached by hand.

## 24. Family results (2026-10-02)

**Falcon-H1-34B at 4 ranks, graph variant, min over repeats, speedup of split22 over tp4:** 1.10x to 1.15x at
512 to 2048 tokens for every batch, 1.19x at batch 4 and 8192, then a sign flip at long context: 0.83x at batch 16
and 32K, 0.68x at batch 64 and 32K. The recipe's graph-curve re-score predicted 1.00x at short context and 0.62x
at batch 64 and 32K; the sign flip and the long-context magnitude were predicted, and the short-context wins are
again about 10 percent above the model, which still under-counts the collective and occupancy savings. The
stream-overlap and fused baselines give 1.04x to 1.18x throughout.

**Falcon-H1-3B at 4 ranks:** homogeneous tensor parallelism is not definable, because 10 query heads do not divide
by 4; vLLM caps the 3B at TP 2. The split layout is feasible, two ranks with 5 query heads each and two ranks with
16 Mamba heads each. The harness now implements uneven query-head shards, 3/2/3/2, so a 4-rank TP baseline
exists for the comparison, and a TP-2 reference is measured as well.

Recipe accuracy so far: direction right in all 32 measured cells across the 7B and 34B; magnitude within about
10 percent except at short context, where the model is conservative.

**Falcon-H1-3B at 4 ranks (uneven 3/2/3/2 query-head tp4 as the baseline), graph variant, split22 over tp4:**
1.08x to 1.16x at 512 to 2048 tokens, 1.27x to 1.43x at 8192, 1.14x to 1.38x at 32K, 1.00x at batch 64 with 512
tokens; repeat-to-repeat CoV below 2 percent in every cell. Same shape as the 7B, as the recipe predicted for a
second configuration of the family.

TP-2 reference (the largest homogeneous layout vLLM supports for the 3B), batch 16, graph variant:

| model | layout | 512 | 8192 | 32768 |
|---|---|---|---|---|
| 3B | tp2 on 2 GPUs | 0.353 ms | 0.457 ms | 0.771 ms |
| 3B | split22 on 4 GPUs | 0.288 ms | 0.291 ms | 0.596 ms |
| 7B | tp2 on 2 GPUs | 0.480 ms | 0.577 ms | 0.893 ms |
| 7B | split22 on 4 GPUs | 0.330 ms | 0.340 ms | 0.664 ms |

So for the 3B the split is the only 4-GPU layout an engine could ship without uneven head shards, and it is
1.23x to 1.57x faster per layer than the TP-2 ceiling.
