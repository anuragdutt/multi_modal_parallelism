# One-collective SSM/gate split for Mamba-1: design record

Status: design fixed 2026-11-01, implementation gated on the measurement in section 7.
Scope: Mamba-1 mixers (`MambaMixer`) in vLLM v0.24.0 on the `mmp/sg` fork branch. GDN /
Mamba-2 later and optional.

## 0. Decision log

| date | decision |
|---|---|
| 2026-11-01 | Goal: one collective per Mamba mixer block, at the end, where the gate path and the SSM path are brought together for the output projection. All SSM-path work on SSM ranks, all gate-path work on gate ranks. |
| 2026-11-01 | The single collective is an **all-gather (concatenation) of channel slices** of the SSM output `y` and of `silu(z)`. Rationale in section 2: the block joins the two paths by an elementwise per-channel product, so the collective has to rebuild both full vectors; nothing is reduced. |
| 2026-11-01 | **Replication is accepted as the price of one collective**, and it is logged here on request: (a) `out_proj` runs replicated on every rank, (b) the x-side of `in_proj`, the causal conv and `x_proj` run replicated on every SSM rank. Section 2 shows both are forced. |
| 2026-11-01 | Role sizes are a flag: `--mamba-split-ssm-gate N+M` (N SSM GPUs, M gate GPUs, N+M = TP size). Default proposal 3+1 (section 4). Flag absent = stock vLLM TP, two all-reduces per block. The old design stays available as `--enable-mamba-sg` for comparison. |
| 2026-11-01 | Throughput is the criterion. Because the design trades a latency-only collective for extra weight bytes and a larger collective (section 5), a measured per-layer model on guppy comes first (section 7); the build starts only if the split has a window at some concurrency, or if the negative result is wanted on record. |
| 2026-11-01 | Benchmark: server/client. `vllm serve` in the `mmp-sg` container (TP 4), AIPerf 0.11.0 on the host, SPEED-Bench throughput_1k prompts, OSL 128 via ignore_eos, CUDA graphs on for every configuration (section 8). |

## 1. The mixer block as implemented (what the split has to preserve)

Stock `vllm_fork/vllm/model_executor/layers/mamba/mamba_mixer.py` (v0.24.0):

1. `in_proj` (`MergedColumnParallelLinear`, line 103) maps `d_model -> 2*d_inner`; the output is
   chunked into the SSM input `hidden_states_BC` and the `gate` (lines 288-289).
   Under TP each rank holds a `d_inner/tp` slice of **both** halves.
2. SSM path: causal conv (+SiLU) on the SSM input, `x_proj` (`RowParallelLinear`,
   `_ssm_transform`) producing `dt, B, C`, then `dt_proj`, then the selective scan with the
   recurrent state. `x_proj` contracts over all channels, so under TP it ends in an all-reduce
   of a `[tokens, dt_rank + 2*d_state]` tensor (first collective of the block).
3. Gate path: nothing but the `in_proj` slice; the SiLU is applied inside the scan kernels.
4. **Join**: the scan kernels take the gate as `z` and multiply it in: prefill
   `selective_scan_fn(..., gate_p, ...)` (lines 374-391), decode
   `selective_state_update(..., z=gate_d, ...)` (lines 431-446). In the Triton decode kernel the
   gate is applied as `out *= z * sigmoid(z)` (`ops/mamba_ssm.py` lines 452-453, guarded by
   `HAS_Z`, line 225); with `z=None` the kernels return the plain scan output (line 843-846).
   So the join is `y * silu(z)`, an elementwise product per channel.
5. `out_proj` (`RowParallelLinear`, line 153) maps `d_inner -> d_model` on the product
   (lines 455-459); under TP it ends in the second all-reduce of the block.

Existing SG design (`--enable-mamba-sg`, `mamba_mixer_ssm.py` / `mamba_mixer_gate.py`): cuts only
at `in_proj`. Gate ranks hold the z-half and send it to their SSM partner; SSM ranks keep conv,
`x_proj` (with a 2-rank all-reduce), `dt_proj`, the scan (gate fused via `z=`), `out_proj` (with a
2-rank all-reduce) and broadcast the output back. Four communication steps per block, so it is
not a one-collective design; it is kept as a comparator.

## 2. Why one collective means "concatenate, then replicate"

The block needs, per channel `k`, both `y_k` (SSM path) and `g_k = silu(z_k)` (gate path) on the
same rank before `out_proj` can consume `y_k * g_k`. With the roles separated, `y` lives only on
SSM ranks and `g` only on gate ranks, each channel-sharded within its role. Three facts follow:

1. **The x-side must be replicated on SSM ranks.** `dt, B, C` come from `x_proj` applied to
   `silu(conv(in_proj_x(h)))`, a contraction over all `d_inner` channels behind a nonlinearity.
   Either the channels are reduced or gathered (a collective before the scan, which the goal
   forbids) or every SSM rank computes all channels of the x-side itself. We take the second:
   each SSM rank computes the full x-half of `in_proj`, the full conv and the full `x_proj`
   locally, then `dt_proj`, the scan and the recurrent state on its channel shard only.
2. **The single collective is an all-gather.** The product cannot be formed in pieces and summed
   (a sum of products is not a product of sums), so no reduction expresses the join. What the
   join needs is the full `y` and the full `g` on the rank that will do `out_proj`. Gathering the
   channel slices in rank order, SSM ranks first, yields exactly `[y_0 .. y_{N-1} | g_0 .. g_{M-1}]
   = [y | g]`, channel-major, which is the concatenation the block's own `in_proj` chunking
   defines. This is why the collective is an all-gather and not an all-reduce: it is the
   concatenation and nothing more.
3. **`out_proj` must be replicated.** After the one collective every rank holds `[y | g]`. The next
   block (residual add, next `in_proj`) needs the block output on every rank. Any sharding of
   `out_proj` would need a second communication to assemble or reduce its output. So every rank
   computes the full product and the full `out_proj`.

Implementation notes that follow from this:

- The scan kernels are called with `z=None` on SSM ranks (gate applied after the gather); both
  kernels support it (`ops/mamba_ssm.py` lines 505, 711). No new kernels, no rebuild.
- The gather uses vLLM's uneven all-gather `GroupCoordinator.all_gatherv`
  (`parallel_state.py:672` -> `cuda_communicator.py:426` -> `pynccl.py:222`, one
  `ncclGroupStart` with a broadcast per root), along dim 0 of channel-major `[c_r, tokens]`
  slices, which is the layout the mixer already uses internally. Equal slices fall back to NCCL
  all-gather automatically.
- `d_inner` is not divisible by 3 for either model (5120, 8192), so SSM shards are uneven
  (`[1707, 1707, 1706]`); `dt_proj`, conv state, A, D and the recurrent state of each rank use
  its own shard size, and the cache spec is per rank. Every rank owns a Mamba cache layer with the
  same layer name, so batch reordering is identical on all ranks (the bug class fixed in
  `558de61` cannot recur).

Alternatives considered and set aside:

| alternative | collectives | why not |
|---|---|---|
| zero-padded all-reduce of `[y \| g]` (each rank adds its slice) | 1 | same result as the all-gather with twice the bytes (ring reduce-scatter + all-gather, and it ships zeros) |
| gate ranks send `g` slices point-to-point to SSM ranks, SSM ranks do the product and a row-sharded `out_proj`, 4-rank all-reduce with gate ranks contributing zeros | 1 all-reduce + 1 P2P step | two communication steps on the critical path; it is the old SG shape with the `x_proj` reduction removed |
| stock TP | 2 | the baseline |

## 3. Data flow per block (N SSM ranks, M gate ranks, TP = N + M)

```
every rank:    h [tokens, d_model]  (replicated, as in TP)
SSM rank i:    x = silu(conv(h @ W_in_x))                 full d_inner, replicated across SSM ranks
               dt_raw, B, C = x @ W_xproj                 full contraction, no collective
               dt_i = dt_raw @ W_dt[:, shard_i]           channel shard i
               y_i = scan(x[:, shard_i], dt_i, A_i, B, C, D_i, state_i)   z = None
gate rank j:   g_j = silu(h @ W_in_z[:, shard_j])         channel shard j
all ranks:     [y | g] = all_gatherv([y_0..y_{N-1}, g_0..g_{M-1}])        the one collective
               out = (y * g) @ W_out                      replicated
```

## 4. Flag and roles

- `--mamba-split-ssm-gate N+M`: parsed in `vllm/config/parallel.py`; validates `N + M ==
  tensor_parallel_size`, `N, M >= 1`. Ranks `0..N-1` are SSM ranks, `N..TP-1` gate ranks.
- Absent: stock `MambaMixer` (two all-reduces). `--enable-mamba-sg`: old design. The two flags
  are mutually exclusive.
- Default proposal **3+1** at TP 4, for throughput: the SSM side's per-token state traffic grows
  with batch while the gate side's work does not, so the SSM side needs the GPUs. At 3+1 both
  roles stream about the same weight bytes per layer (table below). 2+2 leaves gate ranks idle
  for roughly a third of each layer; 1+3 puts four times TP's state traffic on one rank.
- Weight loading: the SSM ranks load the full x-half of `in_proj`, full conv, full `x_proj`,
  and shard `dt_proj` (columns), conv/A/D (channels) by their uneven shard; gate ranks load their
  shard of the z-half of `in_proj`; everyone loads the full `out_proj`.

## 5. What the design costs (structural numbers)

Per layer, bf16 weights, worst rank. Collective bytes are per token per rank: ring all-reduce
traffic `2(n-1)/n * size` for TP, bytes received for the all-gather. State traffic is the fp32
recurrent state read + written by the decode scan per token.

Mamba-2.8B (`d_model` 2560, `d_inner` 5120, `dt_rank` 160, `d_state` 16, 64 layers):

| layout | weights per rank | state traffic per token | collective bytes per token | collective steps |
|---|---|---|---|---|
| TP-4 (stock) | 20.6 MB | 164 KB | 7.7 KB + 0.6 KB | 2 |
| split 3+1 (all-gather) | 54.9 MB (gate 52.4) | 218 KB | 17 KB | 1 |
| split 2+2 (all-gather) | 55.2 MB (gate 39.3) | 328 KB | 15 KB | 1 |
| split 1+3 (all-gather) | 56.0 MB (gate 34.9) | 655 KB | 17 KB | 1 |
| split, zero-padded all-reduce | as above | as above | 31 KB | 1 |
| old SG 2+2 | 28 MB (gate 13.1) | 328 KB | 16 KB | 4 |

Falcon-Mamba-7B (`d_model` 4096, `d_inner` 8192, `dt_rank` 256, `d_state` 16, 64 layers):

| layout | weights per rank | state traffic per token | collective bytes per token | collective steps |
|---|---|---|---|---|
| TP-4 (stock) | 52.5 MB | 262 KB | 12.3 KB + 0.9 KB | 2 |
| split 3+1 (all-gather) | 140.3 MB (gate 134.2) | 350 KB | 27 KB | 1 |
| split 2+2 (all-gather) | 141.0 MB (gate 100.7) | 524 KB | 25 KB | 1 |
| split 1+3 (all-gather) | 143.1 MB (gate 89.5) | 1049 KB | 27 KB | 1 |
| split, zero-padded all-reduce | as above | as above | 49 KB | 1 |

Reading: the split removes the tiny, latency-only `x_proj` collective and keeps one collective
that carries about twice the bytes of TP's `out_proj` reduction, while the SSM ranks stream 2.7x
the weights of a TP-4 rank (and do 2.7x the GEMM work once compute-bound, which is the prefill
regime). Break-even per layer at decode: the fixed latency of one collective must exceed the
time to stream the extra weights (34 MB on Mamba-2.8B, about 50 us on an A6000; 88 MB on
Falcon-Mamba-7B) plus the extra collective bytes. On guppy (four RTX A6000, no NVLink, every pair
routed through the CPU, `nvidia-smi topo -m` all `SYS`) small-message latency is plausibly in
that range at low concurrency; at high concurrency the collective is bandwidth-bound and the split
moves more bytes. Expectation before measuring: a coin flip at concurrency 8 and below, a loss
at 32 and above, worse on the larger model. Prefill (ISL about 1k in SPEED-Bench) is
compute-bound GEMM work plus bandwidth-bound collectives, both of which the split increases.

## 6. Correctness criteria for the implementation

- Split output vs stock TP output on the smoke-test prompts (`scripts/sg/smoke_sg.py`, new
  `--mode split`): first-token agreement 32/32, max |delta first logprob| within the
  noise-bounded criterion already used for TP repeats, two repeats at concurrency 8 and 32 with a
  prefill admitted ahead of decodes (the failure pattern of the old bug).
- Every rank reports the same number of KV-cache groups and the same reorder threshold.
- Output-length check in the benchmark (OSL 128 with ignore_eos: every response has 128
  completion tokens).

## 7. Measurement gate (step 0)

`scripts/sg/microbench_layer.py` (torchrun, 4 GPUs, inside `mmp-sg`) times, in CUDA graphs and
through vLLM's own group coordinators (pynccl), every primitive of every layout at decode batch
1, 8, 32, 128, 256 and a 1024-token prefill chunk, for both models: GEMMs at sharded and
replicated widths, conv update, `selective_state_update` per shard (with and without `z`), silu,
product, prefill conv and scan, the two TP all-reduces, the uneven all-gather for 3+1 / 2+2 /
1+3, the zero-padded all-reduce, and the old SG pair collectives. It composes per-layer times
(slowest role plus collectives) and a request-mix proxy (ISL 1024, OSL 128). Results go to
`results/sg/microbench_layer.json` and section 9 below.

Go/no-go: build the split if its per-layer time is below TP-4's at any concurrency in the
decode table, or if the request-mix ratio is above 1 at any concurrency; otherwise discuss before
building.

## 8. Benchmark protocol (after the build)

- Server: `vllm serve <model> --tensor-parallel-size 4 [--mamba-split-ssm-gate 3+1 |
  --enable-mamba-sg]` inside `mmp-sg`, host port 8001, CUDA graphs on, `--max-num-seqs` at least
  the largest concurrency, one configuration at a time, GPUs otherwise idle.
- Client: AIPerf 0.11.0 on the guppy host (`~/anaconda3/envs/mamba-ssm/bin/aiperf`), SPEED-Bench
  `throughput_1k.jsonl` as the custom dataset, `--osl 128`, `ignore_eos:true`, `temperature:0`,
  completions endpoint, concurrency 1 / 8 / 32 / 128, 512 requests per cell, two trials.
- Cells: TP-4 stock; split 3+1, 2+2, 1+3; old SG 2+2. Models: Mamba-2.8B, then Falcon-Mamba-7B.
- Metrics: output tokens/s (primary), TTFT, ITL, output-length check.

## 9. Results

Pending: filled from `results/sg/microbench_layer.json` once the measurement has run.

## 10. Provenance

Code facts above were read from the fork at `72a5cad` (branch `mmp/sg`, v0.24.0 base) on
2026-11-01; line numbers refer to that commit. Topology from `nvidia-smi topo -m` on guppy,
2026-11-01. Byte counts computed from the model shapes in the HF configs.
