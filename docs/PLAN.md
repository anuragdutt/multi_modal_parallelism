# PLAN.md — Architecture-derived parallelism synthesis for SSM, GDN and hybrid LLMs: validation plan

Written 2026-10-26 after the stage-1 Falcon-H1 experiments (`docs/docs/blog/README.md`, `docs/stage1_elastic_mixer_parallelism_plan.md`
sections 21-24) and the SSM/gate-split deep dive (`NOTES.md`). This is the plan for the next stage: turn the recipe into a
general synthesizer that works on any new model, validate it across families, and use it to find and prove the layouts that
beat tensor parallelism. It is a plan, not a result: every number marked **predicted** is pre-registered here before the
run that tests it.

## 0. Purpose and the claim under test

**Claim.** The parallel layout of an LLM decoder for multi-GPU decode can be derived from its architecture alone: describe
the model as a graph of operators annotated with byte classes, unit axes, reduction signatures and state locality (the
*ontology graph*), cut it into phases at its reductions, enumerate ways to assign operator sets to rank groups and
sharding axes (*executables*), cost each layout per operating regime with a byte model calibrated by micro-benchmarks,
and the result predicts (a) which layout wins, (b) by how much, and (c) where the regime boundaries are, for models the
synthesizer has never seen. A small set of rules (R1-R8, section 2.3) explains every win and every loss.

**What would falsify it.** A model family where the synthesizer's top candidate loses to tensor parallelism while a
layout it ranked lower wins; a measured regime boundary off by more than 2x from the predicted one; or magnitudes wrong by
more than 15% in the graph-captured step time after calibration.

**Why it matters for the thesis.** Stage 1 showed one synthesized layout (the Falcon-H1 branch split) beating vLLM's
tensor parallelism by up to 1.38x per layer, predicted in sign on all 32 measured cells of two models, and correctly
predicted a loss on Falcon-40B. The SSM/gate split line (nine months of work) produced at most +7% and a correctness bug;
the synthesizer's rules explain why in two lines (R1, R5). The chapter is the synthesizer, its rules and its validation
across six families; the elastic attention layout for GDN hybrids (section 6.3) is the demonstration that it finds
something new.

**Scope decisions already taken (2026-10-26):** broader scope than the gate split; Qwen3.5 first, Nemotron second; the
SSM/gate split is closed out with a bug fix and one clean re-measurement; the synthesizer is the claim and the elastic
attention layout the demonstration. Open decisions with the defaults this plan assumes: Qwen3.5-35B-A3B is the GDN
checkpoint (not Qwen3-Next-80B-A3B); the per-step policy with state migration is in scope as a simulation plus one
measured migration, not as an engine feature, until the gate in section 10 passes; the 8-rank H200 result is the headline
only if NVwulf access exists by week 8, otherwise the 4-rank result is the headline and 8 ranks is future work.

## 1. Where we start from

| asset | state | used for |
|---|---|---|
| `harness/` (4-rank layer harness on vLLM v0.30.0 kernels, container `mmp-dev`, image `mmp:stage1`) | works for Falcon-H1, Falcon-40B (parallel residual), Nemotron (sequential: Mamba-2, attention tp/dp, MoE tp/ep); 8-rank code paths untested | the executable runner |
| `harness/synth.py` (recipe v1: templates `falcon_h1`, `parallel_residual`, `sequential_hybrid`; rules R1-R3; byte-model scorer) | predictions right in sign on 32/32 cells (7B, 34B), Falcon-40B negative confirmed; 23% median magnitude error, ~10% conservative at short context | regression baseline for synth v2 |
| `harness/model.py`, `harness/microbench.py` | bandwidth curves (gemv, attention decode, SSM update, conv, all-reduce by group) for 4x A6000 PCIe, eager and graph | cost model calibration |
| results | `results/full` (7B), `results/h1_3b`, `results/h1_34b`, `results/falcon40b`, `results/tp2_reference.csv`; Nemotron sweep raw CSV on guppy (60 configs, composite not computed) | E1 regression set, E3 |
| SG split (vLLM `--enable-mamba-sg`, container `vllm-mamba-sg`) | mechanism understood; reorder-mismatch bug root-caused; valid numbers only for Jamba (+3 to +7%) | E2 close-out (R5 datum) |
| Models on guppy | Falcon-H1 3B/7B/34B (layer-0 shards), Falcon-40B (layer 0), Nemotron 3 Nano (full), Mamba-2.8B, Falcon-Mamba-7B, Jamba-v0.1 (NFS cache), Mamba-Codestral-7B (`~/masarani/mamba2-tp`, 28 GB) | model set |
| Hardware | guppy: 4x RTX A6000 48 GB, PCIe Gen4 only, shared; NVwulf: 8x H200 nodes, no allocation yet | 4-rank now, 8-rank later |
| Literature map | `~/.claude` memory note `hybrid-llm-serving-literature-2026` and `doc/...plan.md` section 3.4: Falcon-H1 mixer parallelism (static branch split, reverses at large batch), vLLM decode context parallelism (position split for attention-only models), OpWeave (operator disaggregation, pipelined), DP attention + EP in vLLM/SGLang (static process topology) | novelty boundary: static layouts exist; architecture-derived synthesis and per-step elasticity inside one TP group do not |

## 2. Definitions

### 2.1 Ontology graph
A model is `pattern` (a string over layer types, e.g. Qwen3.5 `(GGGA)x10` with MoE in every layer, Nemotron
`MEMEM*EMEMEM*...`, Falcon-H1 `H x 44` where H is one parallel-hybrid layer) plus one DAG per layer type. Nodes are
operators; edges carry activation shapes and a composition tag: `seq` (sequential), `sum` (parallel branches added into
the residual, e.g. Falcon-H1 attention + Mamba, Falcon-40B attention + MLP), `route` (token-to-expert).

Each operator node carries exactly these attributes (schema in Appendix A):

| attribute | type | meaning |
|---|---|---|
| `cls` | one of `gemv`, `attn`, `ssu` (recurrent state update), `conv`, `norm`, `moe`, `elem` | selects the bandwidth curve and the launch cost |
| `W` | bytes | weight bytes read per step (0 for state-only ops); for `moe`, per expert |
| `S` | bytes per sequence | state read + write per step (SSM/delta-rule state, conv state) |
| `K` | bytes per token | cache read per step per token of context (KV) |
| `units` | map class -> int | replication-free shard count per byte class: `W: nq` for q/o projections, `W: nkv` for k/v projections, `K: nkv`, `S: n_v_heads` (GDN) or `n_mamba_heads`, `W: n_experts` for MoE, `W: inf` for MLP columns, `W: d_inner` for Mamba-1 channels |
| `divisors` | list | allowed group sizes from head divisibility (e.g. Falcon-H1-3B q heads = 10 excludes 4) |
| `reduce` | list of (size, group-kind) | reductions a unit split induces: `out_proj` -> `[T, H]` on the sharding group; Mamba-1 `x_proj` -> `[T, dt_rank + 2N]`; gated RMSNorm -> `[T, 1]` only if `n_groups % g != 0`; GDN/Nemotron per-head norm -> none |
| `state` | name or null | the per-sequence state the op updates (`kv`, `conv`, `ssm`, `delta`) and therefore the thing that must migrate if the sequence changes shard |
| `phase_role` | `prefill_compute` / `decode_bandwidth` / `both` | whether the op is compute-bound in prefill |
| `touched(B)` | function | for `moe`: expected distinct experts hit per step, `E * (1 - (1 - k/E)^B)` capped at E |

### 2.2 Executables, layouts, policies, regimes
- **Phase**: a maximal set of operators between two reductions in the per-step DAG (so a Falcon-H1 layer has two phases,
  mixer and MLP; a Qwen3.5 GDN layer has two, mixer and MoE; Mamba-1 has three because of `x_proj`).
- **Executable**: `(op set, rank group, axis, replication, boundary collectives)` with `axis` in {`unit`, `seq` (ownership
  of whole sequences), `pos` (position split of each sequence's cache), `expert`}. Replication is the factor by which the
  group exceeds the unit count on each byte class.
- **Layout**: executables covering every operator of the step; valid iff (i) every rank group's collectives are issued
  in the same order on all its members (no deadlock), (ii) the hidden state is replicated on every rank at each phase
  boundary that feeds a unit-sharded phase (owned rows are all-gathered), (iii) state locality holds (a sequence's `kv`,
  `ssm`, `delta` state lives on the ranks of the executable that updates it).
- **Regime**: `(B, context distribution over the batch, prefill token share, hardware)`; we sample regimes on a grid
  (section 7) and on traces (E7).
- **Policy**: `regime -> layout` plus the switch cost between layouts (state migration bytes, R6); reported as three
  numbers per model: best static layout, elastic policy with hysteresis, oracle (switch free).

### 2.3 Rules (the hypotheses the experiments test)
Formulas use: `g` group size, `u` unit count, `r = g / min(g, u)` replication factor, `B` batch, `c` context, `T`
tokens in the step, `W`, `S`, `K` as above, `tau_g` all-reduce latency for a group of `g` at decode sizes (measured: ~40 us
for g=4 and ~25 us for g=2 on PCIe A6000 in graph mode, `harness/synth.py:AR_US`).

- **R1 replication cap.** Per-rank bytes of class X for an op in a group of size g: `X_total / min(g, u_X)`. Beyond
  `u_X` the per-rank traffic no longer shrinks. Evidence: Falcon-H1 7B (2 KV heads at g=4: split wins) vs 34B (4 KV
  heads: split loses at long context). Test: every model in E0's "replication factor" column must match measured
  attention-layer time ratios within 15%.
- **R2 summed branches.** Operators joined by `sum` edges can run on disjoint groups with one reduction of the scaled
  partial sums; internal reductions (gated norm) stay on the branch group. Evidence: Falcon-H1 split22. No new test
  needed; E1 regression keeps it honest.
- **R3 phase balance.** `step = sum over phases of max over groups (time) + collectives`; work can be moved only inside a
  phase. Evidence: swing MLP shards never beat the plain split. Test: E1 regression plus the Qwen3.5 mixed-layout cells in
  E4 (owned attention rows must be all-gathered before the next TP phase; the all-gather must appear in the measured
  collective share).
- **R4 ownership vs sharding (new).** For an op whose per-step bytes are dominated by per-sequence classes (K or S),
  compare per-rank bytes under unit sharding, `W/min(g,u_W) + (S*B + K*B*c)/min(g,u_K)`, with ownership,
  `W + (S*B + K*B*c)/g` (full weights, 1/g of the sequences), and with position split for K,
  `W/min(g,u_W)... + K*B*c/g` plus a merge collective per layer. Ownership wins when `(r_K - 1)/r_K * K*B*c/... ` exceeds the
  extra weight read `(1 - 1/min(g,u_W)) * W` and the sequences balance (at least g sequences of comparable cost); position
  split wins when they do not balance (few long sequences). Predicted for Qwen3.5 (section 8): break-even at ~80K tokens
  of total context in the batch at g=4, ~64K at g=8; up to 1.35x at g=4 and ~2x at g=8 at batch 64 and 32K context; never
  for GDN or Mamba-2 ops (S is sharded for free, weights are not). Test: E4, E5, E6.
- **R5 collective count.** At decode message sizes (`B*H*2` bytes, 256 KB at B=64 for H=2048) every collective costs
  about `tau_g` regardless of bytes, so the number of reductions per step is a first-order term when per-layer compute is
  small; layouts that add a reduction to save a GEMM lose (the gate split), layouts that merge reductions win
  (`tp4_fused` 1.03-1.13x). Evidence: SG split, paper-era all-reduce timings, stage-1 fused baseline. Test: E2 (clean
  re-measurement of the gate split must show no gain beyond noise once outputs are correct) and the Jamba/Falcon-Mamba
  "reductions per step" column in E0 vs measured collective share.
- **R6 state locality (new).** A sequence can change executable only by migrating its state: `S` bytes per layer for
  recurrent state (Qwen3.5: 30 MiB per sequence over 30 GDN layers; Nemotron: 46 MiB fp32), plus KV consolidation when
  moving from unit-sharded to owned attention (one missing KV head per layer: 10 KiB per token on Qwen3.5). A policy
  switches only when `savings per step * expected steps until the next switch > migration bytes / link bandwidth`.
  Test: E7 (measure one consolidation and one state migration on the harness; simulate policies on traces).
- **R7 phase heterogeneity.** Prefill ops are compute-bound and must be unit-sharded; decode follows R4; a mixed step
  may run different axes for its prefill and decode sub-batches when every rank issues the same collective sequence.
  Test: E4 mixed cells (prefill rows TP, decode rows owned).
- **R8 sparse ops.** Expert-partitioned and unit-sharded experts move the same bytes per rank (touched experts / g);
  choose by kernel occupancy (bigger GEMMs under EP) and collective type (all-to-all vs all-reduce); hot experts are
  replicated into measured slack. Test: E3 (Nemotron moe tp4 vs ep4 sweep already run), E5 (Qwen3.5 MoE).

## 3. The synthesizer v2 (what gets built)

### 3.1 Pipeline
```
HF config.json ──► template library ──► ontology graph (JSON) ──► phase cut ──► candidate executables per phase
      │                                                                              │
      │                                                   microbench curves + collective latencies (per hardware)
      │                                                                              │
      └──► regime grid / traces ──────────────────────────────────────────► cost per (layout, regime)
                                                                                     │
                                                      policy synthesis (best static, elastic w/ hysteresis, oracle)
                                                                                     │
                                                   executable specs (JSON) ──► harness runner ──► measurements ──► score
```

### 3.2 Modules (new package `harness/synth2/`, the v1 `harness/synth.py` stays as the regression oracle)
| module | responsibility | inputs / outputs |
|---|---|---|
| `ontology.py` | the schema (dataclasses `Op`, `Edge`, `LayerGraph`, `ModelGraph`), JSON (de)serialisation, validation (unit counts divide, byte classes non-negative, every reduction has a group kind) | config dict -> `ModelGraph`; `model.json` |
| `templates/{mamba1,mamba2,gdn,attention,gated_attention,moe,mlp}.py` | one function per block family turning HF config fields into operator nodes with W/S/K/units/reduce/state (formulas in Appendix B) | config -> `LayerGraph` |
| `templates/compose.py` | composition: sequential pattern strings (`hybrid_override_pattern`, `full_attention_interval`, Jamba's `attn_layer_period/offset`, MoE periodicity), parallel-sum blocks (Falcon-H1, parallel-residual Transformers) | pattern + layer graphs -> `ModelGraph` |
| `extract.py` (later, optional) | build the graph from vLLM's instantiated model on the meta device by walking modules (`QKVParallelLinear`, `MambaMixer2`, `Qwen3NextGatedDeltaNet`, `FusedMoE`, ...) | vLLM model -> `ModelGraph`; cross-checked against the template output |
| `phases.py` | cut each layer DAG at reductions; mark `sum` groups; compute per-phase op sets | `ModelGraph` -> list of `Phase` |
| `enumerate.py` | per phase, enumerate executables: unit-sharded over `g in divisors, g <= world`; ownership (`seq`) for ops with `K` or `S`; position split (`pos`) for `attn`; `expert` for `moe`; disjoint groups for `sum` phases (R2); reject invalid layouts (section 2.2 validity) | phases -> candidate layouts (deduplicated, canonical names such as `attn:seq4|gdn:tp4|moe:tp4`) |
| `cost.py` | byte model + bandwidth curves + launches + collectives (count x `tau_g` + bytes/link BW), `step = sum of phase maxima`; regime-aware (`B`, context histogram, prefill share); replication per R1; all-gather at owned boundaries per R3 | (layout, regime, hw) -> predicted ms with a per-op breakdown |
| `policy.py` | best static layout; elastic policy with hysteresis given migration costs (R6); oracle; expected gain on a trace | cost tables -> policy table + three summary numbers |
| `emit.py` | executable specs the harness consumes: per rank op list with shard slices, group names, collective order, state placement, KV layout | layout -> `exec_<name>.json` |
| `report.py` | the E0 ontology comparison tables and figures (section 4.3), candidate rankings, prediction tables with pre-registration hashes | all of the above -> markdown + PNG |

CLI: `python -m harness.synth2 --model configs/<m>.yaml --world 4 --hw guppy_a6000 --regimes configs/regimes_default.yaml
--out results/synth2/<m>/` producing `model.json`, `phases.json`, `candidates.csv`, `cost_<regime>.csv`, `policy.md`,
`exec_*.json`, `report.md`. Every prediction file is committed before the corresponding measurement runs; the commit hash
is the pre-registration.

### 3.3 Cost model, precisely (Appendix C has the equations)
- Per op per rank: `t = sum over classes of bytes_class / BW_cls(bytes_class) + launches * L` with `BW_cls` the
  log-interpolated achieved-bandwidth curve from `harness/microbench.py` (graph-captured), `L` = 1 us (graph) / 4 us (eager).
- Attention decode: bytes = `K * B_r * c_mean` where `B_r` is the sequences this rank reads (all for unit sharding, its
  owned share for `seq`, all for `pos` but with `c/g` positions each) plus q/k/v/o weights with their own units.
- Recurrent update: `S * B_r / min(g, u_S)` for unit sharding, `S * B_r` for ownership.
- MoE: `touched(B_r) * W_expert / g` under `tp` (every rank reads a 1/g slice of every touched expert), `touched_r *
  W_expert` under `ep` where `touched_r` is the expected experts owned by this rank among those hit (`E/g * (1 - (1 -
  k/E)^B)`), plus the shared expert.
- Collectives: all-reduce `tau_g(bytes)` from the microbench curve by group size; all-gather of owned rows
  `tau_g(B_owned * H * 2)`; position-split merge `tau_g(B * n_heads * (d_head + 1) * 4)`; all-to-all for `ep` modeled as
  two all-reduce-equivalents until measured (E5 measures it).
- Step time: `sum over phases of max over groups of (sum of op times on that group) + collectives`, prefill and decode
  sub-batches summed inside a phase when they share ranks and taken as max when they do not (R7).
- Calibration: the curves are re-measured per hardware (guppy now; NVwulf H200 later) by `harness/microbench.py` extended
  with the GDN recurrence kernel, the fused MoE kernel and all-to-all.

### 3.4 What "generalization" means operationally
A new model enters with only `config.json` plus the name of its block families. The synthesizer must (1) build the graph
without model-specific code beyond the template library, (2) output a ranked candidate list and a policy, (3) and those
predictions, committed before measurement, must satisfy the scoring targets in section 9 on the held-out models (Jamba,
Mamba-Codestral, Falcon-Mamba are held out: their templates are written from the config only, their predictions are
frozen, then measured).

## 4. Model set and the ontology comparison (experiment E0)

### 4.1 The models and why each is in the set
| model | family / composition | KV heads (q heads) | recurrent units | experts | per-token K (all layers) | per-seq S (all layers, bf16) | why it is here | weights status on guppy |
|---|---|---|---|---|---|---|---|---|
| Falcon-H1-7B | parallel hybrid (attention + Mamba-2 summed), 44 layers | 2 (12) | 24 Mamba heads, 1 group | - | 44 KiB | 66 MiB | stage-1 primary; R1/R2/R3 evidence | layer-0 shard |
| Falcon-H1-3B | same, 10 q heads (no even tp4) | 2 (10) | 32 heads | - | ~40 KiB | ~ | divisibility edge case | layer-0 shard |
| Falcon-H1-34B | same, 72 layers, 2 groups | 4 (20) | 32 heads, 2 groups | - | 144 KiB | 144 MiB | R1 sign flip; 8-rank target | layer-0 shard |
| Falcon-40B | parallel-residual Transformer (attention + MLP summed), 60 layers | 8 (128) | - | - | 240 KiB | - | negative control for R2 (no winning split) | layer-0 shard |
| Nemotron 3 Nano 30B-A3B | sequential Mamba-2 / MoE / attention, 52 layers | 2 (32), 6 attention layers | 64 heads x 64, 8 groups, 23 layers | 128 + 1 shared, top-6, 23 layers | 6 KiB | 23 MiB (46 MiB fp32) | R8 (expert traffic dominates), R4 negative (little KV) | full checkpoint |
| Qwen3.5-35B-A3B | sequential GDN / gated attention, MoE every layer, 40 layers, `(GGGA)x10` | 2 (16) x 256, 10 attention layers | 32 v-heads x 128, 16 k-heads, 30 GDN layers | 256 + 1 shared, top-8, 40 layers | 20 KiB | 30 MiB | R4 positive case (KV rivals experts at long context), R7, elastic attention demonstration | **to download** (layer 0 + config; ~1.8 GB) |
| Jamba-v0.1 | sequential Mamba-1 / attention (1 in 8) / MoE (every other layer, 16 experts top-2), 32 layers, 52B total | 8 (32) | Mamba-1 channels (d_inner 8192), d_state 16 | 16, top-2 | 64 KiB (4 layers) | ~16 MiB | held-out: Mamba-1 `x_proj` reductions (R5), replication-free attention (R1 says stay TP), EP for 16 experts | NFS cache (metadata only on guppy; needs download, 104 GB bf16; layer-0 shards suffice: ~3 GB) |
| Falcon-Mamba-7B | pure Mamba-1, 64 layers, d_inner 8192, d_state 16 | - | 8192 channels | - | - | ~40 MiB | held-out pure Mamba-1; SG re-measurement model | NFS cache (14 GB) |
| Mamba-Codestral-7B | pure Mamba-2, 64 layers, 128 heads x 64, 8 groups, d_state 128 | - | 128 heads, 8 groups | - | - | 128 MiB | held-out: state-bound at high batch (S read+write 2 x 128 MiB x B), R4 negative (no K), group alignment for the gated norm at g=8 vs 16 | `~/masarani/mamba2-tp` (state dict), config to confirm |
| Mamba-2.8B | pure Mamba-1, 64 layers, d_inner 5120, d_state 16 | - | 5120 channels | - | - | ~20 MiB | SG re-measurement model; smallest | `/tmp/vllm-hf-cache` in the SG container, host cache |

Dims for Falcon-H1, Falcon-40B, Nemotron and Qwen3.5 come from the configs already in `configs/`; Jamba, Falcon-Mamba and
Mamba-Codestral dims above are from memory of their configs and the `mamba2-tp` shapes and **must be confirmed from
`config.json` before their templates are frozen** (task in week 1). Qwen3.5 attention weights per layer ~54.5 MB, GDN
per layer ~67 MB, expert ~6.3 MB, shared ~6.3 MB; Nemotron expert ~20 MB, shared ~40 MB.

### 4.2 Six families, one synthesizer
`parallel-hybrid` (Falcon-H1), `parallel-residual` (Falcon-40B), `seq-mamba2-moe` (Nemotron), `seq-gdn-moe` (Qwen3.5),
`seq-mamba1-moe` (Jamba), `pure-mamba1` (Falcon-Mamba, Mamba-2.8B), `pure-mamba2` (Mamba-Codestral). The synthesizer
sees families only through the template library; the family label is an output of the phase cut (section 4.3 signature),
not an input.

### 4.3 E0: build, compare and cluster the ontology graphs
Produce for every model and for `g in {2, 4, 8}` the following, as `results/synth2/e0/ontology_comparison.md` plus
figures:
1. **Byte composition per regime** at the reference regimes `(B, c) in {(1, 2K), (16, 8K), (64, 32K)}`: stacked bars of
   per-rank bytes by class (weights by layer type, expert weights touched, K, S) under plain TP. This shows at a glance
   which class dominates where (Nemotron: experts everywhere; Qwen3.5: experts at short context, KV at long;
   Mamba-Codestral: S at high batch; Falcon-H1: K at long context).
2. **Replication table**: for each per-sequence class and op, `u`, `r(g)`, and the replicated bytes per step at the
   reference regimes. R4 candidates are exactly the rows where replicated K bytes exceed the extra replicated weights.
3. **Reduction table**: reductions per layer per layout under unit sharding (Mamba-1: 2 per Mamba layer + 1 per MoE;
   Mamba-2/GDN: 1 + 1; Falcon-H1: 4 under vLLM, 2 + pair under the split), their message sizes, and the predicted
   collective share of the step (R5). Cross-check against measured collective shares where available (stage 1).
4. **Rule-firing matrix**: for each model x regime x `g`, which of R1-R8 change the ranking (boolean), computed by
   ablating the rule in the cost model (e.g. R1 off = pretend replication is free) and checking whether the top candidate
   changes. The matrix is the compact statement of "what the architecture implies".
5. **Structural signature and clustering**: the signature is the per-layer-type tuple (phase count, reduction kinds,
   classes with units < 8, `sum` groups, sparse ops); models with equal signatures must receive the same layout family
   from the synthesizer. Expected clusters: {Falcon-H1 x3}, {Falcon-40B}, {Nemotron, Qwen3.5} (sequential with K-heavy
   attention and sparse MoE; they differ only in magnitudes), {Jamba}, {Falcon-Mamba, Mamba-2.8B}, {Mamba-Codestral}.
6. **Predicted best layout per model x regime x `g`** with the predicted gain over TP and the predicted regime boundary
   (context at which the best layout changes). This table is frozen (committed) before any new measurement.
Time: 2 days once templates exist; no GPU.

## 5. Hardware, software and kernels

### 5.1 Hardware
| | guppy (now) | NVwulf H200 node (pending allocation) |
|---|---|---|
| GPUs | 4x RTX A6000 48 GB, ~768 GB/s HBM each | 8x H200 141 GB, ~4.8 TB/s nominal (~3.3 TB/s achieved assumed) |
| interconnect | PCIe Gen4, no NVLink; 4-rank all-reduce ~40 us, 2-rank ~25 us at decode sizes (graph) | NVLink/NVSwitch; all-reduce ~10 us assumed until measured |
| sharing | shared box: min over repeats, CoV flags, power/util logged per run (stage-1 protocol) | batch job, exclusive node |
| role in the plan | all 4-rank experiments (E1-E7); `world in {2, 4}` | E8: 8-rank cells of every model; TP8 KV replication 4x for 2-KV-head models |
The byte model is re-calibrated per hardware with `harness/microbench.py` (extended in section 6.1); every prediction
table states its hardware.

### 5.2 Software
- Harness container `mmp-dev` (image `mmp:stage1`, vLLM v0.30.0 editable clone at `/opt/vllm`, torch per the image), run
  as the host user with the repo mounted at `/workspace` and the HF cache at `/hf` (`docker/run.sh`).
- vLLM kernels the harness borrows (all already used by stage 1 except the three marked new):
  | family | decode kernel | prefill kernel | source in vLLM v0.30.0 |
  |---|---|---|---|
  | attention (GQA, gated) | paged FlashAttention-2 `flash_attn_varlen_func` (returns LSE on request, needed for position split) | same, varlen | `vllm.vllm_flash_attn` |
  | Mamba-2 | `causal_conv1d_update`, `selective_state_update` | `causal_conv1d_fn`, `mamba_chunk_scan_combined_varlen` | `vllm/model_executor/layers/mamba/ops/` |
  | Mamba-1 | `causal_conv1d_update`, `selective_state_update` (Mamba-1 variant with z) | `causal_conv1d_fn`, `selective_scan_fn` | `vllm/model_executor/layers/mamba/ops/mamba_ssm.py` **new to the harness** |
  | GDN (Qwen3-Next/Qwen3.5) | `fused_recurrent_gated_delta_rule` | `chunk_gated_delta_rule` | `vllm/model_executor/layers/fla/ops/` **new to the harness; verify presence in v0.30.0 in week 1** |
  | gated RMSNorm | `rms_norm_gated` (Mamba-2), per-head gated norm (GDN: `RMSNormGated` with `group_size = head_dim`) | same | `vllm/model_executor/layers/mamba/ops/layernorm_gated.py` |
  | MoE | `fused_moe` (Triton, `fused_experts` + `fused_topk`/sigmoid routing on host) | same | `vllm/model_executor/layers/fused_moe/` **new to the harness** (stage 1 used a per-expert torch loop with host routing) |
  | collectives | vLLM groups via `initialize_model_parallel` + `init_model_parallel_group` for sub-groups; pynccl all-reduce/all-gather; all-to-all via `torch.distributed.all_to_all_single` on the group's process group | - | `harness/dist.py` |
- If a kernel is missing in v0.30.0 (most likely risk: the FLA ops layout), the fallback is to vendor the kernel from the
  `flash-linear-attention` package at the version vLLM pins, inside the container only.

### 5.3 Checkpoints
Layer-0-only downloads (`scripts/download_layer0.sh`) for Qwen3.5-35B-A3B, Jamba-v0.1, Mamba-Codestral-7B; Nemotron and
Falcon-H1 shards exist. Correctness references are fp32 pure-torch layers built from the same shard (stage-1 pattern).

## 6. Harness extensions

### 6.1 Infrastructure
1. **Executable runner** (`harness/exec_runner.py`): consumes `exec_<name>.json` from the synthesizer and builds the
   per-rank layer from it (shard slices, groups, collective order, state placement). Replaces the hand-written
   `plan_for(mode, ...)` switch for new models; stage-1 modes are regenerated through it and must reproduce stage-1
   numbers (E1).
2. **Microbench additions**: GDN recurrence (bytes = state read+write + q/k/v), `fused_moe` (bytes = touched expert
   weights), all-to-all by group size, all-gather by group size, position-split merge kernel; graph-captured curves.
3. **Composite assembler** (`analysis/seq_composite.py` generalised): per-step time = sum over layer types of
   `count x per-layer time` plus embedding/lm_head terms measured once; reports max over ranks.
4. **Regime generator**: batches with a context histogram (uniform, bimodal short/long, trace-sampled) and a prefill
   share; the harness fills KV/state deterministically as in stage 1 (`harness/state.py`).
5. **Correctness gate**: fp32 reference per layer type; `mean_rel <= 1.5e-2`, `cos >= 0.9999`, cross-rank bitwise
   identity for replicated outputs; every mode of every model passes before its sweep (the gate caught three real bugs in
   stage 1 and the SG bug class is exactly what it would miss, so item 6 adds a batch-order check).
6. **Batch-order invariant check**: every rank logs a hash of its row order per step; the runner asserts equality
   across ranks (the lesson of the SG reorder bug applies to any role-asymmetric layout).

### 6.2 Layer templates to add
| template | ops | modes emitted by the synthesizer |
|---|---|---|
| `gdn` (Qwen3.5) | `in_proj_qkvz`, `in_proj_ba`, conv over q/k/v, `fused_recurrent_gated_delta_rule` (decode) / `chunk_gated_delta_rule` (prefill), per-head gated norm, `out_proj` | `tp{g}` by v-head (k-heads aligned), `seq{g}` (ownership; predicted loser), `tp{g}_fused` (n/a: single reduction already) |
| `gated_attention` (Qwen3.5) | q/k/v/gate projections, partial rotary, paged FA2 decode, output gate (sigmoid), `o_proj` | `tp{g}` (q-head sharded, KV replicated beyond 2), `seq{g}` (ownership), `pos{g}` (position split + LSE merge), `mix(f)` (fraction f of sequences owned, rest tp; prefill rows always tp) |
| `moe_deepseek` (Qwen3.5, Nemotron) | router (softmax or sigmoid + bias), top-k, `fused_moe`, shared expert | `tp{g}` (experts column-sharded), `ep{g}` (experts partitioned, all-to-all dispatch/combine), `ep{g}+hot(m)` (m hottest experts replicated on all ranks) |
| `mamba1` (Jamba, Falcon-Mamba, Mamba-2.8B) | `in_proj`, conv, `x_proj` (+ reduction), `dt_proj`, scan/state update with z, `out_proj` | `tp{g}` (2 reductions), `tp{g}_fused` (x_proj reduction kept, out_proj reduction merged with the MoE's where applicable), `sg{a}+{m}` (the gate split, for E2 only) |
| `mamba2_pure` (Mamba-Codestral) | as stage-1 `SsmBranch` with `n_groups = 8` | `tp{g}` with group-aligned norm (`g <= 8`), `tp16` (norm reduction) on NVwulf only |
| `jamba_layer` | composition of `mamba1` + `attention` + `moe` per Jamba's period | per-type modes as above |
Each template provides `reference_<name>()` in fp32 and `bytes_per_op()` for the cost model cross-check.

### 6.3 The elastic attention layout (the demonstration)
Mechanism, per attention layer, per rank, for a decode step with `B` sequences of which a subset `O_r` is owned by rank
`r` and the rest `U` is unit-sharded (f = |O| / B):
1. Rows of `U`: q/k/v projection with this rank's head shard (`nq/g` q heads, 1 KV head replicated on `g/nkv` ranks),
   paged attention over that KV head, `o_proj` partial.
2. Rows of `O_r`: q/k/v projection with the full weights (all heads), paged attention over both KV heads for its own
   sequences, full `o_proj`.
3. One all-reduce over the group for the `U` partials and one all-gather of the `O` rows (every rank needs every row
   for the next unit-sharded phase); both are issued in the same order on all ranks.
4. KV writes: `U` rows by every replica of the head (as in TP); `O_r` rows by the owner only, both heads.
5. Prefill rows are always in `U` (R7); at prefill completion the owner receives the KV head it does not hold (10 KiB
   per token on Qwen3.5, one P2P transfer), measured in E7.
6. Position split (`pos`): every rank holds `1/g` of each sequence's positions for both heads, computes partial attention
   with LSE, and the partials are merged with one all-gather + log-sum-exp per layer; used when `O` cannot be balanced.
The synthesizer chooses f, the ownership assignment (balance by cumulative context) and whether to use `pos` from the
regime; the harness executes the chosen executables. Predicted outcomes are in section 8.

## 7. Experiment matrix

Conventions: all cells are one decode step of one layer (or the composite), CUDA-graph captured with collectives inside,
20 warm-up + 100 timed steps, 3 interleaved repeats, min over repeats with CoV flags, max over ranks; eager runs give the
per-op breakdown. Regime grid unless stated: `B in {1, 4, 16, 64}`, `c in {2K, 8K, 32K}` (plus 128K and 256K for the
single-sequence long-context cells on Qwen3.5), prefill share 0 (decode) and the mixed cells with 1 prefill of 1K tokens
in the batch.

| id | question | models | modes / cells | predictions (section 8) | pass criterion | time |
|---|---|---|---|---|---|---|
| E0 ontology comparison | do the graphs expose the right structure; do families cluster | all 9 | no GPU; tables 1-6 of section 4.3 | clusters as listed in 4.3 | rule-firing matrix and signatures reproduce the expected clusters; replication/reduction tables match stage-1 collective shares within 15% | 2 d |
| E1 synthesizer regression | does synth v2 reproduce synth v1 and stage-1 measurements | Falcon-H1 3B/7B/34B, Falcon-40B | re-score all stage-1 cells from the new graph + emitted executables re-run on 4 cells per model | same rankings as v1; magnitudes within v1's 23% median | sign agreement 100% on the 64 stage-1 cells; emitted executables reproduce stage-1 step times within 5% | 3 d |
| E2 SG close-out | once outputs are correct, is there any gain (R5) | Mamba-2.8B, Falcon-Mamba-7B in `vllm-mamba-sg` | fix `MambaMixerGate.get_kv_cache_spec` in a scratch copy of the installed package; waves test passes; AIPerf TP vs 2+2 at c in {1, 8, 32, 128}, `ignore_eos`, OSL 128 and 512, same session, 2 trials | 2+2 within -5..+3% of TP at all c; no OSL deficit | result recorded either way; closes the line | 2 d (needs the idle server stopped) |
| E3 Nemotron composite | what the existing sweep says about R8 and R4 | Nemotron | `analysis.seq_composite` on `results/nemotron3_nano/raw/sweep_full.csv`; dp4 attention correctness re-check | attention dp4 beats tp4 only at (64, 8K+) by 1.3-1.75x per layer; moe ep4 ~ tp4 within 10%; composite <= 1.11x | matches v1 predictions in sign | 0.5 d |
| E4 Qwen3.5 attention layouts | R4 and R7 | Qwen3.5 | per attention layer: `tp4`, `seq4` (f=1), `mix(f in {0.25, 0.5})`, `pos4`; grid plus bimodal mixes (half the batch at 2K, half at 32K) and the mixed prefill cells; `B=1` at 128K/256K for `pos4` | table 8.1 | synthesized top candidate wins in every cell where predicted gain >= 1.1x; crossover context within 2x of prediction; `mix` never worse than `tp4` by more than 3% at short context | 4 d |
| E5 Qwen3.5 GDN and MoE layouts | R1/R5 on GDN, R8 on MoE | Qwen3.5 | GDN: `tp4` vs `seq4`; MoE: `tp4`, `ep4`, `ep4+hot(8)` over `B` | GDN `seq4` loses everywhere (weights 4x); MoE `ep4` within +/-10% of `tp4`, `hot` no gain at `B <= 16` | synthesizer ranks correctly | 2 d |
| E6 held-out families | generalization | Jamba, Falcon-Mamba, Mamba-Codestral | templates frozen from config; synthesizer emits top-3 layouts per model; measure top-3 + `tp4` on the grid | Jamba: `tp4_fused` + `ep4` best, attention `seq` useless (8 KV heads); Falcon-Mamba: `tp4_fused` best, SG loses; Mamba-Codestral: `tp4` best, state-bound at `B=64` | scoring targets of section 9 on these models | 5 d |
| E7 migration and policy | R6 | Qwen3.5, Nemotron | measure: KV consolidation (P2P, 10 KiB/token) and GDN/Mamba state migration (30/46 MiB per seq) on PCIe; simulate policies on two traces (SPEED-Bench 1k and a synthetic long-context agentic trace with 2K-128K contexts) with the measured step costs from E4/E5 | elastic beats best static by 10-25% on the long-context trace for Qwen3.5, <5% for Nemotron; migration amortised in < 50 steps | gain over best static >= 10% on at least one trace for Qwen3.5 | 4 d |
| E8 8-rank | does the gain scale with replication | Falcon-H1-34B, Qwen3.5, Nemotron | NVwulf H200, `world = 8`: `tp8` vs synthesized layouts | Qwen3.5 attention `seq8`/`pos8` ~2x at (64, 32K); Falcon-H1-34B `split44` 1.1-1.37x | sign agreement; magnitude within 20% (new hardware) | 2 d after allocation |
| E9 engine prototype (stage 2) | does it hold end to end | Qwen3.5 in the vLLM fork | decode-time ownership with prefill under TP, KV consolidation, CUDA graphs, AIPerf with fixed OSL | end-to-end tokens/s gain >= 70% of the harness gain at (64, 32K) | gate in section 10 | 4-6 w |

Order of execution: E2 and E3 first (they need no new code), E0/E1 while the Qwen3.5 template is being written, then E4,
E5, E7, E6, E8 when hardware exists, E9 after the gate.

## 8. Pre-registered predictions

All numbers are from the byte model in section 3.3 with the stage-1 bandwidth curves (A6000 ~0.7 TB/s achieved for
large reads, all-reduce 40 us at g=4) and the dims of section 4.1; they are decode steps, max over ranks, graph variant.
They will be re-scored by synth v2 before each experiment (the re-scored table replaces this one, committed first); if the
re-scored sign differs from the sign here, both are reported.

### 8.1 Qwen3.5-35B-A3B, 4 ranks (A6000): attention layout, composite step speedup over `tp4`
| regime (B, c) | per-rank KV `tp4` | per-rank KV `seq4` | other per-rank bytes (weights, experts touched, state) | predicted `seq4` / `mix` over `tp4` | predicted `pos4` over `tp4` |
|---|---|---|---|---|---|
| (64, 32K) | 21.5 GB | 10.7 GB | 1.0 + 14 + 1.0 GB (+0.41 GB attention weights under `seq`) | **1.35x** | 1.3x (merge cost) |
| (16, 32K) | 5.4 GB | 2.7 GB | 1.0 + 6.5 + 0.25 GB | **1.2x** | 1.2x |
| (64, 8K) | 5.4 GB | 2.7 GB | 1.0 + 14 + 1.0 GB | **1.1x** | 1.1x |
| (16, 8K) | 1.3 GB | 0.7 GB | 1.0 + 6.5 + 0.25 GB | 1.03x | 1.0x |
| (64, 2K) | 1.3 GB | 0.7 GB | 1.0 + 14 + 1.0 GB | 1.0x (policy keeps `tp4`) | 0.98x |
| (4, 32K) | 1.3 GB | 0.7 GB (1 seq per rank) | 1.0 + 1.9 + 0.06 GB | 1.07x | 1.05x |
| (1, 32K) | 0.34 GB | 0.67 GB on one rank | 1.0 + 0.5 + 0.02 GB | **0.75x (loses)** | 1.05x |
| (1, 256K) | 2.7 GB | 5.4 GB on one rank | 1.0 + 0.5 GB | 0.6x (loses) | **1.3x** |
| bimodal: 32 seqs at 2K + 32 at 32K | 11.4 GB | 5.7 GB (balanced owners) | as (64, 32K)/2 | 1.25x with `mix(f=0.5)` owning the long half | 1.2x |
Break-even for ownership: ~80K tokens of total batch context. Predicted crossover context at B=64: ~2.5K (ownership
becomes free of loss) and ~8K (gain >= 1.1x). GDN `seq4` vs `tp4`: 0.6-0.8x everywhere (weights 4x). MoE `ep4` vs `tp4`:
0.9-1.1x (same bytes, kernel efficiency decides); `ep4+hot(8)`: no gain at B <= 16, <= 1.05x at B=64.

### 8.2 Qwen3.5-35B-A3B, 8 ranks (H200)
(64, 32K): `tp8` per rank 29.5 GB and 80 collectives vs `seq8` 13.9 GB -> **~1.9x**; (16, 32K): ~1.5x; (64, 8K): ~1.2x;
(1, 256K): `pos8` ~1.5x. Falcon-H1-34B `split44`: 1.11-1.37x (v1 numbers).

### 8.3 Nemotron 3 Nano, 4 ranks
| regime | predicted best attention layout | composite gain over `tp4` |
|---|---|---|
| (64, 32K) | `seq4` | 1.16x |
| (16, 32K) | `seq4` | 1.06x |
| (64, 8K) | `tp4` (tie) | 1.04x |
| (16, 8K) and shorter | `tp4` | 1.0x |
MoE `ep4` vs `tp4`: within 10%; Mamba-2 `seq4`: loses. The synthesizer must say "stay TP below 32K"; the elastic policy
gains < 5% on any realistic trace. This is the "correct negative" family for R4.

### 8.4 Held-out models, 4 ranks (frozen before their templates are written; re-scored and re-frozen before measurement)
| model | predicted best layout | predicted gain over `tp4` | predicted losers |
|---|---|---|---|
| Jamba-v0.1 | `tp4` for Mamba-1 and attention (8 KV heads: replication-free), `ep4` for MoE at B >= 16 | <= 1.1x (MoE kernel efficiency only) | attention `seq4` (no replication to remove) 0.9-1.0x; `sg2+2` 0.85-0.95x |
| Falcon-Mamba-7B | `tp4` | 1.0x (nothing to gain; 128 reductions per step is the floor) | `sg2+2` 0.85-0.95x, `sg1+3` 0.6-0.8x (matches the clean E2 numbers) |
| Mamba-Codestral-7B | `tp4` (group-aligned norm) | 1.0x | `seq4` 0.4-0.6x (weights 4x; state the same) |
| Mamba-2.8B | `tp4` | 1.0x | `sg2+2` 0.9-1.0x |

### 8.5 Rule-firing predictions (which rules change the top candidate), 4 ranks
| model | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 |
|---|---|---|---|---|---|---|---|---|
| Falcon-H1 7B/3B | yes | yes | yes | no | yes | no | no | - |
| Falcon-H1 34B | yes (flips sign) | yes | yes | no | yes | no | no | - |
| Falcon-40B | no | yes (rejects) | yes | no | yes | no | no | - |
| Nemotron | yes (>= 32K) | - | yes | yes (>= 32K) | yes | yes (policy) | yes | yes |
| Qwen3.5 | yes | - | yes | **yes** | yes | yes | yes | yes |
| Jamba | no | - | yes | no | **yes** | no | no | yes |
| Falcon-Mamba, Mamba-2.8B | no | - | yes | no | yes | no | no | - |
| Mamba-Codestral | no | - | yes | no | yes | no | no | - |

## 9. Scoring the synthesizer

Computed per model and per hardware over every measured cell, by `harness/synth2/report.py`:
| metric | definition | target |
|---|---|---|
| sign accuracy | fraction of cells where (predicted top candidate beats `tp4`) == (measured) | 100% on cells with predicted gain >= 10% or <= -10%; ties (within +/-5%) excluded |
| magnitude error | relative error of the predicted step time of the top candidate vs measured (graph) | median <= 15%, p90 <= 30% (stage 1: 23% median, before the planned re-calibration) |
| ranking | Spearman correlation between predicted and measured step times over all candidates of a cell | >= 0.8 median over cells |
| regret | (measured best - measured predicted-top) / measured best | median <= 5%, max <= 15% |
| crossover error | predicted / measured context at which the best layout changes (per B) | within 2x |
| coverage | fraction of held-out cells where the predicted top is within 5% of the measured best | >= 80% |
| collective share | predicted vs measured share of step time in collectives (eager breakdown) | within 10 points |
Scores are reported separately for the training families (Falcon-H1, Falcon-40B, Nemotron, Qwen3.5) and the held-out
families (Jamba, Falcon-Mamba, Mamba-Codestral, Mamba-2.8B); the held-out numbers are the generalization claim.

## 10. Go/no-go gates and the decision tree

| gate | after | pass condition | if it fails |
|---|---|---|---|
| G1 regression | E1, E3 | synth v2 reproduces synth v1 rankings on all stage-1 cells; emitted executables reproduce stage-1 step times within 5%; Nemotron composite agrees in sign with v1 | fix the runner or the cost model before anything else; nothing new is measured on a broken baseline |
| G2 demonstration | E4, E5 | at least one synthesized Qwen3.5 layout >= 1.2x over `tp4` at B >= 16 and c >= 16K, with `mix` never worse than `tp4` by > 3% at 2K; crossover within 2x | the chapter keeps the synthesizer and the correct negative predictions; the elastic demonstration moves to the 8-rank hardware (E8) or is dropped; investigate the MoE lever (R8 `hot`) as the alternative demonstration |
| G3 generalization | E6 | scoring targets of section 9 on the held-out families | identify the failing rule by ablation, revise the rule set, re-freeze predictions, re-run only the affected cells; report the revision honestly as a second iteration |
| G4 elasticity | E7 | elastic policy >= 10% over best static on at least one Qwen3.5 trace with migration amortised in < 50 steps | static-only synthesizer in the chapter; the engine prototype (E9) is not started |
| G5 hardware | week 8 | NVwulf allocation exists | 8-rank cells become future work; the 4-rank Qwen3.5 result is the headline |
| G6 engine | E9 | end-to-end gain >= 70% of the harness gain at (64, 32K) | report the gap and its cause (scheduler, CUDA graphs, consolidation); the harness result stands on its own |

## 11. Schedule and effort (one person, guppy shared, ~6 h/day; GPU hours in brackets)

| week | work | outputs | GPU |
|---|---|---|---|
| 1 | confirm Jamba / Falcon-Mamba / Mamba-Codestral dims from `config.json` and freeze their yaml; verify FLA and `fused_moe` kernels in `mmp-dev` (`/opt/vllm/vllm/model_executor/layers/fla/ops`, `.../fused_moe`); **E2** SG close-out (fix, waves test, AIPerf re-measurement; needs the idle SG server stopped); **E3** Nemotron composite; start `harness/synth2/` (ontology, templates for the four known families) | E2 and E3 result notes in `doc/`; `configs/{jamba,falcon_mamba,mamba_codestral}.yaml`; `synth2/ontology.py`, 4 templates | 6 h |
| 2 | synth2 `phases`, `enumerate`, `cost`, `policy`, `emit`, `report`; **E1** regression; **E0** tables and figures; download Qwen3.5 layer 0; write `gdn`, `gated_attention`, `moe_deepseek` templates and fp32 references | `results/synth2/e0/`, E1 report, Qwen3.5 harness layers passing the correctness gate | 4 h |
| 3 | executable runner; attention layouts `seq`, `pos`, `mix`; batch-order check; microbench additions (GDN, MoE, all-to-all, all-gather, merge); re-score and commit section 8 predictions from synth v2 | runner reproduces stage-1 modes; prediction commit | 6 h |
| 4 | **E4**, **E5** sweeps and analysis; gate **G2** | `results/qwen35/` figures + summary; G2 decision in `doc/` | 10 h |
| 5 | freeze held-out templates and predictions (commit); **E6** sweeps (Jamba, Falcon-Mamba, Mamba-Codestral); scoring; gate **G3** | `results/{jamba,falcon_mamba,mamba_codestral}/`; score table | 10 h |
| 6 | **E7**: consolidation and migration measurements; trace-driven policy simulation; gate **G4**; results write-up v1 with all figures (blog-style, as `blog/` was) | `results/synth2/policy/`; `doc/synth2_results.md` | 3 h |
| 7-8 | buffer for re-runs; **E8** if NVwulf exists (sbatch package in `scripts/nvwulf/`); chapter draft sections 3-7 | chapter draft | 4 h (NVwulf) |
| 9-14 | **E9** engine prototype if G4 passed: scheduler ownership metadata, attention backend with two sub-batches, consolidation at prefill end, CUDA graphs, AIPerf end to end; otherwise chapter completion | fork branch, end-to-end numbers | 20 h |

Dependencies: E2 needs the user to stop the idle `vllm serve` in `vllm-mamba-sg` for ~4 hours; E6 needs ~5 GB of
layer-0 downloads; E8 needs the allocation; everything else runs in `mmp-dev` on free GPUs.

## 12. Risks and mitigations

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| FLA GDN kernels absent or shaped differently in vLLM v0.30.0 | medium | blocks E4/E5 | week-1 check; vendor `flash-linear-attention` at vLLM's pinned version inside the container; or move the harness image to the vLLM release that ships Qwen3.5 (rebuild `mmp:stage2`) |
| Qwen3.5 gated attention details (partial rotary dims, output gate placement, q/k norms) wrong in the template | medium | correctness gate fails | derive from HF `modeling_qwen3_5.py` and vLLM `qwen3_next.py`, compare layer-0 outputs against the HF module in fp32 before any timing |
| `fused_moe` weight layout and routing variant (softmax top-k with renormalisation vs sigmoid+bias) | medium | wrong MoE numbers | reuse vLLM's `FusedMoE` directly with host-precomputed routing as in stage 1; validate against the per-expert torch loop |
| ownership imbalance in bimodal mixes hides the gain | medium | G2 fails for the wrong reason | the runner assigns owners by cumulative context; report max and mean over ranks; add the `pos` fallback for the long tail |
| position-split merge numerics (LSE in fp32, bf16 partials) | low | correctness gate | validate against the fp32 reference at 256K |
| shared-box contamination | high | noisy cells | stage-1 protocol: min over repeats, CoV flags, power/util logged; re-run flagged cells at night |
| synth v2 regressions vs v1 | medium | G1 fails | keep v1 as oracle; unit tests on the stage-1 cells |
| held-out template error (misread config) | medium | G3 fails for the wrong reason | the runner cross-checks `bytes_per_op` against actual tensor sizes; config fields printed in the report |
| the 2x crossover tolerance is too loose to be meaningful | low | weak claim | also report the absolute predicted vs measured context; the policy simulation uses measured costs, not predicted |
| NVwulf allocation does not arrive | medium | no 8-rank headline | G5: 4-rank headline; 8-rank predictions published as predictions |
| engine integration (E9) exceeds the budget | high | no end-to-end number | gated by G4; the harness result plus the vLLM design doc is the fallback deliverable; the SG reorder bug class is guarded by the batch-order invariant |
| byte model stays ~20% off in magnitude | medium | G3 magnitude target missed | re-calibrate curves per kernel with the new microbench; report sign/ranking/regret as the primary scores (they were right in stage 1 while magnitudes were off) |

## 13. Deliverables and the chapter

**Code (this repo):** `harness/synth2/` (ontology, templates, phases, enumerate, cost, policy, emit, report),
`harness/exec_runner.py`, templates for GDN, gated attention, DeepSeek-style MoE, Mamba-1, pure Mamba-2, Jamba
composition; microbench additions; `analysis/` updates; `configs/*.yaml` for every model; `scripts/run_<model>.sh` chains.
**Data:** `results/synth2/e0/` (ontology comparison), `results/synth2/<model>/` (predictions with their pre-registration
commit hashes, candidate tables, policies), `results/<model>/{raw (ignored), figs, summary}` per sweep, the score tables.
**Documents:** `doc/synth2_design.md` (the schema and the algorithm), `doc/e2_sg_closeout.md`, `doc/synth2_results.md`
(blog-style results with figures), the chapter draft.

**Chapter outline (working title: Architecture-derived parallelism for state-space and hybrid LLM serving)**
1. Introduction: layouts are hand-chosen per model today; hybrids multiply the choices; claim.
2. Background: Mamba-1/2, GDN, GQA attention, MoE; TP, EP, DP attention, decode context parallelism; where bytes go in
   decode.
3. The ontology graph and the rules R1-R8, each with its evidence.
4. The synthesizer: phases, executables, cost model, policy.
5. Evaluation across six families on 4 GPUs (E0, E1, E4-E6): predictions vs measurements, scores, the rule-firing
   matrix.
6. Elastic attention layouts for GDN hybrids (E4, E7, E8): ownership, position split, mixed; policy on traces; 8 ranks.
7. Negative results with root causes: the SSM/gate split (R1, R5; the reorder bug), swing MLP shards (R3), Falcon-40B (R2),
   ownership on Nemotron and Mamba-Codestral (R4 boundaries).
8. Engine prototype (if G4/G6 pass) or the integration design and its measured risks.
9. Related work and conclusion.

## 14. Appendices

### A. Ontology schema (YAML form of `harness/synth2/ontology.py`), Qwen3.5 excerpt
```yaml
model: qwen3_5_35b_a3b
hidden: 2048
pattern: "(GGGA)x10"          # G = gdn layer, A = gated attention layer; every layer has an moe phase
layer_types:
  G:
    phases:
      - name: mixer
        compose: seq
        ops:
          - {name: in_proj_qkvz, cls: gemv, W: 2*H*(2*Dk + 2*Dv), units: {W: n_v_heads}, reduce: [], state: null, phase_role: both}
          - {name: in_proj_ba,   cls: gemv, W: 2*H*2*n_v_heads,   units: {W: n_v_heads}}
          - {name: conv,         cls: conv, S: 2*(d_conv-1)*(2*Dk+Dv)*2, units: {S: n_v_heads}, state: conv}
          - {name: recurrence,   cls: ssu,  S: 2*n_v_heads*dk*dv*2, units: {S: n_v_heads}, state: delta, phase_role: decode_bandwidth}
          - {name: gated_norm,   cls: norm, reduce: []}                     # per head: never cross-rank
          - {name: out_proj,     cls: gemv, W: Dv*H*2, units: {W: n_v_heads}, reduce: [[T, H]]}
      - name: moe
        compose: route
        ops:
          - {name: router, cls: gemv, W: E*H*2, units: {W: inf}}
          - {name: experts, cls: moe, W_expert: 3*I_e*H*2, E: 256, k: 8, units: {W: E}, reduce: [[T, H]]}
          - {name: shared,  cls: gemv, W: 3*I_s*H*2, units: {W: inf}, reduce: [[T, H]]}   # merged with the experts' reduction
  A:
    phases:
      - name: mixer
        ops:
          - {name: qkv_gate_proj, cls: gemv, W: H*(2*nq*dh + 2*nkv*dh)*2, units: {W: nkv}}   # k/v rows cap at nkv
          - {name: attn,          cls: attn, K: 2*nkv*dh*2, units: {K: nkv}, state: kv, phase_role: both}
          - {name: o_proj,        cls: gemv, W: nq*dh*H*2, units: {W: nq}, reduce: [[T, H]]}
      - name: moe   # as above
dims: {Dk: 2048, Dv: 4096, n_v_heads: 32, n_k_heads: 16, dk: 128, dv: 128, d_conv: 4, nq: 16, nkv: 2, dh: 256, E: 256, k: 8, I_e: 512, I_s: 512}
```
Composition for Falcon-H1 uses `compose: sum` with the two branches and their multipliers; Nemotron uses the pattern
string with `M`, `E`, `*`.

### B. Byte formulas per op class (bf16 unless the state dtype says otherwise)
- `gemv`: `W = rows * cols * 2`; per rank `W / min(g, u_W)`.
- `attn` decode: `K = 2 * nkv * dh * 2` per token per layer; per rank `K * sum over sequences read of c`; sequences read =
  all (unit sharding, `1/min(g,nkv)` of heads but every sequence), owned subset (`seq`), all with `c/g` positions (`pos`).
- `ssu`: `S = 2 * heads * dk * dv * dtype_bytes` (read + write); per rank `S * B / min(g, u_S)` or `S * B_owned`.
- `conv`: `S = 2 * (d_conv - 1) * channels * 2`.
- `moe`: touched experts `E * (1 - (1 - k/E)^B)` capped at `E`; per rank `touched * W_expert / g` (`tp`) or
  `(E/g) * (1 - (1 - k/E)^B) * W_expert` (`ep`); shared expert as `gemv`.
- collectives: all-reduce `[T, H]` = `B * H * 2` bytes; gated-norm `[T, 1]` = `B * 4`; all-gather of owned rows
  `B_owned * H * 2`; merge `B * nq * (dh + 1) * 4`.

### C. Cost equations
`t_op = sum_class bytes_class / BW_cls(bytes_class) + n_launch * L`;
`t_phase(group) = sum of t_op over the group's ops in the phase`;
`t_step = sum over phases [ max over groups t_phase(group) + sum over the phase's collectives tau_g(bytes) ]`;
mixed prefill/decode in one phase on the same ranks: sum; on disjoint ranks: max. Policy: for regimes `q`, layouts `l`,
`gain(l, q) = t_step(tp, q) / t_step(l, q)`; elastic value on a trace = sum over steps of the chosen layout's time plus
migration bytes / link bandwidth at each switch, with hysteresis `h` chosen to maximise the trace total.

### D. Commands (to be created; names fixed now so scripts can reference them)
```bash
python -m harness.synth2 --model configs/qwen35_35b_a3b.yaml --world 4 --hw guppy_a6000 --out results/synth2/qwen35/
python -m harness.synth2.report --e0 configs/*.yaml --world 2 4 8 --out results/synth2/e0/
torchrun --nproc_per_node=4 -m harness.run --exec results/synth2/qwen35/exec_attn_seq4.json --batch 64 --ctx 32768 --variant graph
scripts/run_qwen35.sh      # correctness gate, microbench, E4+E5 sweeps, analysis
scripts/run_heldout.sh     # E6 for jamba, falcon_mamba, mamba_codestral
python -m harness.synth2.score --pred results/synth2/<m>/cost_*.csv --meas results/<m>/raw/sweep_full.csv
```

### E. Where things are
`NOTES.md` (SG deep dive, container facts, benchmark audit), `docs/docs/blog/README.md` (stage-1 results), `docs/stage1_elastic_mixer_parallelism_plan.md`
(stage-1 plan and findings), `harness/synth.py` (recipe v1), `harness/seq_model.py` (Nemotron template v1),
`configs/*.yaml` (dims), `results/*` (stage-1 sweeps), guppy `~/masarani` (SG history), guppy `~/backups/vllm-mamba-sg-2026-10-01`
(SG code).
