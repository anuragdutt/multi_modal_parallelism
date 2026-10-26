# Generalization: from per-model tricks to an architecture-derived synthesizer (design note, 2026-10-26)

This note records the discussion that turned the stage-1 recipe into the plan in `PLAN.md`. It is the "why" behind the
ontology graph, the rules and the elastic attention layout; `PLAN.md` is the "what and when".

## 1. The question
Every new model so far (Mamba-1, Falcon-H1, Nemotron, Qwen3.5) was handled with a hand-made layout and a hand-made
experiment. The thesis needs the opposite: a procedure that takes any new SSM, GDN or hybrid model and produces its
parallel layout, the expected gain over tensor parallelism, and the regime where that gain exists, before anything is
measured. The user's framing: divide the model's ontology graph into executables, and derive rules that generalize.

## 2. Why the gate split could not generalize
The SSM/gate split assigns GPUs to the z (gate) path or the x (conv + scan) path of a Mamba-1 block. Measured over nine
months and three model families it never beat tensor parallelism by more than 7%, and only on Jamba, where Mamba layers
are a minority. The reasons are two rules that apply to every model, not one trick:
- R1, replication cap: the per-rank bytes of a byte class stop shrinking once the group exceeds the class's unit count.
  Moving the gate GEMM off the SSM ranks removes one thin projection while concentrating conv, scan and `out_proj` on
  half the ranks, so an SSM rank reads about 37% more weight bytes than a TP-4 rank.
- R5, collective count: at decode sizes every collective is latency-bound, so counts matter and bytes do not; the split
  only trades two 4-rank all-reduces for two 2-rank ones plus a P2P exchange, and pays with idle gate ranks.
The same reasoning kills a gate split for Mamba-2 and GDN (the gate again enters as a thin projection into a gated norm).
It survives only as a balancing move where a layout already leaves ranks idle during a recurrence.

## 3. The objects
- **Ontology graph**: a layer-type pattern plus one operator DAG per layer type. Each operator carries byte classes
  (weights W per step, per-sequence state S read + write, per-token cache K), a unit axis per class (q heads, KV heads,
  Mamba or GDN heads, experts, channels, unbounded), its reduction signature (what cross-shard reduction a unit split
  induces and of what size), its state locality (which per-sequence state it updates) and its phase role (compute-bound
  in prefill, bandwidth-bound in decode). Edges carry shapes and the composition tag: sequential, summed branches, or
  token routing.
- **Phase**: a maximal operator set between two reductions.
- **Executable**: (operator set, rank group, sharding axis in {unit, sequence, position, expert}, replication factor,
  boundary collectives). Validity: identical collective order on every member of a group; the hidden state replicated on
  every rank at a phase boundary that feeds a unit-sharded phase (owned rows are all-gathered); state locality respected.
- **Layout, regime, policy**: a layout covers the step; a regime is (batch, context distribution, prefill share,
  hardware); a policy maps regimes to layouts with switch costs, reported as best static, elastic with hysteresis, oracle.

## 4. The rules
R1 replication cap; R2 summed branches share one reduction; R3 phase balance (step = sum of per-phase maxima, work
moves only within a phase); R4 ownership vs sharding (shard by sequence when replicated per-sequence traffic exceeds the
extra replicated weights and the sequences balance; by position when they do not); R5 collective count; R6 state
locality (a sequence changes shard only by migrating its state; switch with hysteresis); R7 phase heterogeneity (prefill
unit-sharded, decode per R4, mixed steps per sub-batch when collectives line up); R8 sparse ops (expert-partitioned and
sharded experts move the same bytes; kernel occupancy and collective type decide; replicate hot experts into slack).
Each rule is a measured effect or a pre-registered prediction; `PLAN.md` section 2.3 lists the evidence and the test.

## 5. The algorithm (synthesizer v2)
Config -> template library -> graph -> phase cut -> executable enumeration per phase (unit, sequence, position, expert;
disjoint groups for summed branches) -> cost per regime from the byte model with microbench curves and collective
latencies (step = sum of phase maxima) -> policy synthesis -> executable specs the harness runs -> score (sign, magnitude,
ranking, regret, crossover). The family label is an output of the phase cut (a structural signature), not an input.

## 6. What the byte budgets say for the two target models at 4 GPUs
Nemotron 3 Nano: routed expert weights dominate from batch 8 (up to ~56 GB per step over all ranks), KV is 6 KiB per
token, Mamba state 94 MiB per sequence read + write; attention ownership helps only at 32K context and batch 64 (~1.16x).
Qwen3.5-35B-A3B: experts ~56 GB at batch 64, but KV is 20 KiB per token so at 32K context and batch 64 the KV term
(43 GB over all ranks, replicated 2x under TP-4) rivals the experts; attention ownership is predicted at 1.35x on 4
A6000s and ~1.9x on 8 H200s, position split 1.3x for a single 256K sequence, GDN and MoE unchanged (TP). The crossover is
about 80K tokens of total batch context at 4 ranks.

## 7. The proposed layout family for sequential hybrids
GDN and Mamba-2 layers: TP by head (one reduction, state sharded for free; ownership never wins because weights would
replicate). MoE: TP at 4 ranks, EP at 8, measured. Attention layers: three layouts chosen per step inside one TP group
with one scheduler: head-sharded (TP), sequence-owned (each rank holds both KV heads for its sequences, full attention
weights, owned rows all-gathered), position-split (decode context parallelism) for few very long sequences. Prefill always
head-sharded; one-time KV consolidation at prefill end. Switching is cheap because weights for every mode are resident.

## 8. Decisions taken on 2026-10-26
Broader scope than the gate split; Qwen3.5-35B-A3B first, Nemotron second; the gate split closed out with a bug fix (the
batch-reorder mismatch on gate ranks) and one clean re-measurement; the synthesizer is the thesis claim and the elastic
attention layout is the demonstration; held-out families (Jamba, Falcon-Mamba, Mamba-Codestral, Mamba-2.8B) test
generalization with predictions frozen before measurement.

## 9. Open questions carried into the plan
Whether the gate-rank cache-spec fix fully removes the reorder mismatch; which source produced each SG benchmark series;
whether ownership imbalance under bimodal contexts erodes the predicted gain; whether the 2x crossover tolerance is the
right bar; NVwulf availability for the 8-rank headline.
