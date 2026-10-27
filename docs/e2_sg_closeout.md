# E2: SSM/gate split close-out

## Part 1: the batch-reorder fix, validated (2026-10-26)

**Defect.** Under `--enable-mamba-sg`, gate ranks owned no KV-cache groups, so `GPUModelRunner._may_reorder_batch`
returned early on them while SSM ranks moved decode requests ahead of prefills; on every step that needed that reorder
the paired ranks exchanged rows of different tokens and every resident request was corrupted (`NOTES.md` F2). This is
the cause of the shorter, degraded outputs in the August-September 2026 benchmarks on Mamba-2.8B and Falcon-Mamba-7B.

**Fix.** `anuragdutt/vllm` branch `mmp/sg`, commit `558de61`: `MambaMixerGate` inherits `MambaBase`, reports the
same state dtype and shape as its paired SSM mixer and registers under the same layer name, so every TP rank advertises
identical KV-cache groups and performs the same reorder. The gate rank allocates an unused state buffer of the SSM rank's
size; its forward is unchanged.

**Validation** (`scripts/sg/smoke_sg.py`, container `mmp-sg` = image `mmp:sg-dev`, GPUs 0 and 1 of guppy, Mamba-2.8B
bf16, TP 2, `enforce_eager`, `max_num_seqs 8`, 32 prompts x 128 tokens with `ignore_eos`, two repeats per process;
results in `results/sg/smoke_{sg,tp}.json`):

| check | SG (fixed) | regular TP |
|---|---|---|
| per-rank (KV-cache groups, reorder threshold) | rank 0: (1, 1), rank 1: (1, 1) | (1, 1), (1, 1) |
| repeat 0 vs repeat 1 token ids | identical | identical |
| load time | 17.9 s | 11.8 s |
| tokens/s (second repeat, max_num_seqs 8) | 202 | 181 |
| SG vs TP, first generated token identical | 32 / 32 | |
| SG vs TP, max abs difference of the first-token logprob | 0.112 (bf16 noise floor measured earlier: ~0.19) | |
| SG vs TP, full 128-token sequence identical | 15 / 32 (bf16 divergence after some tokens, same as TP vs TP1) | |

Before the fix the gate rank reported (0, None) and the same waves workload diverged between runs with first-token
logprob differences up to 1.2 (`NOTES.md` F4). The throughput difference here is a single run at small batch on GPUs
shared with an idle server and is not a measurement.

## Part 1b: the same validation on the from-source build (2026-10-27)
The container was rebuilt as `mmp:sg-src` with vLLM compiled from the checkout's own kernel sources (no prebuilt
extensions; `docs/dev_workflow.md`). Same test, same outcome: gate rank (1, 1), SG repeats identical, SG vs TP first
token 32/32, max |d logp0| 0.107 (`results/sg/smoke_sg_src.json`, `smoke_tp_src.json`).

## Part 2: clean re-measurement (pending)
Per `PLAN.md` E2: AIPerf, TP 4 vs SG 2+2, Mamba-2.8B and Falcon-Mamba-7B, concurrency 1/8/32/128, `ignore_eos`,
output length 128 and 512, same session, two trials. Requires all four GPUs, i.e. the idle `vllm serve` in the old
container `vllm-mamba-sg` stopped. Scripts will live in `scripts/sg/`.
