# Provenance of `vllm_overlays/sg-2026-07-31`

**What this is.** The files under `vllm/` are byte-exact copies of the hand-patched files inside the installed
vLLM 0.24.0 package (`/usr/local/lib/python3.12/dist-packages/vllm`) of the Docker container `vllm-mamba-sg`
(id `db558a35677a…`, image `vllm/vllm-openai:v0.24.0`, digest `sha256:251eba5c…`) on guppy.fsl.cs.sunysb.edu,
copied out with `docker cp` on 2026-10-26. That installed package, not the `/vllm` git checkout in the container,
is what ran the August-September 2026 SSM/gate-split benchmarks (`docs/NOTES.md` section 4). `SHA256SUMS` covers
every file; `overlay_vs_v0.24.0.patch` is the unified diff against upstream tag `v0.24.0` (`ee0da84ab9e0`).

| file | mtime in container | bytes | changed lines vs v0.24.0 | role |
|---|---|---|---|---|
| `config/parallel.py` | 2026-07-30 23:21 | 43378 | 6 | `enable_mamba_sg` field (the 1+3 size fields are commented out) |
| `distributed/mamba_sg_parallel_state.py` | 2026-07-31 04:18 | 2792 | new (117 lines) | role assignment: first half of each TP group = gate |
| `distributed/parallel_state.py` | 2026-07-31 04:18 | 86515 | 51 | `mamba_sg_branch` and `mamba_sg_pair` groups |
| `engine/arg_utils.py` | 2026-07-31 03:26 | 115857 | 11 | `--enable-mamba-sg` |
| `model_executor/layers/mamba/mamba_mixer_gate.py` | 2026-07-31 04:19 | 5539 | new (139 lines) | gate rank: z shard, pair send, output broadcast |
| `model_executor/layers/mamba/mamba_mixer_ssm.py` | 2026-07-31 04:30 | 23484 | new (576 lines) | SSM rank: x shard, conv, x_proj (+branch all-reduce), scan with z, out_proj (+branch all-reduce), broadcast |
| `model_executor/layers/mamba/mamba_mixer_ssm_old.py` | 2026-07-23 03:53 | 21209 | new, unused | earlier draft left in the package; kept here for the record, not imported |
| `model_executor/models/mamba.py` | 2026-07-31 04:19 | 12655 | 70 | mixer selection by role; weight loading skips the other role's tensors |
| `model_executor/models/jamba.py` | 2026-06-29 08:41 | 21528 | 0 | **identical to the stock v0.24.0 file**: the installed package had no Jamba SG support (open question in `docs/NOTES.md` section 10) |
| `v1/core/kv_cache_utils.py` | 2026-07-31 04:26 | 90696 | 28 | cacheless-worker handling for gate ranks |

**Relation to the git commits.** `hazemmasarani/vllm` `main` holds the same design rebased on upstream `3034c8d38`
(`2195e2a4b` "mamba ssm-Gate TP", `4c0c73684` "falcon-Jamba", which adds Jamba support and makes the gate mixer a
`MambaBase`). The uncommitted 1+3 patch (unequal role sizes via tp-group broadcasts) exists only in the container's
`stash@{0}` and in `~/backups/vllm-mamba-sg-2026-10-01/uncommitted.patch` on guppy. The fork branch
`anuragdutt/vllm:mmp/sg` = `v0.24.0` + this overlay (commit 1) + the batch-reorder fix (commit 2).

**Known defect in this overlay.** Gate ranks own no KV-cache groups, so the v1 runner skips the decode-first batch
reorder on them while SSM ranks perform it; steps that admit a prefill ahead of decoding requests corrupt every
resident request (`docs/NOTES.md` F2). The replica image reproduces this on purpose; the dev image carries the fix.
