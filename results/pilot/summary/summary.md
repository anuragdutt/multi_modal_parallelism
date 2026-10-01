# Stage-1 summary

## Speedup of each mode over tp4 (CUDA-graph variant, median over repeats)

| batch | ctx | tp4 ms | tp4_fused | tp4_streams | split22 | swing best (S, cuts) | swing speedup |
|---|---|---|---|---|---|---|---|
| 4 | 2048 | 0.311 | - | - | - | S=0.2, grid:128 | 1.01x |
| 16 | 512 | 0.371 | 1.09x | 1.16x | 1.11x | S=0.2, neutral | 1.08x |
| 16 | 8192 | 0.475 | 1.07x | 1.17x | 1.37x | S=0.2, neutral | 1.33x |

## Go / no-go

decision cells not measured yet
