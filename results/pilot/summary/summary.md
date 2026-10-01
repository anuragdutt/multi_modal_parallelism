# Stage-1 summary

## Speedup of each mode over tp4 (CUDA-graph variant, median over repeats)

| batch | ctx | tp4 ms | tp4_fused | tp4_streams | split22 | swing best (S, cuts) | swing speedup |
|---|---|---|---|---|---|---|---|
| 16 | 512 | 0.366 | 1.10x | 1.17x | 1.12x | S=0.2, neutral | 1.12x |
| 16 | 8192 | 0.467 | 1.07x | 1.18x | 1.37x | S=0.2, neutral | 1.38x |

## Go / no-go

decision cells not measured yet
