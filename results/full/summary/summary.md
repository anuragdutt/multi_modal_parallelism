# Stage-1 summary

## Speedup of each mode over tp4 (CUDA-graph variant, min over repeats; cells with CoV > 5% flagged *)

| batch | ctx | tp4 ms | tp4_fused | tp4_streams | split22 | swing best (S, cuts) | swing speedup | max CoV |
|---|---|---|---|---|---|---|---|---|
| 1 | 512 | 0.272* | 1.05x | 1.12x | 1.11x | S=0.1, neutral | 1.12x | 0.10 |
| 1 | 2048 | 0.274* | 1.05x | 1.12x | 1.12x | S=0.2, grid | 1.13x | 0.10 |
| 1 | 8192 | 0.286* | 1.05x | 1.16x | 1.16x | S=0.1, neutral | 1.17x | 0.10 |
| 1 | 32768 | 0.318* | 1.04x | 1.21x | 1.30x | S=0.1, neutral | 1.30x | 0.09 |
| 4 | 512 | 0.299* | 1.06x | 1.15x | 1.13x | S=0.3, neutral | 1.13x | 0.09 |
| 4 | 2048 | 0.305* | 1.06x | 1.14x | 1.16x | S=0.2, grid | 1.16x | 0.09 |
| 4 | 8192 | 0.329* | 1.05x | 1.17x | 1.24x | S=0.3, neutral | 1.24x | 0.09 |
| 4 | 32768 | 0.414* | 1.04x | 1.18x | 1.29x | S=0.2, neutral | 1.29x | 0.08 |
| 16 | 512 | 0.366* | 1.09x | 1.16x | 1.11x | S=0.1, neutral | 1.11x | 0.10 |
| 16 | 2048 | 0.388* | 1.09x | 1.17x | 1.18x | S=0.1, neutral | 1.18x | 0.10 |
| 16 | 8192 | 0.469* | 1.07x | 1.17x | 1.38x | S=0.1, neutral | 1.38x | 0.10 |
| 16 | 32768 | 0.790* | 1.04x | 1.12x | 1.19x | S=0.1, neutral | 1.20x | 0.05 |
| 64 | 512 | 0.632 | 1.13x | 1.19x | 1.07x | S=0.1, grid | 1.08x | 0.04 |
| 64 | 2048 | 0.715 | 1.12x | 1.18x | 1.22x | S=0.1, grid | 1.23x | 0.04 |
| 64 | 8192 | 1.028* | 1.08x | 1.12x | 1.31x | S=0.1, grid | 1.32x | 0.31 |
| 64 | 32768 | 2.309* | 1.03x | 1.06x | 1.12x | S=0.4, grid | 1.12x | 0.49 |

## Go / no-go

decision cells not measured yet
