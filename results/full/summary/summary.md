# Stage-1 summary

## Speedup of each mode over tp4 (CUDA-graph variant, median over repeats)

| batch | ctx | tp4 ms | tp4_fused | tp4_streams | split22 | swing best (S, cuts) | swing speedup |
|---|---|---|---|---|---|---|---|
| 1 | 512 | 0.273 | 1.05x | 1.12x | 1.11x | S=0.1, neutral | 1.11x |
| 1 | 2048 | 0.275 | 1.05x | 1.12x | 1.11x | S=0.1, neutral | 1.12x |
| 1 | 8192 | 0.287 | 1.05x | 1.16x | 1.16x | S=0.2, neutral | 1.16x |
| 1 | 32768 | 0.319 | 1.04x | 1.21x | 1.30x | S=0.1, neutral | 1.29x |
| 4 | 512 | 0.301 | 1.06x | 1.15x | 1.13x | S=0.3, neutral | 1.14x |
| 4 | 2048 | 0.307 | 1.06x | 1.15x | 1.16x | S=0.2, neutral | 1.16x |
| 4 | 8192 | 0.331 | 1.05x | 1.17x | 1.24x | S=0.1, neutral | 1.24x |
| 4 | 32768 | 0.416 | 1.03x | 1.18x | 1.30x | S=0.2, neutral | 1.30x |
| 16 | 512 | 0.366 | 1.09x | 1.16x | 1.11x | S=0.1, neutral | 1.11x |
| 16 | 2048 | 0.390 | 1.09x | 1.17x | 1.19x | S=0.1, neutral | 1.19x |
| 16 | 8192 | 0.471 | 1.07x | 1.17x | 1.38x | S=0.1, neutral | 1.38x |
| 16 | 32768 | 0.792 | 1.04x | 1.12x | 1.19x | S=0.4, neutral | 1.20x |
| 64 | 512 | 0.633 | 1.12x | 1.18x | 1.08x | S=0.1, neutral | 1.08x |
| 64 | 2048 | 0.721 | 1.11x | 1.18x | 1.23x | S=0.1, neutral | 1.23x |
| 64 | 8192 | 1.492 | 1.54x | 1.62x | 1.90x | S=0.1, neutral | 1.90x |
| 64 | 32768 | 4.750 | 2.13x | 2.18x | 2.30x | S=0.1, neutral | 2.31x |

## Go / no-go

decision cells not measured yet
