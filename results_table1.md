# Main Quantitative Results (Table 1)

| # | Variant Name | Seeds | Params | FLOPs/Tok | Final Val Loss (Mean +/- Std) | Best Val Loss (Mean) | Val Perplexity | Loss Spikes (Mean) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0. Baseline (Pre-Norm + SwiGLU) | 3/3 | 50,234,880 | 1.452336e+17 | 112.8032 +/- 15.1944 | 35.8606 | N/A (inf) | 78.0 |
| 1 | 1. +AttnRes | 3/3 | 50,237,952 | 1.452551e+17 | 113.0352 +/- 7.8393 | 34.7888 | N/A (inf) | 81.7 |
| 2 | 2. +SiTU | 3/3 | 50,234,880 | 1.452336e+17 | 119.2212 +/- 8.6133 | 39.2441 | N/A (inf) | 79.7 |
| 3 | 3. +Both (AttnRes + SiTU) | 3/3 | 50,234,880 | 1.452551e+17 | 107.7204 +/- 2.2503 | 34.2946 | N/A (inf) | 86.0 |
