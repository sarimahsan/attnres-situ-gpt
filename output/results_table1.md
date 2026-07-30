# Main Quantitative Results (Table 1)

| # | Variant Name | Params | FLOPs/Tok | Final Val Loss (Mean ± Std) | Val Perplexity | Loss Spikes |
|---|---|---|---|---|---|---|
| 0 | 0. Baseline (Pre-Norm + SwiGLU) | 50,234,880 | 3.01e+08 | 2.8371 ± 0.0018 | 17.06 ± 0.03 | 2.0 |
| 1 | 1. +AttnRes | 50,237,952 | 3.01e+08 | 2.7660 ± 0.0078 | 15.89 ± 0.12 | 2.0 |
| 2 | 2. +SiTU | 50,234,880 | 3.01e+08 | 2.7852 ± 0.0074 | 16.20 ± 0.12 | 0.0 |
| 3 | 3. +Both (AttnRes + SiTU) | 50,234,880 | 3.01e+08 | 2.7220 ± 0.0049 | 15.21 ± 0.07 | 0.0 |
