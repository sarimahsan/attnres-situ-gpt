# Main Quantitative Results (Table 1)

| # | Variant Name | Params | FLOPs/Tok | Final Val Loss (Mean ± Std) | Val Perplexity | Loss Spikes |
|---|---|---|---|---|---|---|
| 0 | 0. Baseline (Pre-Norm + SwiGLU) | 50,234,880 | 3.01e+08 | 2.8415 ± 0.0032 | 17.14 ± 0.05 | 2.0 |
| 1 | 1. +AttnRes | 50,237,952 | 3.01e+08 | 2.7667 ± 0.0129 | 15.91 ± 0.21 | 2.0 |
| 2 | 2. +SiTU | 50,234,880 | 3.01e+08 | 2.7853 ± 0.0061 | 16.21 ± 0.10 | 0.0 |
| 3 | 3. +Both (AttnRes + SiTU) | 50,234,880 | 3.01e+08 | 2.7185 ± 0.0089 | 15.16 ± 0.14 | 0.0 |
