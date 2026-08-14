# LingBot V2 Thor latency

| Implementation | PatchEmbed | Attention | CUDA Graph | Mean (ms) | P50 (ms) | P90 (ms) | P99 (ms) | Std (ms) | Peak allocated (GiB) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Official Thor-compatible | conv3d | official_eager | off | 4208.859 | 4209.037 | 4218.356 | 4222.449 | 6.792 | 12.136 |
| PHYAI | gemm | vision=flashinfer; prefix/expert=official_eager | on | 643.208 | 642.904 | 647.003 | 650.705 | 2.167 | 12.207 |

PHYAI speedup vs Official: **6.544x**

Official speedup vs PHYAI: **0.153x**

PHYAI latency change vs Official: **84.72% faster**

Comparison contract: B=1, 3 active views, 256 patches/view, parameters=torch.bfloat16, vision=torch.bfloat16, prompt IDs identical, chunk=50, 10 Euler steps, CUDA Graph Official=off/PHYAI=on, torch.compile off, and identical input SHA256=6bb16c6d6bfbd4c393f94edd278a99f67b11612ba255c5fe78d5bf1bfe4d2239.

Official label means the official LingBot model code and native Robby MoE, with only the hard-coded FlashAttention2 selector changed to the repository's existing eager-attention config so that it can run on the Thor software stack.
