# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Current baseline vs candidate on the exact same runner and verified Q4_0 GGUF; 3 alternating-order pairs.

Hosted-runner regression tolerance: 2.0% for FFN and end-to-end time; candidates still need positive evidence before promotion.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 38144.69 ± 950.17 | 34546.92 ± 841.01 | 9.43% |
| Prefill ms | 49160.67 ± 1660.74 | 45061.27 ± 1305.38 | 8.34% |
| Decode ms | 11429.40 ± 13.85 | 10168.23 ± 28.12 | 11.03% |
| QKV ms | 13800.10 ± 507.86 | 12565.92 ± 386.59 | 8.94% |

- Total improvement: 8.85%
- FFN improvement: 9.43%
- RSS growth: 0.00%
- Gate: **PASS**
