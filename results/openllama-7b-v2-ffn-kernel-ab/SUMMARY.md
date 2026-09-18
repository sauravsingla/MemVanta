# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Current baseline vs candidate on the exact same runner and verified Q4_0 GGUF; 3 alternating-order pairs.

Hosted-runner regression tolerance: 2.0% for FFN and end-to-end time; candidates still need positive evidence before promotion.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 28004.97 ± 14.23 | 27968.82 ± 217.54 | 0.13% |
| Prefill ms | 35795.03 ± 62.54 | 35805.17 ± 275.50 | -0.03% |
| Decode ms | 9765.36 ± 20.89 | 9783.39 ± 2.58 | -0.18% |
| QKV ms | 10307.86 ± 22.19 | 10339.20 ± 49.51 | -0.30% |

- Total improvement: -0.06%
- FFN improvement: 0.13%
- RSS growth: 0.00%
- Gate: **PASS**
