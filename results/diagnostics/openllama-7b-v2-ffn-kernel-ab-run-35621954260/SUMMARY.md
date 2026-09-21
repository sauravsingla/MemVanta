# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Current baseline vs candidate on the exact same runner and verified Q4_0 GGUF; 3 alternating-order pairs.

Hosted-runner regression tolerance: 2.0% for FFN and end-to-end time; candidates still need positive evidence before promotion.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 24902.46 ± 1815.11 | 26860.48 ± 114.72 | -7.86% |
| Prefill ms | 32116.97 ± 2656.80 | 35200.03 ± 152.44 | -9.60% |
| Decode ms | 9657.98 ± 15.74 | 9660.34 ± 14.36 | -0.02% |
| QKV ms | 9292.78 ± 615.03 | 9974.26 ± 26.08 | -7.33% |

- Total improvement: -7.39%
- FFN improvement: -7.86%
- RSS growth: 0.22%
- Gate: **FAIL**
