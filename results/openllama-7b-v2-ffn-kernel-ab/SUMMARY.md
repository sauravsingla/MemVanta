# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Current baseline vs candidate on the exact same runner and verified Q4_0 GGUF; 3 alternating-order pairs.

Hosted-runner regression tolerance: 2.0% for FFN and end-to-end time; pull requests and manual runs remain blocking, while main pushes retain evidence without re-rejecting already-merged code.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 28873.93 ± 145.90 | 28263.08 ± 379.42 | 2.12% |
| Prefill ms | 35364.17 ± 299.54 | 35209.23 ± 489.69 | 0.44% |
| Decode ms | 11846.57 ± 115.56 | 11057.63 ± 52.21 | 6.66% |
| QKV ms | 10735.92 ± 88.15 | 10540.87 ± 82.91 | 1.82% |

- Total improvement: 2.00%
- FFN improvement: 2.12%
- RSS growth: 0.00%
- Gate: **PASS**
- Gate enforcement: **evidence-only on main push**
