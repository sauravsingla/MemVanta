# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Current baseline vs candidate on the exact same runner and verified Q4_0 GGUF; 3 alternating-order pairs.

Hosted-runner regression tolerance: 2.0% for FFN and end-to-end time; pull requests and manual runs remain blocking, while main pushes retain evidence without re-rejecting already-merged code.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 28910.11 ± 348.98 | 28255.33 ± 103.99 | 2.26% |
| Prefill ms | 35094.90 ± 486.79 | 34466.37 ± 225.41 | 1.79% |
| Decode ms | 12316.03 ± 86.90 | 12114.97 ± 186.00 | 1.63% |
| QKV ms | 10736.07 ± 119.05 | 10633.41 ± 14.30 | 0.96% |

- Total improvement: 1.75%
- FFN improvement: 2.26%
- RSS growth: -0.00%
- Gate: **PASS**
- Gate enforcement: **evidence-only on main push**
