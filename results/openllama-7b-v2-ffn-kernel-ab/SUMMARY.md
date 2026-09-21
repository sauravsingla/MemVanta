# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Current baseline vs candidate on the exact same runner and verified Q4_0 GGUF; 3 alternating-order pairs.

Hosted-runner regression tolerance: 2.0% for FFN and end-to-end time; pull requests and manual runs remain blocking, while main pushes retain evidence without re-rejecting already-merged code.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 26407.44 ± 1690.23 | 26682.28 ± 2025.02 | -1.04% |
| Prefill ms | 34755.37 ± 2105.01 | 35126.23 ± 2632.04 | -1.07% |
| Decode ms | 8527.65 ± 692.18 | 8566.97 ± 790.39 | -0.46% |
| QKV ms | 9663.97 ± 627.46 | 9737.93 ± 775.86 | -0.77% |

- Total improvement: -0.95%
- FFN improvement: -1.04%
- RSS growth: -0.00%
- Gate: **PASS**
- Gate enforcement: **evidence-only on main push**
