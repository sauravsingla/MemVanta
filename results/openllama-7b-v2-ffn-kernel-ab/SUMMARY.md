# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Current baseline vs candidate on the exact same runner and verified Q4_0 GGUF; 3 alternating-order pairs.

Hosted-runner regression tolerance: 2.0% for FFN and end-to-end time; pull requests and manual runs remain blocking, while main pushes retain evidence without re-rejecting already-merged code.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 32891.61 ± 828.82 | 33270.17 ± 774.12 | -1.15% |
| Prefill ms | 44000.40 ± 1025.45 | 45224.03 ± 1171.18 | -2.78% |
| Decode ms | 8813.46 ± 10.11 | 8812.26 ± 26.11 | 0.01% |
| QKV ms | 11548.82 ± 238.67 | 12244.32 ± 394.45 | -6.02% |

- Total improvement: -2.31%
- FFN improvement: -1.15%
- RSS growth: -0.00%
- Gate: **FAIL**
- Gate enforcement: **evidence-only on main push**
