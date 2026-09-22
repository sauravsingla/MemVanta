# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Current baseline vs candidate on the exact same runner and verified Q4_0 GGUF; 3 alternating-order pairs.

Hosted-runner regression tolerance: 2.0% for FFN and end-to-end time; pull requests and manual runs remain blocking, while main pushes retain evidence without re-rejecting already-merged code.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 32279.91 ± 606.13 | 32683.12 ± 386.06 | -1.25% |
| Prefill ms | 43230.70 ± 559.67 | 43707.73 ± 825.99 | -1.10% |
| Decode ms | 9858.64 ± 32.45 | 9818.62 ± 20.07 | 0.41% |
| QKV ms | 12597.71 ± 84.37 | 12561.37 ± 333.37 | 0.29% |

- Total improvement: -0.82%
- FFN improvement: -1.25%
- RSS growth: 0.00%
- Gate: **PASS**
- Gate enforcement: **evidence-only on main push**
