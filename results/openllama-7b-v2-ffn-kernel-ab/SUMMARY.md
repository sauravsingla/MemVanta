# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Current baseline vs candidate on the exact same runner and verified Q4_0 GGUF; 3 alternating-order pairs.

Hosted-runner regression tolerance: 2.0% for FFN and end-to-end time; pull requests and manual runs remain blocking, while main pushes retain evidence without re-rejecting already-merged code.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 33428.51 ± 1231.99 | 33776.73 ± 744.04 | -1.04% |
| Prefill ms | 44818.17 ± 1893.85 | 45401.73 ± 1179.28 | -1.30% |
| Decode ms | 8819.35 ± 32.31 | 8790.34 ± 3.69 | 0.33% |
| QKV ms | 11870.29 ± 595.42 | 12058.72 ± 442.92 | -1.59% |

- Total improvement: -1.03%
- FFN improvement: -1.04%
- RSS growth: 0.00%
- Gate: **PASS**
- Gate enforcement: **evidence-only on main push**
