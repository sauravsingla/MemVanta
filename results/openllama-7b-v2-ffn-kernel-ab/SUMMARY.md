# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Current baseline vs candidate on the exact same runner and verified Q4_0 GGUF; 3 alternating-order pairs.

Hosted-runner regression tolerance: 2.0% for FFN and end-to-end time; pull requests and manual runs remain blocking, while main pushes retain evidence without re-rejecting already-merged code.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 33961.87 ± 194.65 | 34402.30 ± 483.18 | -1.30% |
| Prefill ms | 45058.03 ± 504.11 | 45553.97 ± 962.06 | -1.10% |
| Decode ms | 9913.22 ± 47.41 | 9906.61 ± 69.08 | 0.07% |
| QKV ms | 12557.90 ± 337.83 | 12645.35 ± 287.26 | -0.70% |

- Total improvement: -0.89%
- FFN improvement: -1.30%
- RSS growth: 0.00%
- Gate: **PASS**
- Gate enforcement: **evidence-only on main push**
