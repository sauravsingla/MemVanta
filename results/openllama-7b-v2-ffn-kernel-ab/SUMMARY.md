# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Current baseline vs candidate on the exact same runner and verified Q4_0 GGUF; 3 alternating-order pairs.

Hosted-runner regression tolerance: 2.0% for FFN and end-to-end time; pull requests and manual runs remain blocking, while main pushes retain evidence without re-rejecting already-merged code.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 34595.77 ± 434.44 | 33646.02 ± 340.44 | 2.75% |
| Prefill ms | 46007.33 ± 524.70 | 44438.87 ± 241.69 | 3.41% |
| Decode ms | 9822.62 ± 9.65 | 9829.85 ± 15.68 | -0.07% |
| QKV ms | 12826.78 ± 184.59 | 12262.55 ± 106.64 | 4.40% |

- Total improvement: 2.80%
- FFN improvement: 2.75%
- RSS growth: -0.00%
- Gate: **PASS**
- Gate enforcement: **evidence-only on main push**
