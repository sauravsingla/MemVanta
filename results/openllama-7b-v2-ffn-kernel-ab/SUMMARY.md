# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Current baseline vs candidate on the exact same runner and verified Q4_0 GGUF; 3 alternating-order pairs.

Hosted-runner regression tolerance: 2.0% for FFN and end-to-end time; pull requests and manual runs remain blocking, while main pushes retain evidence without re-rejecting already-merged code.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 32664.26 ± 94.49 | 32400.58 ± 16.90 | 0.81% |
| Prefill ms | 43284.83 ± 93.15 | 43102.90 ± 72.28 | 0.42% |
| Decode ms | 9929.38 ± 18.78 | 9649.24 ± 22.85 | 2.82% |
| QKV ms | 11873.72 ± 35.91 | 11783.61 ± 40.00 | 0.76% |

- Total improvement: 0.87%
- FFN improvement: 0.81%
- RSS growth: -0.00%
- Gate: **PASS**
- Gate enforcement: **evidence-only on main push**
