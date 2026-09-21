# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Current baseline vs candidate on the exact same runner and verified Q4_0 GGUF; 3 alternating-order pairs.

Hosted-runner regression tolerance: 2.0% for FFN and end-to-end time; pull requests and manual runs remain blocking, while main pushes retain evidence without re-rejecting already-merged code.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 32828.27 ± 963.14 | 32505.60 ± 461.06 | 0.98% |
| Prefill ms | 43207.07 ± 1388.69 | 42706.00 ± 662.80 | 1.16% |
| Decode ms | 10143.07 ± 25.08 | 10165.60 ± 44.12 | -0.22% |
| QKV ms | 12182.14 ± 555.07 | 12064.25 ± 119.58 | 0.97% |

- Total improvement: 0.90%
- FFN improvement: 0.98%
- RSS growth: -0.00%
- Gate: **PASS**
- Gate enforcement: **evidence-only on main push**
