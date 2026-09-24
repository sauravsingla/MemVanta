# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Current baseline vs candidate on the exact same runner and verified Q4_0 GGUF; 3 alternating-order pairs.

Hosted-runner regression tolerance: 2.0% for FFN and end-to-end time; pull requests and manual runs remain blocking, while main pushes retain evidence without re-rejecting already-merged code.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 30149.41 ± 415.45 | 30324.20 ± 98.86 | -0.58% |
| Prefill ms | 41268.93 ± 443.07 | 41427.67 ± 119.58 | -0.38% |
| Decode ms | 9424.99 ± 19.45 | 9456.84 ± 46.90 | -0.34% |
| QKV ms | 11580.06 ± 6.17 | 11593.35 ± 20.45 | -0.11% |

- Total improvement: -0.38%
- FFN improvement: -0.58%
- RSS growth: 0.00%
- Gate: **PASS**
- Gate enforcement: **evidence-only on main push**
