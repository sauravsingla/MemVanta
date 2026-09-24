# OpenLLaMA 7B FFN kernel repeated same-runner A/B

Current baseline vs candidate on the exact same runner and verified Q4_0 GGUF; 3 alternating-order pairs.

Hosted-runner regression tolerance: 2.0% for FFN and end-to-end time; pull requests and manual runs remain blocking, while main pushes retain evidence without re-rejecting already-merged code.

| Metric | Baseline mean ± SD | Candidate mean ± SD | Improvement |
|---|---:|---:|---:|
| FFN GEMM ms | 32793.97 ± 551.68 | 33336.66 ± 848.03 | -1.65% |
| Prefill ms | 44232.73 ± 374.61 | 45168.50 ± 1135.96 | -2.12% |
| Decode ms | 8820.69 ± 19.73 | 8867.98 ± 65.92 | -0.54% |
| QKV ms | 11968.51 ± 345.06 | 12204.07 ± 446.52 | -1.97% |

- Total improvement: -1.85%
- FFN improvement: -1.65%
- RSS growth: -0.00%
- Gate: **PASS**
- Gate enforcement: **evidence-only on main push**
