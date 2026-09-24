# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- Prefill: 45556.00 ms
- Decode total: 8822.29 ms
- Projection kernels: 53112.10 ms (97.7% of profiled model time)
- FFN GEMM: 33916.13 ms (63.9% of projection-kernel time)
- Non-GEMM core residual: 1265.82 ms
- Dominant component: ffn_gemm_ms (33916.13 ms)
- Dominant profiled kernel kind: ffn_down (12165.32 ms)
- Peak RSS: 3819068 KiB
- Page faults: major=1, minor=94647
- Process CPU: 229%

This profile selects the next optimization target; it is not a universal throughput claim.
