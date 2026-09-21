# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- Prefill: 44338.00 ms
- Decode total: 10215.40 ms
- Projection kernels: 53250.32 ms (97.6% of profiled model time)
- FFN GEMM: 33760.73 ms (63.4% of projection-kernel time)
- Non-GEMM core residual: 1302.68 ms
- Dominant component: ffn_gemm_ms (33760.73 ms)
- Dominant profiled kernel kind: ffn_down (12046.80 ms)
- Peak RSS: 3819260 KiB
- Page faults: major=1, minor=94644
- Process CPU: 234%

This profile selects the next optimization target; it is not a universal throughput claim.
