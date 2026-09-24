# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- Prefill: 33404.60 ms
- Decode total: 7135.63 ms
- Projection kernels: 39542.58 ms (97.5% of profiled model time)
- FFN GEMM: 24709.14 ms (62.5% of projection-kernel time)
- Non-GEMM core residual: 997.18 ms
- Dominant component: ffn_gemm_ms (24709.14 ms)
- Dominant profiled kernel kind: ffn_down (8686.54 ms)
- Peak RSS: 3819152 KiB
- Page faults: major=1, minor=94654
- Process CPU: 230%

This profile selects the next optimization target; it is not a universal throughput claim.
