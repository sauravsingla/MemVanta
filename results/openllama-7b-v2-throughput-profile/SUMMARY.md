# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- Prefill: 30126.50 ms
- Decode total: 4951.60 ms
- Projection kernels: 34171.12 ms (97.4% of profiled model time)
- FFN GEMM: 21079.83 ms (61.7% of projection-kernel time)
- Non-GEMM core residual: 906.67 ms
- Dominant component: ffn_gemm_ms (21079.83 ms)
- Dominant profiled kernel kind: ffn_up (7195.00 ms)
- Peak RSS: 3819032 KiB
- Page faults: major=1, minor=94654
- Process CPU: 224%

This profile selects the next optimization target; it is not a universal throughput claim.
