# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- Prefill: 43074.30 ms
- Decode total: 9646.83 ms
- Projection kernels: 51482.42 ms (97.7% of profiled model time)
- FFN GEMM: 32365.65 ms (62.9% of projection-kernel time)
- Non-GEMM core residual: 1238.18 ms
- Dominant component: ffn_gemm_ms (32365.65 ms)
- Dominant profiled kernel kind: ffn_down (11424.83 ms)
- Peak RSS: 3819208 KiB
- Page faults: major=1, minor=94647
- Process CPU: 233%

This profile selects the next optimization target; it is not a universal throughput claim.
