# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- Prefill: 33373.00 ms
- Decode total: 6926.82 ms
- Projection kernels: 39319.92 ms (97.6% of profiled model time)
- FFN GEMM: 24547.34 ms (62.4% of projection-kernel time)
- Non-GEMM core residual: 979.38 ms
- Dominant component: ffn_gemm_ms (24547.34 ms)
- Dominant profiled kernel kind: ffn_down (8612.96 ms)
- Peak RSS: 3819204 KiB
- Page faults: major=1, minor=94652
- Process CPU: 230%

This profile selects the next optimization target; it is not a universal throughput claim.
