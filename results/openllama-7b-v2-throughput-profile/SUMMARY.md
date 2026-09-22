# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- Prefill: 46018.50 ms
- Decode total: 9814.92 ms
- Projection kernels: 54621.72 ms (97.8% of profiled model time)
- FFN GEMM: 34606.72 ms (63.4% of projection-kernel time)
- Non-GEMM core residual: 1210.99 ms
- Dominant component: ffn_gemm_ms (34606.72 ms)
- Dominant profiled kernel kind: ffn_down (12657.28 ms)
- Peak RSS: 3819244 KiB
- Page faults: major=1, minor=94634
- Process CPU: 232%

This profile selects the next optimization target; it is not a universal throughput claim.
