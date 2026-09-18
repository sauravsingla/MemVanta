# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- Prefill: 46466.10 ms
- Decode total: 12743.40 ms
- Projection kernels: 58157.74 ms (98.2% of profiled model time)
- FFN GEMM: 36391.32 ms (62.6% of projection-kernel time)
- Non-GEMM core residual: 1051.17 ms
- Dominant component: ffn_gemm_ms (36391.32 ms)
- Dominant profiled kernel kind: ffn_down (12638.74 ms)
- Peak RSS: 3810676 KiB
- Page faults: major=1, minor=97616
- Process CPU: 240%

This profile selects the next optimization target; it is not a universal throughput claim.
