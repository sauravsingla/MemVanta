# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- Prefill: 41977.60 ms
- Decode total: 9781.51 ms
- Projection kernels: 50146.02 ms (96.9% of profiled model time)
- FFN GEMM: 31012.47 ms (61.8% of projection-kernel time)
- Non-GEMM core residual: 1612.45 ms
- Dominant component: ffn_gemm_ms (31012.47 ms)
- Dominant profiled kernel kind: ffn_down (10764.57 ms)
- Peak RSS: 3819028 KiB
- Page faults: major=1, minor=94655
- Process CPU: 235%

This profile selects the next optimization target; it is not a universal throughput claim.
