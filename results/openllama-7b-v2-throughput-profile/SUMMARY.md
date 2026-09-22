# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- Prefill: 29626.40 ms
- Decode total: 5839.14 ms
- Projection kernels: 34641.25 ms (97.7% of profiled model time)
- FFN GEMM: 21285.58 ms (61.4% of projection-kernel time)
- Non-GEMM core residual: 823.91 ms
- Dominant component: ffn_gemm_ms (21285.58 ms)
- Dominant profiled kernel kind: ffn_down (7152.84 ms)
- Peak RSS: 3819332 KiB
- Page faults: major=1, minor=94654
- Process CPU: 228%

This profile selects the next optimization target; it is not a universal throughput claim.
