# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- Prefill: 45077.80 ms
- Decode total: 9846.46 ms
- Projection kernels: 53702.25 ms (97.8% of profiled model time)
- FFN GEMM: 33270.94 ms (62.0% of projection-kernel time)
- Non-GEMM core residual: 1221.38 ms
- Dominant component: ffn_gemm_ms (33270.94 ms)
- Dominant profiled kernel kind: ffn_up (11119.39 ms)
- Peak RSS: 3819296 KiB
- Page faults: major=1, minor=94646
- Process CPU: 233%

This profile selects the next optimization target; it is not a universal throughput claim.
