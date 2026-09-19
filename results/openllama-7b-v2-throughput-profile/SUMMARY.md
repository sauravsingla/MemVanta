# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- Prefill: 43742.30 ms
- Decode total: 10174.70 ms
- Projection kernels: 52884.97 ms (98.1% of profiled model time)
- FFN GEMM: 33367.51 ms (63.1% of projection-kernel time)
- Non-GEMM core residual: 1031.47 ms
- Dominant component: ffn_gemm_ms (33367.51 ms)
- Dominant profiled kernel kind: ffn_up (11127.05 ms)
- Peak RSS: 3810540 KiB
- Page faults: major=1, minor=96103
- Process CPU: 235%

This profile selects the next optimization target; it is not a universal throughput claim.
