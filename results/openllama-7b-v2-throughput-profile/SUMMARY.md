# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- Prefill: 40354.60 ms
- Decode total: 12727.20 ms
- Projection kernels: 51961.69 ms (97.9% of profiled model time)
- FFN GEMM: 32668.88 ms (62.9% of projection-kernel time)
- Non-GEMM core residual: 1119.21 ms
- Dominant component: ffn_gemm_ms (32668.88 ms)
- Dominant profiled kernel kind: ffn_gate (10953.27 ms)
- Peak RSS: 3810584 KiB
- Page faults: major=1, minor=96112
- Process CPU: 245%

This profile selects the next optimization target; it is not a universal throughput claim.
