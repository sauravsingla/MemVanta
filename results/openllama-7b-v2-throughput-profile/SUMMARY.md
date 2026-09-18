# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- Prefill: 49100.40 ms
- Decode total: 11414.10 ms
- Projection kernels: 59513.99 ms (98.3% of profiled model time)
- FFN GEMM: 37530.26 ms (63.1% of projection-kernel time)
- Non-GEMM core residual: 1000.18 ms
- Dominant component: ffn_gemm_ms (37530.26 ms)
- Dominant profiled kernel kind: ffn_gate (12520.54 ms)
- Peak RSS: 3810740 KiB
- Page faults: major=1, minor=96107
- Process CPU: 235%

This profile selects the next optimization target; it is not a universal throughput claim.
