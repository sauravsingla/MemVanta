# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- Prefill: 45995.30 ms
- Decode total: 11529.50 ms
- Projection kernels: 56481.95 ms (98.2% of profiled model time)
- FFN GEMM: 35723.08 ms (63.2% of projection-kernel time)
- Non-GEMM core residual: 1042.51 ms
- Dominant component: ffn_gemm_ms (35723.08 ms)
- Dominant profiled kernel kind: ffn_gate (11945.95 ms)
- Peak RSS: 3810664 KiB
- Page faults: major=1, minor=96108
- Process CPU: 237%

This profile selects the next optimization target; it is not a universal throughput claim.
