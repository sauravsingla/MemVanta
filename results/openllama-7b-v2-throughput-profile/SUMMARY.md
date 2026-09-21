# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- Prefill: 37849.50 ms
- Decode total: 12762.20 ms
- Projection kernels: 49214.95 ms (97.2% of profiled model time)
- FFN GEMM: 30917.27 ms (62.8% of projection-kernel time)
- Non-GEMM core residual: 1395.60 ms
- Dominant component: ffn_gemm_ms (30917.27 ms)
- Dominant profiled kernel kind: ffn_gate (10358.18 ms)
- Peak RSS: 3819128 KiB
- Page faults: major=1, minor=94657
- Process CPU: 247%

This profile selects the next optimization target; it is not a universal throughput claim.
