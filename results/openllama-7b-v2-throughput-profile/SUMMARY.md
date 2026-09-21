# OpenLLaMA 7B v2 throughput profile

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, batch 32, F16 KV.

- Prefill: 45459.70 ms
- Decode total: 10154.40 ms
- Projection kernels: 54338.36 ms (97.7% of profiled model time)
- FFN GEMM: 34775.31 ms (64.0% of projection-kernel time)
- Non-GEMM core residual: 1275.41 ms
- Dominant component: ffn_gemm_ms (34775.31 ms)
- Dominant profiled kernel kind: ffn_down (12466.42 ms)
- Peak RSS: 3819004 KiB
- Page faults: major=1, minor=94650
- Process CPU: 233%

This profile selects the next optimization target; it is not a universal throughput claim.
