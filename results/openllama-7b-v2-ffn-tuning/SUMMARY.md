# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | Total ms | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 28359.50 | 5255.33 | 18314.42 | 33614.83 | 3815216 |
| fp32 | 64 | 31854.70 | 5212.61 | 22340.32 | 37067.31 | 3827572 |
| fp32 | 32 | 31760.70 | 5384.54 | 22375.86 | 37145.24 | 3819312 |
| q8act | 64 | 50545.40 | 5229.34 | 35368.86 | 55774.74 | 3828260 |
| q8act | 16 | 51012.50 | 5177.32 | 35585.90 | 56189.82 | 3815340 |
| q8act | 32 | 51003.30 | 5305.56 | 35673.82 | 56308.86 | 3819584 |

Best FP32: batch 16; best Q8-act: batch 64.
Q8-act end-to-end improvement vs best FP32: **-65.92%**.
Q8-act FFN improvement vs best FP32: **-93.12%**.
Q8-act peak-RSS delta: **0.34%**.
Promotion gate (>=3% total, FFN faster, <=2% RSS growth, exact deterministic output): **FAIL**.

A failed promotion gate is valid negative evidence; do not enable Q8 activations by default from this workflow alone.
