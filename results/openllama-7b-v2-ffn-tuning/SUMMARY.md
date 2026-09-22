# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | Total ms | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 39982.20 | 8893.28 | 29779.04 | 48875.48 | 3815080 |
| fp32 | 32 | 43893.40 | 8794.09 | 31928.90 | 52687.49 | 3819212 |
| fp32 | 64 | 44411.60 | 8793.20 | 32401.11 | 53204.80 | 3827312 |
| q8act | 32 | 90574.00 | 8808.79 | 63459.20 | 99382.79 | 3819404 |
| q8act | 64 | 90819.90 | 8798.17 | 63654.69 | 99618.07 | 3828380 |
| q8act | 16 | 90886.50 | 8827.96 | 63613.15 | 99714.46 | 3815144 |

Best FP32: batch 16; best Q8-act: batch 32.
Q8-act end-to-end improvement vs best FP32: **-103.34%**.
Q8-act FFN improvement vs best FP32: **-113.10%**.
Q8-act peak-RSS delta: **0.11%**.
Promotion gate (>=3% total, FFN faster, <=2% RSS growth, exact deterministic output): **FAIL**.

A failed promotion gate is valid negative evidence; do not enable Q8 activations by default from this workflow alone.
