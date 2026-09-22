# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | Total ms | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 32696.80 | 8114.19 | 24619.65 | 40810.99 | 3815040 |
| fp32 | 32 | 33484.40 | 8144.31 | 25422.30 | 41628.71 | 3819136 |
| fp32 | 64 | 36115.90 | 8120.99 | 27006.84 | 44236.89 | 3827488 |
| q8act | 32 | 86856.90 | 8126.49 | 60946.49 | 94983.39 | 3819420 |
| q8act | 64 | 86872.00 | 8138.51 | 60979.56 | 95010.51 | 3828328 |
| q8act | 16 | 86970.20 | 8121.53 | 60994.28 | 95091.73 | 3815112 |

Best FP32: batch 16; best Q8-act: batch 32.
Q8-act end-to-end improvement vs best FP32: **-132.74%**.
Q8-act FFN improvement vs best FP32: **-147.55%**.
Q8-act peak-RSS delta: **0.11%**.
Promotion gate (>=3% total, FFN faster, <=2% RSS growth, exact deterministic output): **FAIL**.

A failed promotion gate is valid negative evidence; do not enable Q8 activations by default from this workflow alone.
