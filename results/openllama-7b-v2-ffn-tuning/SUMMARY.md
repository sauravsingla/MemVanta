# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | Total ms | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 40730.00 | 8808.82 | 30232.86 | 49538.82 | 3815080 |
| fp32 | 32 | 47270.40 | 8809.15 | 34827.91 | 56079.55 | 3819120 |
| fp32 | 64 | 50723.30 | 8811.68 | 37289.88 | 59534.98 | 3827392 |
| q8act | 32 | 90703.90 | 8812.77 | 63542.89 | 99516.67 | 3819428 |
| q8act | 64 | 90817.10 | 8791.19 | 63623.56 | 99608.29 | 3828156 |
| q8act | 16 | 90949.20 | 8810.08 | 63649.81 | 99759.28 | 3815196 |

Best FP32: batch 16; best Q8-act: batch 32.
Q8-act end-to-end improvement vs best FP32: **-100.89%**.
Q8-act FFN improvement vs best FP32: **-110.18%**.
Q8-act peak-RSS delta: **0.11%**.
Promotion gate (>=3% total, FFN faster, <=2% RSS growth, exact deterministic output): **FAIL**.

A failed promotion gate is valid negative evidence; do not enable Q8 activations by default from this workflow alone.
