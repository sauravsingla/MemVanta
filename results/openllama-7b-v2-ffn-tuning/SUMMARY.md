# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | Total ms | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 39766.00 | 9037.73 | 29728.59 | 48803.73 | 3815208 |
| fp32 | 32 | 41973.20 | 8823.41 | 31096.97 | 50796.61 | 3818940 |
| fp32 | 64 | 44064.70 | 8797.16 | 32258.88 | 52861.86 | 3827260 |
| q8act | 32 | 90576.30 | 8823.15 | 63436.67 | 99399.45 | 3819616 |
| q8act | 64 | 90776.30 | 8785.58 | 63598.64 | 99561.88 | 3828104 |
| q8act | 16 | 90973.30 | 8874.68 | 63686.41 | 99847.98 | 3815260 |

Best FP32: batch 16; best Q8-act: batch 32.
Q8-act end-to-end improvement vs best FP32: **-103.67%**.
Q8-act FFN improvement vs best FP32: **-113.39%**.
Q8-act peak-RSS delta: **0.12%**.
Promotion gate (>=3% total, FFN faster, <=2% RSS growth, exact deterministic output): **FAIL**.

A failed promotion gate is valid negative evidence; do not enable Q8 activations by default from this workflow alone.
