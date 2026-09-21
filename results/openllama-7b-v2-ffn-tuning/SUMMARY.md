# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | Total ms | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 40355.20 | 11030.00 | 30206.05 | 51385.20 | 3814976 |
| fp32 | 32 | 42543.10 | 10931.00 | 32362.66 | 53474.10 | 3819112 |
| fp32 | 64 | 46876.90 | 10908.10 | 37356.83 | 57785.00 | 3827284 |
| q8act | 32 | 114878.00 | 10894.40 | 80384.69 | 125772.40 | 3819464 |
| q8act | 64 | 114896.00 | 10958.00 | 80444.38 | 125854.00 | 3828216 |
| q8act | 16 | 115137.00 | 10938.50 | 80508.91 | 126075.50 | 3815116 |

Best FP32: batch 16; best Q8-act: batch 32.
Q8-act end-to-end improvement vs best FP32: **-144.76%**.
Q8-act FFN improvement vs best FP32: **-166.12%**.
Q8-act peak-RSS delta: **0.12%**.
Promotion gate (>=3% total, FFN faster, <=2% RSS growth, exact deterministic output): **FAIL**.

A failed promotion gate is valid negative evidence; do not enable Q8 activations by default from this workflow alone.
