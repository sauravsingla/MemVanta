# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | Total ms | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 39105.20 | 12518.90 | 31728.92 | 51624.10 | 3810812 |
| fp32 | 32 | 39705.30 | 12693.80 | 32282.83 | 52399.10 | 3810716 |
| fp32 | 64 | 45177.30 | 12669.50 | 36745.95 | 57846.80 | 3810632 |
| q8act | 64 | 94010.60 | 12673.70 | 68588.88 | 106684.30 | 3810776 |
| q8act | 32 | 96009.00 | 12493.90 | 69602.72 | 108502.90 | 3810732 |
| q8act | 16 | 98821.90 | 12598.70 | 71602.64 | 111420.60 | 3811068 |

Best FP32: batch 16; best Q8-act: batch 64.
Q8-act end-to-end improvement vs best FP32: **-106.66%**.
Q8-act FFN improvement vs best FP32: **-116.17%**.
Q8-act peak-RSS delta: **-0.00%**.
Promotion gate (>=3% total, FFN faster, <=2% RSS growth, exact deterministic output): **FAIL**.

A failed promotion gate is valid negative evidence; do not enable Q8 activations by default from this workflow alone.
