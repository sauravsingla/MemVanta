# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | Total ms | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 39255.20 | 12862.30 | 32017.43 | 52117.50 | 3811000 |
| fp32 | 32 | 43051.40 | 12686.20 | 34482.24 | 55737.60 | 3810612 |
| fp32 | 64 | 44248.30 | 12829.50 | 36403.14 | 57077.80 | 3810412 |
| q8act | 64 | 93294.10 | 12619.50 | 68021.73 | 105913.60 | 3810864 |
| q8act | 32 | 95809.50 | 12480.20 | 69446.79 | 108289.70 | 3810760 |
| q8act | 16 | 100511.00 | 12518.30 | 72565.61 | 113029.30 | 3810936 |

Best FP32: batch 16; best Q8-act: batch 64.
Q8-act end-to-end improvement vs best FP32: **-103.22%**.
Q8-act FFN improvement vs best FP32: **-112.45%**.
Q8-act peak-RSS delta: **-0.00%**.
Promotion gate (>=3% total, FFN faster, <=2% RSS growth, exact deterministic output): **FAIL**.

A failed promotion gate is valid negative evidence; do not enable Q8 activations by default from this workflow alone.
