# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | Total ms | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 40744.60 | 8820.89 | 30615.26 | 49565.49 | 3815048 |
| fp32 | 32 | 44940.70 | 8846.77 | 33274.91 | 53787.47 | 3818984 |
| fp32 | 64 | 51505.90 | 8966.68 | 37905.81 | 60472.58 | 3827204 |
| q8act | 32 | 90685.90 | 8810.94 | 63494.41 | 99496.84 | 3819420 |
| q8act | 16 | 90925.40 | 8786.65 | 63623.23 | 99712.05 | 3815020 |
| q8act | 64 | 90914.90 | 8813.20 | 63707.33 | 99728.10 | 3828052 |

Best FP32: batch 16; best Q8-act: batch 32.
Q8-act end-to-end improvement vs best FP32: **-100.74%**.
Q8-act FFN improvement vs best FP32: **-107.39%**.
Q8-act peak-RSS delta: **0.11%**.
Promotion gate (>=3% total, FFN faster, <=2% RSS growth, exact deterministic output): **FAIL**.

A failed promotion gate is valid negative evidence; do not enable Q8 activations by default from this workflow alone.
