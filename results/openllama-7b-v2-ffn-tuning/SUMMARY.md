# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | Total ms | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 39750.50 | 9818.55 | 30500.52 | 49569.05 | 3814972 |
| fp32 | 32 | 44048.20 | 9825.50 | 32659.38 | 53873.70 | 3819216 |
| fp32 | 64 | 44776.90 | 9815.90 | 33456.39 | 54592.80 | 3827336 |
| q8act | 32 | 111182.00 | 9809.21 | 77992.91 | 120991.21 | 3819360 |
| q8act | 16 | 111360.00 | 9813.58 | 78069.06 | 121173.58 | 3815024 |
| q8act | 64 | 111381.00 | 9820.47 | 78159.56 | 121201.47 | 3827972 |

Best FP32: batch 16; best Q8-act: batch 32.
Q8-act end-to-end improvement vs best FP32: **-144.09%**.
Q8-act FFN improvement vs best FP32: **-155.71%**.
Q8-act peak-RSS delta: **0.12%**.
Promotion gate (>=3% total, FFN faster, <=2% RSS growth, exact deterministic output): **FAIL**.

A failed promotion gate is valid negative evidence; do not enable Q8 activations by default from this workflow alone.
