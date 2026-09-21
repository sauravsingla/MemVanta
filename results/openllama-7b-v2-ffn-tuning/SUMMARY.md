# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | Total ms | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 40104.70 | 10197.50 | 30820.09 | 50302.20 | 3815120 |
| fp32 | 32 | 45255.10 | 10331.20 | 34596.16 | 55586.30 | 3819312 |
| fp32 | 64 | 51119.10 | 10216.10 | 38516.25 | 61335.20 | 3827376 |
| q8act | 16 | 116475.00 | 10139.60 | 81722.09 | 126614.60 | 3815012 |
| q8act | 32 | 116464.00 | 10168.10 | 81755.03 | 126632.10 | 3819404 |
| q8act | 64 | 116827.00 | 10127.20 | 81979.59 | 126954.20 | 3828072 |

Best FP32: batch 16; best Q8-act: batch 16.
Q8-act end-to-end improvement vs best FP32: **-151.71%**.
Q8-act FFN improvement vs best FP32: **-165.16%**.
Q8-act peak-RSS delta: **-0.00%**.
Promotion gate (>=3% total, FFN faster, <=2% RSS growth, exact deterministic output): **FAIL**.

A failed promotion gate is valid negative evidence; do not enable Q8 activations by default from this workflow alone.
