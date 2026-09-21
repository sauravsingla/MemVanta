# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | Total ms | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 39752.50 | 10168.90 | 30604.68 | 49921.40 | 3814956 |
| fp32 | 32 | 44087.10 | 9860.85 | 32690.05 | 53947.95 | 3819336 |
| fp32 | 64 | 44266.70 | 9920.08 | 32884.52 | 54186.78 | 3827316 |
| q8act | 64 | 111382.00 | 9818.37 | 78155.28 | 121200.37 | 3828048 |
| q8act | 32 | 111349.00 | 9916.16 | 78139.76 | 121265.16 | 3819388 |
| q8act | 16 | 111466.00 | 9920.24 | 78161.11 | 121386.24 | 3815068 |

Best FP32: batch 16; best Q8-act: batch 64.
Q8-act end-to-end improvement vs best FP32: **-142.78%**.
Q8-act FFN improvement vs best FP32: **-155.37%**.
Q8-act peak-RSS delta: **0.34%**.
Promotion gate (>=3% total, FFN faster, <=2% RSS growth, exact deterministic output): **FAIL**.

A failed promotion gate is valid negative evidence; do not enable Q8 activations by default from this workflow alone.
