# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | Total ms | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 46974.30 | 11533.70 | 36085.58 | 58508.00 | 3810868 |
| fp32 | 64 | 48028.50 | 11430.90 | 36815.25 | 59459.40 | 3810648 |
| fp32 | 32 | 48348.30 | 11413.90 | 37013.00 | 59762.20 | 3810512 |
| q8act | 32 | 136066.00 | 11397.60 | 95731.61 | 147463.60 | 3810584 |
| q8act | 64 | 136197.00 | 11498.10 | 95865.30 | 147695.10 | 3810772 |
| q8act | 16 | 136258.00 | 11475.50 | 95791.11 | 147733.50 | 3811128 |

Best FP32: batch 16; best Q8-act: batch 32.
Q8-act end-to-end improvement vs best FP32: **-152.04%**.
Q8-act FFN improvement vs best FP32: **-165.29%**.
Q8-act peak-RSS delta: **-0.01%**.
Promotion gate (>=3% total, FFN faster, <=2% RSS growth, exact deterministic output): **FAIL**.

A failed promotion gate is valid negative evidence; do not enable Q8 activations by default from this workflow alone.
