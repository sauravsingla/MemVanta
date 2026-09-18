# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | Total ms | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 43204.50 | 11501.50 | 33645.71 | 54706.00 | 3810860 |
| fp32 | 32 | 48386.60 | 11485.50 | 37064.20 | 59872.10 | 3810620 |
| fp32 | 64 | 48679.50 | 11496.60 | 37087.57 | 60176.10 | 3810636 |
| q8act | 32 | 136246.00 | 11407.50 | 95869.22 | 147653.50 | 3810876 |
| q8act | 64 | 136329.00 | 11415.70 | 95965.39 | 147744.70 | 3810756 |
| q8act | 16 | 136404.00 | 11416.20 | 95898.92 | 147820.20 | 3811064 |

Best FP32: batch 16; best Q8-act: batch 32.
Q8-act end-to-end improvement vs best FP32: **-169.90%**.
Q8-act FFN improvement vs best FP32: **-184.94%**.
Q8-act peak-RSS delta: **0.00%**.
Promotion gate (>=3% total, FFN faster, <=2% RSS growth, exact deterministic output): **FAIL**.

A failed promotion gate is valid negative evidence; do not enable Q8 activations by default from this workflow alone.
