# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | Total ms | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 40108.10 | 9822.91 | 30798.15 | 49931.01 | 3814936 |
| fp32 | 32 | 44040.80 | 9831.44 | 33715.07 | 53872.24 | 3819308 |
| fp32 | 64 | 50195.70 | 9820.66 | 37743.08 | 60016.36 | 3827372 |
| q8act | 32 | 111257.00 | 9855.77 | 78063.10 | 121112.77 | 3819500 |
| q8act | 64 | 111396.00 | 9829.79 | 78164.00 | 121225.79 | 3828080 |
| q8act | 16 | 111412.00 | 9825.23 | 78103.22 | 121237.23 | 3814996 |

Best FP32: batch 16; best Q8-act: batch 32.
Q8-act end-to-end improvement vs best FP32: **-142.56%**.
Q8-act FFN improvement vs best FP32: **-153.47%**.
Q8-act peak-RSS delta: **0.12%**.
Promotion gate (>=3% total, FFN faster, <=2% RSS growth, exact deterministic output): **FAIL**.

A failed promotion gate is valid negative evidence; do not enable Q8 activations by default from this workflow alone.
