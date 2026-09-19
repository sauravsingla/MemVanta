# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | Total ms | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 39893.60 | 10159.10 | 30911.51 | 50052.70 | 3810800 |
| fp32 | 32 | 43411.70 | 10149.30 | 33112.00 | 53561.00 | 3810716 |
| fp32 | 64 | 44268.10 | 10176.50 | 33460.90 | 54444.60 | 3810552 |
| q8act | 32 | 116245.00 | 10133.00 | 81736.59 | 126378.00 | 3810800 |
| q8act | 16 | 116367.00 | 10146.70 | 81759.86 | 126513.70 | 3811000 |
| q8act | 64 | 116393.00 | 10237.70 | 81906.11 | 126630.70 | 3810972 |

Best FP32: batch 16; best Q8-act: batch 32.
Q8-act end-to-end improvement vs best FP32: **-152.49%**.
Q8-act FFN improvement vs best FP32: **-164.42%**.
Q8-act peak-RSS delta: **0.00%**.
Promotion gate (>=3% total, FFN faster, <=2% RSS growth, exact deterministic output): **FAIL**.

A failed promotion gate is valid negative evidence; do not enable Q8 activations by default from this workflow alone.
