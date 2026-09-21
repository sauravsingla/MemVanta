# OpenLLaMA 7B FFN tuning

Exact verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg16, F16 KV.

| Mode | Batch | Prefill ms | Decode ms | FFN ms | Total ms | Peak RSS KiB |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 16 | 40580.00 | 10159.80 | 31276.61 | 50739.80 | 3815076 |
| fp32 | 32 | 44863.80 | 10186.40 | 34470.52 | 55050.20 | 3818968 |
| fp32 | 64 | 50319.30 | 10156.90 | 37807.84 | 60476.20 | 3827296 |
| q8act | 32 | 116569.00 | 10128.40 | 81781.47 | 126697.40 | 3819364 |
| q8act | 16 | 116571.00 | 10148.60 | 81759.86 | 126719.60 | 3814996 |
| q8act | 64 | 116734.00 | 10163.30 | 81918.24 | 126897.30 | 3828084 |

Best FP32: batch 16; best Q8-act: batch 32.
Q8-act end-to-end improvement vs best FP32: **-149.70%**.
Q8-act FFN improvement vs best FP32: **-161.48%**.
Q8-act peak-RSS delta: **0.11%**.
Promotion gate (>=3% total, FFN faster, <=2% RSS growth, exact deterministic output): **FAIL**.

A failed promotion gate is valid negative evidence; do not enable Q8 activations by default from this workflow alone.
