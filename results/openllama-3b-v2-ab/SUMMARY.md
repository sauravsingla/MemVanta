# OpenLLaMA 3B v2 Q4_0 — MemVanta vs llama.cpp

Generated from the pinned official OpenLLaMA source revision, then benchmarked as the exact same GGUF on both runtimes. CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 5.86 ± 0.01 tok/s
- llama.cpp pp: 22.80 ± 0.01 tok/s
- MemVanta tg: 5.47 ± 0.02 tok/s
- llama.cpp tg: 15.08 ± 0.07 tok/s
- MemVanta peak RSS: 2119488 KiB
- llama.cpp peak RSS: 3885124 KiB
- MemVanta RSS reduction: 45.45%
