# OpenLLaMA 3B v2 Q4_0 — MemVanta vs llama.cpp

Generated from the pinned official OpenLLaMA source revision, then benchmarked as the exact same GGUF on both runtimes. CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 5.95 ± 0.01 tok/s
- llama.cpp pp: 24.51 ± 0.02 tok/s
- MemVanta tg: 4.92 ± 0.01 tok/s
- llama.cpp tg: 15.61 ± 0.06 tok/s
- MemVanta peak RSS: 2119452 KiB
- llama.cpp peak RSS: 3885544 KiB
- MemVanta RSS reduction: 45.45%
