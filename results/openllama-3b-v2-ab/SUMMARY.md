# OpenLLaMA 3B v2 Q4_0 — MemVanta vs llama.cpp

Generated from the pinned official OpenLLaMA source revision, then benchmarked as the exact same GGUF on both runtimes. CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 6.99 ± 0.02 tok/s
- llama.cpp pp: 45.84 ± 1.21 tok/s
- MemVanta tg: 3.39 ± 0.03 tok/s
- llama.cpp tg: 15.31 ± 0.11 tok/s
- MemVanta peak RSS: 2119468 KiB
- llama.cpp peak RSS: 3895576 KiB
- MemVanta RSS reduction: 45.59%
