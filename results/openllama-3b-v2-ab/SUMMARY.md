# OpenLLaMA 3B v2 Q4_0 — MemVanta vs llama.cpp

Generated from the pinned official OpenLLaMA source revision, then benchmarked as the exact same GGUF on both runtimes. CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 7.49 ± 0.00 tok/s
- llama.cpp pp: 33.63 ± 0.03 tok/s
- MemVanta tg: 6.53 ± 0.02 tok/s
- llama.cpp tg: 17.93 ± 0.11 tok/s
- MemVanta peak RSS: 2119468 KiB
- llama.cpp peak RSS: 3883816 KiB
- MemVanta RSS reduction: 45.43%
