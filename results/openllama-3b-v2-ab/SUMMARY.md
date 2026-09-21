# OpenLLaMA 3B v2 Q4_0 — MemVanta vs llama.cpp

Generated from the pinned official OpenLLaMA source revision, then benchmarked as the exact same GGUF on both runtimes. CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 5.78 ± 0.00 tok/s
- llama.cpp pp: 22.80 ± 0.00 tok/s
- MemVanta tg: 4.29 ± 0.01 tok/s
- llama.cpp tg: 15.29 ± 0.01 tok/s
- MemVanta peak RSS: 2119568 KiB
- llama.cpp peak RSS: 3882248 KiB
- MemVanta RSS reduction: 45.40%
