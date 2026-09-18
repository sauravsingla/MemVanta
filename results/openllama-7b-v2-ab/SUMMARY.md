# OpenLLaMA 7B v2 Q4_0 — MemVanta vs llama.cpp

Exact same GGUF on both runtimes. CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 1 warm-up + 5 measured repetitions.

- MemVanta pp: 2.74 ± 0.03 tok/s
- llama.cpp pp: 12.83 ± 0.02 tok/s
- MemVanta tg: 1.38 ± 0.00 tok/s
- llama.cpp tg: 8.08 ± 0.04 tok/s
- MemVanta peak RSS: 3983208 KiB
- llama.cpp peak RSS: 7591032 KiB
- MemVanta RSS reduction: 47.53%
