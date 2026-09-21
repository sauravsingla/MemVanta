# OpenLLaMA 7B v2 Q4_0 — MemVanta vs llama.cpp

Exact same GGUF on both runtimes. CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 1 warm-up + 5 measured repetitions.

- MemVanta pp: 4.04 ± 0.03 tok/s
- llama.cpp pp: 44.84 ± 0.24 tok/s
- MemVanta tg: 2.60 ± 0.03 tok/s
- llama.cpp tg: 11.97 ± 0.09 tok/s
- MemVanta peak RSS: 3982148 KiB
- llama.cpp peak RSS: 7586296 KiB
- MemVanta RSS reduction: 47.51%
