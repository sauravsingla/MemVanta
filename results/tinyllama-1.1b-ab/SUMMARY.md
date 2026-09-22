# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 18.60 ± 0.01 tok/s
- llama.cpp pp: 78.66 ± 0.26 tok/s
- MemVanta tg: 6.39 ± 0.01 tok/s
- llama.cpp tg: 47.15 ± 0.52 tok/s
- MemVanta peak RSS: 644332 KiB
- llama.cpp peak RSS: 1192824 KiB
- MemVanta RSS reduction: 45.98%
