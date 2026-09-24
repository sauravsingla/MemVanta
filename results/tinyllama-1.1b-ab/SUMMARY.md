# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 18.92 ± 0.02 tok/s
- llama.cpp pp: 79.04 ± 0.14 tok/s
- MemVanta tg: 6.66 ± 0.01 tok/s
- llama.cpp tg: 47.11 ± 0.89 tok/s
- MemVanta peak RSS: 644068 KiB
- llama.cpp peak RSS: 1193564 KiB
- MemVanta RSS reduction: 46.04%
