# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 18.79 ± 0.16 tok/s
- llama.cpp pp: 79.02 ± 0.43 tok/s
- MemVanta tg: 6.23 ± 0.01 tok/s
- llama.cpp tg: 49.48 ± 0.14 tok/s
- MemVanta peak RSS: 644576 KiB
- llama.cpp peak RSS: 1193352 KiB
- MemVanta RSS reduction: 45.99%
