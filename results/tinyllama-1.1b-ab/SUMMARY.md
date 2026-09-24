# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 18.70 ± 0.11 tok/s
- llama.cpp pp: 79.13 ± 0.16 tok/s
- MemVanta tg: 6.66 ± 0.01 tok/s
- llama.cpp tg: 48.61 ± 0.75 tok/s
- MemVanta peak RSS: 644120 KiB
- llama.cpp peak RSS: 1193480 KiB
- MemVanta RSS reduction: 46.03%
