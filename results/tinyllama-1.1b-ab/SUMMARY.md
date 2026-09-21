# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 18.76 ± 0.04 tok/s
- llama.cpp pp: 78.73 ± 0.30 tok/s
- MemVanta tg: 6.40 ± 0.02 tok/s
- llama.cpp tg: 47.99 ± 0.17 tok/s
- MemVanta peak RSS: 644188 KiB
- llama.cpp peak RSS: 1193440 KiB
- MemVanta RSS reduction: 46.02%
