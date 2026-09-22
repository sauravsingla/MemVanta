# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 18.65 ± 0.02 tok/s
- llama.cpp pp: 78.72 ± 0.39 tok/s
- MemVanta tg: 6.40 ± 0.01 tok/s
- llama.cpp tg: 47.93 ± 0.49 tok/s
- MemVanta peak RSS: 644064 KiB
- llama.cpp peak RSS: 1193364 KiB
- MemVanta RSS reduction: 46.03%
