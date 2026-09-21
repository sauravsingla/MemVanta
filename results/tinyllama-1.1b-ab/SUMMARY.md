# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 18.52 ± 0.04 tok/s
- llama.cpp pp: 78.89 ± 0.19 tok/s
- MemVanta tg: 6.24 ± 0.01 tok/s
- llama.cpp tg: 48.72 ± 0.07 tok/s
- MemVanta peak RSS: 644268 KiB
- llama.cpp peak RSS: 1192836 KiB
- MemVanta RSS reduction: 45.99%
