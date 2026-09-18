# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 17.17 ± 0.12 tok/s
- llama.cpp pp: 78.99 ± 0.10 tok/s
- MemVanta tg: 5.77 ± 0.01 tok/s
- llama.cpp tg: 47.69 ± 0.18 tok/s
- MemVanta peak RSS: 644004 KiB
- llama.cpp peak RSS: 1190236 KiB
- MemVanta RSS reduction: 45.89%
