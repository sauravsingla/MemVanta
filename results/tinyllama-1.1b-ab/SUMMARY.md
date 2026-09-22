# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 18.55 ± 0.02 tok/s
- llama.cpp pp: 79.28 ± 0.04 tok/s
- MemVanta tg: 6.39 ± 0.00 tok/s
- llama.cpp tg: 48.96 ± 0.11 tok/s
- MemVanta peak RSS: 644160 KiB
- llama.cpp peak RSS: 1190164 KiB
- MemVanta RSS reduction: 45.88%
