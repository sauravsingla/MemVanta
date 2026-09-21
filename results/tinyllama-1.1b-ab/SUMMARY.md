# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 22.61 ± 0.18 tok/s
- llama.cpp pp: 100.25 ± 1.04 tok/s
- MemVanta tg: 5.83 ± 0.09 tok/s
- llama.cpp tg: 48.19 ± 0.32 tok/s
- MemVanta peak RSS: 644384 KiB
- llama.cpp peak RSS: 1226220 KiB
- MemVanta RSS reduction: 47.45%
