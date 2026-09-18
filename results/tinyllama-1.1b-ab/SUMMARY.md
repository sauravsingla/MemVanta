# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 19.46 ± 0.19 tok/s
- llama.cpp pp: 97.42 ± 2.80 tok/s
- MemVanta tg: 5.60 ± 0.09 tok/s
- llama.cpp tg: 47.78 ± 0.35 tok/s
- MemVanta peak RSS: 644220 KiB
- llama.cpp peak RSS: 1226148 KiB
- MemVanta RSS reduction: 47.46%
