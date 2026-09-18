# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 17.22 ± 0.13 tok/s
- llama.cpp pp: 79.05 ± 0.23 tok/s
- MemVanta tg: 5.78 ± 0.01 tok/s
- llama.cpp tg: 48.35 ± 0.20 tok/s
- MemVanta peak RSS: 644284 KiB
- llama.cpp peak RSS: 1194320 KiB
- MemVanta RSS reduction: 46.05%
