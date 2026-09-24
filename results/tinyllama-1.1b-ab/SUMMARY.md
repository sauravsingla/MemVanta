# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 18.47 ± 0.04 tok/s
- llama.cpp pp: 78.93 ± 0.29 tok/s
- MemVanta tg: 6.66 ± 0.01 tok/s
- llama.cpp tg: 48.57 ± 0.28 tok/s
- MemVanta peak RSS: 644308 KiB
- llama.cpp peak RSS: 1194420 KiB
- MemVanta RSS reduction: 46.06%
