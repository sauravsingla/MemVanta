# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 16.15 ± 0.09 tok/s
- llama.cpp pp: 73.68 ± 0.06 tok/s
- MemVanta tg: 5.23 ± 0.00 tok/s
- llama.cpp tg: 46.66 ± 1.76 tok/s
- MemVanta peak RSS: 644092 KiB
- llama.cpp peak RSS: 1190004 KiB
- MemVanta RSS reduction: 45.87%
