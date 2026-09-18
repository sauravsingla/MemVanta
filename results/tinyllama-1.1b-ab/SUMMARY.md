# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 16.73 ± 0.17 tok/s
- llama.cpp pp: 78.92 ± 0.33 tok/s
- MemVanta tg: 5.78 ± 0.02 tok/s
- llama.cpp tg: 48.68 ± 0.06 tok/s
- MemVanta peak RSS: 644564 KiB
- llama.cpp peak RSS: 1190192 KiB
- MemVanta RSS reduction: 45.84%
