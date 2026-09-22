# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 18.64 ± 0.02 tok/s
- llama.cpp pp: 79.10 ± 0.18 tok/s
- MemVanta tg: 6.67 ± 0.01 tok/s
- llama.cpp tg: 48.75 ± 0.08 tok/s
- MemVanta peak RSS: 644204 KiB
- llama.cpp peak RSS: 1193216 KiB
- MemVanta RSS reduction: 46.01%
