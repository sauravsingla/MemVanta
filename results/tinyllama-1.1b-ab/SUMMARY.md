# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 25.69 ± 0.64 tok/s
- llama.cpp pp: 262.63 ± 6.15 tok/s
- MemVanta tg: 12.55 ± 0.35 tok/s
- llama.cpp tg: 71.55 ± 0.33 tok/s
- MemVanta peak RSS: 644404 KiB
- llama.cpp peak RSS: 1190528 KiB
- MemVanta RSS reduction: 45.87%
