# TinyLlama 1.1B Q4_0 — MemVanta vs llama.cpp

Same GGUF, CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 5 repetitions.

- MemVanta pp: 18.42 ± 0.02 tok/s
- llama.cpp pp: 78.72 ± 0.41 tok/s
- MemVanta tg: 6.36 ± 0.01 tok/s
- llama.cpp tg: 48.50 ± 0.34 tok/s
- MemVanta peak RSS: 644160 KiB
- llama.cpp peak RSS: 1194180 KiB
- MemVanta RSS reduction: 46.06%
