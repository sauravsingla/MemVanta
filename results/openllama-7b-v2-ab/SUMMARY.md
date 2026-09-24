# OpenLLaMA 7B v2 Q4_0 — MemVanta vs llama.cpp

Exact same GGUF on both runtimes. CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 1 warm-up + 5 measured repetitions.

- MemVanta pp: 3.73 ± 0.00 tok/s
- llama.cpp pp: 21.56 ± 0.02 tok/s
- MemVanta tg: 2.30 ± 0.00 tok/s
- llama.cpp tg: 9.07 ± 0.07 tok/s
- MemVanta peak RSS: 3982104 KiB
- llama.cpp peak RSS: 7591436 KiB
- MemVanta RSS reduction: 47.54%
