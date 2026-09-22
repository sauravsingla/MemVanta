# OpenLLaMA 7B v2 Q4_0 — MemVanta vs llama.cpp

Exact same GGUF on both runtimes. CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 1 warm-up + 5 measured repetitions.

- MemVanta pp: 2.70 ± 0.01 tok/s
- llama.cpp pp: 12.82 ± 0.01 tok/s
- MemVanta tg: 1.81 ± 0.00 tok/s
- llama.cpp tg: 8.03 ± 0.06 tok/s
- MemVanta peak RSS: 3982004 KiB
- llama.cpp peak RSS: 7590460 KiB
- MemVanta RSS reduction: 47.54%
