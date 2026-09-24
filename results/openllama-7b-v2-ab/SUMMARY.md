# OpenLLaMA 7B v2 Q4_0 — MemVanta vs llama.cpp

Exact same GGUF on both runtimes. CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 1 warm-up + 5 measured repetitions.

- MemVanta pp: 2.78 ± 0.00 tok/s
- llama.cpp pp: 12.86 ± 0.01 tok/s
- MemVanta tg: 1.81 ± 0.00 tok/s
- llama.cpp tg: 8.14 ± 0.01 tok/s
- MemVanta peak RSS: 3981912 KiB
- llama.cpp peak RSS: 7590772 KiB
- MemVanta RSS reduction: 47.54%
