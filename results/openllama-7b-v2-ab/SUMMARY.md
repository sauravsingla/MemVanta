# OpenLLaMA 7B v2 Q4_0 — MemVanta vs llama.cpp

Exact same GGUF on both runtimes. CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 1 warm-up + 5 measured repetitions.

- MemVanta pp: 3.13 ± 0.03 tok/s
- llama.cpp pp: 49.13 ± 1.04 tok/s
- MemVanta tg: 1.27 ± 0.01 tok/s
- llama.cpp tg: 9.05 ± 0.04 tok/s
- MemVanta peak RSS: 3983288 KiB
- llama.cpp peak RSS: 7587096 KiB
- MemVanta RSS reduction: 47.50%
