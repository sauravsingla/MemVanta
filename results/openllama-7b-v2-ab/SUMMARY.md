# OpenLLaMA 7B v2 Q4_0 — MemVanta vs llama.cpp

Exact same GGUF on both runtimes. CPU-only, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, 1 warm-up + 5 measured repetitions.

- MemVanta pp: 3.79 ± 0.02 tok/s
- llama.cpp pp: 21.59 ± 0.02 tok/s
- MemVanta tg: 1.92 ± 0.00 tok/s
- llama.cpp tg: 9.71 ± 0.11 tok/s
- MemVanta peak RSS: 3983264 KiB
- llama.cpp peak RSS: 7588952 KiB
- MemVanta RSS reduction: 47.51%
