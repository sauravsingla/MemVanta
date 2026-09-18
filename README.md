# MemVanta

[![Build](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](CMakeLists.txt)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-blue.svg)](CITATION.cff)

## Fit larger local LLMs into smaller RAM budgets

**MemVanta is a native C++20 CPU inference runtime for quantized Llama-family GGUF models, designed for systems where memory matters more than maximum tokens/sec.**

> **OpenLLaMA 7B v2 Q4_0: ~3.80 GiB peak RSS with MemVanta vs ~7.24 GiB with pinned `llama.cpp` — 47.50% lower in the published test.**

CPU-only · GGUF · Q4_0 / Q6_K / Q8_0 · mmap · paged KV cache · AVX2/FMA

**MemVanta is experimental.** `llama.cpp` is substantially faster in the current 7B benchmark; MemVanta explores the opposite side of the trade-off: **how far can resident memory be reduced while still running useful local LLM inference?**

## Results at a glance

| Metric | MemVanta | pinned `llama.cpp` |
|---|---:|---:|
| Peak RSS — OpenLLaMA 7B v2 Q4_0 | **~3.80 GiB** | ~7.24 GiB |
| Peak RSS reduction | **47.50%** | — |
| Prompt processing | 3.15 ± 0.04 tok/s | **45.82 ± 0.96 tok/s** |
| Token generation | 1.52 ± 0.02 tok/s | **7.76 ± 0.09 tok/s** |
| 3584 MiB constrained-memory test | **Completed** | OOM-killed |

The 7B comparison used the **same GGUF**, CPU-only execution, 4 threads, pp512/tg128, context 768, batch 32, F16 KV, one warm-up and five measured runs.

These are **scoped engineering measurements, not universal performance claims**. Raw evidence is retained in the repository:

- [7B repeated A/B](results/openllama-7b-v2-ab/)
- [7B constrained-memory sweep](results/openllama-7b-v2-ram-constrained/)
- [7B throughput profile](results/openllama-7b-v2-throughput-profile/)
- [3B, 1.1B and 360M evidence](results/)

## 🧪 MemVanta Reproduction Challenge

**Have a Linux or macOS CPU machine? Help independently test the memory result.**

Start in a few minutes:

```bash
git clone https://github.com/sauravsingla/MemVanta.git
cd MemVanta
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Then follow the [external reproduction guide](docs/EXTERNAL_REPRODUCTION.md) to run the same-GGUF comparison against the pinned `llama.cpp` baseline.

Please share:

- CPU and RAM
- operating system and compiler
- model / GGUF SHA
- MemVanta peak RSS
- `llama.cpp` peak RSS
- prompt and generation throughput
- MemVanta commit SHA

**Confirm it. Contradict it. Find a limitation. All three are useful contributions.**

[Reproduction guide](docs/EXTERNAL_REPRODUCTION.md) · [Raw 7B evidence](results/openllama-7b-v2-ab/) · [Benchmark methodology](docs/MEMORY_BENCHMARKING.md) · [Contributing](CONTRIBUTING.md)

## Why MemVanta?

MemVanta is useful for experiments involving:

- **low-RAM CPU inference** on commodity or constrained machines
- **GGUF runtime research** and memory-vs-throughput trade-offs
- **quantized CPU kernels** and AVX2/FMA optimization
- **mmap-backed model access** and bounded caching
- **paged KV cache** and constrained-memory execution
- **reproducible comparison** against a pinned `llama.cpp`

The current real-model executor supports GGUF files with `general.architecture=llama`.

## Build and benchmark

Requires CMake ≥ 3.20, a C++20 compiler, and Linux/macOS.

Benchmark a GGUF model:

```bash
./build/memvanta_real_bench \
  --model model.gguf \
  --threads 4 \
  --ctx 768 \
  --batch 32 \
  --kv f16 \
  --prompt 512 \
  --gen 128 \
  --reps 5 \
  --warmup 1
```

## How it works

```mermaid
flowchart LR
    A[GGUF model] --> B[mmap-backed tensors]
    B --> C[bounded cache / prefetch]
    C --> D[Q4 / Q6 / Q8 CPU kernels]
    D --> E[Transformer execution]
    E --> F[paged KV cache]
    F --> G[tokens]
```

MemVanta currently includes:

- Llama-architecture GGUF execution
- Q4_0, Q6_K, Q8_0, F16 and F32 tensor paths
- AVX2/FMA quantized CPU kernels
- mmap-backed tensor access
- bounded caching and prefetch experiments
- F32 / F16 / Q8 paged KV cache
- batched prefill and token decode
- Llama/SentencePiece-style and GPT-2-style tokenizer support
- real-model benchmarking, profiling and constrained-memory workflows

## Current engineering focus

7B profiling indicates that the dominant bottleneck on the tested host is **projection-kernel compute rather than paging**. Current work therefore focuses on improving **Q4 FFN/kernel throughput without giving back the memory advantage**.

Contributions are especially welcome around:

- Q4/Q8 CPU kernel optimization
- SIMD / AVX performance
- GGUF compatibility
- quantization and KV-cache work
- profiling and performance analysis
- independent benchmark reproduction

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Status and claim boundary

MemVanta is an **active research and engineering prototype** with trained-model evidence up to 7B. APIs and performance characteristics may change.

Published evidence currently includes repeated same-GGUF benchmarks and constrained-memory tests, but **independent third-party reproduction and broader physical-CPU validation are still needed**.

MemVanta does **not** claim a universal memory-scaling law or a throughput advantage over `llama.cpp`.

## Citation

If MemVanta or its benchmark methodology is useful in research, see [CITATION.cff](CITATION.cff).

Apache-2.0 licensed. See [LICENSE](LICENSE).
