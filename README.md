# MemVanta

[![Build](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml)
[![7B A/B](https://github.com/sauravsingla/MemVanta/actions/workflows/sevenb-model-ab.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/sevenb-model-ab.yml)
[![RAM Constrained 7B](https://github.com/sauravsingla/MemVanta/actions/workflows/sevenb-ram-constrained.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/sevenb-ram-constrained.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Run quantized LLMs on CPUs with dramatically lower memory usage.**

MemVanta is an experimental **C++20 CPU inference runtime** for **GGUF** models built around one goal: **fit larger local LLMs into smaller RAM budgets**.

**C++20 · CPU-only · GGUF · Q4_0 · Q6_K · Q8_0 · AVX2/FMA · mmap · paged KV cache**

## Headline benchmark — 47.5% lower peak memory

**OpenLLaMA 7B v2 Q4_0 · exact same GGUF · CPU only · 4 threads · pp512/tg128 · context 768 · batch 32 · F16 KV · 1 warm-up + 5 measured runs**

| Metric | MemVanta | llama.cpp |
|---|---:|---:|
| Peak RSS | **3,983,412 KiB (~3.80 GiB)** | 7,586,960 KiB (~7.24 GiB) |
| Prompt processing | 3.15 ± 0.04 tok/s | 45.82 ± 0.96 tok/s |
| Token generation | 1.52 ± 0.02 tok/s | 7.76 ± 0.09 tok/s |

**Peak resident-memory reduction: 47.50% (~3.44 GiB less).**

> MemVanta optimizes for **memory efficiency**, not raw throughput. `llama.cpp` remains substantially faster in these tests.

Raw evidence: [`results/openllama-7b-v2-ab/`](results/openllama-7b-v2-ab/)

### 7B constrained-memory result

In a separate Linux **cgroup-v2 `MemoryMax` sweep with swap disabled**, using the exact same OpenLLaMA 7B v2 Q4_0 GGUF, CPU-only, 4 threads, pp128/tg32, context 768, batch 32, and F16 KV:

- MemVanta completed at a **3584 MiB tested memory ceiling**.
- pinned `llama.cpp` was **OOM-killed at 3584 MiB**.
- `llama.cpp`'s lowest successful **tested** ceiling was **3840 MiB**.
- tested-ceiling difference: **256 MiB (6.67%)**.

This is an execution-under-pressure result over the tested sweep, **not an exact minimum physical-RAM requirement**.

Raw evidence: [`results/openllama-7b-v2-ram-constrained/`](results/openllama-7b-v2-ram-constrained/)

## Why MemVanta?

Most local-LLM runtimes optimize first for throughput. MemVanta asks a different question:

> **How far can a quantized LLM be pushed on a CPU when RAM — not compute — is the main constraint?**

The project is useful for experiments involving:

- **low-RAM local LLM inference** on CPUs
- **GGUF model execution** and quantized tensor kernels
- **Q4_0 / Q6_K / Q8_0** inference paths
- **mmap and page-cache behavior** under memory pressure
- **paged KV-cache design**
- **CPU inference benchmarking** against `llama.cpp`
- **edge AI / constrained-machine inference**
- reproducible systems research around **memory vs throughput**

MemVanta is **not yet a drop-in replacement for `llama.cpp`** and is not intended to be a polished end-user chatbot runtime.

## Quick start

### Requirements

- CMake 3.20+
- C++20 compiler
- Linux/macOS development environment
- AVX2-capable x86 CPU recommended for the optimized kernel paths

### Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Benchmark a GGUF model

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

For reproducible comparisons, use the same model file, SHA-256, CPU/thread settings, context, batch size, KV type, prompt length and generation length for both runtimes.

## What MemVanta implements

- native **GGUF** model execution
- **Q4_0, Q6_K, Q8_0, F16 and F32** tensor paths
- **AVX2/FMA** quantized CPU kernels
- mmap-backed tensor access
- bounded caching and prefetch experiments
- paged **F32 / F16 / Q8 KV cache**
- batched prefill and token decode paths
- GPT-2 and Llama/SentencePiece-style tokenization
- trained-model CPU benchmarking against pinned `llama.cpp`
- constrained-memory and cgroup-v2 benchmark workflows
- layer/kernel profiling for 7B throughput bottlenecks

## Current engineering focus

Recent 7B profiling shows that the main speed bottleneck is **projection-kernel compute rather than paging** on the tested host:

- projection kernels accounted for about **98% of profiled model time**
- **FFN GEMM** accounted for about **62% of projection-kernel time**
- `ffn_down` was the largest individual profiled kernel class
- only **1 major page fault** occurred in that run

That evidence is steering current optimization work toward the **Q4 FP32/AVX FFN path**, while retaining memory usage as a hard regression guardrail.

Evidence: [`results/openllama-7b-v2-throughput-profile/`](results/openllama-7b-v2-throughput-profile/)

## Benchmark evidence

MemVanta keeps raw benchmark evidence in the repository rather than relying only on transient CI logs.

| Model / experiment | Evidence |
|---|---|
| OpenLLaMA 7B v2 repeated A/B | [`results/openllama-7b-v2-ab/`](results/openllama-7b-v2-ab/) |
| OpenLLaMA 7B v2 constrained RAM | [`results/openllama-7b-v2-ram-constrained/`](results/openllama-7b-v2-ram-constrained/) |
| OpenLLaMA 7B v2 throughput profile | [`results/openllama-7b-v2-throughput-profile/`](results/openllama-7b-v2-throughput-profile/) |
| OpenLLaMA 3B v2 repeated A/B | [`results/openllama-3b-v2-ab/`](results/openllama-3b-v2-ab/) |
| TinyLlama 1.1B constrained RAM | [`results/tinyllama-1.1b-ram-constrained/`](results/tinyllama-1.1b-ram-constrained/) |
| TinyLlama 1.1B repeated A/B | [`results/tinyllama-1.1b-ab/`](results/tinyllama-1.1b-ab/) |
| SmolLM2 360M repeated A/B | [`results/smollm2-360m-ab/`](results/smollm2-360m-ab/) |

## Validation status

| Evidence | Status | Scope |
|---|---|---|
| Repeated same-GGUF 7B A/B | **Published** | 5 measured CPU runs with raw evidence |
| 7B cgroup-v2 memory-pressure sweep | **Published** | Swap disabled; tested memory ceilings retained |
| 7B kernel/throughput profiling | **Published** | Hotspot selection evidence, not a universal performance claim |
| Smaller-model A/B results | **Published** | 360M, 1.1B and 3B evidence retained |
| Physical-CPU reproduction | **Pending** | Current headline evidence should not be generalized beyond tested hosts |
| Independent third-party reproduction | **Pending / invited** | Reproduction guide and issue template are available |
| Universal memory-scaling claim | **Not claimed** | Results are scoped to tested models, settings and hardware |
| Throughput advantage over llama.cpp | **Not claimed** | Throughput is reported as the cost of the memory trade-off |

**Published** means evidence exists in this repository; it does not mean the result has already been independently replicated.

## Reproducing the benchmarks

- [`docs/MEMORY_BENCHMARKING.md`](docs/MEMORY_BENCHMARKING.md) — methodology and claim boundaries
- [`docs/BENCHMARK_CHECKLIST.md`](docs/BENCHMARK_CHECKLIST.md) — publication checklist
- [`docs/EXTERNAL_REPRODUCTION.md`](docs/EXTERNAL_REPRODUCTION.md) — independent reproduction guide

Independent results that confirm, narrow, or contradict the published measurements are welcome.

## Project direction

> **Run larger quantized LLMs within smaller RAM budgets on commodity CPUs.**

Current work focuses on:

- improving **Q4 FFN/kernel throughput** without sacrificing the memory advantage
- tightening the **7B/8B memory boundary**
- reproducing results on physical CPUs
- broadening GGUF model-family coverage
- strengthening numerical validation
- gathering independent third-party reproduction evidence

MemVanta is an **engineering prototype under active development**. Published trained-model evidence currently reaches 7B and shows lower peak RSS on the tested workloads; it does **not** establish a universal scaling law or throughput advantage.

## Contributing

Contributions are welcome in areas such as:

- CPU / AVX kernel optimization
- GGUF compatibility
- quantization
- KV-cache and memory-management work
- benchmark reproduction
- model-family validation
- profiling and performance analysis

See [`CONTRIBUTING.md`](CONTRIBUTING.md). For security reports, see [`SECURITY.md`](SECURITY.md).

## Citation

If MemVanta, its benchmark protocol, or its published measurements are useful in research, please cite the repository using [`CITATION.cff`](CITATION.cff). GitHub can render this metadata through its **Cite this repository** interface.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

---

**Topics:** local LLM inference · CPU LLM inference · GGUF · quantized inference · low-RAM AI · edge AI · C++ inference runtime · Q4_0 · Q6_K · Q8_0 · mmap · paged KV cache · llama.cpp benchmarking