# MemVanta

[![Build](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml)
[![7B A/B](https://github.com/sauravsingla/MemVanta/actions/workflows/sevenb-model-ab.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/sevenb-model-ab.yml)
[![RAM Constrained 7B](https://github.com/sauravsingla/MemVanta/actions/workflows/sevenb-ram-constrained.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/sevenb-ram-constrained.yml)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](CMakeLists.txt)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-blue.svg)](CITATION.cff)

**Memory-efficient local LLM inference in C++20.**

> **Run quantized GGUF models on CPUs when RAM capacity matters more than maximum tokens/sec.**

MemVanta is an experimental CPU inference runtime built around one goal: **fit larger local LLMs into smaller RAM budgets** using quantized kernels, mmap-backed model access, bounded caching, and paged KV cache.

**47.50% lower peak RSS on OpenLLaMA 7B v2 Q4_0 · ~3.80 GiB vs 7.24 GiB**  
**CPU-only · GGUF · Q4_0 / Q6_K / Q8_0 · AVX2/FMA · mmap · paged KV cache**

Status: **Active experimental runtime · trained-model validation up to 7B**

**Quick links:** [7B benchmark](#results-at-a-glance) · [30-second start](#30-second-start) · [Architecture](#architecture) · [Benchmark evidence](#benchmark-evidence) · [Reproduction](#reproducing-the-benchmarks) · [Contributing](#contributing)

> **Want to try it?** Build and run the benchmark executable → [30-second start](#30-second-start)

## When should I use MemVanta?

MemVanta is designed for experiments where **RAM is the primary constraint**: low-memory CPU inference, edge or constrained machines, GGUF runtime research, quantized kernel work, mmap/page-cache behavior, and memory-vs-throughput benchmarking.

If maximum throughput is the goal, `llama.cpp` is substantially faster in the current 7B measurements. MemVanta focuses instead on **reducing resident memory while reporting the throughput cost transparently**.

## What is different?

MemVanta brings four concerns into one systems design:

- **Memory-first execution:** mmap-backed tensor access, bounded caching and prefetch experiments.
- **Quantized CPU inference:** Q4_0, Q6_K, Q8_0, F16 and F32 tensor paths with AVX2/FMA kernels.
- **Paged attention state:** F32, F16 and Q8 KV-cache paths for bounded runtime memory behavior.
- **Auditable evaluation:** same-GGUF comparisons, retained raw evidence, constrained-memory sweeps and explicit claim boundaries.

## Results at a glance

> Published numbers are **reproducible engineering evidence scoped to the tested models, settings and hosts**.

| Evidence | Verified result |
|---|---|
| OpenLLaMA 7B v2 Q4_0 peak RSS | **3.80 GiB MemVanta vs 7.24 GiB llama.cpp — 47.50% lower** |
| 7B constrained-memory sweep | **MemVanta completed at 3584 MiB; llama.cpp was OOM-killed** |
| llama.cpp lowest successful tested ceiling | **3840 MiB** |
| Prompt processing | 3.15 ± 0.04 tok/s vs 45.82 ± 0.96 tok/s |
| Token generation | 1.52 ± 0.02 tok/s vs 7.76 ± 0.09 tok/s |
| Published trained-model evidence | **360M · 1.1B · 3B · 7B** |

### 7B comparison

**OpenLLaMA 7B v2 Q4_0 · exact same GGUF · CPU only · 4 threads · pp512/tg128 · context 768 · batch 32 · F16 KV · 1 warm-up + 5 measured runs**

| Metric | MemVanta | llama.cpp |
|---|---:|---:|
| Peak RSS | **3,983,412 KiB (~3.80 GiB)** | 7,586,960 KiB (~7.24 GiB) |
| Prompt processing | 3.15 ± 0.04 tok/s | **45.82 ± 0.96 tok/s** |
| Token generation | 1.52 ± 0.02 tok/s | **7.76 ± 0.09 tok/s** |

**Peak resident-memory reduction: 47.50% (~3.44 GiB less).**

These results intentionally show the trade-off: **MemVanta wins on peak memory in this test; llama.cpp wins substantially on throughput.**

Raw evidence: [`results/openllama-7b-v2-ab/`](results/openllama-7b-v2-ab/)

### 7B constrained-memory result

In a separate Linux **cgroup-v2 `MemoryMax` sweep with swap disabled**, using the same OpenLLaMA 7B v2 Q4_0 GGUF, CPU-only, 4 threads, pp128/tg32, context 768, batch 32 and F16 KV:

- MemVanta completed at a **3584 MiB tested memory ceiling**.
- pinned `llama.cpp` was **OOM-killed at 3584 MiB**.
- `llama.cpp`'s lowest successful **tested** ceiling was **3840 MiB**.
- tested-ceiling difference: **256 MiB (6.67%)**.

This is an execution-under-pressure result over the tested sweep, **not an exact minimum physical-RAM requirement**.

Raw evidence: [`results/openllama-7b-v2-ram-constrained/`](results/openllama-7b-v2-ram-constrained/)

## 30-second start

Requires **CMake ≥ 3.20**, a **C++20 compiler**, and a Linux/macOS development environment. AVX2-capable x86 is recommended for optimized kernel paths.

```bash
git clone https://github.com/sauravsingla/MemVanta.git
cd MemVanta
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

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

For reproducible comparisons, use the same model file, SHA-256, CPU/thread settings, context, batch size, KV type, prompt length and generation length for both runtimes.

## Architecture

```mermaid
flowchart LR
    A[GGUF model] --> B[mmap-backed tensor access]
    B --> C[bounded cache / prefetch]
    C --> D[Q4 / Q6 / Q8 CPU kernels]
    D --> E[Transformer execution]
    E --> F[paged KV cache]
    F --> G[tokens]
```

The runtime is designed to keep model access and cache behavior bounded while executing quantized transformer workloads on CPU.

## Runtime capabilities

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

Current optimization work therefore targets the **Q4 FP32/AVX FFN path**, while retaining memory usage as a hard regression guardrail.

Evidence: [`results/openllama-7b-v2-throughput-profile/`](results/openllama-7b-v2-throughput-profile/)

## Benchmark evidence

Headline results are backed by raw evidence retained in the repository rather than only transient CI logs.

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
| 7B kernel/throughput profiling | **Published** | Hotspot selection evidence |
| Smaller-model A/B results | **Published** | 360M, 1.1B and 3B evidence retained |
| Physical-CPU reproduction | **Pending** | Headline evidence remains scoped to tested hosts |
| Independent third-party reproduction | **Pending / invited** | Reproduction guide and issue template available |

**Claim boundary:** MemVanta does not claim a universal memory-scaling law or a throughput advantage over `llama.cpp`.

## Reproducing the benchmarks

- [`docs/MEMORY_BENCHMARKING.md`](docs/MEMORY_BENCHMARKING.md) — methodology and claim boundaries
- [`docs/BENCHMARK_CHECKLIST.md`](docs/BENCHMARK_CHECKLIST.md) — publication checklist
- [`docs/EXTERNAL_REPRODUCTION.md`](docs/EXTERNAL_REPRODUCTION.md) — independent reproduction guide

Independent results that confirm, narrow, or contradict the published measurements are welcome.

## Project direction

> **Run larger quantized LLMs within smaller RAM budgets on commodity CPUs.**

Current work focuses on improving **Q4 FFN/kernel throughput** without sacrificing the memory advantage, tightening the **7B/8B memory boundary**, reproducing results on physical CPUs, broadening GGUF model-family coverage, strengthening numerical validation, and gathering independent third-party reproduction evidence.

## Project status

MemVanta is an **active research and engineering prototype**. APIs and performance characteristics may evolve, so pin a commit when using results in reproducible experiments.

## Contributing

Contributions are welcome, particularly around **CPU/AVX kernel optimization, GGUF compatibility, quantization, KV-cache and memory management, benchmark reproduction, model-family validation, profiling and performance analysis**.

See [`CONTRIBUTING.md`](CONTRIBUTING.md). For security reports, see [`SECURITY.md`](SECURITY.md).

## Citation

If MemVanta, its benchmark protocol, or its published measurements are useful in research, cite the repository using [`CITATION.cff`](CITATION.cff).

Apache-2.0 licensed. See [`LICENSE`](LICENSE).

---

**Topics:** local LLM inference · CPU LLM inference · GGUF · quantized inference · low-RAM AI · edge AI · C++ inference runtime · Q4_0 · Q6_K · Q8_0 · mmap · paged KV cache · llama.cpp benchmarking