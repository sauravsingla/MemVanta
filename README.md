# MemVanta

**Run larger local LLMs with less RAM.**

MemVanta is an experimental **C++20 runtime for low-memory CPU LLM inference** with quantized Llama-family GGUF models. It explores mmap-backed model access, quantized CPU kernels, paged KV cache, and byte-bounded adaptive prefetching for memory-constrained local AI.

> **Memory first.** MemVanta trades throughput for a smaller memory footprint. On the canonical repeated OpenLLaMA 7B v2 Q4_0 A/B test, MemVanta used **47.54% less peak RSS** than pinned `llama.cpp`; `llama.cpp` was substantially faster.

## Benchmark

The repeated OpenLLaMA 7B v2 Q4_0 comparison is MemVanta's **primary public memory claim**. Throughput is shown beside memory so the trade-off remains explicit.

<!-- BEGIN_CANONICAL_7B_BENCHMARK -->
| Metric | MemVanta | pinned `llama.cpp` |
|---|---:|---:|
| OpenLLaMA 7B v2 Q4_0 peak RSS | **3.80 GiB** | 7.24 GiB |
| Prompt processing | 3.73 ± 0.00 tok/s | **21.60 ± 0.02 tok/s** |
| Token generation | 1.96 ± 0.00 tok/s | **9.66 ± 0.03 tok/s** |
| Peak-RSS reduction | **47.54%** | baseline |

Source of truth: [`results/openllama-7b-v2-ab/summary.json`](results/openllama-7b-v2-ab/summary.json). The README table is generated from that file; do not edit its numbers by hand.
<!-- END_CANONICAL_7B_BENCHMARK -->

[Raw A/B evidence](results/openllama-7b-v2-ab/) · [Methodology](docs/MEMORY_BENCHMARKING.md) · [External reproduction guide](docs/EXTERNAL_REPRODUCTION.md)

### Constrained-memory evidence

A separate Linux cgroup-v2 experiment, with swap disabled and the same verified 7B model, found a lowest confirmed successful ceiling of **160 MiB for MemVanta** versus **3648 MiB for pinned `llama.cpp`** on the tested hosted runner.

This is **execution-under-pressure evidence**, not a physical-RAM requirement. The 160 MiB value must not be quoted as the RAM required to hold or run a 7B model; the primary memory result is the repeated **3.80 GiB peak-RSS** measurement above.

[Constrained-memory results](results/openllama-7b-v2-ram-constrained/)

## What MemVanta explores

- **Low-memory GGUF inference** for memory-constrained CPUs
- **mmap-backed model access** instead of persistent full-model copies
- **Paged KV cache** with explicit memory bounds
- **Q4/Q8 CPU kernels** with portable fallbacks and AVX2/FMA optimization
- **Byte-bounded adaptive prefetching** driven by usefulness, memory pressure, and latency feedback
- **Repeated same-runner A/B gates** for benchmark-affecting compiler and kernel changes

Performance work is accepted only when it clears the repository's correctness and memory guardrails. Failed optimization candidates are kept as negative results rather than promoted as wins.

## Quick start

Build and run the test suite on Linux or macOS:

```bash
git clone https://github.com/sauravsingla/MemVanta.git
cd MemVanta
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

For benchmark reproduction against pinned `llama.cpp`, use the [external reproduction guide](docs/EXTERNAL_REPRODUCTION.md).

## Supported scope

Trained-model execution currently supports GGUF files with:

```text
general.architecture=llama
```

The GGUF parser itself is architecture-neutral. CI also validates a pinned `general.architecture=qwen2` GGUF with `memvanta_gguf_inspect`; that is **container/parser compatibility evidence only**, not a claim that Qwen2 inference is implemented.

## Correctness and portability

The validation stack includes:

- exhaustive FP16 conversion coverage, including subnormals, rounding boundaries, NaN, and infinity handling
- explicit GGUF parser, string, array, allocation, workspace, KV-page, and offset bounds
- deterministic multi-thread model checks and pinned external-reference comparisons
- concurrency and prefetch lifecycle stress testing
- AddressSanitizer, UndefinedBehaviorSanitizer, ThreadSanitizer, and GGUF fuzz-smoke coverage
- portable x86 runtime dispatch plus AVX2/FMA paths where supported
- ARM64 cross-build and QEMU correctness coverage for ARMv8 SIMD/NEON-capable code generation
- trained-model validation on pinned small and 7B-class Llama-family GGUF models

The goal is to improve memory efficiency without weakening numerical correctness, determinism, portability, or benchmark claim discipline.

## Project status

MemVanta is an **active research prototype** with trained-model evidence up to 7B. Results are scoped to the tested models, settings, and hosts. Independent third-party reproduction is still needed.

## Contributing

Independent benchmark reproductions, CPU kernel optimizations, GGUF compatibility testing, profiling, and well-documented negative results are welcome.

[Contributing](CONTRIBUTING.md) · [Architecture](docs/ARCHITECTURE.md) · [All evidence](results/) · [Citation](CITATION.cff) · [License](LICENSE)
