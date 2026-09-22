# MemVanta

**Low-memory CPU inference for GGUF LLMs.**

MemVanta is an experimental C++20 runtime focused on running quantized Llama-family models with a smaller memory footprint.

## Benchmark

<!-- BEGIN_CANONICAL_7B_BENCHMARK -->
| Metric | MemVanta | pinned `llama.cpp` |
|---|---:|---:|
| OpenLLaMA 7B v2 Q4_0 peak RSS | **3.80 GiB** | 7.24 GiB |
| Prompt processing | 3.73 ± 0.00 tok/s | **21.60 ± 0.02 tok/s** |
| Token generation | 1.96 ± 0.00 tok/s | **9.66 ± 0.03 tok/s** |
| Peak-RSS reduction | **47.54%** | baseline |

Source of truth: [`results/openllama-7b-v2-ab/summary.json`](results/openllama-7b-v2-ab/summary.json). The README table is generated from that file; do not edit its numbers by hand.
<!-- END_CANONICAL_7B_BENCHMARK -->

MemVanta is **memory-first**; pinned `llama.cpp` is substantially faster in this test.

[Benchmark evidence](results/openllama-7b-v2-ab/) · [Methodology](docs/MEMORY_BENCHMARKING.md) · [Reproduce](docs/EXTERNAL_REPRODUCTION.md)

A separate cgroup-v2 test also measured execution under tight memory limits. It is systems evidence, **not a physical-RAM requirement**. [Results](results/openllama-7b-v2-ram-constrained/)

## Build

```bash
git clone https://github.com/sauravsingla/MemVanta.git
cd MemVanta
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Core ideas

- mmap-backed model access
- paged KV cache
- Q4/Q8 CPU kernels
- byte-bounded adaptive prefetching
- repeated A/B performance and memory gates

## Scope

Trained-model execution currently supports GGUF models with `general.architecture=llama`.

The GGUF parser also validates pinned Qwen2 files, but **Qwen2 inference is not implemented**.

CI covers correctness, sanitizers, deterministic model checks, x86 portability, AVX2/FMA paths, and ARM64 cross-build/QEMU validation.

## Status

Active research prototype with trained-model evidence up to 7B. Results apply to the tested models, settings, and hosts; independent reproduction is welcome.

[Contributing](CONTRIBUTING.md) · [Architecture](docs/ARCHITECTURE.md) · [Results](results/) · [Citation](CITATION.cff) · [License](LICENSE)
