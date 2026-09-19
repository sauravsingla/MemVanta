# MemVanta

**Run larger local LLMs with less RAM.**

MemVanta is an experimental C++20 runtime for **low-memory CPU LLM inference** with quantized Llama-family **GGUF models**. It explores mmap-backed model access, quantized CPU kernels, and paged KV cache for memory-constrained local AI.

## Quick Start

Build and test on Linux or macOS:

```bash
git clone https://github.com/sauravsingla/MemVanta.git
cd MemVanta
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

To reproduce the published memory measurements against pinned `llama.cpp`, follow the **[reproduction guide](docs/EXTERNAL_REPRODUCTION.md)**.

## Benchmark: MemVanta vs llama.cpp

<!-- BEGIN_CANONICAL_7B_BENCHMARK -->
| Metric | MemVanta | pinned `llama.cpp` |
|---|---:|---:|
| OpenLLaMA 7B v2 Q4_0 peak RSS | **3.80 GiB** | 7.24 GiB |
| Prompt processing | 3.79 ± 0.02 tok/s | **21.59 ± 0.02 tok/s** |
| Token generation | 1.92 ± 0.00 tok/s | **9.71 ± 0.11 tok/s** |
| Peak-RSS reduction | **47.51%** | baseline |

Source of truth: [`results/openllama-7b-v2-ab/summary.json`](results/openllama-7b-v2-ab/summary.json). The README table is generated from that file; do not edit its numbers by hand.
<!-- END_CANONICAL_7B_BENCHMARK -->

### Constrained-memory boundary

A separate Linux cgroup-v2 `MemoryMax` experiment, with swap disabled and the same verified OpenLLaMA 7B v2 Q4_0 model, narrowed the execution-under-pressure boundary to 32 MiB resolution:

| Boundary metric | MemVanta | pinned `llama.cpp` |
|---|---:|---:|
| Lowest confirmed successful ceiling | **160 MiB** | 3648 MiB |
| Confirmed OOM ceiling | **128 MiB** | 3616 MiB |
| Confirmed-success ceiling difference | **3488 MiB lower** | baseline |
| Lower confirmed-success ceiling | **95.61%** | baseline |

Each final success/OOM edge was repeated twice. This is a **cgroup execution-under-pressure boundary on the tested hosted runner**, not an exact physical-RAM minimum and not a replacement for the peak-RSS/throughput benchmark above.

[Raw A/B evidence](results/openllama-7b-v2-ab/) · [Separate memory-pressure test](results/openllama-7b-v2-ram-constrained/) · [Methodology](docs/MEMORY_BENCHMARKING.md)

`llama.cpp` is substantially faster in this test; MemVanta focuses on the **memory-efficiency side of CPU inference**.

## Low-memory CPU LLM inference

MemVanta focuses on:

- quantized **GGUF inference** on memory-constrained CPUs
- mmap-backed model access and paged KV cache
- Q4/Q8 and AVX2/FMA optimization while preserving memory efficiency

Current real-model execution supports GGUF files with `general.architecture=llama`.

## Contributing

Independent benchmark reproductions, CPU kernel optimizations, GGUF compatibility testing, profiling, and well-documented negative results are welcome.

[Contributing](CONTRIBUTING.md) · [All evidence](results/) · [Citation](CITATION.cff) · [License](LICENSE)

---

**Status:** Active research prototype with trained-model evidence up to 7B. Results are scoped to the tested models, settings, and hosts; independent third-party reproduction is still needed.
