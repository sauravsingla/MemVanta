# MemVanta

**Run larger local LLMs with less RAM.**

MemVanta is an experimental C++20 CPU runtime for quantized Llama-family GGUF models, focused on **memory-efficient inference**.

**Published 7B result:** ~3.80 GiB peak RSS with MemVanta vs ~7.24 GiB with pinned `llama.cpp` — **47.50% lower**.

[Reproduce](#-reproduce-it) · [Evidence](results/openllama-7b-v2-ab/) · [Contribute](CONTRIBUTING.md)

[![Build](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](CMakeLists.txt)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## Benchmark

| | MemVanta | pinned `llama.cpp` |
|---|---:|---:|
| 7B peak RSS | **~3.80 GiB** | ~7.24 GiB |
| Prompt processing | 3.15 ± 0.04 tok/s | **45.82 ± 0.96 tok/s** |
| Token generation | 1.52 ± 0.02 tok/s | **7.76 ± 0.09 tok/s** |
| 3584 MiB memory ceiling | **Completed** | OOM-killed |

Same GGUF · CPU-only · 4 threads · pp512/tg128 · context 768 · batch 32 · F16 KV · 1 warm-up + 5 measured runs.

[Raw evidence](results/openllama-7b-v2-ab/) · [Memory-pressure test](results/openllama-7b-v2-ram-constrained/) · [Methodology](docs/MEMORY_BENCHMARKING.md)

`llama.cpp` is substantially faster in this test; MemVanta targets the memory side of the trade-off.

## 🧪 Reproduce it

Have a Linux or macOS CPU machine? Try to confirm, narrow, or contradict the result.

```bash
git clone https://github.com/sauravsingla/MemVanta.git
cd MemVanta
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Then follow the **[reproduction guide](docs/EXTERNAL_REPRODUCTION.md)** and share your CPU, RAM, compiler, model SHA, peak RSS and throughput.

## What MemVanta explores

- low-RAM CPU inference
- quantized GGUF kernels and AVX2/FMA
- mmap-backed model access and paged KV cache
- throughput improvements that preserve the memory advantage

Current real-model execution supports GGUF files with `general.architecture=llama`.

## Get involved

Independent reproductions, CPU kernel work, GGUF compatibility testing, profiling, and negative results are welcome.

[Contributing](CONTRIBUTING.md) · [All evidence](results/) · [Citation](CITATION.cff) · [License](LICENSE)

---

**Status:** Active research prototype with trained-model evidence up to 7B. Results are scoped to the tested models, settings and hosts, and independent third-party reproduction is still needed.
