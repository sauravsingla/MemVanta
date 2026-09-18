# MemVanta

[![Build](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml/badge.svg)](https://github.com/sauravsingla/MemVanta/actions/workflows/ci.yml)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](CMakeLists.txt)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## Memory-first local LLM inference on CPU

**MemVanta is an experimental C++20 runtime for quantized Llama-family GGUF models when RAM matters more than maximum tokens/sec.**

> **OpenLLaMA 7B v2 Q4_0: ~3.80 GiB peak RSS with MemVanta vs ~7.24 GiB with pinned `llama.cpp` — 47.50% lower in the published test.**

`CPU-only` · `GGUF` · `Q4_0 / Q6_K / Q8_0` · `mmap` · `paged KV cache` · `AVX2/FMA`

`llama.cpp` is substantially faster in the same 7B test. MemVanta explores a different question:

**How little resident memory can local LLM inference use while remaining practical and reproducible?**

## Results

| Metric | MemVanta | pinned `llama.cpp` |
|---|---:|---:|
| 7B peak RSS | **~3.80 GiB** | ~7.24 GiB |
| Prompt processing | 3.15 ± 0.04 tok/s | **45.82 ± 0.96 tok/s** |
| Token generation | 1.52 ± 0.02 tok/s | **7.76 ± 0.09 tok/s** |
| 3584 MiB memory ceiling | **Completed** | OOM-killed |

Same GGUF · CPU only · 4 threads · pp512/tg128 · context 768 · batch 32 · F16 KV · 1 warm-up + 5 measured runs.

[Raw 7B evidence](results/openllama-7b-v2-ab/) · [Constrained-memory evidence](results/openllama-7b-v2-ram-constrained/) · [Methodology](docs/MEMORY_BENCHMARKING.md)

## 🧪 Reproduction Challenge

**Have a Linux or macOS CPU machine? Try to reproduce — or disprove — the result.**

```bash
git clone https://github.com/sauravsingla/MemVanta.git
cd MemVanta
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Then follow the **[external reproduction guide](docs/EXTERNAL_REPRODUCTION.md)** and share your CPU, RAM, compiler, model SHA, peak RSS and throughput.

**Confirm it. Contradict it. Find a limitation. All three help.**

## Why MemVanta?

- Run and study **low-RAM CPU inference**.
- Experiment with **quantized GGUF kernels, mmap and paged KV cache**.
- Measure the **memory ↔ throughput trade-off** with retained raw evidence.
- Contribute to an open systems problem: **more throughput without giving back the memory advantage**.

Current real-model execution supports GGUF files with `general.architecture=llama`.

## Get involved

The most useful contributions right now are:

- independent benchmark reproductions
- Q4/Q8, SIMD and AVX performance work
- GGUF/model compatibility testing
- profiling or well-documented negative results

See **[CONTRIBUTING.md](CONTRIBUTING.md)** or open an issue with your result.

## Project status

MemVanta is an **active research and engineering prototype** with trained-model evidence up to 7B. Independent third-party reproduction and broader physical-CPU validation are still needed.

Published measurements are scoped to the tested models, settings and hosts. MemVanta does **not** claim a universal memory-scaling law or a throughput advantage over `llama.cpp`.

[Benchmark evidence](results/) · [Reproduction guide](docs/EXTERNAL_REPRODUCTION.md) · [Citation](CITATION.cff) · [License](LICENSE)
