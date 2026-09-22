# MemVanta

**Low-memory C++20 LLM inference runtime for quantized GGUF models on CPU.**

MemVanta is a memory-first local LLM runtime for running quantized Llama-family GGUF models on CPUs with limited RAM. It uses mmap-backed model access, paged KV cache, Q4/Q8 kernels, and bounded adaptive prefetching, with reproducible memory and throughput benchmarks against pinned `llama.cpp`.

[Website](https://sauravsingla.github.io/MemVanta/) · [Getting started](https://sauravsingla.github.io/MemVanta/getting-started/) · [7B benchmark](https://sauravsingla.github.io/MemVanta/benchmark/) · [DOI](https://doi.org/10.5281/zenodo.22886357) · [Reproduce](https://sauravsingla.github.io/MemVanta/reproduce/) · [Contributing](CONTRIBUTING.md)

## Why MemVanta?

MemVanta explores a specific systems trade-off: **how much resident memory can CPU LLM inference avoid while still executing a real quantized GGUF model correctly?**

It is designed for experiments where RAM pressure matters more than maximum token throughput, including constrained developer machines, edge systems, and research into memory-aware local inference.

The project is intentionally transparent about the cost of that trade-off. MemVanta is **memory-first**; it does not claim to be faster than `llama.cpp`.

## 7B memory benchmark vs llama.cpp

<!-- BEGIN_CANONICAL_7B_BENCHMARK -->
| Metric | MemVanta | pinned `llama.cpp` |
|---|---:|---:|
| OpenLLaMA 7B v2 Q4_0 peak RSS | **3.80 GiB** | 7.24 GiB |
| Prompt processing | 2.92 ± 0.00 tok/s | **11.98 ± 0.01 tok/s** |
| Token generation | 1.66 ± 0.00 tok/s | **8.00 ± 0.06 tok/s** |
| Peak-RSS reduction | **47.54%** | baseline |

Source of truth: [`results/openllama-7b-v2-ab/summary.json`](results/openllama-7b-v2-ab/summary.json). The README table is generated from that file; do not edit its numbers by hand.
<!-- END_CANONICAL_7B_BENCHMARK -->

The result above is a repeated same-model CPU A/B test on OpenLLaMA 7B v2 Q4_0. It applies to the tested model, workload, host, and pinned comparison runtime; it is not a universal memory-reduction claim.

A separate cgroup-v2 experiment also measured execution under tight memory limits. It is systems evidence, **not a physical-RAM requirement**.

[Benchmark details](https://sauravsingla.github.io/MemVanta/benchmark/) · [Raw evidence](results/openllama-7b-v2-ab/) · [Methodology](docs/MEMORY_BENCHMARKING.md)

## Quick start: run a GGUF model

Build the project:

```bash
git clone https://github.com/sauravsingla/MemVanta.git
cd MemVanta
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Then run trained-model text generation with a supported Llama-family GGUF model that you are licensed to use:

```bash
./build/memvanta_real \
  --model /path/to/model.gguf \
  --prompt "Hello from MemVanta" \
  --n 64 \
  --threads 4 \
  --ctx 2048 \
  --temperature 0
```

`memvanta_real` is the trained-model inference CLI. The separate `memvanta run <file>` command exercises mapped streaming/cache behavior and reports memory telemetry; it is not the text-generation command.

[Full getting-started guide](https://sauravsingla.github.io/MemVanta/getting-started/)

## How low-memory inference works

MemVanta's runtime is organized around explicit memory ownership and bounded data movement:

- **mmap-backed GGUF access** avoids requiring an unconditional full-model copy in a separate heap buffer.
- **Bounded tensor slices and caching** keep model access under explicit memory policy.
- **Paged KV cache** manages attention state with defined bounds.
- **Q4/Q8 quantized CPU kernels** provide compact execution paths for supported tensors.
- **Byte-bounded adaptive prefetching** can change look-ahead behavior without silently expanding the memory budget.
- **Runtime CPU dispatch and AVX2/FMA paths** improve hot paths while portability and correctness remain independently tested.

[Low-memory inference guide](https://sauravsingla.github.io/MemVanta/low-memory-llm-inference/) · [Architecture](https://sauravsingla.github.io/MemVanta/architecture/)

## Current model scope

Trained-model execution currently supports GGUF models with:

```text
general.architecture=llama
```

The GGUF parser also validates pinned Qwen2 files, but **Qwen2 inference is not implemented**. Parser/container compatibility should not be interpreted as trained-model execution support.

The project currently has trained-model evidence up to 7B and remains an active research / engineering prototype rather than a drop-in replacement for a mature general-purpose inference runtime.

## Validation and reproducibility

MemVanta's validation stack includes:

- Release and Debug correctness checks
- AddressSanitizer / UndefinedBehaviorSanitizer and ThreadSanitizer lanes
- parser limits and fuzz smoke
- deterministic trained-model checks
- x86 portability and runtime-dispatch validation
- AVX2/FMA paths
- ARM64 cross-build and QEMU validation
- repeated same-machine A/B memory and throughput measurements

Published benchmark methodology requires the identical GGUF artifact for both runtimes, a pinned comparison-runtime revision, matched workload parameters, warm-up plus repeated measured runs, and throughput reporting beside memory results.

Independent results that confirm, narrow, or contradict the current measurements are useful. Reproduction reports should include model hashes, runtime commits, machine metadata, commands, and raw outputs.

[Reproduction guide](https://sauravsingla.github.io/MemVanta/reproduce/) · [Memory benchmarking protocol](docs/MEMORY_BENCHMARKING.md)

## Project links

- [Project website](https://sauravsingla.github.io/MemVanta/)
- [Getting started](https://sauravsingla.github.io/MemVanta/getting-started/)
- [7B benchmark](https://sauravsingla.github.io/MemVanta/benchmark/)
- [Architecture](docs/ARCHITECTURE.md)
- [Results](results/)
- [Citation](CITATION.cff)
- [Zenodo DOI](https://doi.org/10.5281/zenodo.22886357)
- [Contributing](CONTRIBUTING.md)
- [License](LICENSE)
