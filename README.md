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

## Primary benchmark: MemVanta vs llama.cpp

The canonical repeated peak-RSS comparison below is MemVanta's **primary public memory claim**. Throughput is reported beside memory so the trade-off is explicit.

<!-- BEGIN_CANONICAL_7B_BENCHMARK -->
| Metric | MemVanta | pinned `llama.cpp` |
|---|---:|---:|
| OpenLLaMA 7B v2 Q4_0 peak RSS | **3.80 GiB** | 7.24 GiB |
| Prompt processing | 2.94 ± 0.00 tok/s | **12.82 ± 0.01 tok/s** |
| Token generation | 1.62 ± 0.00 tok/s | **8.10 ± 0.02 tok/s** |
| Peak-RSS reduction | **47.54%** | baseline |

Source of truth: [`results/openllama-7b-v2-ab/summary.json`](results/openllama-7b-v2-ab/summary.json). The README table is generated from that file; do not edit its numbers by hand.
<!-- END_CANONICAL_7B_BENCHMARK -->

### Secondary systems evidence: constrained-memory boundary

A separate Linux cgroup-v2 `MemoryMax` experiment, with swap disabled and the same verified OpenLLaMA 7B v2 Q4_0 model, narrowed the execution-under-pressure boundary to 32 MiB resolution:

| Boundary metric | MemVanta | pinned `llama.cpp` |
|---|---:|---:|
| Lowest confirmed successful ceiling | **160 MiB** | 3648 MiB |
| Confirmed OOM ceiling | **128 MiB** | 3616 MiB |
| Confirmed-success ceiling difference | **3488 MiB lower** | baseline |
| Confirmed-success ceiling reduction vs pinned `llama.cpp` | **95.61%** | baseline |

Each final success/OOM edge was repeated twice. This is a **cgroup execution-under-pressure boundary on the tested hosted runner**, not an exact physical-RAM minimum and not a replacement for the peak-RSS/throughput benchmark above. In particular, **160 MiB must not be quoted as the physical RAM required to hold or run a 7B model**; the primary memory result remains the repeated **3.80 GiB peak-RSS** measurement above.

At each runtime's lowest confirmed successful ceiling, the two confirmation runs averaged approximately:

| Pressure-workload throughput | MemVanta @ 160 MiB | pinned `llama.cpp` @ 3648 MiB |
|---|---:|---:|
| Prompt processing (pp128) | 6.17 tok/s | **19.21 tok/s** |
| Token generation (tg32) | 1.39 tok/s | **6.10 tok/s** |

These pressure-run throughput values use the pp128/tg32 boundary workload and are not directly comparable to the canonical pp512/tg128 throughput table above.

[Raw A/B evidence](results/openllama-7b-v2-ab/) · [Separate memory-pressure test](results/openllama-7b-v2-ram-constrained/) · [Methodology](docs/MEMORY_BENCHMARKING.md)

`llama.cpp` is substantially faster in these tests; MemVanta focuses on the **memory-efficiency side of CPU inference**.

## Low-memory CPU LLM inference

MemVanta focuses on:

- quantized **GGUF inference** on memory-constrained CPUs
- mmap-backed model access and paged KV cache
- Q4/Q8 and AVX2/FMA optimization while preserving memory efficiency

Current trained-model execution supports GGUF files with `general.architecture=llama`.

The GGUF container/parser is intentionally architecture-neutral. CI also validates parsing a pinned `general.architecture=qwen2` GGUF with `memvanta_gguf_inspect`; this is **container compatibility evidence only**, not a claim that Qwen2 inference is implemented.

## Adaptive prefetch and performance guardrails

MemVanta uses **byte-bounded adaptive weight prefetching** with usefulness, memory-pressure, and latency feedback. Look-ahead is constrained by an explicit hot-set budget, and real-model validation checks that prefetching preserves deterministic output while staying inside the configured memory bound.

The adaptive decision logic is isolated from cache ownership and I/O so policy behavior can be unit-tested independently. Performance profiling separates **prefill, decode, FFN, QKV projection, attention output projection, and output-head costs**.

Release builds can use interprocedural optimization where supported to attack hot Q4/Q8 projection overhead without introducing persistent model copies or an unbounded cache. CPU-kernel/compiler changes are evaluated with repeated same-runner OpenLLaMA 7B A/B tests so changes that regress throughput or memory are rejected before promotion. Experimental ideas that fail these gates are treated as negative results rather than promoted optimizations.

## Correctness, reliability, and portability

The validation stack includes:

- exhaustive FP16 conversion coverage, including subnormals, rounding boundaries, NaN, and infinity handling
- explicit GGUF parser, string, array, allocation, workspace, KV-page, and offset bounds
- deterministic multi-thread model checks and pinned external-reference comparisons
- concurrency and prefetch lifecycle stress testing
- AddressSanitizer, UndefinedBehaviorSanitizer, ThreadSanitizer, and GGUF fuzz-smoke coverage
- portable x86 runtime dispatch plus optimized AVX2/FMA paths where supported
- ARM64 cross-build and QEMU correctness coverage compiled for ARMv8 SIMD/NEON-capable code generation
- trained-model validation on pinned small and 7B-class Llama-family GGUF models
- non-Llama Qwen2 GGUF container/parser validation with claims explicitly scoped away from execution support

These guardrails are designed to keep memory-efficiency work from weakening numerical correctness, determinism, portability, or benchmark claim discipline.

## Contributing

Independent benchmark reproductions, CPU kernel optimizations, GGUF compatibility testing, profiling, and well-documented negative results are welcome.

[Contributing](CONTRIBUTING.md) · [Architecture](docs/ARCHITECTURE.md) · [All evidence](results/) · [Citation](CITATION.cff) · [License](LICENSE)

---

**Status:** Active research prototype with trained-model evidence up to 7B. Results are scoped to the tested models, settings, and hosts; independent third-party reproduction is still needed.
