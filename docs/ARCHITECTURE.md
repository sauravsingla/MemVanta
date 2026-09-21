# MemVanta architecture

MemVanta is a memory-first CPU inference research runtime. The core design rule is that throughput work must not silently turn into full-model residency or weaken correctness evidence.

## Runtime layers

### GGUF container and mapped storage

- `gguf.*` validates GGUF structure, metadata limits, tensor ranges and quantized byte sizes.
- `mmap_file.*` owns mapped model access and advisory page hints.
- `tensor_store.*` exposes bounded slices without requiring full-model copies.

The GGUF parser is architecture-neutral. Real-model execution is currently narrower than parser compatibility and is documented separately in the README.

### Memory policy

- `lru_cache.*` owns the bounded copy cache.
- `prefetcher.*` owns asynchronous data movement/lifecycle.
- `prefetch_policy.*` owns adaptive look-ahead decisions.
- `runtime.*` coordinates slices, telemetry and the selected policy.

The adaptive controller is intentionally separated from I/O and cache ownership. It receives one completed observation window and returns only a new depth plus an adjustment direction. This keeps policy tests deterministic and prevents benchmark-specific state from leaking into storage code.

### Numerical kernels

- `quant.*` and `quant_kernels.*` cover compact quantization primitives.
- `gguf_kernels.*` covers trained-model tensor projections and SIMD paths.
- `worker_pool.*` owns reusable CPU parallelism.

Release-only compiler/link optimization is permitted only when deterministic correctness remains green. Hot-path changes are evaluated with the existing repeated same-runner 7B A/B workflows, and peak RSS is reported with throughput.

### Model execution

- `llama_model.*` owns the currently supported trained-model execution graph.
- paged KV cache and tokenizer logic live beside that graph today; future architecture expansion should move these behind smaller interfaces rather than growing one monolithic model file.

## Validation layers

1. **Unit and sanitizer correctness**: Release/Debug, ASan/UBSan, TSan, parser limits, FP16 coverage, concurrency stress and fuzz smoke.
2. **Portable ISA validation**: native x86, portable x86 runtime dispatch, macOS, and an ARM64 cross/QEMU lane compiled for ARMv8 SIMD/NEON-capable code generation.
3. **Container architecture validation**: architecture-neutral GGUF inspection is exercised on a non-Llama GGUF in the model matrix. This validates parser/container compatibility only; it is not an execution-support claim.
4. **Trained-model correctness**: deterministic Llama-family model checks and pinned external-reference/tokenizer comparisons.
5. **Performance evidence**: repeated same-machine/same-workload A/B runs with throughput and peak RSS kept together.

## Benchmark claim hierarchy

The primary public memory result is the canonical repeated OpenLLaMA 7B peak-RSS comparison. The cgroup `MemoryMax` boundary is secondary systems evidence about reclaimable execution under pressure; it is not an exact physical-RAM minimum and must not replace the peak-RSS result in headlines.

## CI maintenance rule

Prefer adding jobs or matrix entries to an existing workflow over creating a new workflow. A dedicated workflow is justified only when it has materially different permissions, triggers, hardware requirements, or publication semantics. When a stricter matrix fully supersedes an older smoke workflow, remove the redundant workflow rather than keeping both indefinitely.
