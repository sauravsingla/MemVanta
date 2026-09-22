# 2025–2026 CPU LLM inference research notes

This note records research directions considered for MemVanta's memory-first CPU inference roadmap. It is intentionally separated from benchmark claims: repository measurements remain the authority for MemVanta performance.

## Research reviewed

### Pushing the Envelope of LLM Inference on AI-PC — 2025

Georganas, Kalamkar, and Heinecke take a bottom-up CPU approach to 1-bit and 2-bit LLM microkernels and report up to 2.2x end-to-end speedup over bitnet.cpp on the tested platforms. The useful lesson for MemVanta is methodological: optimize the actual low-bit CPU microkernel first, then validate the change end-to-end rather than inferring model speed from a synthetic kernel alone.

- arXiv:2508.06753
- https://arxiv.org/abs/2508.06753

### Vec-LUT — 2025

Vec-LUT identifies memory-bandwidth under-utilization in scalar lookup-table inference and introduces vector LUTs, a LUT-centric tensor layout, and cache-aware streamed lookup. The paper reports up to 4.2x improvement over its tested baselines and integrates the implementation into llama.cpp.

For MemVanta this is a longer-term research direction rather than a drop-in Q4_0 optimization: adopting it would require a new optional kernel/layout path plus compatibility, correctness, RSS, and portability validation.

- arXiv:2512.06443
- https://arxiv.org/abs/2512.06443

### T-SAR — 2025

T-SAR explores in-register LUT generation and SIMD-register reuse for ternary CPU inference. It reinforces the value of keeping hot low-bit operations in registers and reducing memory traffic, but the proposed design includes SIMD hardware modifications and targets ternary inference, so its reported gains must not be transferred to commodity MemVanta Q4_0 execution.

- arXiv:2511.13676
- https://arxiv.org/abs/2511.13676

### FairyFuse — 2026

FairyFuse fuses ternary sub-GEMVs into AVX-512 kernels and reports 32.4 tokens/s on its Xeon test system and 1.24x throughput versus its llama.cpp Q4_K_M comparison. Its key lesson for MemVanta is to remove memory passes and keep work in registers where possible. It is not directly comparable to MemVanta because it uses ternary weights, AVX-512, a different runtime/model path, and a different benchmark environment.

- arXiv:2604.20913
- https://arxiv.org/abs/2604.20913

### T-MAC — EuroSys 2025 reference

T-MAC uses lookup tables to execute low-bit mixed-precision CPU matrix operations without conventional weight dequantization. The preprint originated in 2024 and the work was accepted at EuroSys 2025; the implementation has continued to evolve and includes low-bit CPU paths and GGUF-related integrations.

This is relevant to a future optional MemVanta kernel track, but any adoption must preserve standard model compatibility and must be benchmarked under MemVanta's own memory envelope rather than importing external speedup claims.

- arXiv:2407.00088
- https://arxiv.org/abs/2407.00088
- https://github.com/microsoft/T-MAC

## What the research means for MemVanta

The common themes are consistent with MemVanta's own profiling evidence:

1. **Profile real model execution first.** MemVanta's 7B profile identifies projection/FFN work as the dominant cost, so kernel changes must be judged on the real workload and not only on microbenchmarks.
2. **Reduce data movement and register round-trips.** This is the most directly applicable near-term theme. PR #50 validated one such change: keeping the native AVX2 Q4×Q8 dot block in registers improved repeated same-runner 7B decode by 10.51% and FFN GEMM by 2.47% with effectively flat RSS.
3. **Do not replace the current OpenMP decode path wholesale with the existing custom worker pool.** PR #50 tested that hypothesis first; decode regressed materially and the code was reverted. Future scheduling work should preserve the utilization characteristics of the proven OpenMP path unless evidence shows otherwise.
4. **Treat LUT-based ultra-low-bit kernels as an optional research track.** Vec-LUT, T-MAC, and T-SAR are promising, but their formats, hardware assumptions, or execution models differ from MemVanta's current GGUF Q4_0 path.
5. **Prefer bounded fusion only after measurement.** Fusion can reduce memory traffic, but MemVanta's earlier fused batch-8 Q4 FFN candidate regressed and was reverted. Fusion is a hypothesis, not an automatic optimization.
6. **Do not repeat Q8 activation prefill blindly.** Existing MemVanta 7B evidence shows that tested Q8 activation prefill was materially slower.

## Candidate order after PR #50

Near term:

- further profile-guided AVX2/FMA work in the Q4×Q8 FFN decode kernels, especially `ffn_down`;
- investigate whether grouped QKV and gate/up projections can share one OpenMP parallel region without reducing CPU utilization or changing arithmetic;
- inspect compiler output and hardware counters for register pressure, load stalls, cache misses, and integer-SIMD reduction overhead;
- test thread-count scaling, affinity, and false-sharing behavior on representative CPUs;
- improve cache-local traversal only where residency remains bounded and peak RSS stays flat;
- keep portable runtime-dispatch and ARM64 fallbacks independently validated.

Longer-term research track:

- optional LUT-based low-bit kernels behind runtime dispatch;
- additional ISA-specific kernels, including AVX-512 only as an optional path with portable fallback;
- bounded operator fusion only where repeated real-model A/B evidence supports it;
- format/layout experiments only when they can coexist with GGUF compatibility and explicit memory accounting.

No external paper result should be presented as a MemVanta performance claim. Canonical README benchmark values must continue to come only from MemVanta's machine-readable benchmark pipeline.
