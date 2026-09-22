# 2025–2026 CPU LLM inference research notes

This note records research directions considered for MemVanta's memory-first CPU inference roadmap. It is intentionally separated from benchmark claims: repository measurements remain the authority for MemVanta performance.

## Directions relevant to MemVanta

Recent CPU/edge LLM systems research continues to emphasize that low-bit inference performance is dominated by data movement, cache reuse, specialized low-bit microkernels, and minimizing conversion/dispatch overhead. Work on vector/table-lookup low-bit kernels (for example Vec-LUT/T-MAC-style approaches) suggests a possible longer-term route for ultra-low-bit CPU execution, while 2025–2026 edge/CPU systems work also reinforces careful fusion and in-register reuse.

For MemVanta, these ideas must be filtered through two constraints: standard GGUF compatibility and the existing low-residency architecture. A research result that assumes a different quantization format, full weight residency, AVX-512-only hardware, or model retraining is not an immediate drop-in optimization.

## Immediate implications

1. **Profile real model execution first.** MemVanta's own 7B profile currently identifies projection/FFN work as the dominant cost, so kernel/scheduling improvements should be judged there rather than on synthetic microbenchmarks alone.
2. **Reduce orchestration and data movement before changing formats.** Reusing existing persistent worker threads, reducing repeated dispatch, and improving cache-friendly traversal are low-risk because they can preserve the current GGUF representation and memory envelope.
3. **Treat lookup-based ultra-low-bit kernels as experimental.** They are promising for a future optional path, but adopting them would require format/runtime-dispatch work plus correctness and RSS validation.
4. **Prefer bounded fusion.** Fusion that removes repeated memory passes can be valuable, but prior MemVanta evidence already shows that a fused batch-8 Q4 FFN candidate regressed, so fusion must be validated rather than assumed beneficial.
5. **Do not repeat rejected activation quantization blindly.** Existing MemVanta evidence shows Q8 activation prefill was materially slower in the tested 7B workload.

## Candidate order

Near term:
- persistent decode scheduling / reduced parallel-region overhead;
- grouped projection dispatch with one worker-pool synchronization point;
- Q4/Q8 AVX2/FMA accumulator/load scheduling guided by compiler/perf evidence;
- thread-count scaling and false-sharing checks;
- cache-local traversal improvements that do not enlarge residency.

Longer term research track:
- optional lookup-based low-bit kernels inspired by recent CPU/edge work;
- bounded operator fusion only where real-model A/B evidence supports it;
- additional ISA-specific kernels behind runtime dispatch, with portable fallback.

No research idea should change the canonical README benchmark until reproduced by the repository's machine-readable benchmark pipeline.
