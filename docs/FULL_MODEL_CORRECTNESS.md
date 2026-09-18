# Full-model correctness policy

MemVanta treats real-model correctness as a separate requirement from throughput and memory benchmarks.

## Blocking checks

The `Full Model Correctness` workflow downloads the exact pinned TinyStories Llama GGUF used by the existing benchmark suite and verifies its SHA-256 before inference. It then requires:

- the pinned greedy continuation in `tests/reference/stories15m-greedy.txt` to remain unchanged for both 1-thread and 4-thread execution;
- the complete greedy output to be identical between 1-thread and 4-thread execution;
- `memvanta_eval` to produce finite NLL/perplexity values;
- greedy token IDs to match exactly across 1-thread and 4-thread execution; and
- average NLL to remain within a 0.1% relative tolerance across those thread counts.

The 0.1% NLL guardrail is intentionally numerical rather than bitwise: the first hosted validation observed about 0.068% average-NLL drift between 1-thread and 4-thread evaluation while greedy token IDs remained exactly equal. The workflow records the observed drift in its evidence artifact and fails if it exceeds the guardrail.

The golden continuation is a **MemVanta regression fixture**, not a claim that MemVanta is numerically identical to another runtime. Any intentional model-semantics change that updates this fixture should include evidence explaining why the new sequence is more correct.

## External reference evidence

The same workflow builds the repository's pinned llama.cpp revision and runs deterministic greedy generation against the exact same GGUF. Its decoded continuation and the MemVanta continuation are written to `external-reference-status.json` with an explicit equality flag.

Decoded-text equality is currently recorded as evidence rather than asserted as a blocking gate. Different decoded text can arise from a genuine model-math/tokenizer bug, but also from runtime-specific tokenization or CLI semantics. A future blocking cross-runtime parity gate should compare token IDs or logits from explicitly aligned prompts/tokenization and document numerical tolerances rather than silently assuming two CLI text streams are equivalent.

## Benchmark relationship

Correctness gates do not replace the existing A/B performance, memory, sanitizer, fuzzing, or real-model benchmark workflows. Performance optimizations should pass the correctness workflow before their speed or memory results are considered for promotion.
