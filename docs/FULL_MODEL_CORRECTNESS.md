# Full-model correctness policy

MemVanta treats real-model correctness as a separate requirement from throughput and memory benchmarks.

The `Full Model Correctness` workflow runs for relevant pull-request changes and for relevant source/configuration pushes to `main`. Result-only benchmark publication under `results/**` is intentionally outside its path filter, so evidence publishing does not repeatedly trigger the expensive full-model job.

## Blocking checks

The workflow downloads the exact pinned TinyStories Llama GGUF used by the existing benchmark suite and verifies its SHA-256 before inference. It then requires:

- the pinned greedy continuation in `tests/reference/stories15m-greedy.txt` to remain unchanged for both 1-thread and 4-thread execution;
- the complete greedy output to be identical between 1-thread and 4-thread execution;
- `memvanta_eval` to produce finite NLL/perplexity values;
- greedy token IDs to match exactly across 1-thread and 4-thread execution; and
- average NLL to remain within a 0.1% relative tolerance across those thread counts.

The 0.1% NLL guardrail is intentionally numerical rather than bitwise: the first hosted validation observed about 0.068% average-NLL drift between 1-thread and 4-thread evaluation while greedy token IDs remained exactly equal. The workflow records the observed drift in its evidence artifact and fails if it exceeds the guardrail.

The golden continuation is a **MemVanta regression fixture**, not a claim that MemVanta is numerically identical to another runtime. Any intentional model-semantics change that updates this fixture should include evidence explaining why the new sequence is more correct.

## External reference evidence

The same workflow builds the repository's pinned llama.cpp revision with nonessential tests, examples, and web UI disabled. It builds both `llama-cli` and `llama-tokenize`.

Before comparing generated text, the workflow records exact prompt-token IDs for `Once upon a time` from:

- MemVanta with BOS requested;
- MemVanta without BOS;
- llama.cpp with the model's default BOS policy; and
- llama.cpp with BOS disabled.

Those arrays and equality flags are stored in `tokenizer-parity.json`. This makes a cross-runtime generation mismatch diagnosable as a tokenizer/BOS issue versus a later model-math divergence. The initial tokenizer comparison is evidence-producing rather than blocking so that the pinned model's exact semantics can be established without hiding a discrepancy; once aligned semantics are demonstrated, the matching configuration should become a blocking parity gate.

The workflow also runs deterministic greedy generation against the same GGUF and writes the llama.cpp and MemVanta decoded continuations to `external-reference-status.json` with an explicit equality flag.

Decoded-text equality remains evidence rather than a blocking assertion until prompt token IDs and numerical/model semantics are explicitly aligned. A blocking cross-runtime model gate should then compare aligned token IDs and, where practical, logits/top-k values with documented numerical tolerances rather than assuming two CLI text streams are equivalent.

## Benchmark relationship

Correctness gates do not replace the existing A/B performance, memory, sanitizer, fuzzing, or real-model benchmark workflows. Performance optimizations should pass the correctness workflow before their speed or memory results are considered for promotion.
