# Contributing to MemVanta

Thanks for contributing to MemVanta.

## New contributor? Start here

These starter issues are deliberately small and useful to the project:

- [#53 — Add a benchmark system-information collector for reproduction reports](https://github.com/sauravsingla/MemVanta/issues/53)
- [#54 — Add a one-command local smoke script for new contributors](https://github.com/sauravsingla/MemVanta/issues/54)
- [#55 — Add a supported-platform and validation matrix for contributors](https://github.com/sauravsingla/MemVanta/issues/55)

Comment on the issue if you want to coordinate before starting. For concrete systems questions that do not yet belong in an issue, use [GitHub Discussions](https://github.com/sauravsingla/MemVanta/discussions) — useful topics include benchmark reproduction, CPU/compiler behavior, memory measurements, model compatibility and hardware results. Security reports should still follow `SECURITY.md`.

## Development setup

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Pull requests

- Keep changes focused and explain the motivation.
- Add or update tests for behavioral changes.
- For performance changes, include reproducible benchmark commands and raw results.
- Do not present synthetic-model results as trained-model results.
- Do not claim superiority over another runtime unless the comparison uses the same model, hardware, thread settings, context, and benchmark protocol.
- Preserve model-license and third-party attribution requirements.

## Benchmark evidence

For performance-related PRs, please report:

- CPU and memory configuration
- compiler and build type
- model name, exact file hash, quantization, and context size
- thread count and batch settings
- repetitions and warm-up policy
- prefill throughput, decode throughput, and peak RSS where applicable

Raw benchmark output is preferred over screenshots.

Independent benchmark reproductions — including results that narrow or contradict existing measurements — are welcome when the full environment and raw evidence are included.

## Coding style

MemVanta targets modern C++20. Prefer clear ownership, bounded memory use, explicit error handling, and deterministic tests. Avoid unnecessary dependencies in the core runtime.
