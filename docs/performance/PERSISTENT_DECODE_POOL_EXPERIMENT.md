# Persistent decode worker-pool experiment

## Hypothesis

The single-token decode path performs repeated projection matvecs for QKV, attention output, FFN gate/up/down, and the output head. MemVanta already owns a persistent `WorkerPool`, but the decode path previously passed `nullptr` to these kernels, causing `parallel_rows` to fall back to OpenMP regions.

The candidate reused the existing persistent pool for decode projections only. It did not change quantization math, tensor layout, model residency, prefill batching, KV-cache policy, or weight-cache budgets.

## Evidence motivating the test

The published 7B throughput profile identifies projection kernels as the dominant cost, with FFN projection work the largest component. Prior repository experiments also show that Q8 activation prefill and a fused batch-8 Q4 FFN kernel were not wins, so those approaches were intentionally not repeated here.

## Result: rejected and reverted

PR #50 profiled the candidate on the exact verified OpenLLaMA 7B v2 Q4_0 workload used by the repository throughput-profile workflow.

Candidate profile:

- prefill: 46214.00 ms
- decode total: 19668.10 ms
- projection kernels: 64720.53 ms
- FFN GEMM: 39508.62 ms
- peak RSS: 3819144 KiB
- process CPU: 194%

The most recent published baseline profile before the experiment reported roughly 9.8 s decode with the same benchmark shape, while the candidate took 19.7 s. The regression is far larger than normal hosted-runner noise. RSS remained effectively flat, but throughput failed the experiment's acceptance criteria.

The code change was therefore reverted. The result suggests that the existing OpenMP path is substantially better at sustaining CPU utilization for these decode projections than the current custom `WorkerPool` implementation.

## Follow-up implication

Do not replace OpenMP projection scheduling wholesale with the existing worker pool. Future scheduling work should instead profile narrower opportunities such as reducing synchronization inside grouped QKV/gate-up dispatch while preserving OpenMP's current utilization, or focus directly on the dominant Q4/Q8 FFN microkernels and data movement.
