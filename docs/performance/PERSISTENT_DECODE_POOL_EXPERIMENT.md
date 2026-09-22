# Persistent decode worker-pool experiment

## Hypothesis

The single-token decode path performs repeated projection matvecs for QKV, attention output, FFN gate/up/down, and the output head. MemVanta already owns a persistent `WorkerPool`, but the decode path previously passed `nullptr` to these kernels, causing `parallel_rows` to fall back to transient OpenMP regions.

This candidate reuses the existing persistent pool for decode projections only. It does not change quantization math, tensor layout, model residency, prefill batching, KV-cache policy, or weight-cache budgets.

## Evidence motivating the test

The published 7B throughput profile identifies projection kernels as the dominant cost, with FFN projection work the largest component. Prior repository experiments also show that Q8 activation prefill and a fused batch-8 Q4 FFN kernel were not wins, so those approaches are intentionally not repeated here.

## Acceptance criteria

Keep the candidate only if:

- correctness and sanitizer/model validation remain green;
- repeated real-model evidence does not show a meaningful throughput regression;
- decode improves beyond normal hosted-runner noise where possible;
- peak RSS remains effectively flat (target <=2% growth);
- no new full-model or unbounded cache residency is introduced.

If the candidate is neutral or regresses, revert it rather than weakening performance gates.
