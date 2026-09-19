# OpenLLaMA 7B v2 fine RAM-boundary experiment

Linux cgroup-v2 `MemoryMax` search, swap disabled. Same verified Q4_0 GGUF, CPU-only, 4 threads, pp128/tg32, context 768, batch 32, F16 KV.

- Resolution: 32 MiB
- MemVanta: OOM at 128 MiB; succeeds at 160 MiB
- llama.cpp: OOM at 3616 MiB; succeeds at 3648 MiB
- Lowest-confirmed-success ceiling difference: 3488 MiB
- MemVanta lower confirmed-success ceiling: 95.61%

Each final edge is repeated 2 times. This measures an execution-under-pressure boundary on this hosted runner; it is not an exact physical-RAM minimum and not a throughput benchmark.
