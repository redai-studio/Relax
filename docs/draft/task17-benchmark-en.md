# No.17 — Frozen Condition Cache Benchmark

Reusing frozen conditions improved median steady-state training throughput by **7.7778% for SenseNova** and **14.5436% for Qwen-Image-Edit**. Low-reuse SenseNova throughput fell by **20.6834%**, so reuse should be enabled for workloads that repeat conditions and fit the working set in memory.

## Results

Each entry compares the same model, initial LoRA/optimizer state, seed, sample order and precision. Gain is `online_seconds / cached_seconds - 1`, a throughput increase rather than a percentage reduction in latency. All three pairs are retained.

| Workload | Pair 1 | Pair 2 | Pair 3 | Median |
|---|---:|---:|---:|---:|
| SenseNova, 128 repeated pairs | +9.2914% | +5.4866% | +7.7778% | **+7.7778%** |
| Qwen-Image-Edit, 128 repeated pairs | +14.4331% | +15.2332% | +14.5436% | **+14.5436%** |
| SenseNova, 640 distinct samples | −20.6834% | −19.3819% | −22.3100% | **−20.6834%** |

The 18 segments each contain 16 cold-cache warmup updates followed by 64 measured updates. B8 gives 640 sample occurrences per segment. The repeated workloads traverse 128 real editing pairs five times; the low-reuse workload consumes 640 distinct samples from a 1,024-row dataset. The actual order is AB / BA / AB for each workload, with A online and B cached.

Both repeated workloads start empty, encode 128 conditions and then serve 512 hits. Neither evicts entries. SenseNova retains about 17.994 GiB; Qwen retains about 0.692 GiB. Low reuse has zero hits, 640 writes and 201 evictions, ending at 63.955 GiB within the 64 GiB per-process limit.

## Measurement conditions

The measurements cover real `FSDPSFTTrainer` updates: image preparation waits, forward, backward, gradient reduction/clipping, Adam and scheduler. Model initialization is reported separately; Ray/TQ orchestration and checkpoint I/O are outside the steady-state interval.

The run used H200 GPUs with 143,771 MiB each, Torch 2.9.0+cu128, SDPA, deterministic algorithms, BF16 computation, FP32 master weights/reduction, LoRA rank/alpha 32, learning rate 1e-4 and seed 20261030. SenseNova and Qwen used separate world1 processes on GPU2 and GPU3, with CPU affinity 40–59 and 60–79. The lanes ran concurrently and shared NUMA0 host bandwidth. Qwen pair 3 A overlapped the SenseNova lane for about 81.95% of its process duration; B ran after that lane ended. The first two Qwen pairs were almost fully concurrent and showed comparable gains.

The new reproduction runner defaults to one GPU with all 28 stages executed serially. The recorded run above used two concurrent world1 lanes.

## Cold costs and resources

Cold-cache cumulative training time becomes faster and remains faster from updates 24/28/57 in the three SenseNova pairs, and 18/1/13 for Qwen. All six repeated-workload cached paths finish the 80-update window sooner, including when comparing initialization-plus-training reports and whole-process receipts. No low-reuse pair reaches sustained payback within the window.

| Resource | SenseNova repeated A / B | Qwen repeated A / B | SenseNova low reuse A / B |
|---|---|---|---|
| Peak GPU allocation | about 38.50 / 38.50 GiB | about 64.35 / 64.35 GiB | about 38.50 / 38.50 GiB |
| Maximum sampled process RSS | 2.64 / 20.76 GiB | 2.78 / 3.55 GiB | 2.65 / 68.75 GiB |
| Cache residency at end | 0 / 17.994 GiB | 0 / 0.692 GiB | 0 / 63.955 GiB |

RSS is sampled during updates, model-loading high-water marks are recorded separately, and storage samples capture run-file sizes plus global filesystem/SHM availability at stage boundaries.

## Numerical evidence and reproduction

The numerical reports compare online A1/A2 and cached cold/warm trajectories from the same initial state. Loss, per-parameter gradients and FP32 update deltas are reported bitwise exact for both models, with matching RNG states and exact cached tensor payloads. The comparison covers one update on eight real samples at seed 20261030 and B8.

The public evidence is under `examples/frozen_condition_cache/results/h200-20261103/`. It preserves the comparison reports and 18 step-by-step performance records. The public copies redact machine paths and identifiers; original file hashes are recorded in the evidence manifest. See that directory's README and `summary.json` for the file index.

The review bundle omits original trajectory tensors, initial-state tensors and large DCP files. A downloadable full-tensor evidence URL remains to be added; the JSON reports alone cannot independently reconstruct those tensor comparisons.

To reproduce, follow [Frozen Condition Cache](../en/guide/frozen-condition-cache.md) and the README in `examples/frozen_condition_cache/`. Model/data versions and file hashes are pinned in that example.
