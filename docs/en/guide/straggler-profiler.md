# Straggler Analysis

The Megatron backend ships an always-on, low-overhead straggler profiler. Every training rank brackets its own forward / backward / optimizer / gradient-sync / parameter all-gather segments with CUDA events on the compute stream (so "GPU time" below includes any host launch gaps inside the bracket), and every K rollouts all ranks Gloo-all-gather one fixed-length statistics vector to the primary rank, which decides whether a rank is holding the group back, why, and writes the verdict to metrics, the timeline and the log.

This is not `torch.profiler`: no kernel tracing, no `cuda.synchronize()`, no change to compute/communication overlap. It is meant to stay enabled in production runs.

## Enabling

The profiler is off by default. Pass the environment variables to the training actors through the Ray runtime environment:

```yaml
# configs/env.yaml
env_vars:
  RELAX_STRAGGLER_PROFILER: "1"
  RELAX_STRAGGLER_REPORT_INTERVAL: "10"
```

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RELAX_STRAGGLER_PROFILER` | bool | false | Master switch. |
| `RELAX_STRAGGLER_REPORT_INTERVAL` | int | 10 | Rollouts per cross-rank gather + analysis (one "window"). A rollout may contain multiple optimizer steps; reported ms/step values are per rollout. |
| `RELAX_STRAGGLER_Z_THRESHOLD` | float | 3.0 | Robust z-score (median / MAD within the peer group) a candidate must exceed. |
| `RELAX_STRAGGLER_REL_THRESHOLD` | float | 0.10 | Minimum relative excess over the peer median. |
| `RELAX_STRAGGLER_PERSIST_WINDOWS` | int | 3 | Consecutive windows a rank must qualify before it is flagged (applies to every reason); filters one-off spikes (checkpointing, GC). |

Each actor-role process logs `Straggler profiler enabled: report_interval=… primary=… meta=RankMeta(rank=…, dp=…, tp=…, pp=…, cp=…, ep=…, host=…, device=…)` at start-up. Critic / reference / `actor_fwd` actors never run training steps and are not profiled.

## What you get

### Log (primary rank)

One INFO line per window without alerts:

```text
[straggler] step=29 no straggler; self median 734.6 ms max 745.6 ms (rank 0, +2%),
latest to grad-sync rank 0 by 13.4 ms (peers idle 45.9 ms), pp_stage_imbalance 1.00, overhead 0.10 ms/step
```

When there are alerts, the window in which a rank's reason changes logs one WARNING, and every window that still has alerts logs an INFO per-rank table. Every reason needs `PERSIST_WINDOWS` consecutive windows:

```text
[straggler] step=14 rank 0 (rank0_dp0_tp0_pp0, host=…, gpu=0) reaches the DP grad-sync 433.0 ms/step
after its DP peers (peers idle 451.4 ms/step = 44% of their compute, z=8.9) for 3 windows;
its own GPU compute is 0.99x peers, gc 1.8 ms, cpu/gpu fwd 1.26x -> late_arrival (host-side stall)
[straggler] step=14 per-rank window (ms/step):
 rank | tag               | host | device | self_ms | wait_ms | late_ms | tokens  | ms_per_ktok | gc_ms | reason
    0 | rank0_dp0_tp0_pp0 | …    |      0 |  1021.2 |    26.1 |   433.0 | 15610.8 |        65.4 |   1.8 | late_arrival
    1 | rank1_dp1_tp0_pp0 | …    |      1 |  1083.4 |   404.3 |    54.7 | 15534.4 |        69.7 |   2.2 | none
```

### Metrics (same step key as `perf/*`; TensorBoard / WandB / ClearML)

These scalars go out in the same `tracking_utils.log` call as the step's `perf/*` metrics, never in a request of their own: with `--use-metrics-service` every log call is a synchronous HTTP request on the training thread.

| Metric | Meaning |
|--------|---------|
| `straggler/{fwd,bwd,optim,pp_recv,pp_send,dp_grad_sync,dp_param_gather,lp_fwd,lp_pp_recv,lp_pp_send}/{median_ms,max_ms,max_rank,spread}` | Per-segment GPU time (ms/step): median and global max across ranks; `max_rank` / `spread` are the rank with the largest excess over *its PP-stage peers* and that excess (`value/peer_median − 1`). `lp_*` are the same segments during log-prob / forward-only passes. `dp_param_gather` is only populated when `--overlap-param-gather` is off (Megatron does not time the overlapped path). |
| `straggler/self/*`, `straggler/self_per_ktok/*` | Own compute time (`fwd + bwd + optim`) and its per-token normalisation; the latter is omitted, like `tokens/*`, when any rank reports zero tokens. |
| `straggler/wait/{median_ms,max_ms,max_rank,spread}` | Time spent waiting for peers (`pp_recv + dp_grad_sync + dp_param_gather`). |
| `straggler/late/max_ms`, `straggler/late/max_rank`, `straggler/late/peer_idle_ms` | The rank that reaches the DP gradient sync last, by how much, and how long its peers idle for it per step. |
| `straggler/tokens/{median,max,spread}` | Token-count imbalance: `median` / `max` are global, `spread` is the largest excess within a stage; omitted when any rank reports zero tokens. |
| `straggler/pp_stage_imbalance` | `max/min` of per-stage median compute time; independent of any slow device. |
| `straggler/gc/{median_ms,max_ms}` | Python GC pauses. |
| `straggler/flagged/count`, `straggler/flagged/rank` (−1 if none), `straggler/flagged/reason` | Alert state. Reason codes: 0 none, 1 slow_device, 2 data_imbalance, 3 upstream_wait, 4 cpu_bound, 5 late_arrival. |
| `straggler/waiting/count` | Ranks waiting on a slow upstream PP stage in this window (victims, not culprits; not counted in `flagged`). |
| `straggler/self_overhead_ms`, `straggler/gather_ms`, `straggler/analyze_ms`, `straggler/dropped_events` | The tool's own cost: CPU time per step spent draining events; gather time per window on the primary rank (the first window also includes one metadata gather); analysis time per window on the primary rank; event pairs dropped because the queue was full (normally 0). |

### Timeline

With `--timeline-dump-dir` set (the timeline is flushed through the metrics-service adapter, so `--use-metrics-service` must be on as well), every window appends one `straggler/<seg> rank{g}_dp{d}_tp{t}_pp{p}` event per rank per non-zero segment (`pid` = 2³⁰ + global rank, above any real pid; `tid` = segment index). In Perfetto that is one row per rank, so a longer `bwd` bar or a shorter `dp_grad_sync` bar is visible at a glance.

## Reading the reason

| reason | Rule | Where to look |
|--------|------|---------------|
| `slow_device` | Own GPU time well above the other ranks of the same PP stage, token count normal | `nvidia-smi -q -d CLOCK,PERFORMANCE,TEMPERATURE`, ECC, neighbours on the node, NVLink topology |
| `data_imbalance` | Own time high but normal per token; token count clearly higher | `--balance-data`, dynamic batch splitting, oversize samples |
| `cpu_bound` | Own time high *and* Python GC pauses during the step ≥ 5 % of it | `gc.freeze()`, data threads contending for the GIL, per-token Python loops |
| `upstream_wait` | Own time normal but `pp_recv` well above the same-stage peers, by at least 5 % of the stage's median own time | The flagged rank on the upstream stage, or `pp_stage_imbalance` (uneven layer split) |
| `late_arrival` | Own GPU time normal, but its `dp_grad_sync` bracket is much *shorter* than its DP peers' — a collective finishes for everyone at once, so the last rank to arrive has the shortest bracket and everyone else's bracket contains the wait for it | Host-side gaps between kernels: data fetch, synchronous I/O / HTTP, GIL, launch gaps. `py-spy dump --pid <pid>` on that rank is the quickest confirmation |

`late_arrival` is easy to miss: any rule that only looks at GPU time cannot see it. It showed up in the very first real job the prototype ran on — rank 0's GPU time matched its peers, yet it reached every gradient sync 0.4–1.5 s late because the primary rank was posting logging metrics over synchronous HTTP on the training thread.

## Overhead

- One pair of non-blocking `cudaEventRecord` per Megatron timer call site (events are pooled): one pair per micro-batch for forward and for backward, plus roughly ten pairs per step for gradient sync, parameter all-gather and the optimizer phases.
- At the end of each step completed events are read lazily with `event.query()` and accumulated into the window vector: `straggler/self_overhead_ms` measures 0.12–0.15 ms/step on 8×A100.
- One Gloo `all_gather` (18 float64 per rank) plus the analysis on the primary rank every `REPORT_INTERVAL` rollouts, reported as `gather_ms` and `analyze_ms`: < 1 ms/step amortised. The analysis is O(n log n) in the stage size: a single-machine CPU micro-benchmark gives 2.4 ms per window at 8 ranks, 10 ms at 512 and 35 ms at 2048. The other ranks wait for the primary at the next collective.
- Each timer call (start / stop plus two `event.record()`) costs about 30 µs. On an idle GPU of the A100 test machine, replaying Megatron's call sequence with the worst case of 17 timers per step (3 micro-batches): in a kernel-launch-bound loop the timer calls plus the end-of-step readout add 0.80 ms/step of wall time (max 0.97 ms); in a compute-bound loop they add only 0.09 ms.
- Adding everything up and assuming all of it sits on the critical path gives about 1.2 ms/step typical and 1.9 ms/step worst case, i.e. 0.10 % and 0.17 % of a 1.15 s step. This is a summed upper-bound estimate.
- End to end: 8×A100 / A800, Qwen3-0.6B SFT (DP8, step ≈ 1.15 s — a small model, the case most sensitive to timing overhead). Comparing separate profiler-on and profiler-off runs is limited by ~1 % run-to-run variation, so the profiler is instead toggled every 10 rollouts within a run, with a second run in the opposite phase, so each 10-step block is measured once on and once off on identical data. With the metrics service off, four pairs give −0.04 % / +2.04 % / −0.16 % / −0.15 %; A/A controls (profiler off in both runs) on three hosts give +0.41 % / −0.70 % / −1.41 %. Turning the profiler on does not increase generation-2 GC count. Summing every cost item and assuming they all sit on the critical path bounds the overhead at ≈ 0.17 % for 8 ranks.

## Where it lives

- `relax/utils/straggler/timers.py`: the non-blocking drop-in for Megatron's `config.timers` (Megatron's own `Timer.start/stop` call `cuda.synchronize()`, which is why Relax sets it to `None` when the profiler is off).
- `relax/utils/straggler/stats.py`: the statistics vector layout and the Megatron timer name → segment map.
- `relax/utils/straggler/collector.py`: per-rank event pool, lazy read-out, GC callbacks, Gloo gather.
- `relax/utils/straggler/detector.py`: pure-numpy window analysis, unit-testable without a GPU.
- `relax/utils/straggler/reporter.py`: metrics / timeline / log output.
- Wiring: `relax/backends/megatron/model.py` (`config.timers = straggler_timers(...)` at three sites) and `relax/backends/megatron/actor.py` (`install_straggler_collector`, per-step `_straggler_end_step`).
