# Straggler Analysis

The Megatron backend ships an always-on, low-overhead straggler profiler. Every training rank brackets its own forward / backward / optimizer / gradient-sync / parameter all-gather segments with CUDA events on the compute stream (so "GPU time" below includes any host launch gaps inside the bracket), and every K rollouts all ranks Gloo-all-gather one fixed-length statistics vector to the primary rank, where a background thread decides whether a rank is holding the group back, why, and writes the verdict to metrics, the timeline and the log.

This is not `torch.profiler`: no kernel tracing, no `cuda.synchronize()`, no change to compute/communication overlap. It is meant to stay enabled in production runs. No exception inside the profiler ever reaches the training loop: failures are counted, and repeated failures switch the profiler off on every rank together.

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
| `RELAX_STRAGGLER_RECOVER_WINDOWS` | int | 2 | Consecutive clean windows before a flagged rank's alert clears, so alerts do not flap; uncertain windows are not clean. |

Each actor-role process logs `Straggler profiler enabled: report_interval=… primary=… meta=RankMeta(rank=…, dp=…, tp=…, pp=…, cp=…, ep=…, host=…, device=…)` at start-up. Critic / reference / `actor_fwd` actors never run training steps and are not profiled.

## What you get

### Log (primary rank)

The primary rank's analysis thread logs as soon as a window's analysis ends; `step` is the step of the window's last rollout, without waiting for the next training step. One INFO line per window without alerts:

```text
[straggler] step=29 no straggler (uncertain 0); self median 734.6 ms max 745.6 ms (rank 0, +2%),
latest to grad-sync rank 0 by 13.4 ms (peers idle 45.9 ms), pp_stage_imbalance 1.00, overhead 0.10 ms/step
```

When there are alerts, the window in which a rank's reason changes logs one WARNING, and every window that still has alerts logs an INFO per-rank table. Every reason needs `PERSIST_WINDOWS` consecutive windows to be raised and `RECOVER_WINDOWS` consecutive clean windows to clear (the clearing logs another WARNING: `rank 5 (…) recovered from slow_device after 2 clean windows`):

```text
[straggler] step=14 rank 0 (rank0_dp0_tp0_pp0, host=…, gpu=0) reaches the DP grad-sync 433.0 ms/step
after its DP peers (peers idle 451.4 ms/step = 44% of their compute, z=8.9) for 3 windows;
its own GPU compute is 0.99x peers, gc 1.8 ms, cpu/gpu fwd 1.26x -> late_arrival (host-side stall)
[straggler] step=14 per-rank window (ms/step):
 rank | tag               | host | device | self_ms | wait_ms | late_ms | tokens  | ms_per_ktok | gc_ms | reason
    0 | rank0_dp0_tp0_pp0 | …    |      0 |  1021.2 |    26.1 |   433.0 | 15610.8 |        65.4 |   1.8 | late_arrival
    1 | rank1_dp1_tp0_pp0 | …    |      1 |  1083.4 |   404.3 |    54.7 | 15534.4 |        69.7 |   2.2 | none
```

A rank whose window cannot support a verdict is reported as uncertain with a cause (one WARNING when the cause changes) instead of silently passing as healthy:

```text
[straggler] step=19 rank 3 (rank3_dp3_tp0_pp0) is uncertain: 2 event pairs were still in flight when the window
closed; their time lands in the next window
```

| Cause | Meaning |
|-------|---------|
| `dropped_events` / `unread_events` / `profiler_errors` | That rank's timing for the window is incomplete: event pairs were dropped because the queue was full, were not yet readable when the window closed (they count in the next window), or the profiler caught errors on that rank. An unread `dp_grad_sync` bracket would make the rank look like a late arriver, so no verdict is drawn. |
| `no_peers` | Its PP stage has no other rank to compare with. |
| `missing_tokens` | It looks slow, but its stage has no token counts, so data imbalance cannot be told apart from a slow device. |

An uncertain window counts neither way: it breaks candidate streaks and is not a clean window for clearing an alert; an existing alert stays as it is.

### Metrics (same step key as `perf/*`; TensorBoard / WandB / ClearML)

These scalars go out in the same `tracking_utils.log` call as the `perf/*` metrics of the training thread's next `log_perf_data`, never in a request of their own: with `--use-metrics-service` every log call is a synchronous HTTP request on the training thread. Because the analysis runs on a background thread, a window's scalars normally ride along with the next rollout's `perf/*`; `straggler/window/{first,last}_rollout` say which window they belong to. The last rollout of a run waits for the last window's analysis before logging.

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
| `straggler/uncertain/count`, `straggler/uncertain/rank` (−1 if none), `straggler/uncertain/cause` | Number of uncertain ranks, the first one and its cause. Cause codes: 0 none, 1 dropped_events, 2 unread_events, 3 profiler_errors, 4 no_peers, 5 missing_tokens. |
| `straggler/recovered/count` | Ranks whose alert cleared in this window. |
| `straggler/window/first_rollout`, `straggler/window/last_rollout` | The window these scalars belong to (they are usually emitted one rollout later). |
| `straggler/latency/analyzed_ms`, `straggler/latency/emitted_ms`, `straggler/latency/prev_delivered_ms` | Report latency, all measured from the window close (before the gather): analysis done (the alert lines are logged at that moment), scalars handed to `log_perf_data` on the training thread, and the previous window's `log_perf_data` returned (with the metrics service: the service acknowledged it; a window cannot carry its own delivery time). Each delivered window also logs an INFO line `window rollouts 10-19: closed -> analyzed +… ms (alerts logged), emitted +… ms, delivered +… ms`. |
| `straggler/health/state`, `straggler/health/errors`, `straggler/health/degraded_ranks` | The profiler's own health: worst state across ranks (0 active, 1 degraded, 2 disabled), errors caught in the window, and ranks currently degraded. |
| `straggler/self_overhead_ms`, `straggler/gather_ms`, `straggler/analyze_ms`, `straggler/dropped_events`, `straggler/unread_events` | The tool's own cost: CPU time per step spent draining events; gather time per window on the primary rank (the first window also includes one metadata gather); analysis time per window on the background thread; event pairs dropped because the queue was full (normally 0); event pairs still unread when the window closed (normally 0). |

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

## When the profiler itself fails

Exceptions in the timer calls, the event read-out, the window gather, the analysis and the log callback are caught and counted; none reaches the training loop. A rank that has caught 3 errors, or has had dropped event pairs or errors in 3 consecutive windows, asks for the profiler to be switched off. The request travels in that rank's window vector (the `health` field), so every rank sees the same gathered table and switches off at the same window; no rank leaves on its own and strands the others in the next gather. Once off, the timer calls are skipped and no more gathers run; the primary rank emits `straggler/health/state = 2` and logs one WARNING naming the requesting rank (the reason is in that rank's log). Training carries on. A failed gather is only visible to the rank it failed on, so that rank stops joining later gathers on its own.

## Overhead

- One pair of non-blocking `cudaEventRecord` per Megatron timer call site (events are pooled): one pair per micro-batch for forward and for backward, plus roughly ten pairs per step for gradient sync, parameter all-gather and the optimizer phases.
- At the end of each step completed events are read lazily with `event.query()` and accumulated into the window vector: `straggler/self_overhead_ms` measures 0.12–0.15 ms/step on 8×A100.
- One Gloo `all_gather` (21 float64 per rank) every `REPORT_INTERVAL` rollouts, reported as `gather_ms`: < 1 ms/step amortised. The analysis runs on a background thread of the primary rank, reported as `analyze_ms`; it takes no training-thread time and no rank waits for it at the next collective. It is O(n log n) in the stage size: a single-machine CPU micro-benchmark gives 2.4 ms per window at 8 ranks, 10 ms at 512 and 35 ms at 2048.
- Each timer call (start / stop plus two `event.record()`) costs about 30 µs. On an idle GPU of the A100 test machine, replaying Megatron's call sequence with the worst case of 17 timers per step (3 micro-batches): in a kernel-launch-bound loop the timer calls plus the end-of-step readout add 0.80 ms/step of wall time (max 0.97 ms); in a compute-bound loop they add only 0.09 ms.
- Adding everything up and assuming all of it sits on the critical path gives about 1.2 ms/step typical and 1.9 ms/step worst case, i.e. 0.10 % and 0.17 % of a 1.15 s step. This is a summed upper-bound estimate.
- End to end: 8×A100 / A800, Qwen3-0.6B SFT (DP8, step ≈ 1.15 s — a small model, the case most sensitive to timing overhead). Comparing separate profiler-on and profiler-off runs is limited by ~1 % run-to-run variation, so the profiler is instead toggled every 10 rollouts within a run, with a second run in the opposite phase, so each 10-step block is measured once on and once off on identical data. With the metrics service off, four pairs give −0.04 % / +2.04 % / −0.16 % / −0.15 %; A/A controls (profiler off in both runs) on three hosts give +0.41 % / −0.70 % / −1.41 %. Turning the profiler on does not increase generation-2 GC count. Summing every cost item and assuming they all sit on the critical path bounds the overhead at ≈ 0.17 % for 8 ranks.

## Where it lives

- `relax/utils/straggler/timers.py`: the non-blocking drop-in for Megatron's `config.timers` (Megatron's own `Timer.start/stop` call `cuda.synchronize()`, which is why Relax sets it to `None` when the profiler is off).
- `relax/utils/straggler/stats.py`: the statistics vector layout and the Megatron timer name → segment map.
- `relax/utils/straggler/collector.py`: per-rank event pool, lazy read-out, GC callbacks, the per-window gather; torch-free, with events and gathers injected.
- `relax/utils/straggler/worker.py`: the background thread on the primary rank that runs the analyses in order.
- `relax/utils/straggler/detector.py`: pure-numpy window analysis (including uncertain verdicts and alert clearing), unit-testable without a GPU.
- `relax/utils/straggler/health.py`: the profiler's own health (active → degraded → disabled).
- `relax/utils/straggler/reporter.py`: metrics / timeline output (training thread), log output (background thread) and report latency.
- `relax/utils/straggler/runtime.py`: the only module that touches the accelerator (through `relax.utils.device`, so CUDA and NPU alike), the process groups or Megatron's parallel state; `StragglerProfiler` does the per-step reporting around `log_perf_data`.
- Wiring: `relax/backends/megatron/model.py` (`config.timers = straggler_timers(...)` at three sites) and `relax/backends/megatron/actor.py` (`install_straggler_profiler`, per-step `_log_perf_data`).
