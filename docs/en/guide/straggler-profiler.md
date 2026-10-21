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
| `RELAX_STRAGGLER_PERSIST_WINDOWS` | int | 3 | Consecutive windows a rank must qualify before it is flagged (applies to every reason); filters one-off spikes (checkpointing, GC); must be at least 1. |
| `RELAX_STRAGGLER_RECOVER_WINDOWS` | int | 2 | Consecutive clean windows before a flagged rank's alert clears, so alerts do not flap; uncertain windows are not clean; must be at least 1. |
| `RELAX_STRAGGLER_GATHER_TIMEOUT_S` | float | 60.0 | Timeout in seconds of the profiler's dedicated Gloo group. A gather is a few milliseconds of small CPU tensors; this only bounds how long the remaining ranks block once a rank has disappeared. |

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
[straggler] step=29 rank 3 (rank3_dp3_tp0_pp0, host=…, gpu=3) reaches the DP grad-sync 195.6 ms/step
after its DP peers (peers idle 201.4 ms/step = 21% of their compute, z=120.1) for 3 windows;
its own GPU forward is 0.99x peers, gc 1.8 ms, cpu/gpu fwd 1.55x -> late_arrival (host-side stall)
[straggler] step=29 per-rank window (ms/step):
 rank | tag               | host | device | self_ms | wait_ms | late_ms | tokens  | ms_per_ktok | gc_ms | reason       | uncertain
    0 | rank0_dp0_tp0_pp0 | …    |      0 |   941.0 |     6.4 |     0.4 | 14802.1 |        63.6 |   1.8 | none         |
    3 | rank3_dp3_tp0_pp0 | …    |      3 |   743.6 |     5.8 |   195.6 | 14592.7 |        51.0 |   1.8 | late_arrival |
```

The table is cut down to ranks 0 and 3; the real one has a row per rank.

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
| `no_wait_reference` | It waits on PP peers, but every other rank of its stage is flagged, late or uncertain, so there is no wait to compare with. |

An uncertain window counts neither way: it breaks candidate streaks and is not a clean window for clearing an alert; an existing alert stays as it is.

### Metrics (same step key as `perf/*`; TensorBoard / WandB / ClearML)

These scalars go out in the same `tracking_utils.log` call as the `perf/*` metrics of the training thread's next `log_perf_data`, never in a request of their own: with `--use-metrics-service` every log call is a synchronous HTTP request on the training thread. Because the analysis runs on a background thread, a window's scalars normally ride along with the next rollout's `perf/*`; `straggler/window/{first,last}_rollout` say which window they belong to. The last rollout of a run waits for the last window's analysis before logging. If the analysis falls behind and several windows finish in the same step (this does not happen in normal runs), each window still reaches the timeline and the log, but only the scalars of the latest window whose reporting succeeded reach the metrics (one log call carries one value per key); each window it supersedes logs a WARNING `… windows ready at once; only rollouts …-… reach the metrics` and counts as one profiler error, and a window whose reporting fails counts once, as that failure (see *When the profiler itself fails*).

| Metric | Meaning |
|--------|---------|
| `straggler/{fwd,bwd,optim,pp_recv,pp_send,dp_grad_sync,dp_param_gather,lp_fwd,lp_pp_recv,lp_pp_send}/{median_ms,max_ms,max_rank,spread}` | Per-segment GPU time (ms/step): median and global max across ranks; `max_rank` / `spread` are the rank with the largest excess over *its PP-stage peers* and that excess (`value/peer_median − 1`). `lp_*` are the same segments during log-prob / forward-only passes. `dp_param_gather` is only populated when `--overlap-param-gather` is off (Megatron does not time the overlapped path). |
| `straggler/self/*`, `straggler/self_per_ktok/*` | Own compute time (`fwd + bwd + optim`) and its per-token normalisation; the latter is omitted, like `tokens/*`, when any rank reports zero tokens. |
| `straggler/wait/{median_ms,max_ms,max_rank,spread}` | Time spent waiting for peers (`pp_recv + dp_grad_sync + dp_param_gather`). |
| `straggler/late/max_ms`, `straggler/late/max_rank`, `straggler/late/peer_idle_ms` | The rank that reaches the DP gradient sync last, by how much (compared on `bwd + dp_grad_sync`, see `late_arrival` below), and how long its peers idle for it per step. |
| `straggler/tokens/{median,max,spread}` | Token-count imbalance: `median` / `max` are global, `spread` is the largest excess within a stage; omitted when any rank reports zero tokens. |
| `straggler/pp_stage_imbalance` | `max/min` of per-stage median compute time; independent of any slow device. |
| `straggler/gc/{median_ms,max_ms}` | Python GC pauses. |
| `straggler/flagged/count`, `straggler/flagged/rank` (−1 if none), `straggler/flagged/reason` | Alert state. Reason codes: 0 none, 1 slow_device, 2 data_imbalance, 3 upstream_wait, 4 cpu_bound, 5 late_arrival. |
| `straggler/waiting/count` | Ranks waiting on a slow upstream PP stage in this window (victims, not culprits; not counted in `flagged`). |
| `straggler/uncertain/count`, `straggler/uncertain/rank` (−1 if none), `straggler/uncertain/cause` | Number of uncertain ranks, the first one and its cause. Cause codes: 0 none, 1 dropped_events, 2 unread_events, 3 profiler_errors, 4 no_peers, 5 missing_tokens, 6 no_wait_reference. |
| `straggler/recovered/count` | Ranks whose alert cleared in this window. |
| `straggler/window/first_rollout`, `straggler/window/last_rollout` | The window these scalars belong to (they are usually emitted one rollout later). |
| `straggler/latency/analyzed_ms` | Milliseconds from the window close (before the gather) to the end of its analysis, when the alert lines are logged. |
| `straggler/health/state`, `straggler/health/errors`, `straggler/health/degraded_ranks` | The profiler's own health: worst state across ranks (0 active, 1 degraded, 2 disabled), errors caught in the window, and ranks currently degraded. |
| `straggler/self_overhead_ms`, `straggler/gather_ms`, `straggler/analyze_ms`, `straggler/dropped_events`, `straggler/unread_events` | The tool's own cost: CPU time per step spent draining events; gather time per window on the primary rank (the first window also includes one metadata gather); analysis time per window on the background thread; event pairs dropped because the queue was full (normally 0); event pairs still unread when the window closed (normally 0). |

### Timeline

With `--timeline-dump-dir` set (the timeline is flushed through the metrics-service adapter, so `--use-metrics-service` must be on as well), every window appends one `straggler/<seg> rank{g}_dp{d}_tp{t}_pp{p}` event per rank per non-zero segment (`pid` = 2³⁰ + global rank, above any real pid; `tid` = segment index). In Perfetto that is one row per rank, so a longer `bwd` bar or a shorter `dp_grad_sync` bar is visible at a glance.

## Coverage

The timings reuse the 22 timer call sites Megatron already has, so they cover the **scheduling plane**:

| Covered | Detail |
| --- | --- |
| `fwd` / `bwd` | One micro-batch's forward / backward (whole block, not split into layers) |
| `pp_recv` / `pp_send` | PP point-to-point (9 upstream names folded into 2 segments; direction and rank pair are not distinguished) |
| `dp_grad_sync` | DP gradient sync (4 upstream names folded into 1 segment; the CP and DP portions are not separable) |
| `dp_param_gather` | The distributed optimizer's parameter all-gather |
| `optim` | The optimizer phases (6 upstream names) |

**Not covered** (no upstream call site exists; adding one is new work):

- **The internals of attention, MoE, head/logits and MTP.** Upstream Megatron has no timer call site inside these modules, so today they are only visible *inside* `fwd` / `bwd` and cannot be attributed separately. Drilling into them needs new call sites and is out of scope for this version.
- **EP all-to-all** (expert-parallel communication in MoE). It is a "wait for a peer" kind of cost and is currently mixed into the compute segment.
- **The real compute/communication overlap ratio.** A smaller `dp_grad_sync` does **not** read as "better overlap" — it is the sum of the waiting segments, a proxy, not the overlap ratio itself.

**Known granularity limits** (see `reason` below): ranks of the same TP group wait for each other inside forward and backward, so when one of them slows down the whole group is reported and which one it was cannot be told apart; when every rank slows down at once (whole-machine clock throttle, shared storage stall) nothing is detected.

## Reading the reason

| reason | Rule | Where to look |
|--------|------|---------------|
| `slow_device` | Own GPU time or forward time well above the other ranks of the same PP stage, token count normal | `nvidia-smi -q -d CLOCK,PERFORMANCE,TEMPERATURE`, ECC, neighbours on the node, NVLink topology |
| `data_imbalance` | Own or forward time high but normal per token; token count clearly higher | `--balance-data`, dynamic batch splitting, oversize samples |
| `cpu_bound` | Own or forward time high *and* Python GC pauses during the step ≥ 5 % of its own time | `gc.freeze()`, data threads contending for the GIL, per-token Python loops |
| `upstream_wait` | Own time normal but `pp_recv` well above the same-stage ranks that are neither flagged nor late, by at least 5 % of the stage's median own time; a rank that reaches the gradient sync late because it waits on PP is classed here too | The flagged rank on the upstream stage, or `pp_stage_imbalance` (uneven layer split) |
| `late_arrival` | Own GPU time normal, but its `bwd + dp_grad_sync` is much *shorter* than its DP peers' — a collective finishes for everyone at once, so the last rank to arrive waits least and everyone else's time contains the wait for it | Host-side gaps between kernels: data fetch, synchronous I/O / HTTP, GIL, launch gaps. `py-spy dump --pid <pid>` on that rank is the quickest confirmation |

`late_arrival` is easy to miss: any rule that only looks at GPU time cannot see it. The rank's GPU time matches its peers; it just reaches every gradient sync late. A common cause is a blocking call on the training thread, such as the primary rank posting metrics over synchronous HTTP.

With `--overlap-grad-reduce` the gradient reduce-scatter is issued bucket by bucket during backward, so the wait for the slowest rank lands in every DP peer's `bwd` and only a short tail is left in `dp_grad_sync`. That is why late arrival is compared on `bwd + dp_grad_sync` (which ends when the gradient sync completes, with or without the overlap), and why slow ranks are also compared on forward alone (forward has no DP collective). The waiting peers' own time grows too, so a `late_arrival` alert quotes the rank's forward against its peers, not its own time.

## When the profiler itself fails

Exceptions in the timer calls, the event read-out, the window gather, the analysis and the log callback are caught and counted; none reaches the training loop. A rank that has caught 3 errors, or has had dropped event pairs or errors in 3 consecutive windows, asks for the profiler to be switched off. The request travels in that rank's window vector (the `health` field), so every rank sees the same gathered table and switches off at the same window; no rank leaves on its own and strands the others in the next gather. Once off, the timer calls are skipped, but the rank **keeps joining every window gather**: it sends an all-zero row of `NUM_FIELDS` values with only `health` set to 2, so the collective keeps its shape and the switched-off state keeps reaching ranks that have not seen it yet.

A failed gather is only visible to the rank it failed on. It no longer leaves the collective: the failure is counted, the window is reset as usual, and the next window gathers again with a normal row (or the switched-off row above). After repeated failures the rank switches off locally and starts sending that row, and the other ranks read `health = 2` in the next gather that completes and switch off in the same window. The metadata gather of the first window runs once and is never retried: the ranks that got the metadata only gather rows from then on, so a retry on the failed rank alone would misalign the collectives. That rank sends its row for the window as usual with `health = 2`, and every rank switches off in that window. The metadata gather itself is a single collective of one fixed 1 KiB byte row per rank: a rank that fails to serialize its metadata, or has too much of it, still joins with an empty row, and every rank then sees the same missing entry and switches off in that window the same way. To bound the case where a rank process is really gone and can send nothing, the profiler creates a dedicated Gloo group for these gathers at install time, with a timeout set by `RELAX_STRAGGLER_GATHER_TIMEOUT_S` (60 s by default); this does not change the timeout of the shared Gloo group or the training path. **Known limitation:** a single gather still blocks for that timeout when a member has disappeared, and if the whole group is gone every rank can only switch off locally, with no cross-rank propagation.

## Overhead

- One pair of non-blocking `event.record()` calls per Megatron timer call site (events are pooled). At the end of each step completed events are read lazily with `event.query()`, never `cuda.synchronize()`; the cost is reported as `straggler/self_overhead_ms`.
- One Gloo `all_gather` every `REPORT_INTERVAL` rollouts, reported as `straggler/gather_ms`. Each rank contributes 168 B per window (21 float64), independent of the micro-batch count; because this is an `all_gather`, every rank materializes the full `N × 168 B` table, and the whole job receives `N × (N−1) × 168 B` per window (excluding Gloo protocol overhead). The first window also has a separate metadata gather of 1 KiB per rank.
- The analysis runs on a background thread of the primary rank, reported as `straggler/analyze_ms`. It is O(n log n) in the stage size: a few milliseconds per window at 8 ranks, about 35 ms at 2048 ranks in a CPU benchmark.
- On 8×A100 with Qwen3-0.6B DP8 (step ≈ 1 s), the summed worst-case upper bound is about 1.9 ms/step (≈ 0.19 %), and end-to-end on/off comparisons are indistinguishable from zero; turning it on adds no synchronizing calls and leaves the NCCL/compute kernel overlap unchanged. The method and all measurements are in [RFC #362](https://github.com/redai-studio/Relax/issues/362).

## Alert latency

Detection runs only when a window closes, and an alert needs `PERSIST_WINDOWS` consecutive candidate windows. With the defaults (`REPORT_INTERVAL=10`, `PERSIST_WINDOWS=3`) an alert appears about 21–30 rollouts after a rank starts slowing down; if the slowdown starts so late in a window that the window aggregate stays below the thresholds, add one more window. Lowering either setting alerts sooner, at the cost of reacting to one-off spikes.

The alert line is logged as soon as the analysis finishes (`straggler/latency/analyzed_ms`, typically a few milliseconds); the scalars go out with the next `perf/*` report and are shown by whichever output backend is in use (metrics service, W&B, TensorBoard or ClearML).

## Where it lives

- `relax/utils/straggler/timers.py`: the non-blocking drop-in for Megatron's `config.timers` (Megatron's own `Timer.start/stop` call `cuda.synchronize()`, which is why Relax sets it to `None` when the profiler is off).
- `relax/utils/straggler/stats.py`: the statistics vector layout and the Megatron timer name → segment map.
- `relax/utils/straggler/collector.py`: per-rank event pool, lazy read-out, GC callbacks, the per-window gather; torch-free, with events and gathers injected.
- `relax/utils/straggler/worker.py`: the background thread on the primary rank that runs the analyses in order.
- `relax/utils/straggler/detector.py`: pure-numpy window analysis (including uncertain verdicts and alert clearing), unit-testable without a GPU.
- `relax/utils/straggler/health.py`: the profiler's own health (active → degraded → disabled).
- `relax/utils/straggler/reporter.py`: metrics / timeline output (training thread) and log output (background thread).
- `relax/utils/straggler/runtime.py`: the only module that touches the accelerator (through `relax.utils.device`, so CUDA and NPU alike), the process groups or Megatron's parallel state; `StragglerProfiler` does the per-step reporting around `log_perf_data`.
- Wiring: `relax/backends/megatron/model.py` (`config.timers = straggler_timers(...)` at three sites) and `relax/backends/megatron/actor.py` (`install_straggler_profiler`, per-step `_log_perf_data`).
