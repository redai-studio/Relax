# Straggler Analysis

The Megatron backend ships a low-overhead straggler profiler that can stay enabled in production runs. It times every training rank's forward, backward, optimizer, gradient-sync and similar segments with CUDA events, compares the ranks every K rollouts, and reports the rank holding the group back and why. This is not `torch.profiler`: it traces no kernels; ordinary training steps never call `cuda.synchronize()`, and their measured compute/communication overlap stays within the A/A run-to-run spread; with tracing enabled, the sampled step synchronizes once per window from the second window onward to align ranks in time. Errors inside the profiler never affect training.

## Features

- **Straggler verdicts**: every window (K rollouts) gathers each rank's segment times, compares ranks only within the same PP stage, and gives a reason (see *Reading the reason*); a rank without enough data is reported as uncertain. An alert needs `PERSIST_WINDOWS` consecutive abnormal windows and clears after `RECOVER_WINDOWS` consecutive clean windows.
- **PP waits and micro-batches**: under non-interleaved 1F1B the PP wait is split into fill, steady-state and drain phases; the extremes of single micro-batch forward / backward times and token counts are recorded.
- **Sampled-step trace**: every window puts one rollout's timings of all ranks on a common time axis and writes a Perfetto file: PP bubbles, each micro-batch's forward / backward (with the model chunk under VPP), sends, gradient syncs and the optimizer.
- **Module timing**: in the sampled step, the forward and backward of every layer's attention and MLP / MoE block, with a MoE layer split into router, dispatch, experts and combine.
- **Level switching while training**: paused, window statistics, plus trace, plus module timing, switched from a web page or an endpoint.
- **Self-shutdown on errors**: the profiler's own errors are counted and never reach the training loop. When a rank has caught 3 errors within its last 10 windows, or has lost spans or caught errors in 3 consecutive windows, every rank switches the profiler off in the same window and training carries on. See `straggler/health/*`.

## Enabling

The profiler is off by default. Pass the environment variables to the training actors through the Ray runtime environment:

```yaml
# configs/env.yaml
env_vars:
  RELAX_STRAGGLER_PROFILER: "1"
  RELAX_STRAGGLER_REPORT_INTERVAL: "10"
```

Every training rank must see the same values (and the same `--timeline-dump-dir`): the window length and whether traces are written decide when each rank joins the collectives. On several machines set them through the Ray runtime environment, not per node.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RELAX_STRAGGLER_PROFILER` | bool | false | Master switch. |
| `RELAX_STRAGGLER_REPORT_INTERVAL` | int | 10 | Rollouts per gather and analysis (a window); how to choose it is below the table. ms/step values below are per rollout. |
| `RELAX_STRAGGLER_Z_THRESHOLD` | float | 3.0 | Robust z-score (median / MAD within the peer group) a rank must exceed to count as abnormal. |
| `RELAX_STRAGGLER_REL_THRESHOLD` | float | 0.10 | Minimum relative excess over the peer median. |
| `RELAX_STRAGGLER_PERSIST_WINDOWS` | int | 3 | Consecutive abnormal windows before a rank is flagged; filters one-off spikes. Must be at least 1. |
| `RELAX_STRAGGLER_RECOVER_WINDOWS` | int | 2 | Consecutive clean windows before an alert clears; uncertain windows are not clean. Must be at least 1. |
| `RELAX_STRAGGLER_GATHER_TIMEOUT_S` | float | 60.0 | Timeout in seconds of the profiler's dedicated Gloo group; it only bounds how long the other ranks wait once a rank process has disappeared. |
| `RELAX_STRAGGLER_TRACE_DIR` | str | empty | Output directory of the sampled-step trace; defaults to `straggler/` under `--timeline-dump-dir`. Without either, no trace is written. |
| `RELAX_STRAGGLER_TRACE_KEEP` | int | 50 | Number of newest trace files to keep. |
| `RELAX_STRAGGLER_LEVEL` | int | −1 | Start level (see *Levels*); −1 means 2 with a trace directory and at most 256 ranks, 1 otherwise. |
| `RELAX_STRAGGLER_CONTROL_POLL_S` | float | 5.0 | How often, in seconds, the primary rank asks the metrics service for a level. |

`REPORT_INTERVAL` trades alert lag against overhead. An alert needs `PERSIST_WINDOWS` windows in a row: about 21–30 rollouts at the default 10, 3–4 at 1 (two Qwen3-0.6B DP8 runs; a 40-rollout clean run raised no alert). Each window has a fixed cost: one statistics gather (about 1 ms on 8 GPUs, 1.3–6.5 ms on two 8-GPU nodes), plus with the trace on a device synchronize (about 1 ms), the sampled-step gather and a trace file, about 3 ms in all, so 3 ms / `REPORT_INTERVAL` per rollout.

- Long rollouts (RL, for example, tens of seconds to minutes each): set 1–2; the extra cost stays under 0.01%.
- Short rollouts (small-model SFT, for example, about 1 s): keep the default 10. At 1 every rollout costs about 3 ms more (about 0.3%, more across machines), which with the direct cost comes close to or above 0.5%. A one-rollout window may also close before its sampled step is read, and then has no trace.

The default does not follow the rollout length and the value cannot change while training runs: every rank must join the gather at the same rollout, and the lengths the ranks measure may differ.

Each actor process logs `Straggler profiler enabled: report_interval=… primary=… level=… control=… trace_dir=… meta=RankMeta(…)` at start-up; `trace_dir=-` means no trace is written. Critic, reference and `actor_fwd` actors run no training steps and are not profiled.

## Levels

| Level | Content |
| --- | --- |
| 0 `paused` | Nothing is timed or analyzed, alert counts are frozen; the only scalar is `straggler/control/level` |
| 1 `coarse` | Window statistics and verdicts |
| 2 `detail` | Plus the sampled-step trace; needs a trace directory and runs as `coarse` without one |
| 3 `modules` | Plus module timing: only during the sampled step, trace only, never in the window statistics or the verdicts; needs a trace directory and must be set explicitly |

With a trace directory the start level is `detail`, which can stay on (see Overhead); `modules` costs more, so turn it on only while investigating.

`RELAX_STRAGGLER_LEVEL` sets the start level. With `--use-metrics-service`, set `HEAD_NODE` to the reachable address of the head node. While training runs, open `http://${HEAD_NODE}:8000/metrics/straggler` in a browser and press a button, or send the request directly:

```bash
curl -X POST "http://${HEAD_NODE}:8000/metrics/straggler/control" -H 'Content-Type: application/json' -d '{"level": 0}'
```

Every rank switches from the next window on, at most one window plus one poll interval later; the page shows the level in effect. Without the metrics service the level stays at its start value, and a profiler that has switched itself off after errors ignores level requests. The page and the endpoints are served by the Ray Serve server on port 8000 of the head node and have no authentication, so do not expose them outside the cluster.

On the page, "Requested level" is the last level asked for and "Level in effect" the level the primary rank reports; they can differ until the next window boundary. The primary rank reports every `RELAX_STRAGGLER_CONTROL_POLL_S` seconds from a background thread: if reports keep arriving but "Latest finished rollout" stops advancing, training is stuck; if "Last report from the primary rank" keeps growing, the primary rank has exited or cannot reach the metrics service.

## What you get

### Log (primary rank)

`step` is the step of the window's last rollout. A window without alerts logs one INFO line:

```text
[straggler] step=29 no straggler (uncertain 0); self median 734.6 ms max 745.6 ms (rank 0, +2%),
latest to grad-sync rank 0 by 13.4 ms (peers idle 45.9 ms), pp_stage_imbalance 1.00, overhead 0.10 ms/step
```

When there are alerts, the window in which a rank's reason changes logs one WARNING, and every later window that still has alerts logs a table with one row per rank; clearing logs another WARNING, such as `rank 5 (…) recovered from slow_device after 2 clean windows`:

```text
[straggler] step=29 rank 3 (rank3_dp3_tp0_pp0, host=…, gpu=3) reaches the DP grad-sync 197.4 ms/step
after its DP peers (peers idle 203.3 ms/step = 22% of their compute, z=41.0) for 3 windows;
its own GPU forward is 0.99x peers, gc 1.8 ms, cpu/gpu fwd 1.56x -> late_arrival (host-side stall)
[straggler] step=29 per-rank window (ms/step):
 rank | tag               | host | device | self_ms | wait_ms | late_ms | tokens  | ms_per_ktok | gc_ms | steady_ms | mb_fwd_max | reason       | uncertain
    0 | rank0_dp0_tp0_pp0 | …    |      0 |   933.7 |     6.4 |     6.1 | 14802.1 |        63.1 |   1.8 |       0.0 |      157.8 | none         |
    3 | rank3_dp3_tp0_pp0 | …    |      3 |   737.1 |     5.9 |   197.4 | 14592.7 |        50.5 |   1.8 |       0.0 |      148.0 | late_arrival |
```

Only two rows are shown; the real table has a row per rank. `steady_ms` is the steady-state PP wait (`pp_wait_up + pp_wait_down`) and `mb_fwd_max` the slowest single forward of the window (ms).

Every window also logs an INFO line with the numbers that are not sent as scalars (the second table under Metrics), for example `[straggler] step=29 other metrics: analyze_ms 2.9, bwd/max_ms 401.2, …`.

A rank whose data cannot support a verdict is reported as uncertain, with a cause:

```text
[straggler] step=19 rank 3 (rank3_dp3_tp0_pp0) is uncertain: 2 event pairs were still in flight when the window
closed; their time lands in the next window
```

| Cause | Meaning |
|-------|---------|
| `dropped_events` / `unread_events` / `profiler_errors` | That rank's timing for the window is incomplete: spans were dropped from a full queue or still unread when the window closed, or the profiler caught errors on that rank |
| `no_peers` | Its PP stage has no other rank to compare with |
| `missing_tokens` | It looks slow, but there are no token counts, so data imbalance cannot be told apart from a slow device |
| `no_wait_reference` | It waits on PP, but every other rank of its stage is flagged, late or uncertain, so there is no wait time to compare with |

An uncertain window breaks the run of abnormal windows and does not count toward clearing an alert.

### Metrics

A window sends only the scalars in the first table below; they use the same step as `perf/*` and go to TensorBoard / WandB / ClearML. Every extra scalar adds training-thread time to that step's flush. The analysis runs on a background thread, so a window's scalars normally go out with the next rollout's `perf/*`; `straggler/window/{first,last}_rollout` name the window they belong to. The background thread logs the rest of the window's numbers in the `other metrics` line (the second table, keys without the `straggler/` prefix).

The timings reuse the 22 timer call sites Megatron already has, grouped into these segments:

| Segment | Content |
| --- | --- |
| `fwd` / `bwd` | One micro-batch's forward / backward |
| `pp_recv` / `pp_send` | PP point-to-point (9 timers folded into 2 segments) |
| `dp_grad_sync` | Gradient sync (4 timers folded into 1 segment; the CP and DP portions cannot be separated) |
| `dp_param_gather` | The distributed optimizer's parameter all-gather; only with `--overlap-param-gather` off |
| `optim` | The optimizer (6 timers; includes the grad-norm reduction when gradient clipping is on) |
| `lp_fwd` / `lp_pp_recv` / `lp_pp_send` | The same segments during log-prob / forward-only passes |

| Scalar | Meaning |
|--------|---------|
| `straggler/flagged/{count,rank,reason}` | Alert state; rank is −1 if none. Reason codes: 0 none, 1 slow_device, 2 data_imbalance, 3 upstream_wait, 4 cpu_bound, 5 late_arrival |
| `straggler/uncertain/count` | Number of uncertain ranks |
| `straggler/self/{median_ms,max_ms,max_rank}` | Own compute time (`fwd + bwd + optim`, ms/step): median and maximum across all ranks; `max_rank` is the rank that exceeds the other ranks of its PP stage the most, not necessarily the one with the maximum |
| `straggler/late/max_ms` | How late the rank that reaches the DP gradient sync last is |
| `straggler/health/{state,errors,requested_by}` | The profiler's own health: the worst state across ranks (0 active, 1 degraded, 2 disabled; degraded: spans lost or errors caught in the window, or Megatron calls the timers in a way the profiler does not know, see Limits), errors caught in the window, and the rank that asked to switch the profiler off (only then) |
| `straggler/{dropped_events,unread_events}`, `straggler/trace/lost` | Spans dropped from a full queue, spans still unread when the window closed, and spans lost from the sampled-step trace (all normally 0) |
| `straggler/control/level` | The window's level: 0 paused, 1 coarse, 2 detail, 3 modules |
| `straggler/self_overhead_ms` | CPU time per step the profiler spends reading the timings |
| `straggler/window/{first_rollout,last_rollout}` | The window these scalars belong to |
| `straggler/latency/analyzed_ms` | Milliseconds from the window close to the end of its analysis, when the alert lines are logged |

| In the log | Meaning |
|--------|---------|
| `<segment>/{median_ms,max_ms,max_rank,spread}` | Per-segment GPU time (ms/step): median and maximum across all ranks; `max_rank` / `spread` are the rank with the largest excess over the other ranks of its PP stage and that excess (`value/peer_median − 1`) |
| `self/spread`, `self_per_ktok/*` | How far own compute time exceeds the other ranks of the PP stage at most; own compute time per 1000 tokens, omitted when any rank reports zero tokens |
| `wait/{median_ms,max_ms,max_rank,spread}` | Time spent waiting for other ranks (`pp_recv + dp_grad_sync + dp_param_gather`) |
| `late/{max_rank,peer_idle_ms}` | The rank that reaches the DP gradient sync last, and how long the rest of its DP group idles per step waiting for it |
| `tokens/{median,max,spread}` | Token-count imbalance across ranks; omitted when any rank reports zero tokens |
| `pp_stage_imbalance` | `max/min` of the per-stage median compute time |
| `{pp_wait_fill,pp_wait_up,pp_wait_down,pp_wait_drain}/{max_ms,max_rank}` | The training-phase PP wait split by non-interleaved 1F1B phase: `fill` waits for upstream activations while the pipeline fills, `up` / `down` wait for upstream activations / downstream gradients in the steady state, `drain` waits for downstream gradients while it drains. A balanced pipeline does not idle in the steady state, so `up + down` beyond the transfer time is extra bubble; an upstream rank stalling at the start of the step shows up in the downstream stages' `fill`. Only with PP |
| `pp_bubble/ratio` | `pp_recv` as a share of the timed training time; the largest per-stage median. Only with PP |
| `micro_batch/{fwd_max_ms,fwd_max_rank,fwd_argmax,fwd_argmax_tokens,fwd_min_ms,bwd_max_ms,bwd_max_rank,tokens_max,tokens_min}` | Extremes of single training micro-batches over the window: the slowest forward and its rank, its micro-batch number in its rollout (from 0) and its token count (−1 when not reported), the fastest forward, the slowest backward and its rank, and the largest / smallest token count of a micro-batch (omitted without token counts) |
| `step_max/{fwd,bwd,optim,pp_recv,dp_grad_sync}_{ms,rank}` | The slowest single rollout of each segment over the window and its rank: a slowdown the window's averages hide (one slow rollout, or one that began late in the window) shows here; omitted for a segment that was never timed |
| `gc/{median_ms,max_ms}` | Python GC pauses |
| `waiting/count` | Ranks waiting on a slow upstream PP stage; they are held up, not slow, and do not count in `flagged` |
| `uncertain/{rank,cause}` | The first uncertain rank (−1 if none) and its cause. Cause codes: 0 none, 1 dropped_events, 2 unread_events, 3 profiler_errors, 4 no_peers, 5 missing_tokens, 6 no_wait_reference |
| `recovered/count` | Ranks whose alert cleared in this window |
| `health/degraded_ranks` | Ranks currently degraded |
| `gather_ms`, `analyze_ms` | Gather and analysis time per window on the primary rank |
| `trace/{ranks,module_brackets,align_spread_ms,clock_offset_ms_max,unaligned_hosts,bubble_excess_ms_max,bubble_excess_rank}` | Sampled-step trace numbers, see the next section |

### Sampled-step trace (Perfetto)

With a trace directory, the first rollout of every window after the first (the sampled step) is written as `straggler_trace_rollout{N}.json`; open it in [ui.perfetto.dev](https://ui.perfetto.dev). The metrics service is not needed.

Each rank is a row, ordered by (dp, pp, tp) so the stages of a pipeline sit next to each other. The row name carries the rank's measured PP wait and the bubble a balanced 1F1B pipeline would have with the rank's own forward and backward times; the difference is extra bubble. On the slowest stage the difference is negative; look at the other stages there. Within a row, slices nest by time:

| Level | Content |
| --- | --- |
| 1 (PP) | `busy`: back-to-back compute and communication; `bubble fill` / `up` / `down` / `drain`: PP waits named by 1F1B phase, with `waits_for` saying whether the previous or the next stage is awaited; `untimed`: more than 0.5 ms covered by no timed span (host-side stalls) |
| 2 (fwd / bwd) | Inside `busy`: `F{i}` / `B{i}` (`i` is the micro-batch number; with VPP the model chunk follows: `F{i} chunk {c}`; forwards carry their token count), `send`, `grad sync` (the DP gradient sync, Megatron's `all-grads-sync`; the other gradient syncs keep their Megatron timer names), `param gather`, `optimizer`; log-prob passes show `lp F{i}` and so on |
| 3 (modules) | Only at the `modules` level. Inside `F{i}` / `B{i}`: the forward and backward of the attention and of the MLP (`moe` for a MoE layer) of every transformer layer, e.g. `attn fwd L3`, `moe bwd L3`, with `layer` and `micro_batch` in the arguments (and `model_chunk` with VPP); inside a MoE layer, `router`, `dispatch` (where the EP all-to-all happens), `experts`, `combine` and `shared experts` (with `--moe-shared-expert-overlap`, Megatron does not call the shared experts' forward but runs them in pieces on a separate CUDA stream, overlapped with the token exchange, so there is no `shared experts` slice and that time is not timed separately); MTP layers are named `attn fwd MTP L1` and so on (with `mtp` in the arguments); a forward inside `B{i}` is an activation recompute and is named `… (recompute)`. To time the blocks of another model, add specs to `MODULE_SPECS` in `relax/utils/straggler/megatron_adapter.py` |

The primary rank's own CPU-side timers (such as `actor_train` and the per-micro-batch `get_data_batch_*`) get a separate `primary CPU` row.

Of the numbers below only `straggler/trace/lost` is a scalar; the rest are in the `other metrics` log line.

| Number | Meaning |
|--------|---------|
| `straggler/trace/ranks`, `straggler/trace/lost` | Ranks that delivered a trace; timed spans lost (from a full queue, beyond the per-step cap of 4096 spans plus 8192 module spans, or skipped by an error; module spans queue separately, take at most half of the queue, are pushed out first when it is full, and count here only, not in `dropped_events`) |
| `straggler/trace/module_brackets` | Module spans in the trace, summed over ranks (`modules` level only) |
| `straggler/trace/align_spread_ms` | After the cross-machine clock correction, the largest difference between the end of `all-grads-sync` on the ranks of a DP group, i.e. the alignment error left; it includes the order in which the all-reduce itself ends on the ranks (about 1 ms on two 8-GPU nodes, 0.02–0.25 ms on one node); ranks that lost spans are left out |
| `straggler/trace/clock_offset_ms_max` | The largest absolute clock offset of a machine to the primary's machine (ms), which the trace is shifted by; only when offsets were estimated for at least two machines |
| `straggler/trace/unaligned_hosts` | Machines no clock offset was estimated for, drawn on their own wall clock, so their slices do not line up with the other machines'; only when not 0 |
| `straggler/trace/bubble_excess_ms_max`, `straggler/trace/bubble_excess_rank` | The largest excess of measured PP wait over the balanced 1F1B bubble, and its rank |

### Timeline

With both `--timeline-dump-dir` and `--use-metrics-service` set, every window appends one `straggler/<segment> rank{g}_dp{d}_tp{t}_pp{p}` event per rank and non-zero segment; in Perfetto that is one row per rank.

## Reading the reason

| reason | Rule | Where to look |
|--------|------|---------------|
| `slow_device` | Own compute time or forward time well above the other ranks of the same PP stage, token count normal | `nvidia-smi -q -d CLOCK,PERFORMANCE,TEMPERATURE`, ECC, neighbours on the node, NVLink topology |
| `data_imbalance` | Own compute or forward time high but normal per token; token count clearly higher | `--balance-data`, dynamic batch splitting, oversize samples |
| `cpu_bound` | Own compute or forward time high *and* Python GC pauses during the step ≥ 5 % of the own compute time | `gc.freeze()`, data threads contending for the GIL, per-token Python loops |
| `upstream_wait` | Own compute time normal, but `pp_recv` well above the same-stage ranks that are neither flagged nor late, by at least 5 % of the stage's median own compute time; a rank that reaches the gradient sync late because it waits on PP is classed here too. When its stage has no such rank (for example two ranks per stage and the other one flagged), its `pp_recv` is compared with its gradient-sync peers' instead, and the excess must account for at least half its lateness | The flagged rank on the upstream stage, or `pp_stage_imbalance` (uneven layer split) |
| `late_arrival` | Own GPU time normal, but its `bwd + dp_grad_sync` is much *shorter* than that of the rest of its DP group: a collective finishes for everyone at once, so the last rank to arrive waits least | Host-side gaps between kernels: data fetch, synchronous I/O / HTTP, GIL, launch gaps. `py-spy dump --pid <pid>` on that rank is the quickest confirmation |

`late_arrival` cannot be seen from GPU time alone: the rank's GPU time matches the others', it just reaches every gradient sync late. A common cause is a blocking call on the training thread, such as posting metrics over synchronous HTTP. With `--overlap-grad-reduce`, the wait for the slowest rank lands in the other ranks' `bwd`, which is why late arrival is compared on `bwd + dp_grad_sync` and slow ranks are also compared on forward time alone. For the same reason the peers of a late rank have a longer backward: a rank whose own compute time but not its forward stands out is not reported as slow when a late rank of its gradient-sync group accounts for at least half the excess. When a whole TP group is reported as slow, the alerts say so, since its ranks wait for each other and the one that slowed down first is not known. An alerted rank's reason changes only after the new one has held for `PERSIST_WINDOWS` windows; until then the alert keeps the old reason.

## Overhead

- One pair of non-blocking CUDA events per timer call site; at the end of each step only completed events are read. One Gloo all-gather per window, 312 B per rank.
- On 8×A100 with Qwen3-0.6B DP8 (step ≈ 1 s) and full functionality (level 2, a trace directory, the metrics service, overlap), the profiler's direct cost on rank 0 is about 1.5 ms per step (≈ 0.15 %), about 60 % of it in the timers; the other ranks wait for rank 0 at the next collective. An end-to-end in-run crossover gives +0.13 % (95 % CI −0.05 % to +0.30 %, four pairs), with the upper bound below 0.5 %.
- Sampled-step trace: one extra device synchronize and anchor per window (about 1.1 ms), and the sampled step is gathered to the primary rank when the window closes (about 2 ms for the two window-close gathers); at most about 0.03 % per step with the default window.
- `modules` level: the same end-to-end crossover at the `modules` level gives +0.34 % (95 % CI +0.20 % to +0.48 %, four pairs). Each block costs about 21 µs of CPU time in the forward and 15 µs in the backward (single-GPU microbenchmark), and reading the module spans adds about 3.5 ms per window; on Qwen3-0.6B TP1×PP4×DP2 the sampled step's forwards and backwards cannot be told apart from the detail level (within ±3 %).
- The method and all measurements are in [RFC #362](https://github.com/redai-studio/Relax/issues/362).

## Limits

- The timings reuse Megatron's existing timer call sites. In the window statistics, the internals of attention, MoE, head / logits and MTP are part of `fwd` / `bwd`, and the `modules` level only adds the attention, the MLP / MoE block and the parts of a MoE layer to the trace; EP all-to-all counts as compute; there is no real compute/communication overlap ratio, and a smaller `dp_grad_sync` does not mean better overlap.
- Ranks of the same TP group wait for each other inside forward and backward, so when one of them slows down the whole group is reported, and which one slowed down first is not known; when every rank slows down at once (whole-machine clock throttling, a shared-storage stall, the whole group's communication slowing down) nothing is detected.
- The PP phase split, the phase names in the trace and the bubble estimate assume non-interleaved 1F1B; the transfers of the interleaved (VPP) schedule count in `pp_recv` only. With VPP, the micro-batch number and model chunk come from Megatron's interleaved schedule table (`get_schedule_table`, built from the model's own config) by call order; training calls under `combined_1f1b` merge forward and backward steps and are not numbered.
- The trace costs grow with the number of ranks, all on the primary: each window puts the sampled step of every rank into one file, which in a CPU benchmark at 2048 ranks was about 10 MB and took about 0.25 s to build (`modules` level: about 80 MB and 2 s), holding the primary's Python interpreter and slowing its training thread. Above 256 ranks the trace is therefore off by default; estimate this cost before setting level 2 or 3 by hand.
- Ranks are aligned by their wall clocks, shifted per machine by an estimated clock offset: from the end of the DP gradient sync when a DP group spans machines (it ends together on a DP group); when only PP crosses machines, from the order of the forwards and backwards across that hop, assuming the same transfer time both ways; ranks that lost spans take no part; machines neither links are not shifted and are counted in `unaligned_hosts`. On two nodes the first estimate was within 0.3 ms of the offset measured over ssh and the second 0.2–0.6 ms off. Nothing is timed under CUDA graph capture.
- Alerts lag: with the defaults an alert appears about 21–30 rollouts after a rank starts slowing down; with long rollouts lower `REPORT_INTERVAL` (see "Enabling"). Lowering `PERSIST_WINDOWS` also alerts sooner, at the cost of reacting to one-off spikes.
- Timing goes through Megatron's timer interface (`config.timers`). If a Megatron version calls it differently (a timer name the profiler does not know, a new argument to `timers()` / `start` / `stop`, a `Timers.log` / `write` call), or the first rollout never reaches one of the timers every step calls (`forward-compute`, `backward-compute`, `all-grads-sync`, `optimizer-copy-to-main-grad`; `forward-compute` for log-prob passes), training is not interrupted: the call is ignored, a warning is logged once and that rank reports `straggler/health/state` 1 (degraded) from then on, without counting towards a switch-off. That time is then missing from the statistics. The adapter's CPU contract tests check the installed Megatron for these changes. Switching the profiler off (`RELAX_STRAGGLER_PROFILER=0`) gives `config.timers = None` back.
- When a rank process disappears entirely, every gather on the other ranks waits for the `RELAX_STRAGGLER_GATHER_TIMEOUT_S` timeout, and after repeated failures each rank switches the profiler off locally.
