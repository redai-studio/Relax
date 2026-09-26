# Straggler Profiler

The straggler profiler observes Megatron training-timer intervals across data-parallel ranks and reports when one rank is slow relative to equivalent peers.

## What the straggler profiler is

The profiler attaches a drop-in replacement for Megatron's `config.timers` and records every timer interval Megatron emits (`forward-backward`, `forward-compute`, backward phases, the optimizer phases and the send/receive pairs). Ranks that execute the same program are compared with each other inside a time window, and a rank that is measurably slower than its fastest equivalent peer is reported with the measured facts behind the verdict.

It is observe-only. It never mitigates, reschedules, aborts a step, or changes training in any way. It adds no collective and no synchronisation to the training path, and every entry point is fail-open: a profiler failure is counted and degrades to a no-op rather than propagating into Megatron's schedule.

## How to enable it

The profiler is **off by default**. Enable it with the master switch:

```bash
export RELAX_STRAGGLER_ENABLE=1
```

Enabling the profiler in a single process keeps the observation rank-local: each process compares nothing and a rank-local run reports `uncertain`. To compare ranks, also point every process at the rank-0 collector socket:

```bash
export RELAX_STRAGGLER_ENABLE=1
export RELAX_STRAGGLER_COLLECTOR_ADDR=127.0.0.1:29741
export RELAX_STRAGGLER_OUTPUT_DIR=/path/to/straggler-evidence
```

The disabled path is designed and exercised by the test suite to add no threads, sockets, CUDA events or extra work. With the switch off, the runtime is never constructed, `get_straggler_timers()` returns `None`, and the training backend leaves Megatron's `config.timers = None`, so the default deployment keeps its upstream behaviour. The switch accepts `1`, `t`, `true`, `y`, `yes` and `on` (and the matching false values); an unrecognised value is rejected at startup instead of quietly resolving to off.

## Configuration

Every knob is an environment variable read at process start. Values are resolved once per process; a value that is unparsable or below its minimum falls back to the default and the adjustment is recorded in the startup log as `clamped`.

| Variable                              | Default      | Meaning                                                                                                                                                                                                                                          |
| ------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `RELAX_STRAGGLER_ENABLE`              | `False`      | Master switch. The profiler is inert unless this is a true value.                                                                                                                                                                                |
| `RELAX_STRAGGLER_WINDOW_S`            | `5.0`        | Length in seconds of one bounded **time** window. Windows are anchored to the cohort's first observation and advance by wall-clock time; they are **not** aligned to optimizer steps, so a window has no step order. Minimum `0.1`.              |
| `RELAX_STRAGGLER_REPORT_INTERVAL_S`   | `10.0`       | Cadence of the collector's periodic summary line and of the status-file writer. Minimum `0.1`.                                                                                                                                                   |
| `RELAX_STRAGGLER_OUTPUT_DIR`          | unset        | Directory for per-rank JSONL streams and status JSON files. Unset keeps the data in memory only (still counted and summarised).                                                                                                                  |
| `RELAX_STRAGGLER_COLLECTOR_ADDR`      | unset        | `host:port` of the same-host collector socket. Unset makes each process keep its own collector, so observation is rank-local. The reference observer recipe exports `127.0.0.1:29741`; that port is the recipe's choice, not a built-in default. |
| `RELAX_STRAGGLER_WARMUP_WINDOWS`      | `2`          | Number of opening windows that are never judged. The first intervals carry lazy CUDA context/event allocation and the first data batch, which look like a slow rank. Minimum `0`; maximum `10`.                                                  |
| `RELAX_STRAGGLER_MIN_COHORT`          | `2`          | Minimum number of reporting ranks per `(cohort, stage)` pair in a window. A smaller cohort is reported as `uncertain` instead of judged. Minimum `2`.                                                                                            |
| `RELAX_STRAGGLER_TOPOLOGY_EPOCH`      | `""` (empty) | Identifies one parallel layout inside the cohort key. It is **schema-reserved**: nothing derives or updates it, so a re-shard is invisible to the cohort key unless an operator sets it by hand.                                                 |
| `RELAX_STRAGGLER_TIMER_LOG_LEVEL`     | `2`          | Highest Megatron timer level captured. Megatron uses 1 for coarse phases and 2 for fine ones; the default captures both. Range 1-2.                                                                                                              |
| `RELAX_STRAGGLER_EVENT_POOL`          | `512`        | CUDA events preallocated per rank. One interval needs two, so pool exhaustion degrades that interval to host timing only. Minimum `2`.                                                                                                           |
| `RELAX_STRAGGLER_QUEUE_MAX`           | `4096`       | Bound of the per-rank envelope queue between the observer thread and the sender. Minimum `1`.                                                                                                                                                    |
| `RELAX_STRAGGLER_WORK_TOLERANCE`      | `0.05`       | Relative tolerance used both for the timing deviation and for the work-comparability test before a rank is judged. Minimum `0.0`.                                                                                                                |
| `RELAX_STRAGGLER_MIN_STAGE_MS`        | `5.0`        | Absolute magnitude floor in milliseconds. A stage whose observed or peer-fastest magnitude, or whose absolute gap, is not above this floor is classified `uncertain` rather than `straggler`. Minimum `0.0`.                                     |
| `RELAX_STRAGGLER_PERSIST_WINDOWS`     | `3`          | Number of consecutive anomalous windows required before a rank is reported as a straggler. Minimum `1`.                                                                                                                                          |
| `RELAX_STRAGGLER_DEBUG_HOST_DELAY_MS` | `0.0`        | **Debug-only** injection: sleep this many milliseconds inside the measured interval. Inert at `0.0`.                                                                                                                                             |
| `RELAX_STRAGGLER_DEBUG_RANK`          | `-1`         | **Debug-only** injection target rank; `-1` means every rank.                                                                                                                                                                                     |
| `RELAX_STRAGGLER_DEBUG_STAGE`         | `""` (empty) | **Debug-only** injection target timer name; empty means every stage.                                                                                                                                                                             |

The three `RELAX_STRAGGLER_DEBUG_*` knobs exist to measure detector sensitivity with a controlled slowdown. The delay lands inside the measured interval, exactly like a real slow stage, and they change nothing except the sleep. They are not part of a normal deployment configuration.

## Architecture

The data path has four stages and runs out of band with the training step:

1. **Timer shim** (`StragglerTimers`) replaces `config.timers`. `start`/`stop` record a host timestamp and, when CUDA is available, a pair of CUDA events on the current stream. `start`/`stop` never call `torch.cuda.synchronize()` and never call `torch.distributed.barrier()`.
2. **In-process observer** (`StragglerObserver`) owns a fixed CUDA event pool, a bounded pending list and one daemon readout thread. It reads the events back off the training thread, classifies each interval as `device` or `host_only`, and builds a `TimingEnvelope` carrying the rank identity and the optional workload counters.
3. **Rank-0 out-of-band collector** (`TimingCollector` plus `EnvelopeSender`/`EnvelopeReceiver`) receives envelopes from the other ranks over a same-host TCP socket. Rank 0 binds the socket and judges every rank's envelopes; the other ranks ship and judge nothing. There is no collector process and no cross-node transport.
4. **Detector and reporter** (`StragglerDetector`, `report_once`). The detector compares ranks inside a cohort and produces verdicts; the reporter turns the runtime counters and drained verdicts into platform metrics and merges them into the existing perf path (`relax/utils/training/train_metric_utils.py`).

There is **no new training-path collective** and **no new HTTP request**. The timer shim adds no collective, the observer reads events off the training thread, the transport is a same-host socket owned by background daemon threads, and the reporter only reads in-process counters. Every failure in the pipeline is counted rather than raised.

## Metrics

The reporter emits flat scalar keys under the `perf/straggler/` prefix. A value that cannot be measured is omitted rather than reported as zero, so a missing key reads as "not measured" and never as a false "none observed".

| Key                                            | Meaning                                                                                                                             |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `perf/straggler/active_stragglers`             | Number of `(cohort, stage, rank)` triples currently flagged.                                                                        |
| `perf/straggler/verdicts`                      | Verdicts produced so far.                                                                                                           |
| `perf/straggler/windows_closed`                | Time windows closed and judged so far.                                                                                              |
| `perf/straggler/coverage`                      | Cohort coverage, emitted only when the summary publishes an explicit coverage ratio.                                                |
| `perf/straggler/judged_fraction`               | Fraction of received envelopes the collector judged. This is a transport ratio, not cohort coverage.                                |
| `perf/straggler/dropped`                       | Sum of every drop counter the runtime reports (queue full, pending full, output full, capped detector structures, malformed input). |
| `perf/straggler/worst_deviation`               | Relative deviation of the largest measured slowdown among the drained verdicts.                                                     |
| `perf/straggler/worst_rank`                    | Rank of that largest measured slowdown.                                                                                             |
| `perf/straggler/rollout_id`                    | Rollout id from the published training context.                                                                                     |
| `perf/straggler/optimizer_step`                | In-rollout optimizer-step index from the published training context.                                                                |
| `perf/straggler/step_ordinal`                  | Monotonic run step ordinal assigned by the profiler.                                                                                |
| `perf/straggler/workload_incomparable_windows` | Windows in which at least one rank's token workload exceeded the tolerance.                                                         |
| `perf/straggler/workload_missing_windows`      | Windows in which at least one rank published no workload.                                                                           |
| `perf/straggler/workload_publish_skipped`      | Workload publications withheld by the caller's self-consistency guard.                                                              |
| `perf/straggler/workload_publish_errors`       | Workload publication attempts that raised.                                                                                          |
| `perf/straggler/collector_status_available`    | `1` when this process owns the collector, `0` when it does not.                                                                     |

`collector_status_available` matters under pipeline parallelism: the platform merges metrics only on the Megatron primary rank (`tp0`, pipeline-last, `dp0`), which owns the collector only when `pp_size == 1`. Under `pp_size > 1` the exporting rank owns no collector, and the explicit `0` is the "straggler counters unavailable here" marker instead of silent omission.

When `RELAX_STRAGGLER_OUTPUT_DIR` is set, the collector also persists evidence under `<output_dir>/run_<run_id>/`:

| File                                                     | Written by                                                           | Content                                                                             |
| -------------------------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `straggler_envelopes.jsonl`                              | collector                                                            | One judged timing interval per line.                                                |
| `straggler_verdicts.jsonl`                               | collector                                                            | One verdict per line.                                                               |
| `collector_status.json`, `collector_status_<label>.json` | rank 0                                                               | Collector summary, including the deduplication counters and the detector counters.  |
| `runtime_status.json`, `runtime_status_<label>.json`     | rank 0 for the unsuffixed name, every rank for its own suffixed name | Per-process snapshot of the shim, observer, sender/receiver and collector counters. |

Status files are written atomically (unique temp file, then rename) by a background status-writer thread, so a process killed without a graceful shutdown leaves a status file at most one interval stale.

## Verdict semantics

A verdict describes **one rank in one stage (one Megatron timer name) in one time window**. It keeps measured facts and inferred causes in separate fields:

- `facts` contains everything that was observed or counted: the rank and cohort, the stage, the window index, per-rank and per-peer sample counts, the observed and peer-fastest and peer-median host medians, the ratio, the absolute delta, the tolerance, persistence, cohort size and coverage, the device medians and their availability, and the workload deltas.
- `candidate_causes` is the only forward statement the verdict makes about *why*. It starts at `("undetermined",)` and is replaced only by a statement the facts can carry. It is `host_side_delay_possible` when the host interval grew while the GPU-timeline interval did not, and `device_or_stream_visible_delay_possible` when both grew. A stage name never contributes to the cause.

The `reason` field is a **pure measurement classification**, never a cause:

- `gpu_stream_stall` — the host interval and the GPU-timeline interval both grew, so the extra time is inside the GPU timeline.
- `host_only_stall` — the host interval grew while the GPU-timeline interval did not, so the extra time is demonstrably outside the stream (host work or waiting).
- `attribution_unknown` — the device interval is unavailable (no CUDA events), so the host/device split cannot be made.
- `within_tolerance`, `cohort_below_min_size`, `below_absolute_floor` and `workload_incomparable` are the other measurement classifications.

The distinction is not proof of slow hardware. A CUDA-event pair measures **elapsed time on the GPU stream**, which is a GPU timeline and not a statement of GPU-busy time: a host stall that leaves the stream idle is indistinguishable from slow kernels. Separating a slow kernel from an idle stream needs per-kernel data, which this profiler does not collect, so `gpu_stream_stall` reports the observable gap and stops.

The verdict `kind` is `straggler`, `recovered` or `uncertain`. A rank is reported as a straggler only after it deviates by more than `RELAX_STRAGGLER_WORK_TOLERANCE` for `RELAX_STRAGGLER_PERSIST_WINDOWS` consecutive windows. The reference is the **fastest** peer in the window, not the mean, so one slow rank cannot drag the baseline towards itself. Comparison happens only inside a cohort of ranks that differ only in the data-parallel dimension.

## What it does NOT mean

- **It is not a root-cause claim.** The verdict reports the measured gap and the causes the facts support. It does not collect per-kernel durations or peer-to-peer transfer times, so it never claims a network, fault or "slow GPU" cause.
- **It never mitigates.** The profiler does not reschedule, abort, slow down or otherwise change training. A `straggler` verdict is information for an operator, not an action.
- **Attention and MoE groups are schema-reserved only.** They are declared as capabilities in the stage taxonomy but are not measured: the pinned Megatron version emits no attention or MoE timer name, so attention/MoE costs cannot be split out of `forward-compute` without a deep hook, which is deliberately out of scope.
- **The comparison requires an equivalent cohort.** Two ranks are compared only when they run the same program and differ only in the data-parallel dimension (same TP/PP/VPP/CP/EP/ETP position, same model chunk, same topology epoch and stage schema). A rank is never compared against a peer that legitimately does different work, and a cohort smaller than `RELAX_STRAGGLER_MIN_COHORT` yields `uncertain`.
- **The absolute floor can hide a real slowdown.** A gap must exceed both the relative tolerance and the absolute floor (`RELAX_STRAGGLER_MIN_STAGE_MS`). Below the floor the relative deviation is dominated by host/launch jitter, so a large *relative* slowdown on a very short stage is classified `uncertain` (`below_absolute_floor`) rather than `straggler` and can therefore be missed.

## Known limitations

- **Restart contract.** The runtime is a process-wide singleton built on first use and is never rebuilt within a process, and the observer's `disabled` state is terminal by design. There is no production restart or reset hook (the reset helper is test-only), so a profiler that fails to start or self-disables stays off until the process restarts.
- **Late packets are lost.** The detector closes window `W` once an envelope from `W + 1 + WINDOW_GRACE` arrives, where `WINDOW_GRACE` is one window. A packet delayed by more than that grace cannot be placed in its window and is dropped as late transport noise rather than judged.
- **The workload stamp can be stale.** The observer looks the workload up when it *builds* an envelope, that is on the readout thread after the interval closed. The stamped workload can therefore belong to a later rollout or optimizer step than the interval; the envelope's only temporal anchor is its host start time, and the workload is advisory transport metadata.
- **The topology epoch is inert.** `RELAX_STRAGGLER_TOPOLOGY_EPOCH` is part of the cohort key, but nothing derives or updates it. A re-shard is invisible to the cohort key unless an operator sets the variable by hand.
- **The no-CUDA path judges on the training thread.** Without CUDA events the observer has no event to poll back, so it hands each finished interval directly to the collector, and the detector runs on the training thread. File I/O and network transport stay on their own threads, but the judgement itself is not off-thread in that mode.
- **No rollout or topology reset.** A window that spans a rollout boundary is drained into the next rollout's perf log. Coverage gaps are counted (`incomplete_windows`) rather than guessed, and a window holding a single rank of a multi-rank cohort produces no verdict.

## Example output

The snippets below are **illustrative**: they show the shape of a verdict and of a collector summary, with measurement values elided as `"..."`. They are not the output of any particular run.

One verdict, as written to `straggler_verdicts.jsonl`:

```json
{
  "kind": "straggler",
  "cohort": "topo0:dense:...",
  "name": "forward-compute",
  "rank": 3,
  "label": "rank3/tp0/pp0/.../dp3/chunk-1",
  "window_index": "...",
  "deviation": "...",
  "consecutive_windows": "...",
  "rank_host_ms": "...",
  "reference_host_ms": "...",
  "rank_device_ms": "...",
  "reference_device_ms": "...",
  "host_only": false,
  "cohort_size": "...",
  "reason": "gpu_stream_stall",
  "measurement_kind": "device_and_host",
  "candidate_causes": ["device_or_stream_visible_delay_possible"],
  "facts": {
    "stage": "forward-compute",
    "observed_ms": "...",
    "peer_fastest_ms": "...",
    "peer_median_ms": "...",
    "ratio": "...",
    "absolute_delta_ms": "...",
    "cohort_size": "...",
    "cohort_expected": "...",
    "coverage_ratio": "...",
    "device_available": true,
    "device_ms": "...",
    "peer_device_ms": "...",
    "workload_delta": "...",
    "workload_evidence_degraded": false
  }
}
```

A collector summary, as written to `collector_status.json`:

```json
{
  "envelopes": "...",
  "judged_packets": "...",
  "invalid_packets": "...",
  "duplicate_packets": "...",
  "late_packets": "...",
  "windows_closed": "...",
  "stragglers_reported": "...",
  "recoveries_reported": "...",
  "uncertain_judgements": "...",
  "incomplete_windows": "...",
  "warmup_windows_skipped": "...",
  "workload_incomparable_windows": "...",
  "workload_missing_windows": "...",
  "active_stragglers": [
    { "cohort": "topo0:dense:...", "name": "forward-compute", "rank": 3, "label": "..." }
  ],
  "dedup": { "new": "...", "duplicate": "...", "late": "...", "expired": "...", "evicted": "..." },
  "persist_windows": 3,
  "work_tolerance": 0.05,
  "min_stage_ms": 5.0
}
```

## Performance note

Overhead is measured with an **AB/BA paired campaign**: the same recipe is run with the profiler off (arm A) and on (arm B), in both orders across sessions, and the paired difference is estimated at the session level so step-to-step noise is not treated as independent samples.

Measured in-process on CPU, the per-interval cost of the recording path is on the order of microseconds. That figure is a **CPU proxy and is not a GPU measurement**: it does not capture device-side effects, and the campaign's end-to-end result is the authoritative measurement of the profiler's effect on a real training run. The campaign is the reference for any overhead claim; this page deliberately does not restate a headline percentage.

## Troubleshooting

**The collector is unreachable.** Check that `RELAX_STRAGGLER_COLLECTOR_ADDR` is set to the same `host:port` in every process, that the host and port are reachable on that machine, and that rank 0 could bind it. A bind failure is logged and counted (`accept_errors` on the receiver). On the sending side, `sender.send_errors` counts failed connects and sends, `sender.connected` reports whether the socket is up, and `sender.dropped_queue_full` counts envelopes dropped because the send queue was full while the collector was away.

**No verdicts appear.** The common reasons are, in order:

- **Cohort too small.** Fewer than `RELAX_STRAGGLER_MIN_COHORT` ranks reported for the `(cohort, stage)` pair, so the window yields `uncertain` instead of a verdict. A single-rank run reports nothing by design.
- **Warmup.** The first `RELAX_STRAGGLER_WARMUP_WINDOWS` windows are never judged; `warmup_windows_skipped` counts them.
- **Equivalence and workload gates.** A rank whose own token workload exceeds `RELAX_STRAGGLER_WORK_TOLERANCE` past its peer median yields `uncertain` with reason `workload_incomparable` rather than a straggler verdict, and `workload_missing_windows` counts windows with no workload to compare. Missing workload does not suppress the timing verdict; it marks the workload evidence as degraded.
- **The absolute floor.** A deviation that does not exceed `RELAX_STRAGGLER_MIN_STAGE_MS` yields `uncertain` with reason `below_absolute_floor`. Short metadata stages commonly land here.
- **Persistence.** A rank must deviate for `RELAX_STRAGGLER_PERSIST_WINDOWS` consecutive windows before it is reported.
- **Coverage gaps.** A window holding one rank of a multi-rank cohort increments `incomplete_windows` and produces no verdict.

**Where the status JSON lives.** With `RELAX_STRAGGLER_OUTPUT_DIR` set, the collector writes `collector_status.json` and the rank-0 `runtime_status.json` under `<output_dir>/run_<run_id>/`; every rank also writes `runtime_status_<label>.json`, and rank 0 writes `collector_status_<label>.json`. These files are refreshed atomically by a background thread. If `RELAX_STRAGGLER_OUTPUT_DIR` is unset, no status file is produced and the same counters are available only through the `perf/straggler/*` metrics and the logs.
