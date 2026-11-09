# Agentic Session export: event-loop responsiveness report

Reproduce with `scripts/benchmarks/benchmark_session_export.py`; its module docstring defines every
metric. Every run checks that all variants exported byte-identical tokens, rollout tokens, loss
mask, rollout logprobs, weight versions, prompt and response.

## Where the time went

A 64-turn trajectory (163,456 tokens; 2,048-token responses, 512-token observations) took about
290 ms of event-loop time to finalize, all of it blocking every other Session on the Shard:

| step                                                                     | share | note                                                                         |
| ------------------------------------------------------------------------ | ----- | ---------------------------------------------------------------------------- |
| `copy.deepcopy` of the whole Sample payload in `_compact_sample_payload` | ~78%  | ~750k list elements copied, then converted to NumPy and discarded            |
| two `tokenizer.decode` calls (prompt, response)                          | ~17%  | half of it is `transformers` input validation; the Rust decode holds the GIL |
| Forest walk, list concatenation, NumPy conversion                        | ~5%   |                                                                              |

Moving this work to a thread alone does not free the loop: the thread still holds the GIL for the
whole Python-level deep copy (released only every 5 ms switch interval) and for each `decode` call
(~43 ms per call here, never released). The change therefore removes work first, then moves what
is left off the loop:

1. Build training arrays directly from the Sample's lists and shallow-copy lists of immutable
   scalars such as `weight_versions`; everything else is still deep-copied with one shared memo.
2. Decode through the tokenizer backend's `decode_batch`, which releases the GIL. It is used only
   when it returns the same text as `tokenizer.decode` (a fast tokenizer with
   `clean_up_tokenization_spaces=False`, e.g. Qwen); otherwise `tokenizer.decode` is kept.
3. Run sample assembly and compaction on one worker thread per Shard while the finalizer keeps
   holding the Session lock; export IDs, trajectory records and publication stay on the loop.

Single-export cost drops from ~290 ms to ~62 ms, of which ~43 ms is GIL-free decode.

## Setup

- CPU only: Intel Xeon @ 2.30 GHz, both variants pinned to the same 2 logical CPUs (`taskset -c 0,1`).
  Python 3.12.3, NumPy 1.26.4, transformers 5.22.2, tokenizers 0.23.3, Qwen2 BPE tokenizer (same
  vocabulary as Qwen3). `OMP_NUM_THREADS=MKL_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`.
- Baseline is `main` at `b01e7e7`. Variants run in fresh processes, alternating order per repeat;
  5 repeats each. Table cells are the median and \[min, max\] over repeats.
- Load: waves of Sessions finish together at evenly spaced times inside an 8 s window while
  `health()` requests arrive at 200/s on an absolute schedule. Forests are built before the window;
  one export and one short request warm up each process.
- Latency counts from the scheduled arrival, so queueing behind a blocked loop is included. The
  loop-lag probe records every missed 1 ms deadline, so a long stall is weighted by its length.

## Results

### Long trajectories, 8 concurrent exports per wave (4 waves, 163,456 tokens each)

The ablation columns isolate the two halves of the change: `copy-only` keeps exports on the loop,
`thread-only` moves the unchanged export to the worker thread.

| metric                  | base                         | copy-only                 | thread-only                  | optimized                 | change  |
| ----------------------- | ---------------------------- | ------------------------- | ---------------------------- | ------------------------- | ------- |
| loop lag P99 (ms)       | 2468.07 \[2361.27, 2728.43\] | 465.82 \[426.78, 521.68\] | 32.68 \[30.59, 36.62\]       | 10.14 \[8.57, 10.35\]     | -99.6%  |
| loop lag max (ms)       | 2538.76 \[2414.27, 2808.43\] | 513.63 \[481.85, 601.68\] | 49.95 \[46.41, 56.85\]       | 18.70 \[17.04, 26.98\]    | -99.3%  |
| short request P50 (ms)  | 2123.15 \[1946.40, 2221.83\] | 0.77 \[0.73, 0.78\]       | 3.89 \[3.75, 4.06\]          | 0.65 \[0.63, 0.66\]       | -100.0% |
| short request P99 (ms)  | 3745.25 \[3379.70, 3948.87\] | 467.29 \[425.67, 518.84\] | 33.11 \[30.02, 37.74\]       | 9.66 \[8.09, 10.81\]      | -99.7%  |
| export latency P50 (ms) | 5851.26 \[5480.55, 6051.08\] | 482.79 \[442.44, 492.44\] | 2251.67 \[2055.15, 2430.07\] | 263.49 \[239.64, 278.41\] | -95.5%  |
| export latency P99 (ms) | 7832.26 \[7466.25, 8035.83\] | 528.26 \[495.97, 624.08\] | 4139.46 \[3853.11, 4644.58\] | 497.66 \[486.74, 572.41\] | -93.6%  |
| export throughput (/s)  | 3.24 \[3.18, 3.37\]          | 4.94 \[4.90, 4.97\]       | 3.16 \[3.01, 3.25\]          | 4.95 \[4.92, 4.96\]       | +52.8%  |
| process CPU (s)         | 9.85 \[9.48, 10.05\]         | 2.49 \[2.28, 2.54\]       | 10.12 \[9.84, 10.53\]        | 2.40 \[2.34, 2.52\]       | -75.7%  |
| peak RSS delta (MiB)    | 94.27 \[91.65, 96.07\]       | 53.56 \[48.30, 56.77\]    | 128.20 \[95.04, 139.63\]     | 131.19 \[127.29, 138.88\] | +39.2%  |

The baseline needs ~9.9 s of CPU for 8 s of offered load, so its exports queue up; the optimized
Shard finishes each wave well before the next one. `thread-only` frees the loop but keeps all of
the CPU cost and still stalls it for 30-50 ms at a time on the GIL; short requests also pay ~4 ms
at the median because the loop waits for the GIL switch interval.

### Long trajectories, one export at a time (8 waves, 163,456 tokens)

| metric                  | base                      | optimized              | change  |
| ----------------------- | ------------------------- | ---------------------- | ------- |
| loop lag P99 (ms)       | 305.40 \[280.22, 344.13\] | 4.95 \[4.29, 5.18\]    | -98.4%  |
| short request P99 (ms)  | 303.27 \[281.03, 343.57\] | 4.49 \[3.30, 5.06\]    | -98.5%  |
| export latency P50 (ms) | 305.05 \[289.15, 313.58\] | 54.89 \[54.12, 60.77\] | -82.0%  |
| process CPU (s)         | 2.90 \[2.72, 3.02\]       | 1.03 \[0.94, 1.09\]    | -64.3%  |
| peak RSS delta (MiB)    | 11.81 \[11.18, 12.61\]    | 29.19 \[28.73, 30.41\] | +147.2% |

### Short trajectories, one export at a time (16 waves, 20,096 tokens)

| metric                  | base                   | optimized              | change  |
| ----------------------- | ---------------------- | ---------------------- | ------- |
| loop lag P99 (ms)       | 36.83 \[34.66, 37.14\] | 1.19 \[1.17, 1.44\]    | -96.8%  |
| short request P99 (ms)  | 36.76 \[34.91, 37.44\] | 1.29 \[1.17, 1.78\]    | -96.5%  |
| export latency P50 (ms) | 38.42 \[37.95, 39.25\] | 8.53 \[8.41, 8.92\]    | -77.8%  |
| process CPU (s)         | 1.19 \[1.14, 1.22\]    | 0.72 \[0.70, 0.75\]    | -39.6%  |
| peak RSS delta (MiB)    | 1.46 \[1.45, 1.47\]    | 10.44 \[10.32, 10.57\] | +614.7% |

Every run completed all scheduled exports and short requests; no request was dropped or retried.

## Resource changes

- Threads: one `agentic-session-export` thread per Shard, started by the first export and joined
  at Shard shutdown. No new processes, dependencies, or configuration.
- CPU: 40-76% less process CPU for the same exports.
- Memory: peak RSS rises by 9-37 MiB in these runs. This is glibc keeping a separate malloc arena
  for the worker thread, not retained data: with `MALLOC_ARENA_MAX=1` the optimized peak delta in
  the 8-concurrent case falls to 36.73 MiB \[35.21, 38.24\] versus 54.28 MiB for `copy-only`, below
  the baseline's 94 MiB. The Shard's existing `trim_memory()` (`malloc_trim`) can hand it back.
- Ordering: exports on one Shard run one at a time, so a Session waits behind at most the exports
  ahead of it instead of blocking the loop. Waiting finalizers hold only their own Session lock.

## Reproduce

```bash
git worktree add --detach ../relax-base origin/main
taskset -c 0,1 python scripts/benchmarks/benchmark_session_export.py \
    --root base=../relax-base --root optimized=. --tokenizer /path/to/Qwen3-0.6B \
    --repeats 5 --output export-bench.json
# one export at a time / short trajectories
... --sessions 1 --waves 8
... --turns 8 --sessions 1 --waves 16
```
