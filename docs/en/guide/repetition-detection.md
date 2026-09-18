# Repetition Detection

Long responses sometimes degenerate into repeated text. Relax reports this
online as the `rollout/repetition_frac` metric, and ships an offline entry
that pinpoints *where* the repetition is inside a dumped response.

## What is detected

Detection runs on `Sample.response` — the full response text, which for
agentic rollouts may also contain tool observations. It uses a compression
ratio: a chunk of text that `zlib` compresses far better than normal prose
is flagged as low-entropy suspected repetition; structured tool output can
also trigger the detector.

The scan covers the **whole response** with overlapping windows:

| Parameter | Default | Meaning |
| --- | --- | --- |
| window size | 10,000 chars | Size of each scanned window |
| stride | 5,000 chars | Distance between consecutive window starts |
| threshold | 10.0 | A window hits when its compression ratio is **strictly greater** |

`stride` may not exceed the window size: a wider stride would leave unscanned
gaps between windows, and repetition falling in a gap would be missed with no
indication. Passing one raises `ValueError`. The threshold must be finite and
strictly positive; NaN, infinity, zero and negative values are rejected even
when the input is empty.

A nonempty response shorter than one window is scanned as one window. An empty
response has no windows, is not repetitive, and has `max_compression_ratio=None`
(`null` in JSON). When the last stride-aligned window does not reach the end of
the text, the exact trailing 10,000-character suffix is scanned as an extra
window, so the tail is never left uncovered. Every window is scanned: cost
is never reduced by silently skipping windows in the middle.

::: tip Why overlapping windows
Overlapping windows reduce boundary misses but do not guarantee detection of
a repetitive run one window long. For example, a run at `[2500, 12500)`
shares each scanned window with clean context, which may dilute its compression
ratio below the threshold. With the defaults, a run of at least 15,000 characters
guarantees that it contains a complete scanned window; a hit still depends on
the compression ratio. The
previous implementation looked only at the final 10,000 characters, so a
response that repeated in the middle and ended cleanly was never flagged.
:::

## Offsets and intervals

Reported intervals are `[start, end)` half-open slices into `Sample.response`,
counted in **characters** (Unicode code points, i.e. `len(text)`), not bytes.
`response[start:end]` gives back exactly the scanned window, which keeps the
offsets meaningful for Chinese and other non-ASCII text.

An interval marks a **suspected repetitive window**, not the exact boundary
of the repetition — the repeated span may be shorter than, or extend beyond,
the window that flagged it.

When a report counts covered characters, it counts the **union** of the hit
intervals, so overlapping windows are never double-counted.

## Online metric

`rollout/repetition_frac` is the fraction of samples in a rollout batch whose
response contains repetition anywhere. The aggregation is still a sample
fraction, but the detector now covers the full response and also scans short
responses and responses exactly one window long. The old suffix-only detector
skipped responses of at most 10,000 characters. Values from the old and new
detectors are therefore not directly comparable; rescan the same dumps with
one implementation for comparisons.

The boolean API is preserved:

```python
from relax.utils.metrics.metric_utils import has_repetition

has_repetition(sample.response)  # -> bool
```

For the detailed result, use `scan_repetition`:

```python
from relax.utils.metrics.metric_utils import scan_repetition

report = scan_repetition(sample.response)
report.has_repetition        # bool
report.hit_intervals         # [(start, end), ...] suspected repetitive windows
report.max_compression_ratio # largest ratio seen, or None if nothing was scanned
report.covered_chars         # union length of the hit intervals
report.to_dict()             # JSON-ready summary
```

`has_repetition` short-circuits at the first hit; `scan_repetition` scans
every window by default so the offline report is complete.

Both names can also be imported straight from `relax.utils.repetition`. The two
paths are the same implementation, but `relax.utils.repetition` imports only the
standard library, which suits offline scripts;
`relax.utils.metrics.metric_utils` pulls in `torch` and the rest of the training
dependencies.

## Offline diagnosis

The entry reads both dump formats, and several files may be mixed in one run:

| Format | How it is produced | Notes |
| --- | --- | --- |
| `.jsonl` | **Always written every step** once `--rollout-result-dir` is set | Streamed line by line, no `torch` needed; usually the file at hand |
| `.pt` | Requires `--save-debug-rollout-data` | Pickled objects, so read only from a trusted source; `torch` is imported lazily on this path |

```bash
# The always-on JSONL results, diagnosable with no training dependencies
python -m relax.entrypoints.diagnose_repetition /path/result/train/0.jsonl -o repetition.json

# A .pt dump works too, as do several files at once
python relax/entrypoints/train.py ... --save-debug-rollout-data /path/dump/{rollout_id}.pt
python -m relax.entrypoints.diagnose_repetition /path/dump/*.pt --only-hits
```

The detection core lives in `relax.utils.repetition` and imports only the
standard library, so JSONL results can be diagnosed on a plain CPU box with no
`torch`, Ray or metrics service installed.

Options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `-o, --output` | stdout | Write the JSON report to this path |
| `--window-size` | 10000 | Window size in characters |
| `--stride` | 5000 | Stride in characters |
| `--threshold` | 10.0 | Compression-ratio threshold |
| `--only-hits` | off | Report only repetitive samples |

`--output` may not point at any input dump: the dumps are training output that
cannot be re-created, so overwriting one would destroy the data being diagnosed.

The report identifies each sample by its position in the dump plus its
`dataset`, `index`, `sample_index`, `group_index` and `rollout_id`, and its
`repetition_frac` matches the online metric for the same samples and
thresholds. Every sample must have a string `response`; missing, `null`, or
non-string values fail with a file/line or sample-position diagnostic instead
of silently lowering the fraction. An empty string is valid and counts as a
nonrepetitive sample. Reports use strict JSON serialization.

```json
{
  "config": {
    "window_size": 10000,
    "stride": 5000,
    "threshold": 10.0,
    "offset_unit": "characters",
    "interval_convention": "half-open [start, end)"
  },
  "num_samples": 2,
  "num_repetitive_samples": 1,
  "repetition_frac": 0.5,
  "files": [
    {
      "path": "/path/dump/7.pt",
      "rollout_id": 7,
      "num_samples": 2,
      "num_repetitive_samples": 1,
      "repetition_frac": 0.5,
      "samples": [
        {
          "position": 1,
          "index": 1,
          "group_index": 0,
          "has_repetition": true,
          "text_length": 45600,
          "threshold": 10.0,
          "max_compression_ratio": 147.05882352941177,
          "num_windows_scanned": 9,
          "hit_windows": [
            { "start": 15000, "end": 25000, "compression_ratio": 147.05882352941177 },
            { "start": 20000, "end": 30000, "compression_ratio": 147.05882352941177 },
            { "start": 25000, "end": 35000, "compression_ratio": 11.481056257175661 }
          ],
          "covered_chars": 20000
        }
      ]
    }
  ]
}
```

The three hit windows overlap and span characters 15,000-35,000, so
`covered_chars` is their union (20,000), not the sum of their lengths (30,000).

## Cost

Full-scan time is linear in response length. Window bounds are generated
lazily, and both entry points share one scan: `has_repetition` delegates to
`scan_repetition` with the early exit enabled, so it stops at the first hit and
retains nothing per window; excluding the input, its auxiliary memory is
`O(window_size)`. The detailed report additionally keeps one entry per hit
window, so `scan_repetition` uses `O(window_size + H)` for `H` hit windows —
`H` is 0 for clean text, whatever the response length.

Measured on 2026-09-18 with macOS 27.0 arm64, Python 3.14.7, and five timing
repeats. All individual cases use the full `scan_repetition`, including fully
repetitive text; no early exit is enabled.

| Response length | Case | Windows | Mean wall ms | Mean CPU ms | Traced peak MiB | Process peak RSS MiB |
| --- | --- | --- | --- | --- | --- | --- |
| 10,000 | Clean | 1 | 0.09 | 0.08 | 0.30 | 25.59 |
| 10,000 | Middle | 1 | 0.02 | 0.02 | 0.30 | 25.89 |
| 10,000 | Repetitive | 1 | 0.02 | 0.02 | 0.30 | 25.77 |
| 100,000 | Clean | 19 | 2.13 | 2.13 | 0.31 | 25.66 |
| 100,000 | Middle | 19 | 1.89 | 1.80 | 0.31 | 25.95 |
| 100,000 | Repetitive | 19 | 0.26 | 0.26 | 0.31 | 25.52 |
| 1,000,000 | Clean | 199 | 24.08 | 23.88 | 0.31 | 28.41 |
| 1,000,000 | Middle | 199 | 24.23 | 24.11 | 0.31 | 30.30 |
| 1,000,000 | Repetitive | 199 | 2.84 | 2.83 | 0.34 | 26.56 |

Clean input is deterministic SHA-256 hexadecimal filler. The middle case
replaces the central `min(length, 20000)` characters with repeated `spam`, so at
10,000 characters it is equivalent to the fully repetitive case. The repetitive
case contains only repeated `spam`. Wall time uses `perf_counter`; CPU time uses
`process_time`. Both exclude input construction and warmup.

Memory is measured separately. `tracemalloc` starts after input construction
and warmup, recording incremental tracked allocations but excluding input
storage, imports and untracked native allocations. Each case's peak RSS uses
`resource.ru_maxrss` in a fresh subprocess and includes the interpreter, imports,
input construction and the full scan. macOS bytes and Linux KiB are converted
to MiB. RSS is a lifetime process peak, not the scan's incremental memory cost.
Only macOS was measured here; the Linux path was not rerun.

The batch benchmark measures the online `has_repetition` predicate and mean
aggregation only, excluding other rollout metrics, Ray and dump I/O. Each batch
reuses one immutable response. Repetitive cases replace 20,000 characters with
`"spam" * 5000` at the indicated position; remaining context is clean filler.
The table reports mean wall time:

| Batch size | Response length | Clean | Beginning repeat | Middle repeat | End repeat |
| --- | --- | --- | --- | --- | --- |
| 512 | 40,000 | 329.72 ms | 7.18 ms | 52.80 ms | 136.36 ms |
| 1024 | 40,000 | 710.88 ms | 14.76 ms | 120.31 ms | 310.84 ms |
| 512 | 200,000 | 2369.85 ms | 7.36 ms | 1030.33 ms | 2147.32 ms |

Short-circuit savings depend on the first hit's position. A hit near the end
still requires scanning preceding clean windows. Timings and RSS vary with the
machine and system load; these results do not replace training-environment measurements.

Run from the repository root without training dependencies (RSS measurement
requires macOS or Linux):

```bash
PYTHONPATH=. python3 tests/utils/benchmark_repetition_scan.py --batches --json
```

Output includes environment, repeat count, single-response CPU/wall timings,
traced allocation peaks, independent-process peak RSS, and batch CPU/wall
timings. `wall_mean_ms` is mean wall time and `peak_mem_mib` is peak traced
allocations. Omit `--batches` to run only
the three single-response cases at 10,000 / 100,000 / 1,000,000 characters.

CPU integration tests load the complete rollout module and call the real
`compute_metrics_from_samples`, replacing only deployment dependencies so a
missing SGLang installation no longer skips the test. They cover train/eval
metrics, real Sample objects, and dump writing/reading. They do not start Ray
or a GPU and do not substitute for multi-node training validation.

## Limitations

- Detection is statistical, not semantic: it finds low-entropy repeated text,
  not paraphrased or semantically redundant content.
- Roles are not distinguished. If a response embeds tool observations, a
  repetitive observation counts as repetition.
- Nothing is truncated and no reward is changed — this is a diagnostic only.

## Next Steps

- [Debugging](./debugging.md)
- [Configuration](./configuration.md)
