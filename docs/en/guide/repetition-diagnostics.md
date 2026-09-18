# Full-response Repetition Diagnostics

## Overview

Relax detects highly compressible windows throughout `Sample.response`, including tool observations. This identifies repetition near the beginning or middle even when the response ends normally. Compression is a diagnostic measurement: the detector does not summarize context, modify responses or rewards, or stop generation. It does not distinguish roles or detect semantic repetition.

The existing `repetition_frac` metric counts samples with at least one hit, divided by all samples. Production aggregation lives in `relax/utils/metrics/rollout_metrics.py` and is imported by the distributed rollout module. The boolean interface remains available from `relax.utils.metrics.metric_utils`.

## Detection Rules

| Setting | Default and meaning |
|---|---|
| Window | 10,000 Python string characters |
| Stride | 5,000 characters; no window cap or adaptive skipping |
| Compression | UTF-8 byte length divided by zlib-compressed byte length, level 9 |
| Hit | Compression ratio **strictly greater than 10**; equality is not a hit |
| Offsets | Zero-based Unicode code points, inclusive start and exclusive end: `[start, end)` |
| Short response | Every nonempty response up to the window size is scanned once |
| Empty response | No windows, no hits, maximum ratio `null`, covered characters 0 |

Offsets are neither UTF-8 bytes nor tokens nor grapheme clusters: a combining mark counts separately. Inspect a reported interval using the original `response[start:end]` before any display transformations.

For 22,000 characters the windows are `[0, 10000)`, `[5000, 15000)`, `[10000, 20000)`, and `[12000, 22000)`. An unaligned suffix gets a full, end-aligned window; an already aligned final window is not duplicated. Every character is covered, regardless of response length.

::: warning Interpretation and compatibility
A hit marks a suspicious window, not exact repetition boundaries. Naturally compressible content can also trigger it, and a small repeated span mixed with normal text may remain below threshold. Unlike the previous tail-only implementation, short responses and responses exactly 10,000 characters long are now evaluated. Historical and new `repetition_frac` values therefore have different detection coverage.
:::

Detailed diagnostics scan every window and report the maximum ratio over **all** windows, including non-hits. The boolean API can stop after a proven hit; a negative result requires a complete scan. `covered_chars` is the length of the union of hit windows, so overlaps count once. It is not the number of precisely repeated characters.

## Quick Start

Training metrics also use the full-response boolean scan without a window limit. For nonrepetitive samples, every window is compressed; with fixed window and stride sizes, CPU cost grows linearly with the total response characters in the batch. Each rollout step pays the sum of its samples' scan costs. The detector benchmarks below do not measure whole-batch aggregation or training throughput. Measure these separately for your batch size and response lengths when evaluating training overhead.

Run from a checkout with Python available. The JSONL path uses the standard library and does not require Ray, Megatron, or PyTorch:

```bash
python -m relax.entrypoints.repetition tests/fixtures/repetition/middle_repetition.jsonl --output /tmp/repetition-report.json
```

The fixed fixture contains repetition in the middle followed by a normal suffix. For existing dumps, pass one or more files or directories; directories are searched recursively and paths are deduplicated:

```bash
python -m relax.entrypoints.repetition /data/rollout-results --output /tmp/repetition-report.json
python -m relax.entrypoints.repetition /data/debug.pt --input-format torch --output /tmp/debug-report.json
```

## Configuration

| CLI argument | Meaning |
|---|---|
| `inputs` | One or more dump files or directories |
| `--output` | Required JSON report destination |
| `--window-size` | Positive integer; default 10,000 |
| `--stride` | Positive integer no larger than the window; default 5,000 |
| `--threshold` | Positive finite number; default 10.0 |
| `--input-format` | `auto` (default), `jsonl`, or `torch` |
| `--trusted-torch` | Allow pickle objects in trusted legacy Torch dumps |

These options configure the offline invocation, not training arguments. Auto detection recognizes `.jsonl`, `.pt`, and `.pth`; use an explicit format for files with other extensions. Directory discovery includes only the corresponding recognized extensions.

## Dump Formats and Reports

JSONL input is one sample object per nonblank line with a string `response`. It accepts training and evaluation summaries written by `save_rollout_result_jsonl` and `save_eval_summary_jsonl`. Torch input is the dictionary produced by `save_debug_rollout_data`, containing `samples` and optionally `rollout_id`; PyTorch is imported only for this format and tensors load on CPU.

The report preserves `source`, zero-based `record_index`, one-based JSONL `line_number`, and available `rollout_id`, `sample_index`, `index`, `group_index`, and `dataset`. Missing identifiers remain `null`. `sample_index` and `index` retain their original meanings and are not substituted for one another. Use source plus record position to distinguish records with duplicate or missing IDs.

Each sample includes `has_repetition`, `response_chars`, `scanned_windows`, `hits` (`start`, `end`, `compression_ratio`), `max_compression_ratio`, and `covered_chars`. Raw responses are not copied into the report. The top level records `schema_version`, offset conventions, detection settings, sources, samples, and a summary with sample counts, `repetition_frac`, scanned/covered character statistics, scanned windows, global maximum ratio, and elapsed seconds. Hit summaries are also logged.

Malformed JSON, non-object records, and absent/non-string responses fail with the source location rather than being silently omitted. Reports are streamed to a temporary file and published atomically after success; failure preserves an existing report. An explicit input cannot be overwritten by the report. JSONL memory scales with one record and its hits, whereas Torch deserialization loads one entire dump.

On POSIX systems, a new report is private to its owner (`0600`). Replacing an existing report preserves its read/write/execute permission bits; ownership, ACLs, and extended attributes are not copied. For shared access, set the report's desired permissions explicitly before subsequent updates. Temporary files remain private while the report is being generated. Windows access is governed by its filesystem ACLs.

::: warning Legacy Torch dumps
The default uses restricted `torch.load(..., weights_only=True)`. If your own trusted legacy dump contains custom pickled objects, explicitly add `--trusted-torch`. This permits arbitrary pickle execution; never enable it for untrusted files. A restricted-load error does not trigger an automatic unsafe fallback.
:::

## API Reference

```python
from relax.utils.repetition import RepetitionConfig, analyze_repetition, has_repetition
from relax.utils.rollout_dump import iter_rollout_records

config = RepetitionConfig(window_size=10_000, stride=5_000, threshold=10.0)
for record in iter_rollout_records("tests/fixtures/repetition/middle_repetition.jsonl"):
    detected = has_repetition(record.response, config)
    result = analyze_repetition(record.response, config)
    report_record = {**record.identity(), **result.to_dict()}
    assert detected == result.has_repetition
```

`analyze_repetition` returns an immutable `RepetitionResult`; each entry of `hits` is a `RepetitionWindow`. `iter_repetition_windows(length, config)` exposes the same coverage policy without compression. `diagnose_dumps(inputs, output, *, config=RepetitionConfig(), input_format="auto", trusted_torch=False)` in `relax.entrypoints.repetition` writes the full report and returns its summary.

## Validation and Benchmarks

```bash
python -m pytest tests/utils/test_repetition.py tests/utils/test_repetition_dump.py tests/utils/test_repetition_metrics.py
python -m relax.tools.benchmark_repetition --output benchmarks/repetition/local.json --repeats 7
```

Tests cover beginning/middle/end repetition, the fixed middle-repeat/normal-suffix fixture, empty and short strings, exact windows, overlap boundaries, unaligned suffixes, Chinese offsets, controls, threshold equality, and overlap-union counts. Integration tests exercise the production sample aggregation and actual dump writers/readers, including the CLI. The optional distributed logger test requires the training image's Ray/SGLang/TransferQueue/Megatron stack and reports a skip when absent; CPU integration does not establish GPU or multinode execution.

Committed measurements are in `benchmarks/repetition/windows-python313.json`. The following excerpt is the **detailed API, nonrepetitive control**, on Windows 11, Python 3.13.9, 12 logical CPUs, zlib runtime 1.3.1, seven timing batches:

| Characters | Windows | Median CPU ms | Median wall ms | Process peak MiB | Scan tracemalloc peak KiB |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 1 | 0.053 | 0.068 | 24.06 | 304.14 |
| 100,000 | 19 | 1.645 | 1.690 | 25.69 | 314.23 |
| 1,000,000 | 199 | 18.229 | 18.701 | 37.19 | 314.23 |

Each length/case/API runs in a fresh subprocess. Timing excludes imports, text generation, and tracemalloc, and batches complete calls to reduce CPU-clock quantization. Native process peak includes interpreter, input generation, warmup, and native zlib memory; it is not incremental scan memory. A separate tracemalloc pass measures only tracer-visible allocations. The full report also contains boolean, fully repeated, middle-repeat, and Chinese middle-repeat cases, raw runs, and p95 values. Detailed scans always visit every window. A mixed 10,000-character middle-repeat case can remain below threshold because it is a single diluted window. These are detector benchmarks, not dump I/O or end-to-end training throughput measurements.

## Troubleshooting

- **No hit despite some repetition:** inspect the reported maximum and original window text; compression thresholds do not guarantee detection of every repeated span.
- **Invalid configuration:** require `0 < stride <= window_size` and a positive finite threshold.
- **Unknown format or bad record:** select the input format explicitly when needed and repair the identified source record; do not discard invalid samples to make the report succeed.
- **High memory on Torch input:** the serializer loads the whole dump. JSONL supports streaming individual records.

## Next Steps

- [Rollout Result Viewer](./rollout-result-viewer.md)
- [Debugging Guide](./debugging.md)
- [Metrics Service](./metrics-service-detailed.md)
