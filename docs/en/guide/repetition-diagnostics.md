# Repetition Detection and Offline Diagnostics

Use offline diagnostics to locate suspicious regions when rollout responses contain long repeated passages. The tool scans the entire `response` field, including any tool observations stored in it, so it can locate repetition near the beginning, middle or end.

## Quick Start

Run from the repository root with an existing training/evaluation JSONL result or a trusted `.pt` rollout dump:

```bash
python -m relax.entrypoints.diagnose_repetition /path/to/rollout_result/train/42.jsonl --output report.json
python -m relax.entrypoints.diagnose_repetition /path/to/rollout_data/42.pt --output report.json
```

Omit `--output` to write JSON to stdout. JSONL needs only Python's standard library; `.pt` also requires PyTorch. Both formats can be analyzed on CPU. Only load trusted `.pt` files because their serialization format can execute code when loaded.

## Configuration

| Parameter | Default | Meaning |
|---|---|---|
| `--window-size` | `10000` | Characters per window; a positive integer. |
| `--stride` | `5000` | Characters between window starts; positive and no greater than window size. |
| `--threshold` | `10.0` | Compression ratio must be strictly greater than this finite positive value to count as a hit. |

These parameters apply to the offline report. The training `repetition_frac` metric uses the default settings. The tool scans every window, uses a single window for short nonempty responses, and adds a final window when needed to cover the end. Compression ratio is the number of UTF-8 bytes divided by the size after zlib compression at level 9.

The metric stops scanning a sample after its first hit; the offline report always scans every window to collect complete diagnostics. Both produce the same repetition decision with the default settings. Compared with the previous tail-only metric, repetition anywhere in a response can now count, including responses of 10,000 characters or fewer that previously always returned false. Do not directly compare `repetition_frac` across runs using the old and new detectors.

## Reading the Report

| Field | Meaning |
|---|---|
| `summary.repetition_frac` | Fraction of samples with at least one hit. |
| `has_repetition` | Whether this sample has any hit windows. |
| `hit_windows` | Suspicious windows, each with `start`, `end` and `compression_ratio`. |
| `max_compression_ratio` | Largest ratio across all scanned windows; `0.0` for an empty response. |
| `window_count` | Number of windows scanned for this sample. |

Use `source` and `sample_position` to locate the original record: the position is a zero-based JSONL line number or `.pt` sample-list index. Original rollout/sample IDs and available dataset names are also included.

Window offsets are zero-based Unicode code-point indices with an exclusive end. For example, `start=10000`, `end=20000` identifies `response[10000:20000]`; these are character positions, not byte or token positions. Hit windows can overlap and do not indicate exact repetition boundaries. Highly compressible text, such as structured tool output, may also trigger a hit, so inspect the reported region before treating it as unwanted repetition.

## Next Steps

- [Rollout Result Viewer](./rollout-result-viewer.md) — Browse the original prompts and responses.
- [Debugging Guide](./debugging.md) — Investigate training accuracy issues.
