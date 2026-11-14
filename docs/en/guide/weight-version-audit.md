# Weight Version Staleness Audit

In asynchronous training, multi-turn agentic generation can span several weight updates, and a finished sample may then wait a long time before training consumes it. The staleness audit separates the two lag sources: the **generation span** (version changes crossed while generating, typically weight releases during tool waits) and the **post-generation lag** (versions aged between generation completion and training consumption), alongside token-weighted lag and version-missing rates. The first release is a bypass audit capability — existing sampling and GRPO group semantics are untouched.

## Data Sources

The audit consumes two streams of the Agentic observability JSONL (directory layout in [Agentic Rollout](./agentic-rollout.md), schema `3.2.0`):

| Stream | Field | Meaning |
|---|---|---|
| `trajectories/*.jsonl` | `lineage[].weight_version_spans` | Per resp node and per backend generate attempt: the committed token interval and the engine's weight version at generation time; `null` when the engine did not report. Coordinates are **node-relative** (each turn starts at 0). |
| `train/*.jsonl` | `samples[].consumed_weight_version` | The actor's weight version (publication ordinal, monotonically increasing per policy) when training actually consumed the sample; `null` when not captured. |

The streams join on `sample_export_id`. Version numbers share one publication sequence: colocate/hybrid modes use `weight_updater.weight_version`, DCS uses `DeviceDirectBackend.weight_version`.

## Quick Start

Run offline against a produced observation directory (`<run_dir>` is `<dump_dir>/<run_id>`, the directory holding `trajectories/` and `train/`):

```bash
python -m relax.utils.agentic_obs.version_audit <run_dir> [--session S] [--group G] [--missing-only] [--jsonl]
```

| Flag | Meaning |
|---|---|
| `--session S` | Audit only samples of this `session_id` |
| `--group G` | Audit only samples of this `group_id` |
| `--missing-only` | Report only samples with unknown spans or unknown tokens |
| `--jsonl` | Emit one JSON detail row per sample instead of the report |

The default output is a run-level summary JSON plus per-sample detail lines. A reproducible end-to-end example (multi-turn generation with two weight updates during tool waits) is in `docs/superpowers/examples/weight-version-audit-report.md`, backed by the test `tests/cpu/test_weight_version_scenarios.py`.

## Metrics

Let V be the set of known versions among a sample's spans, token counts by span length, and `current` the `consumed_weight_version`:

| Metric | Definition |
|---|---|
| Generation span `generation_span` | `max(V) − min(V)`, known versions only; `null` when none |
| Consumption lag (oldest) `oldest_lag` | `current − min(V)` |
| Consumption lag (newest) `newest_lag` | `current − max(V)` |
| Token-weighted lag `token_weighted_lag` | `Σ((current − v_i) × len_i) / Σ len_i`, known-version spans only |
| Post-generation lag `post_generation_lag` | Equal to `newest_lag` (the aging after generation completed); reported separately from the generation span |
| Missing rate (span) | Unknown-version spans / total spans |
| Missing rate (token-weighted) | Unknown-version tokens / total tokens |
| Missing rate (sample) | Samples whose spans are all unknown / total samples |

Reading the numbers:

- **Unknown versions stay explicit**: `null` (engine did not report) and non-numeric version strings are both unknown — never guessed as version 0 or the newest version. Lag means/medians/maxima cover **only samples with known versions**; missing data is quantified separately through the three missing rates.
- **Interpretation**: a large generation span means generation itself crossed several releases (e.g., multiple `update_weights` during tool waits); a large post-generation lag means the sample queued a long time after completion before consumption.
- Detail-line token intervals are resp-node-relative; whole-response flattened coordinates are produced by `build_sample` (`Sample.weight_version_spans`).

## Detail-Line Format

```text
sample=<export_id> session=<session_id> turn=<resp node ordinal> attempt=<attempt_seq> tokens=[<start>,<end>) version=<source version|None> age=<age at consumption|None>
```

`age = current − version`, `None` when either side is unknown. Multiple attempts of one turn after abort recovery are distinguished by `attempt`, traceable to the individual generation segment.

## Fault Tolerance

- **Torn tail lines**: the writer contract allows a crashed process to leave a partial final JSON line; the reader skips it with a warning instead of aborting — post-mortem staleness analysis of a crashed run is a primary use case.
- **Overflow shards**: `overflow_step_*_dp_*.jsonl` shards written under train-side backpressure are separate updates for the same `(rollout_id, dp_rank)`; they are read in numeric order so the newest consumption stamp wins.
- **Missing streams**: a non-directory `run_dir` fails fast at the CLI; with no stream files, the missing rates in the summary are `null` (no denominator), never 0.
