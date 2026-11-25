# Weight Version Staleness Audit

In asynchronous training, multi-turn agentic generation can span several weight updates, and a finished sample may then wait a long time before training consumes it. The staleness audit separates the two lag sources: the **generation span** (version changes crossed while generating, typically weight releases during tool waits) and the **post-generation lag** (versions aged between generation completion and training consumption), alongside token-weighted lag and version-missing rates. The first release is a bypass audit capability — existing sampling and GRPO group semantics are untouched.

## Data Sources

The audit consumes two streams of the Agentic observability JSONL (directory layout in [Agentic Rollout](./agentic-rollout.md), schema `3.2.0`):

| Stream | Field | Meaning |
|---|---|---|
| `trajectories/*.jsonl` | `lineage[].weight_version_spans` | Per resp node and per backend generate attempt: the committed token interval and the engine's weight version at generation time; `null` when the engine did not report. Coordinates are **node-relative** (each turn starts at 0). |
| `train/*.jsonl` | `samples[].consumed_weight_version` | The actor's weight version (publication ordinal, monotonically increasing per policy) when training actually consumed the sample; `null` when not captured. |
| `train/*.jsonl` | `samples[].loss_mask.ranges` | The loss mask training actually applied (half-open intervals over the response). Token-weighted metrics count only these tokens. |

The streams join on `sample_export_id`. The audit first shifts every resp node's spans into response coordinates (by the response tokens committed before that node, including the interleaved tool observations), so a span and the loss mask cover the same tokens. Version numbers share one publication sequence: colocate/hybrid modes use `weight_updater.weight_version`, DCS uses `DeviceDirectBackend.weight_version` (a sync that only refreshes actor_fwd and publishes nothing to rollout does not advance it, so one real publish never reads as two).

## Reported Population

**Only samples with a train-stream record enter the training population.** Sessions write trajectories before reward/filtering, and evaluation exports land in the same stream, so exports that were filtered out, evaluation-only, or whose update never reached disk are absent from `train/*.jsonl`. Those exports are listed separately as `unconfirmed` (`unconfirmed_sample_count` plus detail lines tagged `reason=no-train-record`) and never contribute to `sample_count`, lag means, or missing rates.

Conversely, a sample with a train record whose `consumed_weight_version` is `null` is still confirmed consumption — only its consumption point is unknown, and its lags read as `null`.

## Quick Start

Run offline against a produced observation directory (`<run_dir>` is `<dump_dir>/<run_id>`, the directory holding `trajectories/` and `train/`):

```bash
python -m relax.utils.agentic_obs.version_audit <run_dir> [--session S] [--group G] [--missing-only] [--jsonl]
```

| Flag | Meaning |
|---|---|
| `--session S` | Audit only samples of this `session_id` |
| `--group G` | Audit only samples of this `group_id` |
| `--missing-only` | Report only samples with unknown spans or unknown tokens (including unknown spans among unconfirmed exports) |
| `--jsonl` | Emit one JSON detail row per sample instead of the report |

The default output is a run-level summary JSON plus per-sample detail lines. `--jsonl` rows carry a `confirmed` field separating consumed training samples from unconfirmed exports.

## Metrics

Let V be the set of known versions among a sample's spans, token counts by **how many span tokens fall inside the effective loss mask**, and `current` the `consumed_weight_version`:

| Metric | Definition |
|---|---|
| Generation span `generation_span` | `max(V) − min(V)`, known versions only; `null` when none |
| Consumption lag (oldest) `oldest_lag` | `current − min(V)` |
| Consumption lag (newest) `newest_lag` | `current − max(V)` |
| Token-weighted lag `token_weighted_lag` | `Σ((current − v_i) × len_i) / Σ len_i`, known-version spans only, `len_i` the effective token count |
| Post-generation lag `post_generation_lag` | Equal to `newest_lag` (the aging after generation completed); reported separately from the generation span |
| Effective tokens `effective_token_count` | Span tokens that fall inside the loss mask |
| Missing rate (span) | Unknown-version spans / total spans |
| Missing rate (token-weighted) | Unknown-version effective tokens / effective tokens |
| Missing rate (sample) | Samples whose spans are all unknown / total samples |

Reading the numbers:

- **Unknown versions stay explicit**: `null` (engine did not report) and non-numeric version strings are both unknown — never guessed as version 0 or the newest version. Lag means/medians/maxima cover **only samples with known versions**; missing data is quantified separately through the three missing rates.
- **Only trained tokens count**: `--rollout-sample-filter-path` can zero a sample's loss mask while the sample still goes through GRPO group processing, and the train record keeps such rows. Token-weighted lag and token missing rate count only tokens inside the mask; otherwise a zeroed sample would dilute the lag with tokens training never used. Version-level metrics (`oldest_lag` / `newest_lag` / `generation_span`) still cover all known spans.
- **Zero-effective samples are separate**: when a sample keeps no token (all-zero mask), its `token_weighted_lag` and `token_missing_rate` are `null`, it is counted in the summary's `zero_effective_sample_count`, and it does not enter the token-weighted mean.
- **Interpretation**: a large generation span means generation itself crossed several releases (e.g., multiple `update_weights` during tool waits); a large post-generation lag means the sample queued a long time after completion before consumption.
- Detail-line token intervals are in response coordinates and compare directly against the training loss mask; `Sample.weight_version_spans` uses the same coordinates.

### Example

An agent generates over several turns while two weight updates land during tool waits (v1 → v3), and training consumes it at `current = v5`: turn 1's 2 tokens come from v1, turn 2's 3 tokens from v3.

```json
{
  "sample_count": 1,
  "lag_sample_count": 1,
  "oldest_lag_mean": 4.0,
  "oldest_lag_median": 4,
  "oldest_lag_max": 4,
  "newest_lag_mean": 2.0,
  "newest_lag_median": 2,
  "newest_lag_max": 2,
  "token_weighted_lag_mean": 2.8,
  "generation_span_mean": 2.0,
  "effective_token_count": 5,
  "zero_effective_sample_count": 0,
  "span_missing_rate": 0.0,
  "token_missing_rate": 0.0,
  "sample_missing_rate": 0.0,
  "unconfirmed_sample_count": 0
}
```

```text
sample=1001 session=sess-1 turn=1 attempt=0 tokens=[0,2) version=1 age=4
sample=1001 session=sess-1 turn=2 attempt=0 tokens=[2,5) version=3 age=2
```

The generation span is 2 (v1 → v3), the post-generation lag is 2 (v3 → v5), and the token-weighted lag is `(4×2 + 2×3) / 5 = 2.8`. The example has no tool-observation tokens between its two turns, so turn 2 starts at 2; with observations in between, the coordinates shift further by their length. Backed by the test `tests/cpu/test_weight_version_scenarios.py`.

## Detail-Line Format

```text
sample=<export_id> session=<session_id> turn=<resp node ordinal> attempt=<attempt_seq> tokens=[<start>,<end>) version=<source version|None> age=<age at consumption|None>
sample=<export_id> session=<session_id> unconfirmed=true reason=no-train-record spans=<span count>
```

`age = current − version`, `None` when either side is unknown. Multiple attempts of one turn after abort recovery are distinguished by `attempt`, traceable to the individual generation segment.

## Fault Tolerance

- **Torn tail lines**: the writer contract allows a crashed process to leave a partial final JSON line; the reader skips it with a warning instead of aborting — post-mortem staleness analysis of a crashed run is a primary use case.
- **Overflow shards**: `overflow_step_*_dp_*.jsonl` shards written under train-side backpressure are separate updates for the same `(rollout_id, dp_rank)`; they are read in numeric order so the newest consumption stamp wins.
- **Missing masks**: an older record without `loss_mask` is not treated as an all-zero mask; its spans count at full length, so unknown evidence never reads as "did not train".
- **Missing streams**: a non-directory `run_dir` fails fast at the CLI; with no stream files, the missing rates in the summary are `null` (no denominator), never 0.
