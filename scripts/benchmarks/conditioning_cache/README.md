# Frozen-conditioning cache validation

Use the normal training entrypoints and the shared numerical verifier. Fixed
experiment plans, cluster orchestration and raw logs belong in the separately
versioned experiment evidence, not the training library.

## Launch and verify

Prepare worker-visible models, data and output directories as described in
[SenseNova SFT](../../../docs/en/guide/frozen-conditioning-cache.md) or
[Qwen-Image SFT](../../../docs/en/guide/t2i-sft.md). Select the corresponding
launcher:

| Model      | Training script                                                    |
| ---------- | ------------------------------------------------------------------ |
| SenseNova  | `scripts/training/diffusion/run-sensenova-condition-cache.sh`      |
| Qwen-Image | `scripts/training/diffusion/run-qwen-image-sft-condition-cache.sh` |

Submit through `bash scripts/entrypoint/ray-job.sh "$CONDITIONING_LAUNCHER"`
on an existing Ray cluster after setting `CONDITIONING_LAUNCHER` to one of the
paths above. Set `MODEL_PATH`, `DATA_ROOT`, `METADATA_PATH`, `DATASET_SIZE`,
`SAVE_DIR` and the Ray endpoint for the workload. Use a new experiment name and
`RELAX_CONDITION_CACHE_RUN_DIR` for every run. `DRY_RUN=1` previews a launcher
without submitting a job.

Before measuring performance, enable `RELAX_CONDITION_CACHE=1`,
`RELAX_CONDITION_VERIFY=1`, `RELAX_CONDITION_VERIFY_RTOL=0` and
`DETERMINISTIC_ATTENTION=1`. For a nonzero first-step update, set
`LR_WARMUP_ITERS=0` for Qwen; for SenseNova append `--lr-warmup-iters 0`
after the launcher path in the submission command. The shared verifier restores
initial model, optimizer and RNG state, exercises the actual cache branch, and
compares loss, complete effective-batch reduced gradients and actual FP32
parameter updates. It writes `conditioning-verification.json` in the run
directory. Every compared value must be finite; zero references require exact
zeros. Deterministic runs require exact agreement. A non-deterministic result
needs repeated online references and a justified absolute tolerance in addition
to the 1% relative ceiling. Verification performs extra training work and must
not be included in performance measurements.

## Measure repeated workloads

Disable verification and run separate online (`RELAX_CONDITION_CACHE=0`) and
cached (`RELAX_CONDITION_CACHE=1`) jobs. Keep data order, effective batch,
resolution, optimizer, initialization, encoder placement and other training
settings identical; only the cache switch changes. Use multiple independent
pairs, alternate their order, and retain low-reuse results as well.

```bash
python scripts/tools/benchmark_condition_cache.py \
  --online /path/to/online-r01 /path/to/online-r02 /path/to/online-r03 \
  --cached /path/to/cached-r01 /path/to/cached-r02 /path/to/cached-r03 \
  --cold-steps "${COLD_STEPS:?Set the predeclared cold-window step count}" \
  --output /path/to/summary.json
```

The parser reads per-rank events, counts the global batch once and uses the
slowest rank's complete step time. It reports cold and steady windows, repeated
run variation and missing evidence; local encoding time is only diagnostic.
Add `--low-reuse-online` and `--low-reuse-cached` directories for separate
low-reuse comparisons. Set the cold window from the data size and experiment
plan rather than choosing it after seeing results.

Preserve the exact source commit, model/data revisions, launch configuration and
raw events. Full job timing additionally needs Ray's `job.json` with `status`,
`start_time` and `end_time` in milliseconds. Trainer events alone exclude
startup and shutdown. Keep external CPU/GPU/shared-memory/disk measurements,
cache build and payback analysis, and cancellation/error cleanup evidence with
the experiment reports; throughput summaries do not certify those properties.
