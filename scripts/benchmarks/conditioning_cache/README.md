# Frozen-conditioning cache benchmark tools

This directory contains the reusable parts of the multi-model SFT cache
validation workflow. Raw measurements, fixed experimental plans, model/data
manifests, and large tensor snapshots belong in a separately versioned evidence
package. Do not add model weights or historical experiment directories here.

The training entry points remain:

- `scripts/entrypoint/ray-job.sh` with
  `scripts/training/diffusion/run-sensenova-condition-cache.sh`.
- `scripts/entrypoint/ray-job.sh` with
  `scripts/training/diffusion/run-qwen-image-sft-condition-cache.sh`.

## Tools

| Tool                     | Responsibility                                                                                                                                                                      |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prepare_experiments.py` | Validate a frozen model-neutral plan and render reviewable launch requests; no network or Ray calls.                                                                                |
| `run_experiments.py`     | Execute a prepared plan on an existing dedicated single-node **8-GPU** Ray cluster, preserving source/request/harness identities. Defaults to preparation only.                     |
| `owned_serve.py`         | Internal ownership checks for the runner. Identify applications using names, deployment timestamps/configuration, and live replica run markers before any name-scoped cleanup.      |
| `summarize_plan.py`      | Check declared run coverage/configuration and call the existing [raw benchmark parser](../../tools/benchmark_condition_cache.py); do not pool independent models or configurations. |
| `compare_snapshots.py`   | Verify snapshot manifests and compare complete FP32 gradient/update tensors across all shards, including initial-state/input/RNG preconditions. CPU only.                           |

The runner deliberately retains its tested dedicated eight-GPU restriction. It
is not a new general cluster scheduler. It requires one shared node lock, no
other active jobs or Serve applications at launch, and idle GPU/Ray resources.
It does not start or stop Ray, kill training processes, or globally delete Serve
applications. Successful jobs permit cleanup only of applications whose live
ownership was captured and is still valid. Failed/timed-out jobs remain for
inspection. Cancellation, component probes, stage profiling, and archive
packaging use the separately pinned evidence harness appropriate to that run.

## Plan and execution

Use the fixed plan supplied with the evidence as a schema example; create a new
plan with new output directories and job names for every reproduction. Do not
rerun a completed plan in place. A plan records:

- `schema_version: 1` and `status: "ready"` after preparation is complete;
- `cluster`: Ray/dashboard addresses, expected GPUs, dependency Python paths,
  shared node lock, and exact expected Serve deployment names;
- `workloads`: model identity/path/content SHA, dataset selection/metadata SHA,
  frozen source/launcher SHA, effective batch/training configuration, and
  independent payload/prefetch byte limits;
- `runs_root` and ordered `runs`: unique name, workload, purpose, repeat, cache
  setting, number of updates, and predeclared cold window.

Performance groups require complete online/cache pairs. Keep all relevant
training settings identical within each pair, including existing optimizations;
only the cache switch differs. Preserve low-reuse and failed results. A prepared
request is not evidence that execution or acceptance passed.

```bash
python scripts/benchmarks/conditioning_cache/prepare_experiments.py \
  --plan /path/to/new-plan.json --check-local --output /path/to/new-prepared.json

# The default runner command validates/prepares without submitting jobs.
python scripts/benchmarks/conditioning_cache/run_experiments.py \
  --plan /path/to/new-plan.json

# Execute only on the authorized idle cluster after reviewing the new plan.
python scripts/benchmarks/conditioning_cache/run_experiments.py \
  --plan /path/to/new-plan.json --execute
```

`--check-local` must run on the execution host, where the source, launcher,
models, and dataset paths exist. The runner submits the canonical training
scripts through `ray-job.sh`. Every run archives its request, source identity,
job result, per-rank events, and the exact harness files used for execution.
Do not hash models, download assets, profile, or copy large evidence while
collecting formal performance measurements.

## Offline verification

```bash
python scripts/benchmarks/conditioning_cache/summarize_plan.py \
  --plan /path/to/fixed-plan.json --runs-root /path/to/completed-runs \
  --repo /path/to/frozen-checkout --output /path/to/new-summary.json

python scripts/benchmarks/conditioning_cache/compare_snapshots.py \
  --reference /path/to/reference-snapshots \
  --actual /path/to/actual-snapshots --world-size 8 --rtol 0 \
  --output /path/to/new-comparison.json
```

The snapshot comparison uses elementwise tensor differences aggregated across
shards, not a comparison of tensor norms. Every parameter tensor must be finite;
a zero reference norm requires an exactly zero result. Updates are the actual
FP32 `theta_after - theta_before`, not the relative error of final full weights.
Use `--rtol 0` for the deterministic exact path. The tool accepts at most 1%
relative tolerance; a non-deterministic experiment still needs an independently
established repeated reference and justified absolute-error policy.

The plan summary establishes run provenance and paired training throughput. It
does not independently certify lifecycle, random sampling semantics, resource
growth, confidence intervals, or every acceptance item. Preserve the raw logs
and separate resource/lifecycle/advanced-statistics reports in the evidence
package, with the exact scripts that produced them. Historical evidence must
use its original harness; this directory uses the neutral `conditioning-cache-`
job prefix for new runs and does not rewrite old requests.

## Tests

```bash
python scripts/benchmarks/conditioning_cache/test_harness.py -v
CUDA_VISIBLE_DEVICES= python -m pytest -q \
  scripts/benchmarks/conditioning_cache/test_compare_snapshots.py
```

The first suite is stdlib-only and mocks ownership/plan/numerical gates without
Ray, network, or GPU execution. Snapshot tests use existing CPU PyTorch/pytest
dependencies. These tests do not replace real distributed SFT verification.

Model contracts and cache lifecycle are documented in
[frozen-conditioning-cache](../../../docs/en/guide/frozen-conditioning-cache.md)
and [Qwen T2I SFT](../../../docs/en/guide/t2i-sft.md).
