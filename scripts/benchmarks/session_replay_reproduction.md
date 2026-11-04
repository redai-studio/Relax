# Complete Task14 CPU replay reproduction

Run this recipe in a **private Kaggle CPU notebook** with Internet enabled. It
uses no GPU, model weights or recorded tool commands. Keep trajectory inputs and
tokenizer files under `/tmp`, outside notebook outputs. Keep the reviewed public
candidate checkout for the driver, watchdog, logger and protocol assets. Build a
separate measurement-input checkout at the protocol's pinned `657f8fb` base as
shown below; the public candidate's HEAD is not the measurement base. The driver
checks that base plus the benchmark and candidate pipeline hashes before starting.
Run the shell snippets in the same Bash session so their path variables persist.

The selected candidate retains synchronous reduced-copy export and adds a
compatibility guard. Exact builtin, unaliased value trees use the reduced-copy
path; custom types, copy hooks, repeated containers, cycles and excessive depth
use the original deepcopy path. Its scan cost belongs in the measured candidate.
The protocol pins the guarded pipeline; historical performance numbers from an
earlier pipeline do not establish this revision's performance.

## Prepare the environment and inputs

From the reviewed public candidate checkout containing this recipe:

```bash
export TASK14_PUBLIC_ROOT=$(pwd -P)
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TASK14_PUBLIC_ROOT"
python -c 'import sys; from importlib.metadata import version; sys.stdout.write("torch==" + version("torch") + "\n")' > /tmp/task14-torch-constraint.txt
python -m pip install -c /tmp/task14-torch-constraint.txt \
  -r "$TASK14_PUBLIC_ROOT/scripts/benchmarks/session_replay_evidence/cpu-replay-requirements.txt"
python "$TASK14_PUBLIC_ROOT/scripts/benchmarks/prepare_session_replay_inputs.py" \
  --root /tmp/task14-replay-inputs --fetch-corpus
```

The requirements pin the replay packages used by the inspected study. Keep the
source checkout on `PYTHONPATH` for the runner and watchdog's repository logger.
JSON state and event files remain available even when the configured log level
suppresses informational progress messages. Keep the
provided Kaggle PyTorch build and record the newly allocated environment:
Kaggle images and CPU allocations can change. The latest public evidence retains
the actual run's source, provided Torch and command records. The preparer
downloads only the pinned SWE-Gym parquet and five tokenizer files, verifies
their SHA256 values and writes provenance. It never downloads model weights.

Prepare these inputs in Kaggle; trajectories need not reach the maintainer's
computer.

## Prepare the separate measurement-input source

The runner requires measurement-input HEAD `657f8fb`; passing the public candidate
checkout directly would fail that guard. Obtain a fresh base and copy only the
two reviewed measurement inputs, without changing its detached HEAD:

```bash
TASK14_PERF_BASE=657f8fb961af9ab4b648e7d58280055dbfe4b626
export TASK14_PERF_ROOT=$(mktemp -d /tmp/task14-performance-input.XXXXXX)
git -C "$TASK14_PERF_ROOT" init
git -C "$TASK14_PERF_ROOT" fetch --depth 1 \
  https://github.com/redai-studio/Relax.git "$TASK14_PERF_BASE"
git -C "$TASK14_PERF_ROOT" checkout --detach FETCH_HEAD
test "$(git -C "$TASK14_PERF_ROOT" rev-parse HEAD)" = "$TASK14_PERF_BASE"
mkdir -p "$TASK14_PERF_ROOT/scripts/benchmarks"
cp "$TASK14_PUBLIC_ROOT/scripts/benchmarks/benchmark_session_replay.py" \
  "$TASK14_PERF_ROOT/scripts/benchmarks/benchmark_session_replay.py"
cp "$TASK14_PUBLIC_ROOT/relax/agentic/pipeline/__init__.py" \
  "$TASK14_PERF_ROOT/relax/agentic/pipeline/__init__.py"
(cd "$TASK14_PERF_ROOT" && sha256sum -c <<'MEASUREMENT_PINS'
c8d99872fead81a02ec759d26edcc6a9c599bb325c8ee18a3d984f5b018c2e9b  scripts/benchmarks/benchmark_session_replay.py
2455c0ad053cc54025a8b618d1e03fa99854b54e13dd6ff75128bb92c23559f6  relax/agentic/pipeline/__init__.py
MEASUREMENT_PINS
)
```

Do not copy the public checkout's full tree, tests, HTTP support or later thread
experiments into this measurement input. The driver creates its own baseline and
candidate checkouts from `657f8fb`, copies the same benchmark to both and applies
the pipeline only to candidate.

## Run the complete fixed comparison

```bash
PYTHONPATH="$TASK14_PUBLIC_ROOT" \
  python "$TASK14_PUBLIC_ROOT/scripts/benchmarks/run_session_replay_comparison.py" \
  --source-root "$TASK14_PERF_ROOT" \
  --corpus /tmp/task14-replay-inputs/corpus \
  --tokenizer /tmp/task14-replay-inputs/tokenizer \
  --output /kaggle/working/task14-study-001
```

Protocol and schedule assets are resolved next to the public driver file, not
from `--source-root` or the caller's current directory. Its sibling watchdog and
summarizer also remain in the public checkout. The outer `PYTHONPATH` supplies the
repository logger to the driver and independent watchdog. For each measured arm,
the driver replaces the child's `PYTHONPATH` with that arm's own checkout and
sets its working directory there, so the public candidate cannot shadow baseline
imports. The summary collector uses only the standard library.

Use a fresh output directory for every explicitly requested study. The driver
creates two separate temporary checkouts at the pinned base, copies the same
frozen benchmark into both and applies only the optimized pipeline to candidate.
It fingerprints the full sources and requires that compaction is their sole
difference. A filename containing `baseline` does not select baseline code; the
driver selects the actual source checkout as each subprocess's working directory.
It validates all source hashes again after completion.

The packaged schedule has four paired repeats, short/long inputs, concurrency
1/4, and two revisions: **32 fresh processes**, 64 main exports and one warmup
export each, totaling 2,048 main and 32 warmup exports. Even repeats use baseline
then candidate; odd repeats reverse both arm and workload order. Each pair uses
its repeat index as the ordering seed. The benchmark preserves all due probes,
including arrivals drained after the finite export window. The collector checks
work counts, actual concurrency, selections, tokenizer identity and core training
hashes before calculating paired statistics.

For source/command readiness only, use the same invocation with `--plan-only`
and a different fresh output directory. This still builds and fingerprints the
two sources and writes all 32 commands, but measures **zero arms**. It cannot
establish a performance gain or substitute for the archived executed study.

## Background execution and progress

In a Kaggle code cell, `subprocess.Popen` can start the command above with
`start_new_session=True` and stdout/stderr redirected to a file. Keep the
notebook cell alive until the child exits so Kaggle does not stop its runtime.
The CLI also re-enters an isolated OS session when invoked by an ordinary shell.
Its independent `session_replay_watchdog.py` checks liveness, new log error
keywords, stalled command progress, command deadlines and the 1,800-second study
deadline. State, command identities and cleanup events persist in the output
directory. Commands are released only after their process ownership is recorded.
Automatic arm retries are zero; failures remain failures for diagnosis.

Read progress without starting another run:

```bash
cat /kaggle/working/task14-study-001/state.json
cat /kaggle/working/task14-study-001/watchdog-state.json
cat /kaggle/working/task14-study-001/active-command.json
```

Use the notebook log for cloud notifications. No local resident monitor is
required. These scripts do not provide an event-triggered model wakeup; do not
interpret persisted errors as proof that a model has already intervened.
Successful completion requires both the study and the independent supervisor's
terminal verification in `reproduction-summary.json`.

## Independently recompute the summary

```bash
PYTHONPATH="$TASK14_PUBLIC_ROOT" \
  python "$TASK14_PUBLIC_ROOT/scripts/benchmarks/summarize_session_replay.py" \
  --directory /kaggle/working/task14-study-001 \
  --output /kaggle/working/task14-study-001/recomputed-summary.json
```

Keep all `arm-PP-LENGTH-cC-ARM.json` files, command/order receipts, source
provenance, machine/environment records and supervisor terminal records together.
The collector can read an existing complete 32-arm metadata bundle without any
new replay. Recomputing historical metadata is arithmetic validation, not a new
performance measurement. The original study's raw artifact index binds its
compressed and uncompressed metadata hashes.

## Same-shard mixed fault integration

```bash
RELAX_TASK14_CORPUS=/tmp/task14-replay-inputs/corpus \
RELAX_TASK14_TOKENIZER=/tmp/task14-replay-inputs/tokenizer \
RELAX_TASK14_FAULT_OUTPUT=/kaggle/working/task14-fault-evidence \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=. \
  python -m pytest -q tests/agentic/test_session_export_concurrent_faults.py \
  --junitxml=/kaggle/working/task14-fault-junit.xml
```

Execute once per isolated revision with identical inputs; require all three
exact test IDs and no skips. Each case first obtains a clean four-Session control,
then exercises four long real-text Sessions on one shard: a cancelled finalizing
watcher with completing peers; a malformed logprob reaching the real compactor
while peer exports match control; and shutdown mixing finalizing, backend-inflight,
PREPARE-queued and completed Sessions. Receipts retain exact source rows, native
worker/descendant ownership, outcome publication, task/permit/FD/session-directory
cleanup and daemon reaping. Run the tests under a bounded command owner with the
provided watchdog; pass the evidence directory as its process-harness ledger.

The replay and fault tests use the real tokenizer, compiler, Forest, Session IR,
builder and compactor. Inference is deterministic mock inference, token IDs are
reconstructed, masks are generated and logprobs are synthetic. Fault integration
uses native managed processes and a local permit RPC adapter. No actual model,
HTTP/Ray deployment, GPU, multi-node or production steady-state claim follows.
The malformed logprob is explicitly injected fault input, not a dataset defect.

Historical v41 metrics and raw observations are published in the
[Task14 v41 evidence release](https://github.com/Timothy-kira/Relax/releases/tag/task14-v41-evidence-20261102).
They apply to the earlier reduced-copy pipeline, not the guarded candidate or
the thread-offload experiments. Neither plan-only nor the fault tests rerun or
re-score that saved study. Preserve each study's exact source and protocol pins.

## Copy compatibility and HTTP/Ray coverage

Run the eight additional semantic regressions in the same isolated candidate
environment:

```bash
PYTHONPATH=. python -m pytest -q tests/agentic/test_training_artifact_semantics.py \
  --junitxml=/kaggle/working/task14-copy-semantics-junit.xml
```

These cover public copy hooks, shared-rollout decisions, metadata traversal
order, root cycles, repeated container ownership, error precedence and admission
without unknown hooks. They complement the original three test modules.

The separate [HTTP/Ray functional recipe](session_export_http_ray_reproduction.md)
uses frozen external support files against the actual selected checkout. It
adds real HTTP/Ray correctness and cleanup evidence with deterministic generation
fixtures; it measures zero performance arms and supplies no real-model or GPU
training claim. Selected-source acceptance requires its 12 serializer controls
and all eight HTTP/Ray scenarios; missing or skipped coverage remains incomplete.
Require a fresh run after a production source change.
