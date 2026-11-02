# Complete Task14 CPU replay reproduction

Run this recipe in a **private Kaggle CPU notebook** with Internet enabled. It
uses no GPU, model weights or recorded tool commands. Keep trajectory inputs and
tokenizer files under `/tmp`, outside notebook outputs. The pinned public source
commit is recorded in `session_replay_evidence/replay-protocol.json`; obtain that
checkout and apply the supplied Task14 source overlay without changing its HEAD.
The driver checks the benchmark and candidate pipeline hashes before starting.

## Prepare the environment and inputs

From the supplied source checkout:

```bash
python -c 'from importlib.metadata import version; print("torch==" + version("torch"))' > /tmp/task14-torch-constraint.txt
python -m pip install -c /tmp/task14-torch-constraint.txt -r scripts/benchmarks/session_replay_evidence/cpu-replay-requirements.txt
python scripts/benchmarks/prepare_session_replay_inputs.py --root /tmp/task14-replay-inputs --fetch-corpus
```

The requirements pin the replay packages used by the inspected study. Keep the
provided Kaggle PyTorch build and record the newly allocated environment:
Kaggle images and CPU allocations can change. The latest public evidence retains
the actual run's source, provided Torch and command records. The preparer
downloads only the pinned SWE-Gym parquet and five tokenizer files, verifies
their SHA256 values and writes provenance. It never downloads model weights.

Prepare these inputs in Kaggle; trajectories need not reach the maintainer's
computer.

## Run the complete fixed comparison

```bash
python scripts/benchmarks/run_session_replay_comparison.py \
  --corpus /tmp/task14-replay-inputs/corpus \
  --tokenizer /tmp/task14-replay-inputs/tokenizer \
  --output /kaggle/working/task14-study-001
```

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
python scripts/benchmarks/summarize_session_replay.py \
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

The latest v41 metrics and raw observations are published in the
[Task14 v41 evidence release](https://github.com/Timothy-kira/Relax/releases/tag/task14-v41-evidence-20261102).
Only this latest run is included. Neither plan-only nor the fault tests reruns
or re-scores the saved performance study.
