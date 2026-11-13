# Qwen3.5-35B RL with native Megatron OpenTelemetry

This example is a **pre-upgrade development baseline** for observability work.
It runs text prompts from DAPO Math on Qwen3.5-35B-A3B with eight colocated GPUs
and exposes the candidate Megatron dependency's own OpenTelemetry spans.

The recipe is intentionally short: two rollouts, four prompts per rollout and
four responses per prompt, with a default response budget of 512 tokens. It uses
GRPO group advantages, asymmetric policy clipping and TIS, like Relax's existing
Qwen3.5 colocate recipe. It does not enable dynamic reward filtering or change the
reward to make a smoke test pass. Truncated responses may all receive zero reward.
This is not an RL quality or full DAPO algorithm benchmark.

## Dependency identity

The candidate image uses the following exact source revisions:

| Component       | Revision                                   |
| --------------- | ------------------------------------------ |
| Megatron Bridge | `6862ee170a2569607e8ebc5dde91f119779093ca` |
| Megatron Core   | `9e68a11bdd380610396a17594ba4b17c09108cc4` |
| NeMo Lens       | `b85578fc2b736a1804705e537001b5f45e9c715d` |

The Core snapshot declares version `0.20.0` and is the above Bridge revision's
`.main.commit`. **It is not the final `core_v0.20.0` release tag.** Keep the source
SHAs with the image digest when reporting issues; the version string is not
sufficient to identify this development baseline.

## Build the candidate image

The candidate uses the existing `docker/Dockerfile` and its `relax` target, so
the normal CI image pipeline can build this branch without a custom Dockerfile
or additional build flags. The default build selects the clean paired Megatron
snapshots above, omits the regular Megatron patches, and installs Lens. The
existing train-stage and CUDA runtime build process still applies.

```bash
docker build \
    -f docker/Dockerfile \
    --target relax \
    -t relax:megatron-otel-qwen35 .
```

Existing CI overrides such as `BASE_IMAGE` and a cached `TRAIN_IMAGE` keep their
normal meaning. The Megatron/Lens selection is in the final `relax` stage and
does not require rebuilding a cached train stage.

The resulting checkout is `/root/Relax`. Dependency source identities are
recorded in `/opt/relax/megatron-otel-versions.txt`. Keep the build's Relax commit,
base image and output digest alongside this file. Build-time imports do not
establish that GPU training works.

An operator-built image from Relax `1e72874b76ec6a92a1b9f68344e096fb9cf4972b`
completed the GPU smoke run described below. That run supplied the same runtime
flags now used by this recipe. Image builds remain operator-owned.

## Prepare assets

Run commands from the Relax checkout included in the candidate image. Provide
an exclusive eight-GPU node and mount model/data directories into the container.
The accepted run used one node with eight H20-3e 141 GB GPUs; this document does
not claim that every eight-GPU system fits.

```bash
export MODEL_DIR=/path/to/models
export DATA_DIR=/path/to/data

hf download Qwen/Qwen3.5-35B-A3B \
    --local-dir "${MODEL_DIR}/Qwen3.5-35B-A3B"
hf download --repo-type dataset zhuzilin/dapo-math-17k \
    --local-dir "${DATA_DIR}/dapo-math-17k"
```

Existing copies can be reused and mounted read-only. Keep all model config, tokenizer, processor and
weight files. The original checkpoint's architecture is detected through
AutoBridge, including its vision-language wrapper when present; text-only data
does not automatically convert a conditional-generation checkpoint into a pure
causal-LM checkpoint. This recipe does not supply images or test vision behavior.
The dataset is JSONL with `prompt` and `label` fields. The built-in `dapo` reward
returns a dictionary, and the recipe selects its `score` field.

`HF_CHECKPOINT` and `PROMPT_SET` can override the two default paths. `MODEL_DIR`
and `DATA_DIR` remain the model and dataset roots.

## Run the smoke recipe

```bash
bash examples/megatron_otel/scripts/run-qwen35-35B-A3B-dapo-8xgpu-colocate.sh
```

Direct execution uses `scripts/entrypoint/local.sh`, which cleans local
Ray/Python training processes before starting a local Ray head. Use an exclusive
node. If the candidate image is already present on every node of an existing
cluster, skip local cluster startup and supply its Ray head address:

```bash
RELAX_ENTRYPOINT_MODE=existing \
HOST_IP=<ray-head-ip> \
RAY_NO_WAIT=1 \
    bash \
    examples/megatron_otel/scripts/run-qwen35-35B-A3B-dapo-8xgpu-colocate.sh
```

Submission blocks and follows logs by default. For an explicitly nonblocking
submission, set `RAY_NO_WAIT=1`; then inspect the returned job ID using
`ray job logs --address="http://<ray-head-ip>:8265" <job-id>`. A successful submission alone
is not proof of a completed optimizer step. Per-launch logs are under this
example's `scripts/log/` directory. With no-wait, that file contains only the
submission output.

The default layout is BF16, training TP4/SP/PP1/CP1/EP8/ETP1, and one TP8 SGLang
engine. It retains CPU optimizer offload and full activation recomputation, and
uses selective parameter/gradient offload and the native all-to-all MoE
dispatcher. The candidate image's TMS preload reproduced a CUDA IPC
`_share_cuda_()` invalid-argument failure on the tested driver; selective offload
avoids that preload while preserving colocate sleep/wake. No dependency patch
is required for this workaround.

Streaming input, two reward workers, disabled SGLang CUDA Graph/custom all-reduce
and disabled TransferQueue RDMA keep this smoke run small. The script also
forwards `RAY_OVERRIDE_JOB_RUNTIME_ENV=1` so the job and driver runtime environments
can merge when using an existing Ray cluster. It enables no FP8, MTP, custom Qwen
whole-pack optimizations, fully asynchronous training, evaluation or checkpoint
save/load. Every default invocation initializes from the HF checkpoint.

ClearML is optional: `USE_CLEARML=1` enables it using existing credentials.
`PROJECT_NAME` and `EXP_NAME` control metric/task naming. This recipe only starts
from HF. Checkpoint saving and resuming are not supported by this smoke recipe.

For a launch preview without starting Ray or loading the model:

```bash
DRY_RUN=1 bash examples/megatron_otel/scripts/run-qwen35-35B-A3B-dapo-8xgpu-colocate.sh
```

## Observe native spans

Console export is enabled by default, on training rank 0. The script copies all
`MEGATRON_OTEL_*`, `NEMO_LENS_*` and `OTEL_*` variables into the Ray job runtime
environment, so they reach workers instead of remaining only in the submitting
shell. `NEMO_LENS_RUN_ID` is generated once per invocation and shared by all ranks.
Set it explicitly when a particular run identifier is required.

The default span-group setting is `per_step,microbatch`. Look for actual native
spans such as `megatron.microbatch.forward`, `megatron.microbatch.backward`,
`megatron.grad_sync.start` and `megatron.grad_sync.finish`, with the intended run
and rank identifiers. A hand-written probe span is not sufficient evidence that
the native training instrumentation was exercised. These spans describe CPU-side
framework call intervals, not a complete CUDA kernel timeline.

To send telemetry to an OTLP receiver reachable **from the training workers**:

```bash
MEGATRON_OTEL_EXPORTER=otlp \
OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector>:4317 \
OTEL_EXPORTER_OTLP_PROTOCOL=grpc \
    bash examples/megatron_otel/scripts/run-qwen35-35B-A3B-dapo-8xgpu-colocate.sh
```

Use `OTEL_EXPORTER_OTLP_HEADERS` only if the receiver requires authentication.
The recipe disables shell xtrace before passing the runtime environment. To
inspect all training ranks, set `MEGATRON_OTEL_EXPORT_STRATEGY=all_ranks`; rank 0
export is the smaller default for development.

### Check a captured training log

Choose an explicit run ID before submitting so the same value can be supplied
to the verifier:

```bash
export NEMO_LENS_RUN_ID=qwen35-otel-smoke-001
bash examples/megatron_otel/scripts/run-qwen35-35B-A3B-dapo-8xgpu-colocate.sh

python examples/megatron_otel/verify_traces.py \
    examples/megatron_otel/scripts/log/<launch-log>.log \
    --run-id "${NEMO_LENS_RUN_ID}"
```

With a nonblocking Ray submission, save `ray job logs` output after the job
finishes and pass that file instead. The no-wait submission log alone does not
contain the training spans. The script disables Ray log deduplication so similar
span records are not suppressed; the verifier tolerates duplicate exported
records and strips Ray prefixes and ANSI color codes.

The verifier prints a bounded JSON summary and exits with status 0 only when a
single exported actor `relax.train.step` has native forward **and** backward
descendants, plus a `relax.optimizer.step` descendant whose
`relax.optimizer.update_successful` attribute is true, all under the requested
run ID. It follows intermediate spans using `trace_id` and `parent_id`; matching
span names in unrelated traces do not pass. Gradient sync span counts are also
reported. Missing evidence exits with status 1. This helper checks Lens console
JSONL; use the receiver's own query/export facilities when using OTLP.

## Validation boundary

The intended acceptance is one or more real policy optimizer updates on DAPO
data and observable native Megatron spans. The second default rollout also
exercises the normal colocate return to rollout after training. No tensor-level
weight comparison or numerical equivalence assertion is part of acceptance.

| Check                                                            | Status                                                                                                                                 |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Shell syntax and argument/environment forwarding                 | Passed locally with `bash -n` and an isolated Ray CLI stub; no GPU or Ray cluster started                                              |
| Clean dependency imports and text provider configuration         | Passed on CPU; no weight loading or GPU model construction                                                                             |
| Lens setup, native DDP decorators and exception-path trace flush | Passed on CPU with empty DDP buckets; no GPU communication                                                                             |
| Candidate image and actual import provenance                     | Operator-built image exercised; Core/Bridge import paths and relevant source digests matched the pinned revisions                      |
| Qwen3.5-35B DAPO generation, backward and optimizer update       | Passed: two rollouts and two actor steps on eight H20-3e 141 GB GPUs; job succeeded                                                    |
| Native Megatron spans exported from the real training run        | Passed: verifier exited 0; two actor traces each contained native forward/backward and an optimizer span with `update_successful=true` |
| Numerical accuracy, RL quality and weight equivalence            | Not validated; both smoke steps had zero loss and grad norm                                                                            |
| Throughput, memory regression and long-running stability         | Not validated                                                                                                                          |
| Checkpoint save/load, old-checkpoint compatibility and recovery  | Not validated                                                                                                                          |
| Other models, vision inputs, PP/CP > 1 and fully async           | Not validated                                                                                                                          |

The accepted console capture contained 14 unique spans: two `relax.train.step`,
two `relax.optimizer.step`, four `megatron.microbatch.forward`, two
`megatron.microbatch.backward` and two `megatron.grad_sync.finish`, plus two Relax
forward/backward parent spans. Two native forward spans were standalone log-prob
passes; the verifier did not count them as descendants of a training step.

The draft branch is for observability development. This validates the execution
path and console export, not nonzero parameter changes or useful learning. OTLP
collector delivery has not been validated.

An exploratory run of the existing checkpoint/HDO CPU tests returned 25 passes
and one failure in `test_hybrid_cpu_hf_reload_refreshes_bf16_and_native_fp32_copies`
with both the installed dependency and the clean candidate. This prototype does
not claim HDO FP32 reload correctness; the failure is not being treated as an
accuracy acceptance gate for this observability smoke run.
