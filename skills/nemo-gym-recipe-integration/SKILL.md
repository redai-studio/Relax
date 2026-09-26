---
name: nemo-gym-recipe-integration
description: Integrate a new NVIDIA NeMo Gym environment into Relax as a three-step recipe. Use when adding or debugging a recipe under examples/nemo_gym_agentic/recipes; covers data preparation, a local private Gym service, direct Ray training launch, verifier validation, callback networking, lifecycle cleanup, and failure triage.
argument-hint: "<nemo-gym-environment>"
---

# Integrating a NeMo Gym recipe

Build a new recipe by following the proven Calendar flow and the accumulated PITFAIL records. Keep the user-facing workflow to exactly three steps:

1. prepare data;
2. start the local NeMo Gym service;
3. start remote Relax training.

## Non-negotiable rules

- Deliver executable scripts, not a prose-only procedure.
- Run steps 1 and 2 locally. Fix and retry until data, readiness, and the verifier all pass.
- Put only deployment-specific values in `env.sh`: image, model path, Gym host/port, shared data path, and optionally remote Ray address.
- Write training hyperparameters directly in the training script. Do not create `NEMO_GYM_NUM_ROLLOUT`, `NEMO_GYM_N_SAMPLES_PER_PROMPT`, context-length, batch-size, or parser environment variables.
- The training script must contain the complete parameter arrays and final `ray job submit`. It may use `scripts/run_training.sh` as the remote job entry and `scripts/run_agent_app.sh` as the thin agent client, but must not delegate to another model training recipe.
- Never enable `--no-wait` by default. Add it only when the caller explicitly sets `RAY_NO_WAIT=1`.
- Data preparation must create the exact shared file checked by training. A file left only in a Docker volume is not prepared data.
- Training consumes the raw NeMo Gym JSONL. Do not pass an already converted `*_relax.jsonl` back through `convert_dataset.py`.
- Do not claim success from `/readyz`, a Ray Job `SUCCEEDED` state, or a one-sample rollout alone.
- A NeMo HTTP 500 must leave a traceback in `docker logs`; do not accept access-log-only failures.
- Preserve user changes and replace only the named recipe container after verifying its ownership label.
- Never run broad `ray stop`, `pkill`, Docker prune, or sandbox cleanup on a shared system.

## Read before editing

Read these files completely because they evolve with real failures:

```bash
rg --files examples/nemo_gym_agentic/recipes | rg '/PITFAIL\.md$' | sort
```

Then read:

- every returned `PITFAIL.md`;
- the target environment config in the pinned NeMo Gym checkout;
- the target agent, resource server, dataset declaration, verifier, and cleanup behavior;
- `examples/nemo_gym_agentic/service/Dockerfile`;
- `examples/nemo_gym_agentic/scripts/convert_dataset.py`;
- `examples/nemo_gym_agentic/scripts/run_training.sh`;
- `examples/nemo_gym_agentic/scripts/run_agent_app.sh`;
- the Calendar recipe as the simple-agent reference;
- Workplace Assistant for stateful tool/resource cleanup;
- R2E-Gym for sandbox, artifact, and multi-process callback propagation.

Do not infer graph names from directory names. Record the exact contract before implementing:

```text
config path:
agent name and type:
resource server name:
dataset repo/artifact/split:
raw verifier fields:
stateful cleanup requirement:
tool-call parser:
reasoning parser:
maximum agent steps:
```

## Required recipe files

Create the following under `examples/nemo_gym_agentic/recipes/<recipe>/`:

```text
env.sh
prepare_<recipe>.sh
start_<recipe>_gym.sh
run-<model>-nemo-gym-<recipe>.sh
verify_<recipe>.py
README.md
PITFAIL.md
```

Use the exact graph names from the pinned Gym config. If the agent/resource needs a dedicated venv, patch, or runtime asset, wire it into the shared Dockerfile and build it into the image.

## Step 1: prepare data

The host entry script must source its sibling `env.sh` and require an absolute `NEMO_GYM_SOURCE_DATA` ending in the expected raw filename.

Implementation requirements:

1. Download or materialize the selected split inside the pinned Gym image.
2. Write the raw JSONL directly to a filesystem visible at the same absolute path from every remote Ray node.
3. Do not assume that a path visible in the development container is bind-mountable by an outer Docker daemon. Probe the mount from a temporary container when Docker-in-Docker is possible.
4. Preserve all verifier and agent fields. In particular, keep `responses_create_params`, tools, ground truth, expected answers, environment-specific metadata, repository identity, and unknown passthrough fields.
5. Validate that every non-empty line is a JSON object and that raw/conversion checks retain the expected row count.
6. Do not hardcode a dataset row count or tool count from a README; validate the downloaded artifact.
7. Use the correct split. A held-out benchmark/test split is smoke/evaluation data, not an RL training set.

Conversion has one source of truth:

```text
shared raw Gym JSONL
    -> Ray job calls convert_dataset.py once
    -> ${EXP_DIR}/data/<recipe>_train.jsonl
```

An optional converted file produced during preparation is only a schema check. Document that clearly and never use it as `NEMO_GYM_SOURCE_DATA`.

Before continuing, run the prepare script and assert the exact raw path is non-empty.

## Step 2: start the local Gym service

Prefer a local Docker service with a private Ray cluster. The remote Relax cluster communicates with it only over HTTP.

Keep the two Ray systems separate:

```text
Gym private Ray: owned by the local container
Relax Ray:       RAY_ADDRESS on the remote training cluster
```

The Gym launcher must unset inherited `RAY_ADDRESS` and `RAY_JOB_SUBMISSION_ID` before starting its private Ray.

Networking requirements:

- `GYM_HOST` must be assigned to the Docker daemon host and reachable from every Relax worker.
- The Gym host must reach the Relax head's Agentic Chat API on port 8000.
- The callback base URL must include `/agentic_api`.
- The callback allowlist contains the bare host/IP used by the Relax callback URL. A CIDR is an allowlist, not a Ray address.
- Put Gym and Relax internal addresses in `NO_PROXY`; do not rely on an HTTP proxy to bridge internal traffic.
- Choose a unique service port block and a non-overlapping private Ray port block. Avoid the host ephemeral port range.
- The service port base and the training `NEMO_GYM_GATEWAY_PORT` must resolve to the same Gateway port.

The launcher must build `NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON` with the exact environment, config, agent URL, readiness URLs, interrupt policy, concurrency, queue, and deadline.

Stateful environments require a real cleanup contract keyed by opaque rollout ID. Add cleanup and probe endpoints if the resource server retains sessions, databases, sandboxes, or containers. Do not report cancellation as clean merely because the local HTTP task was cancelled.

Container lifecycle:

1. build a fresh image containing the current checkout;
2. verify the existing named container has the expected recipe label;
3. replace only that container;
4. wait with a bounded readiness loop and show logs on timeout;
5. verify `/readyz` reports the pinned Gym commit, healthy janitor, and zero active trials.

Run `verify_<recipe>.py` against the real resource server. It must prove that a known-correct response receives the success reward and a known-wrong response does not.

For multi-turn or tool environments, also run a deterministic full trial and verify callback history, tool-result feedback, final reward, artifacts, and cleanup. A direct resource verifier alone does not validate callback routing.

## Step 3: write the direct training script

Follow the Calendar/R2E layout:

1. resolve `SCRIPT_DIR`, `EXAMPLE_DIR`, `RELAX_ROOT`, `run_training.sh`, and `run_agent_app.sh`;
2. source the Relax entrypoint and model config;
3. validate model and raw data paths;
4. prepend the current `RELAX_ROOT` to `RUNTIME_ENV_JSON.env_vars.PYTHONPATH` and set `py_executable=/usr/bin/python3`;
5. derive the Ray Jobs dashboard from `RAY_ADDRESS`, unless `RAY_DASHBOARD_ADDRESS` is explicit;
6. generate a unique submission ID;
7. define complete checkpoint, rollout, GRPO, optimizer, performance, SGLang, tracking, and misc argument arrays in this script;
8. submit `run_training.sh` with the Gateway URL, raw source data, converted prompt output, and actual data count.

Hard-code recipe training choices in the script, for example:

```bash
--num-rollout 3
--rollout-batch-size 1
--n-samples-per-prompt 8
--global-batch-size 8
--rollout-max-prompt-len 6144
--rollout-max-response-len 2048
--rollout-max-context-len 8192
--agentic-reasoning-parser qwen3
```

Change these values by editing the recipe script, not by adding environment-variable wrappers.

Set `--agent-env` with the exact Gateway URL, environment, config, model alias, interrupt policy, deadline, and lease. Preserve the opaque rollout prefix across every subprocess, Ray boundary, sandbox, and model callback. Model callbacks must use:

```text
/ng-rollout/<opaque-rollout-id>/v1/responses
```

or the corresponding `/v1/chat/completions` route.

Select parsers from actual model output:

- Qwen reasoning output needs `--agentic-reasoning-parser qwen3` when the verifier rejects inline `<think>`.
- Tool-call parsers must match the model/chat-template format and only matter when tools are present.

For GRPO, one sample only validates plumbing. Use at least four samples per prompt for a meaningful group advantage and confirm reward variance. Keep prompt, response, context, and per-GPU token budgets compatible.

Do not use Ray `WORKING_DIR=./` when every cluster node already sees the same shared checkout. Resolve agent commands and working directories to identical absolute shared paths.

## Validation ladder

Do not skip levels or substitute one level for another.

1. **Static:** `bash -n`, focused unit tests, converter preservation tests, and Docker patch apply checks.
2. **Data:** raw path exists, rows parse, required verifier fields survive, and train does not accidentally use a test split.
3. **Service:** fresh image/container, `/readyz`, expected commit, zero initial trials, and deterministic verifier pass/fail behavior.
4. **Network:** every remote Ray node can reach the Gateway; the Gym container can reach the active Relax `/agentic_api` route.
5. **Trial:** one real trial completes with expected turns/tools/artifacts/reward and returns `active_trials` to zero.
6. **Training:** multi-sample rollout produces reward variance, non-zero advantage where expected, an optimizer step, no Actor OOM/traceback, and a checkpoint.

Ray Job `SUCCEEDED` is not sufficient. Inspect rollout JSONL, Actor logs, optimizer metrics, checkpoint output, Gateway trial counts, and environment-specific sandbox cleanup.

## Failure triage

| Symptom | Check first |
|---|---|
| Gateway 404/410 callback | Opaque rollout prefix, correct `/agentic_api`, current session state, old image/container |
| Agent `/run` 500 | `docker logs` traceback and inner response body; do not stop at the access log |
| `cleanup_unverified` | Missing abort/force-cleanup/probe contract or a still-running remote sandbox |
| Reward always zero | Run clean verifier, inspect exact model final answer, reasoning/tool parsing, expected metadata, evaluator execution |
| Correct answer still zero | Inline `<think>`, answer format, parser mismatch, lost metadata, wrong base state/commit |
| `policy_model finished unexpectedly` | Stale container/process, occupied port, Ray component/worker-port overlap |
| No container/artifact found | Docker mount namespace, filename/instance mapping, manifest, prefix, shared path |
| Job succeeded but no learning | One-sample GRPO, zero reward variance, Actor OOM, missing optimizer step |
| Trials appear after fresh Gateway | A stale Relax client is still submitting; stop the requester before restarting Gateway |

Reward zero is not automatically an integration failure. Separate these cases:

- verifier never ran or could not parse the response;
- environment/evaluator failed;
- callback/lifecycle failed;
- the model completed the environment but produced a genuinely wrong answer.

## Definition of done

The recipe is complete only when all of the following are true:

- The seven required recipe files exist and use the pinned graph names.
- One data command creates the exact shared raw source path.
- One service command starts a fresh labeled container and reaches ready.
- The verifier distinguishes correct and incorrect responses.
- One direct training script contains the actual hyperparameters and waits by default.
- Remote nodes reach Gym and Gym reaches the active Relax callback route.
- A real multi-sample training run reaches an optimizer step and writes a checkpoint.
- Actor logs contain no hidden OOM/traceback.
- Gateway active trials and environment resources return to a clean state.
- New environment-specific failures are recorded concisely in that recipe's `PITFAIL.md`.

When handing off, show only the three commands the user needs, the resulting data/checkpoint paths, and any remaining unverified item.
