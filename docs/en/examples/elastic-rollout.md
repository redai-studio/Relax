# Qwen3-4B Elastic Rollout Recipe

This recipe starts Qwen3-4B with four Actor GPUs and four initial one-GPU
SGLang Rollout engines. The included Autoscaler config scales the Rollout pool
between 4 and 8 engines while training continues.

## Required Elastic Settings

| Setting | This recipe | Purpose |
| --- | --- | --- |
| Execution mode | `--fully-async` | Keeps Actor and Rollout in separate resource pools so Rollout engines can scale independently. |
| Autoscaler | `--autoscaler-config <yaml>` with `enabled: true` | Enables automatic scaling decisions. |
| Initial floor | `"rollout": [1, 4]` in `--resource` and `min_engines: 4` | Starts and protects four initial Rollout engines. |
| Scaling unit | `--rollout-num-gpus-per-engine 1` | Each added engine requests 1 GPU and 1 CPU. |

The run script already passes these flags; `AUTOSCALER_CONFIG` selects the
YAML file. The baseline needs 8 GPUs: 4 for Actor and 4 for the initial Rollout
engines.

Relax submits Ray resource demand but does not create nodes or Pods. The
cluster provisioner must add each engine's additional 1 GPU and 1 CPU to the
same Ray cluster.

## Launch

Prepare these paths:

```text
<MODEL_DIR>/Qwen3-4B/
<DATA_DIR>/dapo-math-17k/dapo-math-17k.jsonl
<DATA_DIR>/aime-2024/aime-2024.jsonl
```

Then launch from the repository root:

```bash
RAY_ADDRESS=ray-head.example.com:6379 \
RAY_DASHBOARD_ADDRESS=http://ray-head.example.com:8265 \
MODEL_DIR=/path/to/models DATA_DIR=/path/to/data EXP_DIR=/path/to/output \
AUTOSCALER_CONFIG=examples/elastic_rollout/autoscaler.yaml \
WORKING_DIR=./ \
bash scripts/entrypoint/ray-job.sh \
  examples/elastic_rollout/run-qwen3-4B-8xgpu-elastic-async.sh
```

The marker prefix comes from `RELAX_INITIAL_NODE_GROUP` and defaults to
`stable`, which requires `stable_gpu` and `stable_cpu`. For example,
`RELAX_INITIAL_NODE_GROUP=baseline` requires `baseline_gpu` and `baseline_cpu`.
This affinity pins the fixed baseline to that node group. Rollout engines added
by scale-out have no node-group affinity and can use any eligible capacity,
including preemptible capacity. Relax does not make the pinned group
non-preemptible; the Ray, Kubernetes, or resource platform must provide its
stability. On a plain Ray cluster without marker resources, set
`ENABLE_AFFINITY=0`.

## Confirm the Result

1. Training advances with four initial Relax `ACTIVE` Rollout engines.
2. Under load, external capacity becomes ready, the Ray placement group becomes
   ready, and Relax reports more than four `ACTIVE` engines.
3. After load falls, the Autoscaler returns to four engines and training keeps
   advancing.

For external eviction, the platform must deliver `SIGTERM` to the elastic
`SGLangEngine` Ray actor and allow enough time to drain; Relax does not install
a Pod `preStop` hook.

## Advanced Configuration

For Autoscaler policies, scaling APIs, lifecycle states, and troubleshooting,
see [Elastic Rollout Scaling](../guide/elastic-rollout.md).
