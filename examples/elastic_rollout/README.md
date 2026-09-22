# Elastic Rollout Recipe

[English](#english) | [中文](#中文)

<a id="english"></a>

## English

This example runs Qwen3-4B with four Actor GPUs and four initial one-GPU
SGLang Rollout engines. The included Autoscaler config scales the Rollout pool
between 4 and 8 engines.

Required elastic settings:

| Setting        | This recipe                                              | Purpose                                                                                        |
| -------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Execution mode | `--fully-async`                                          | Keeps Actor and Rollout in separate resource pools so Rollout engines can scale independently. |
| Autoscaler     | `--autoscaler-config <yaml>` with `enabled: true`        | Enables automatic scaling decisions.                                                           |
| Initial floor  | `"rollout": [1, 4]` in `--resource` and `min_engines: 4` | Starts and protects four initial Rollout engines.                                              |
| Scaling unit   | `--rollout-num-gpus-per-engine 1`                        | Each added engine requests 1 GPU and 1 CPU.                                                    |

The run script already passes these flags; `AUTOSCALER_CONFIG` selects the
YAML file. The baseline needs 8 GPUs, while added capacity must be supplied by
an external cluster provisioner.

Prepare these paths before launching:

```text
<MODEL_DIR>/Qwen3-4B/
<DATA_DIR>/dapo-math-17k/dapo-math-17k.jsonl
<DATA_DIR>/aime-2024/aime-2024.jsonl
```

Launch from the repository root:

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

A successful run trains continuously, adds an external node or Pod, reaches
Ray placement-group readiness, scales above four Relax `ACTIVE` engines, and
later returns to four without stopping.

Relax does not create nodes or install `preStop`; the platform must provide
capacity and deliver graceful `SIGTERM` eviction with enough time to drain.
See the [full recipe walkthrough](../../docs/en/examples/elastic-rollout.md) or
the [advanced configuration and troubleshooting guide](../../docs/en/guide/elastic-rollout.md).

______________________________________________________________________

<a id="中文"></a>

## 中文

此示例使用 4 张 Actor GPU 和 4 个各占 1 张 GPU 的初始 SGLang Rollout
引擎运行 Qwen3-4B。随附的 Autoscaler 配置在 4 到 8 个引擎之间扩缩容。

弹性必需配置：

| 配置       | 本 recipe                                                      | 作用                                                           |
| ---------- | -------------------------------------------------------------- | -------------------------------------------------------------- |
| 执行模式   | `--fully-async`                                                | Actor 与 Rollout 使用独立资源池，才能单独扩缩 Rollout Engine。 |
| Autoscaler | `--autoscaler-config <yaml>`，且 YAML 中 `enabled: true`       | 启用自动扩缩容决策。                                           |
| 初始下限   | `--resource` 中 `"rollout": [1, 4]`，同时设置 `min_engines: 4` | 启动并保护 4 个初始 Rollout Engine。                           |
| 扩缩容单位 | `--rollout-num-gpus-per-engine 1`                              | 每增加 1 个 Engine，申请 1 GPU 和 1 CPU。                      |

启动脚本已经传入这些 flag；`AUTOSCALER_CONFIG` 用于选择 YAML 文件。固定
基线需要 8 张 GPU，扩容资源需要由外部集群资源提供方补充。

启动前准备以下路径：

```text
<MODEL_DIR>/Qwen3-4B/
<DATA_DIR>/dapo-math-17k/dapo-math-17k.jsonl
<DATA_DIR>/aime-2024/aime-2024.jsonl
```

在仓库根目录启动：

```bash
RAY_ADDRESS=ray-head.example.com:6379 \
RAY_DASHBOARD_ADDRESS=http://ray-head.example.com:8265 \
MODEL_DIR=/path/to/models DATA_DIR=/path/to/data EXP_DIR=/path/to/output \
AUTOSCALER_CONFIG=examples/elastic_rollout/autoscaler.yaml \
WORKING_DIR=./ \
bash scripts/entrypoint/ray-job.sh \
  examples/elastic_rollout/run-qwen3-4B-8xgpu-elastic-async.sh
```

标记资源的前缀由 `RELAX_INITIAL_NODE_GROUP` 设置，默认为 `stable`，对应
`stable_gpu` 和 `stable_cpu`。例如设置 `RELAX_INITIAL_NODE_GROUP=baseline` 时，
集群需要提供 `baseline_gpu` 和 `baseline_cpu`。该 affinity 将固定基线 pin 到
指定节点组；scale-out 新增的 Rollout Engine 不带节点组 affinity，可使用任意
满足资源需求的容量，包括可抢占资源。Relax 本身不会让固定节点组变成不可抢占，
其稳定性和不可抢占性需要由 Ray、Kubernetes 或资源平台保障。普通 Ray 集群
没有标记资源时，设置 `ENABLE_AFFINITY=0`。

成功时训练持续推进，外部节点或 Pod ready，Ray Placement Group ready，Relax
`ACTIVE` 引擎数扩到 4 以上，随后缩回 4 且任务不中断。

Relax 不创建节点，也不安装 `preStop`；外围平台需要提供资源，并用有足够排空
时间的 `SIGTERM` 实现优雅抢占。继续阅读[完整 recipe 说明](../../docs/zh/examples/elastic-rollout.md)
或[高级配置与故障排查](../../docs/zh/guide/elastic-rollout.md)。
