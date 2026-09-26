# Qwen3-4B 弹性 Rollout recipe

本 recipe 使用 4 张 Actor GPU 和 4 个各占 1 张 GPU 的初始 SGLang Rollout
引擎启动 Qwen3-4B。随附的 Autoscaler 配置在训练持续运行时，将 Rollout
引擎数控制在 4 到 8 个之间。

## 弹性必需配置

| 配置 | 本 recipe | 作用 |
| --- | --- | --- |
| 执行模式 | `--fully-async` | Actor 与 Rollout 使用独立资源池，才能单独扩缩 Rollout Engine。 |
| Autoscaler | `--autoscaler-config <yaml>`，且 YAML 中 `enabled: true` | 启用自动扩缩容决策。 |
| 初始下限 | `--resource` 中 `"rollout": [1, 4]`，同时设置 `min_engines: 4` | 启动并保护 4 个初始 Rollout Engine。 |
| 扩缩容单位 | `--rollout-num-gpus-per-engine 1` | 每增加 1 个 Engine，申请 1 GPU 和 1 CPU。 |

启动脚本已经传入这些 flag；`AUTOSCALER_CONFIG` 用于选择 YAML 文件。固定
基线需要 8 张 GPU：Actor 4 张，4 个初始 Rollout Engine 各 1 张。

Relax 只向 Ray 提交资源需求，不创建节点或 Pod。集群资源提供方需要为每个
新增 Engine 提供 1 GPU 和 1 CPU，并将其加入同一个 Ray 集群。

## 启动

准备以下路径：

```text
<MODEL_DIR>/Qwen3-4B/
<DATA_DIR>/dapo-math-17k/dapo-math-17k.jsonl
<DATA_DIR>/aime-2024/aime-2024.jsonl
```

然后在仓库根目录启动：

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

## 确认结果

1. 训练持续推进，初始 4 个 Rollout 引擎在 Relax 中为 `ACTIVE`。
2. 负载升高后，外部资源 ready、Ray Placement Group ready，Relax 中
   `ACTIVE` 引擎数超过 4。
3. 负载下降后，Autoscaler 将引擎数缩回 4，训练仍继续推进。

发生外部抢占时，平台需要把 `SIGTERM` 传递到弹性 `SGLangEngine` Ray Actor，
并提供足够的排空时间；Relax 不安装 Pod `preStop` hook。

## 高级配置

Autoscaler 策略、扩缩容 API、生命周期状态和故障排查请参阅
[弹性 Rollout 扩缩容](../guide/elastic-rollout.md)。
