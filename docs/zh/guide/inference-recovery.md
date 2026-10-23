# 推理服务局部自愈

**故障自愈会丢弃已经读取但尚未训练的样本，以及生成到一半的 Rollout 样本。恢复后的数据批次与策略版本可能变化，因此该功能应按 off-policy 场景使用，不保证严格 on-policy。对 on-policy 敏感的任务请不要开启，并显式设置 `RELAX_ENABLE_INFERENCE_RECOVERY=false`（或 `0`）。**

Relax 支持在 Rollout 或独立 ActorFwd 运行时抛出异常后，仅重建出错的推理服务并继续训练。恢复期间，框架暂停新的数据生产和消费，等待旧操作结束，再将原训练 Actor 的当前权重同步到新实例。局部恢复成功时，训练 Actor、optimizer 和 scheduler 保持不变，Controller 不执行全局重启。

这项能力针对已经返回的明确异常，例如 HTTP 请求重试耗尽后抛出的错误。它不新增卡死检测，也不保证所有推理错误都能局部恢复；恢复不安全、失败或预算用尽时，会转回原全局重启流程。

## 配置

以下是环境变量，不是命令行参数。需要在启动训练入口前，设置到 **Controller 所在的训练 Job** 中；只在 SGLang 子进程中设置不会生效。

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `RELAX_ENABLE_INFERENCE_RECOVERY` | 按训练模式决定 | `true` 或 `1` 开启，`false` 或 `0` 关闭，同时控制 Rollout 和独立 ActorFwd 的局部恢复。 |
| `RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS` | `3` | 非负整数。Rollout 和 ActorFwd 共用的累计局部恢复尝试次数；`0` 表示第一次故障就转全局重启。负数在启用恢复时报错，非整数由启动时的环境校验拒绝。 |
| `RELAX_INFERENCE_RECOVERY_DRAIN_TIMEOUT` | `180` | 正数，单位秒。等待旧工作退出的时间上限，也用于局部恢复与全局重启的交接；不是整个恢复过程的总超时。 |

| 训练模式 | 未设置开关时 | 可局部恢复的角色 |
|---|---|---|
| full-async（`--fully-async`） | 配置满足条件时开启，否则保留原行为 | Rollout；实际部署的独立 ActorFwd |
| hybrid-async（`--hybrid`） | 配置满足条件时开启，否则保留原行为 | Rollout。forward 在训练 Actor 内执行，没有独立 ActorFwd 服务。 |
| colocate（不含 hybrid） | 关闭，需要显式设为 `true` 或 `1` | Rollout |

开关值忽略大小写和首尾空白，例如 `TRUE` 与 `true` 等效。显式设为 `true` 或 `1` 时，如果配置不受支持，启动会报 `Unsupported inference recovery configuration`，列出冲突字段。未设置开关时，不支持的配置会保留原行为。设为 `false` 或 `0` 时关闭本功能，跳过局部恢复的配置检查。

这三个变量遵循统一的环境配置规则：空值等同于未设置，使用默认值；开关还接受 `on`/`off`、`yes`/`no`、`t`/`f`、`y`/`n`。非法布尔值或无法转换为声明类型的数值会在启动校验时报错，即使局部恢复已关闭。

### 启用与关闭

在自己的训练启动脚本中、执行 `python -m relax.entrypoints.train` 之前，可以这样设置：

```bash
export RELAX_ENABLE_INFERENCE_RECOVERY=true
export RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS=3
export RELAX_INFERENCE_RECOVERY_DRAIN_TIMEOUT=180
```

需要关闭时，将 `RELAX_ENABLE_INFERENCE_RECOVERY` 设置为 `false`。这些设置在训练启动时读取；修改运行中进程外部的环境变量不会改变当前任务的行为。

使用 Ray Jobs 时，需要把变量传给远端训练入口。若启动脚本已经使用 `RUNTIME_ENV_JSON`，在提交前将配置合并到现有 `env_vars` 中，保留其他环境配置：

```bash
RUNTIME_ENV_JSON="$(python - <<'PY'
import json
import os

runtime_env = json.loads(os.environ["RUNTIME_ENV_JSON"])
runtime_env.setdefault("env_vars", {}).update({
    "RELAX_ENABLE_INFERENCE_RECOVERY": "true",
    "RELAX_INFERENCE_RECOVERY_MAX_ATTEMPTS": "3",
    "RELAX_INFERENCE_RECOVERY_DRAIN_TIMEOUT": "180",
})
print(json.dumps(runtime_env))
PY
)"
export RUNTIME_ENV_JSON
```

保持原提交命令使用 `ray job submit --runtime-env-json="${RUNTIME_ENV_JSON}"`。`scripts/entrypoint/ray-job.sh` 会生成基础的 `RUNTIME_ENV_JSON`，因此应在它生成之后、实际提交 Job 之前合并；仅在提交端 shell 中 `export`，不能保证变量进入远端 Job。

## 支持条件与配置冲突

当前支持框架自行管理的静态资源、Megatron 训练后端、标准 SGLang Rollout 和 GRPO。训练 Actor、数据队列与恢复状态服务需要保持可用；局部恢复不负责修复训练 Actor 本身、节点丢失或 GPU 故障，也不负责在故障后迁移或减少资源。

| 条件 | 必须满足的配置 |
|---|---|
| 训练与算法 | `train_backend=megatron`、`advantage_estimator=grpo`、`kl_coef=0`；不启用 `--use-kl-loss`、Critic 或 OPD。 |
| 生成入口 | 使用默认 `relax.engine.rollout.sglang_rollout.generate_rollout`；不设置 `--custom-generate-function-path`、`--dynamic-sampling-filter-path`，不启用 Agentic Rollout。 |
| 模型与奖励 | 不设置 `--multimodal-keys`；不启用 `--lora-adapter-mode`，不配置 GenRM 实例。 |
| 资源和路由 | 不使用 `--rollout-external`、`--autoscaler-config`、`--sglang-config`、`--sglang-router-ip` 或 `--use-slime-router`；保留框架管理的引擎和路由。 |
| 其他执行选项 | 不启用 `--use-fault-tolerance`、`--use-dynamic-global-batch-size`、`--debug-train-only` 或 `--debug-rollout-only`。`--use-fault-tolerance` 是另一条 Rollout 容错路径，不是本功能的开关。 |
| colocate | 必须启用训练卸载、Rollout 卸载及权重备份：`offload_train=True`、`offload_rollout=True`、`enable_weights_backuper=True`。`--colocate` 的参数校验会自动启用卸载，权重备份应保持开启。 |
| hybrid | 不允许 `offload_train=True` 或 `offload_rollout=True`；不要与 `--offload`、`--offload-train`、`--offload-rollout` 搭配。 |
| 路由重放 | full-async 和 hybrid 的局部恢复均不支持 `--use-routing-replay`；hybrid 参数校验会自动设置 `fully_async=True`，同样受此限制。 |

表中的字段是最终解析后的配置。例如启动报错显示 `enable_weights_backuper=False`，应检查启动脚本中的 `--disable-weights-backuper`。完整训练模式配置见 [full-async](./fully-async-training.md) 和 [hybrid](./hybrid-training.md)。

`--partial-rollout` 本身不在冲突项中，但恢复时同样可能丢弃半成品和待处理样本，不保证续接故障前生成到一半的内容。

## 恢复流程

1. Controller 收到 Rollout 或独立 ActorFwd 的运行异常后，检查剩余恢复次数，并暂停新工作。
2. 等待在途训练和旧推理操作结束，避免旧实例继续写数据，或在权重同步尚未结束时拆除服务。
3. 以原训练 Actor 的下一训练位置为准，清理尚未训练的数据，丢弃待生成样本；输入读取位置不回退。
4. 确认旧推理子实例退出，在原资源组上重建出错的推理服务。健康的训练 Actor 和其他服务保留。
5. 将原训练 Actor 当前权重同步给替代实例，等待权重同步以及所需显存恢复完成，再恢复数据生产和训练。

预算按尝试次数累计，成功恢复不会清零。例如预算为 `3` 时，Rollout 恢复一次、ActorFwd 恢复一次、Rollout 再恢复一次，就已经用完；下一次异常转全局重启。某次局部恢复本身失败也会直接升级，不会为了用满剩余预算而重复重建。全局重启重新创建恢复状态，局部预算重新计数。

::: warning 数据与训练策略
恢复允许丢弃生成一半的样本，以及已经生成完成但尚未训练的待处理批次；不承诺数据零丢失。恢复后的数据顺序、批次和策略版本可能变化，不保证严格 on-policy 或与无故障运行逐步一致。这也是纯 colocate 默认不开启的原因；对这一点敏感的任务应使用 `RELAX_ENABLE_INFERENCE_RECOVERY=false`。
:::

## 与健康检查、全局重启的关系

`--use-health-check` 控制原有的健康检查，不是局部恢复的启用条件。建议保留它作为兜底；本功能不增加对静默卡死的检测。

局部恢复与健康检查可能看到同一次故障。只有对应角色正在由局部恢复线程处理时，健康回调才会暂缓重复重启；否则仍走原重启路径。健康检查先接管故障时，也可能直接进入全局重启，不能理解为“开启后所有推理故障都必定局部恢复”。

以下情况会转回原全局重启流程：预算耗尽、局部重建或权重恢复失败、多个角色同时异常、权重同步期间发生故障等。如果旧工作无法安全退出，会终止任务，避免新旧实例同时操作资源。独立服务进程完全失联也不保证能完成局部停止和重建。

`--max-global-restart` 仍单独限制全局重启次数，默认 `3`，超过后任务结束；它与局部预算不是同一个计数。局部升级和健康检查触发的全局重启都受此限制。全局重启会重建训练状态，是否从已有 checkpoint 恢复取决于原有的保存、加载配置；局部恢复不额外保存 checkpoint，也不保证全局重启保住最近的训练进度。

排查时可查找日志 `Local recovery completed`，其中包含出错角色、累计尝试次数、替代实例代次、下一训练位置和耗时。预算耗尽会出现 `Local recovery budget (...) exhausted`；随后是否完成全局重启及继续训练，需要继续查看训练日志，不能只依据服务状态判断。健康检查的更多说明见 [健康检查管理器](./health-check-manager.md)。
