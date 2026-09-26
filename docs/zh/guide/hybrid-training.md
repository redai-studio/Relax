# Hybrid 混合训练模式

## 概述

**Hybrid 模式** 是 Relax 在 [Colocate（同步）](./architecture.md) 与 [Fully Async（全异步）](./fully-async-training.md) 之间的第三种执行模式。它将以下两者结合：

- Fully Async 的 **流式数据流水线**（TransferQueue + `max-staleness` 控制 off-policy 容忍度），以及
- Colocate 的 **进程内权重共享**（TensorBackuper + `_switch_model`，使 ref / actor_fwd / advantages 全部在 actor 自身 GPU 上完成）。

具体来说，Actor 与 Rollout 仍然部署在 **独立的 GPU placement group** 上（与 Fully Async 一致），但 actor 不再把权重广播到独立的 ActorFwd / Reference / Advantages 服务，而是通过 CPU/GPU `TensorBackuper` 在 `actor`、`ref`、`old_actor`、`teacher` 等 tag 之间切换同一套权重，所有 forward 在本地完成，最后通过同步的 `UpdateWeightFromTensor` 路径把权重推给 rollout。

### 模式对比

| 维度                | Colocate（同步）                          | Fully Async（全异步）                                        | Hybrid                                                                                |
| ------------------- | ----------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| **GPU 布局**        | Actor 与 Rollout 分时复用同一组 GPU       | Actor / Rollout / ActorFwd / Reference 各自独立 GPU          | Actor 与 Rollout 独立 GPU；ref / actor_fwd / adv 复用 actor 的 GPU                    |
| **数据流水线**      | TransferQueue，批同步                     | TransferQueue + StreamingDataLoader，完全流式                | TransferQueue + 子批次流式（`num-iters-per-train-update`）                            |
| **权重同步**        | 进程内 tensor 拷贝                        | 通过 DCS（Checkpoint Engine）做 NCCL broadcast               | 同步的 `UpdateWeightFromTensor` 推给 rollout；ref/actor_fwd 走 TensorBackuper          |
| **Staleness**       | `max_staleness = 0`（严格 on-policy）     | 可配置 `max_staleness`                                       | 可配置 `max_staleness`                                                                 |
| **部署的角色**      | `actor`, `critic`, `rollout`              | `actor`, `critic`, `rollout`, `advantages`, `reference`, `actor_fwd` | `actor`, `critic`, `rollout`（与 Colocate 相同；ref/actor_fwd 在 actor 内部）         |
| **DP token 均衡**   | 静态/序列长度路径通过 `--balance-data` 支持 | 动态 batch 自动做 DP token 均衡；`--balance-data` 可传入但无额外效果 | 动态 batch 自动做 DP token 均衡；静态/序列长度路径可使用 `--balance-data`              |

### 何时选择 Hybrid

选择 **Hybrid** 的场景：

- 希望获得独立 rollout GPU 与流水线数据带来的吞吐收益，但
- 模型较大，单独部署 ref / actor_fwd 服务会浪费 GPU，或者
- 希望 ref / actor_fwd 在 actor 本地完成，同时保留会自动做 DP token 均衡的流式动态 batch 路径。

选择 **Fully Async**：拥有充足的 GPU 单独运行 ref / actor_fwd / advantages 服务，并希望在 step 之间做真正的并行流水线。

选择 **Colocate**：GPU 紧张，可以接受 rollout → train 的串行执行。

______________________________________________________________________

## 架构

### 角色布局

Hybrid 与 Colocate 使用相同的角色集合 —— 只部署 `actor`、`critic`（可选）、`rollout` 三个 Ray Serve 服务。判断逻辑位于 `relax/core/registry.py`：

```python
def process_role(config):
    if config.hybrid:
        # hybrid mode: actor handles ref/actor_fwd internally
        # via _switch_model, only need actor + rollout services
        return ROLES_COLOCATE
    if config.fully_async:
        ...
```

但与 Colocate 不同，actor 与 rollout 的 placement group 是 **互相独立** 的，这与 Fully Async 一致。参见 `relax/core/controller.py`：

```python
if colocate and not self.config.hybrid:
    # Sync colocate: actor and rollout share GPUs via time-sharing (offload/onload)
    actor_rollout_pgs = create_placement_group(num_gpus=num_gpus)
else:
    # fully_async (pure or hybrid): actor and rollout use separate GPUs
    actor_rollout_pgs = None
```

### 参数解析

`--hybrid` 是唯一对外暴露的开关。`relax/utils/arguments.py` 将其展开为下游已识别的两个底层参数：

```python
if args.hybrid:
    args.fully_async = True
    args.colocate = True
```

直接传入 `--fully-async --colocate` 会被拒绝，必须使用 `--hybrid`。单一开关的设计使 `args.hybrid` 成为 registry、controller 分发以及 `train_hybrid` 调用点中识别 hybrid 模式的唯一权威标志。

### 架构图

ASCII 图中保留英文以避免框线对齐错乱：

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Controller (Orchestrator)                           │
│                     relax/core/controller.py                               │
│                                                                            │
│       ┌───────────────────────────────────┐    ┌────────────────────┐      │
│       │            Actor Service          │    │  Rollout Service   │      │
│       │  (own placement group, N GPUs)    │    │ (own PG, M GPUs)   │      │
│       │                                   │    │   SGLang engines   │      │
│       │  ┌────────────────────────────┐   │    └─────────┬──────────┘      │
│       │  │ TensorBackuper             │   │              ▲                 │
│       │  │  tags: actor / ref /       │   │              │                 │
│       │  │        old_actor / teacher │   │              │                 │
│       │  │  _switch_model(tag) swaps  │   │              │                 │
│       │  │  weights on the same GPUs  │   │              │                 │
│       │  └────────────────────────────┘   │              │                 │
│       │  train_hybrid():                  │              │                 │
│       │    ├─ ref forward   (switch:ref)  │              │                 │
│       │    ├─ actor forward (switch:actor)│                                │
│       │    ├─ advantages    (in-process)  │              │                 │
│       │    └─ train         (switch:actor)│                                │
│       └──────────────┬────────────────────┘              │                 │
│                      │ UpdateWeightFromTensor (sync) ───┘                  │
└──────────────────────┼─────────────────────────────────────────────────────┘
                       ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      TransferQueue (Data Plane)                           │
│  Rollout writes train_N partition incrementally ──► Actor consumes        │
│  in sub-batches via get_meta(batch_size, batch_index) with max-staleness  │
└───────────────────────────────────────────────────────────────────────────┘
```

______________________________________________________________________

## `train_hybrid` 训练流程

`relax/backends/megatron/actor.py:708` 实现的 Hybrid 训练步骤分为三个阶段：

1. **采集子批次并完成小批次 forward 计算（峰值显存小）**

   全局 batch 被切分为 `num_iters_per_train_update` 份子批次。对每个子批次，actor 会：

   - 从 TransferQueue 拉取数据（`_get_data_from_transfer_queue("train", rollout_id, fields, batch_size, batch_index)`）
   - 若已备份 ref 权重，执行 `_switch_model("ref")` 并计算 ref log-probs
   - 若已备份 teacher 权重（OPD 场景），执行 `_switch_model("teacher")` 并计算 teacher log-probs
   - 执行 `_switch_model("old_actor" 或 "actor")` 并计算当前 actor 的 log-probs
   - 把扩充后的子批次追加到内存列表

2. **合并子批次并做全局 Advantages 归一化**

   所有子批次 dict 被拼接成一个 `rollout_data`，随后 `compute_advantages_and_returns(self.args, rollout_data)` 在合并后的整批数据上执行一次。这是两阶段设计的 **核心正确性要求** —— Advantages 归一化必须看到完整的 DP-group 批次，而不是各个子批次切片。

3. **在合并批次上训练并推送权重**

   一次 `train(...)` 调用基于合并后的 batch 完成优化器步进。随后 actor 把新权重备份到 `actor` tag（如果到达 ref 更新间隔，也刷新 `ref` tag），然后调用 `self.update_weights()` 通过 `UpdateWeightFromTensor` 把最新权重同步给 rollout。

子批次 forward 控制了激活峰值显存（与 Fully Async 行为一致），而合并后的训练步则保留了 Colocate 风格的全局统计量。

______________________________________________________________________

## 配置

### 必需参数

| 参数                            | 用途                                                                                            |
| ------------------------------- | ----------------------------------------------------------------------------------------------- |
| `--hybrid`                      | 启用 Hybrid 模式（内部展开为 `fully_async=True, colocate=True`）                                |
| `--resource '{...}'`            | 分别声明 `actor` 与 `rollout` 的 placement group，例如 `{"actor":[1,4],"rollout":[1,4]}`        |
| `--num-iters-per-train-update`  | 每个全局 batch 切分的子批次数量（越大 → 峰值显存越小，TransferQueue 轮询次数越多）              |
| `--max-staleness`               | Off-policy 容忍度（0 = 严格 on-policy，>0 允许一定程度滞后）                                    |

### 常用可选参数

| 参数                          | 说明                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| `--balance-data`              | Hybrid 模式下可传入。启用 `--use-dynamic-batch-size` 时，DP token 均衡由流式 sampler 自动完成；在静态/序列长度路径上，该参数会启用 DP token 均衡。 |
| `--num-data-storage-units`    | TransferQueue 存储 actor 的数量。                                                                       |
| `--use-streaming-dataset`     | 从磁盘流式读取 prompts，而不是全量载入内存。                                                            |
| `--ref-update-interval`       | 周期性地用最新 actor 权重刷新缓存的 ref 权重。                                                          |

### 默认值覆盖

启用 `--hybrid` 后，`relax/utils/arguments.py` 会按以下方式设置默认值（除非用户显式传入）：

- `offload_train = False` 且 `offload_rollout = False` —— actor 与 rollout GPU 独立，不需要 offload
- `compute_advantages_and_returns = True` —— actor 必须在本地计算 advantages
- `fully_async = True`、`colocate = True` —— 由 `--hybrid` 推导得出

::: tip
Hybrid 和纯 fully async 在启用 `--use-dynamic-batch-size` 时都会使用 `StreamingTokenBudgetSampler`，因此动态 batch 路径会自动做 DP token 均衡。`--balance-data` 主要用于静态/序列长度均衡 batch；在动态 batch 路径上没有额外效果。
:::

______________________________________________________________________

## 快速开始

8 GPU 多模态 Hybrid 训练的参考启动脚本位于
`scripts/training/multimodal/run-qwen35-9B-8xgpu-openr1mm-hybrid-async.sh`。

它构建的 Hybrid 调用命令为：

```bash
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 4], "rollout": [1, 4]}' \
    --max-staleness 2 \
    --num-data-storage-units 1 \
    --num-iters-per-train-update 8 \
    --balance-data \
    --hybrid \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"
```

该配置的关键点：

- 8 GPU 总量，actor 与 rollout 各占 4 张
- `max-staleness 2` —— actor 可以消费比最新权重落后最多 2 个 step 的 rollout 输出
- `num-iters-per-train-update 8` —— 每个全局 batch 在 forward 阶段被切分为 8 个子批次
- `balance-data` —— 该脚本可传入；DP token 均衡由动态 batch 路径处理
- 算法采用 GRPO，附带 `--use-kl-loss` 与 `--use-tis`（这些是算法参数，与 Hybrid 正交）

______________________________________________________________________

## 故障排除

### `train_hybrid(rollout_id=N) batch_index=K stalled for ... seconds`

该警告由 `relax/backends/megatron/actor.py` 抛出，发生在 actor 不断尝试拉取下一个子批次、partition 却始终未被标记 `all_consumed` 时。常见原因：

- Rollout 漏填了当前 partition（丢弃样本后未补齐）。
- Rollout 因健康检查失败或重启而处于暂停状态。
- Staleness 预算耗尽：rollout 必须等待新权重，因此无法继续产出数据。

在认定为代码 bug 前，请先排查 rollout 侧日志及 partition 状态。

### 动态 batch 的 DP token 均衡

同时启用 `--fully-async` 或 `--hybrid` 与 `--use-dynamic-batch-size` 时，DP token 均衡会通过 `StreamingTokenBudgetSampler` 自动完成。如果仍观察到 token 负载不均，请先检查 token 预算（`--max-tokens-per-gpu`）、DP size 和 streaming sampler 的 stall 日志，再调整执行模式。

### Rollout 长时间看到旧权重

Hybrid 在每次 `train_hybrid` 结束时通过同步的 `UpdateWeightFromTensor` 路径推送权重。如果观察到权重更新间隔过大，请检查：

- Actor 日志中的 `update_weights()` 耗时
- Rollout 健康检查是否触发了 actor 等待（权重同步前会调用 `_check_services_health()`）

______________________________________________________________________

## 下一步

Hybrid 模式计划推进的工作：

- **接入 DCS 做权重同步** —— 用 Distributed Checkpoint Service 替换当前同步的 `UpdateWeightFromTensor` 路径，使权重向 rollout 广播能够与下一轮训练迭代重叠，消除每次 `train_hybrid` 结束时残留的同步阻塞。
- **将 `train_actor` 拆分为 `num_iters_per_train_update` 次训练迭代** —— 目前 `num_iters_per_train_update` 只对 forward 阶段做了切分，合并后的训练步仍然在整个全局 batch 上跑一次。下一步将训练步同样切成 `num_iters_per_train_update` 次，让优化器更新与 TransferQueue 数据消费形成流水线，并进一步压低训练侧峰值显存。

相关文档：

- [全异步训练流水线](./fully-async-training.md) —— Hybrid 借用的流式数据引擎
- [架构设计](./architecture.md) —— Relax 服务分层总览
- [权重更新流水线优化](./update-weights-pipeline.md) —— `UpdateWeightFromTensor` 与 DCS 的差异
