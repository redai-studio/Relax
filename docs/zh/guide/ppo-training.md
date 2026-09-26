# PPO 训练

Relax 为 Megatron 后端提供了一等支持的同步 PPO（Proximal Policy Optimization）Actor-Critic 训练路径。

## 概述

通过 `--advantage-estimator ppo` 选择 PPO。与 GRPO、GSPO、CISPO 和 SAPO 不同，PPO 会训练独立的 Critic 模型。Critic 预测 token 级 value，Advantages 服务计算 Generalized Advantage Estimation（GAE），Actor 则优化裁剪后的策略目标。

当前支持的 PPO 拓扑是同步 colocate 模式。提供的入门配置运行在 8 张 GPU 上：Actor、Critic 与 SGLang 共享同一个 placement group 并分时使用 GPU，CPU-only 的 Advantages 服务通过 TransferQueue 交换 `values`、`advantages` 和 `returns`。

::: warning 当前范围
暂不支持 fully-async PPO。请使用本文介绍的同步 colocate 配置。
:::

## 架构

```text
┌─────────────┐   rollout data   ┌───────────────┐
│   Rollout   │ ───────────────> │ TransferQueue │
└──────▲──────┘                  └───────┬───────┘
       │                                 │
       │                         ┌───────▼───────┐
       │                         │    Critic     │
       │                         │ values + loss │
       │                         └───────┬───────┘
       │                                 │ values
       │                         ┌───────▼───────┐
       │                         │  Advantages   │
       │                         │  GAE outputs  │
       │                         └───────┬───────┘
       │                    advantages + returns
       │                                 │
       │                         ┌───────▼───────┐
       └──── weight update ───── │     Actor     │
                                 │ PPO-Clip loss │
                                 └───────────────┘
```

| 组件 | 职责 | TransferQueue 字段 |
|---|---|---|
| **Rollout** | 生成回复、reward 和 rollout log-probability | `tokens`、`rewards`、`rollout_log_probs`、mask 与长度 |
| **Critic** | 预测 value，并使用裁剪 value loss 训练 | 产生 `values`；使用本地计算的 `returns` 训练 value |
| **Advantages** | 在 Critic value 就绪后计算 GAE | 消费 `values`；产生 `advantages` 和 `returns` |
| **Actor** | 使用 PPO-Clip 训练策略并发布新权重 | 消费 `advantages`、`returns` 和旧策略 log-probability |

## 快速开始

入门脚本要求存在以下目录：

- `${MODEL_DIR}/Qwen3.5-9B`
- `${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl`
- `${DATA_DIR}/aime-2024/aime-2024.jsonl`

启动 8 卡 colocate 配置：

```bash
MODEL_DIR=/path/to/models \
DATA_DIR=/path/to/data \
EXP_DIR=/path/to/experiments \
bash scripts/training/text/run-qwen35-9B-8xgpu-ppo.sh
```

该配置使用以下参数启用 PPO：

```bash
PPO_ARGS=(
   --advantage-estimator ppo
   --gamma 1.0
   --lambd 0.95
   --eps-clip 0.2
   --eps-clip-high 0.2
   --entropy-coef 0.0
   --use-rollout-logprobs
   --kl-coef 0.0
   --value-clip 0.5
   --critic-lr 1e-5
   --num-critic-only-steps 5
   --critic-lr-warmup-iters 5
)
```

它要求以下资源拓扑：

```bash
--resource '{"actor": [1, 8], "critic": [1, 8], "rollout": [1, 8], "advantages": [1, 0]}' \
--colocate
```

在 colocate 模式下，`actor`、`critic` 与 `rollout` 的资源形状相同时会共享同一个 placement group，并分时占用 GPU 显存。示例降低了 `--sglang-mem-fraction-static`，并启用 optimizer CPU offload，为该调度方式留出足够显存。

## 配置

### PPO 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--advantage-estimator ppo` | `grpo` | 选择 PPO 服务图并启用 Critic 路径 |
| `--gamma` | `1.0` | GAE 折扣因子 |
| `--lambd` | `1.0` | GAE lambda |
| `--eps-clip` | `0.2` | PPO-Clip 下侧裁剪边距 |
| `--eps-clip-high` | `None` | PPO-Clip 上侧裁剪边距；未设置时跟随 `--eps-clip` |
| `--value-clip` | `0.2` | Critic value 裁剪范围 |
| `--entropy-coef` | `0.0` | 熵奖励系数 |
| `--num-critic-only-steps` | `0` | 初始仅训练 Critic 的 rollout 步数 |
| `--critic-lr` | `None` | Critic 学习率；未设置时跟随 `--lr` |
| `--critic-lr-warmup-iters` | `0` | Critic 线性 warmup 迭代数 |
| `--critic-load` | `None` | Critic 加载的 checkpoint；未设置时跟随 `--load` |
| `--critic-save` | `None` | Critic checkpoint 输出目录 |
| `--use-rollout-logprobs` | 关闭 | 使用 SGLang rollout log-probability 作为旧策略值 |

### 资源要求

所有当前支持的 PPO 配置都必须在 `--resource` 中包含 `critic` 和 `advantages`。

- 使用 `--use-rollout-logprobs`；同步 PPO 不会部署独立 `actor_fwd` 服务。
- 保持关闭 `--use-kl-loss` 并设置 `--kl-coef 0.0`。同步 PPO 服务图没有为 `ref_log_probs` 提供 Reference producer。

::: warning Colocate KL 配置
同步 PPO 收到 `--use-kl-loss` 或 `--kl-coef != 0` 时，参数处理会记录 warning、关闭 `--use-kl-loss`，并将 `--kl-coef` 重置为 `0.0`。从 GRPO 脚本复制配置时应直接移除这些选项，不要依赖自动归一化。
:::

### Checkpoint 恢复

Actor 与 Critic 必须从同一迭代恢复。Relax 会读取 `--load` 和 `--critic-load` 下的 `latest_checkpointed_iteration.txt`；两者迭代不一致时，会在服务启动前直接报错。

请使用以下任一一致状态：

1. Actor 与 Critic 都从 `--hf-checkpoint` 冷启动。
2. 两者都加载相同迭代的 Megatron checkpoint。

需要持久化 Critic checkpoint 时，请设置 `--critic-save`。

## 最佳实践

1. 使用当前提供的同步 colocate 拓扑；暂不支持 fully-async PPO。
2. Critic value head 从策略 checkpoint 初始化时，可使用 `--num-critic-only-steps` 进行预热。
3. 为 Actor 与 Critic 设置不同的学习率；示例分别使用 `1e-6` 和 `1e-5`。
4. 联合观察 `value_loss`、`value_clipfrac`、`pg_loss`、`pg_clipfrac` 和 `ppo_kl`。
5. 同时为 Actor、Critic、optimizer state 与 SGLang 规划内存。资源形状相同可通过分时复用节省 GPU 数量，但不会降低 host memory 需求。

## 故障排除

### 缺少 `critic` 或 `advantages` 资源

PPO 会在部署前验证这两个角色。请在 `--resource` 中同时添加它们；Advantages 是 CPU-only 服务，可以使用 `[1, 0]`。

### 缺少旧策略 log-probability

在当前支持的同步 colocate 拓扑中使用 `--use-rollout-logprobs`。

### Actor 与 Critic 恢复状态不一致

将 `--load` 和 `--critic-load` 指向 tracker 迭代相同的 checkpoint，或移除两边的 Megatron checkpoint，让两个模型都从 HF checkpoint 冷启动。

### 服务切换时 GPU OOM

检查 Actor、Critic 与 Rollout 是否使用相同的 colocate 资源形状，降低 `--sglang-mem-fraction-static`，并保留 optimizer CPU offload。PPO 路径会强制启用训练模型 offload，使 Actor 与 Critic 能在不同阶段之间释放 GPU 显存。

## 下一步

- [算法参考](../examples/algorithms.md) — 对比 PPO、GRPO、GSPO、CISPO 和 SAPO
- [配置说明](./configuration.md) — 查看全部训练参数
- [架构设计](./architecture.md) — 了解 Ray Serve 组件模型
