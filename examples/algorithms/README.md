# Algorithm Examples

本目录包含 Relax 支持的各种策略梯度算法的启动脚本。

## 概述

Relax 框架集成了多种策略梯度算法，均通过 `--advantage-estimator` 参数选择。GRPO、RLOO、CISPO、GSPO 与 SAPO 共享 Actor/Rollout 服务拓扑；RLOO 仅支持同步固定批量，PPO 还需要 Critic 与 Advantages 服务，应分别遵守专用配置。

## 支持的算法

| 算法                     | 启用参数                                             | 推荐场景                    |
| ------------------------ | ---------------------------------------------------- | --------------------------- |
| **PPO**                  | `--advantage-estimator ppo`                          | Actor-Critic、token 级 GAE  |
| **GRPO**                 | `--advantage-estimator grpo`                         | 默认、大多数场景            |
| **REINFORCE++**          | `--advantage-estimator reinforce_plus_plus`          | token KL-to-go 与全局归一化 |
| **REINFORCE++-baseline** | `--advantage-estimator reinforce_plus_plus_baseline` | group baseline 与独立 k2 KL |
| **RLOO**                 | `--advantage-estimator rloo`                         | 无偏基线、非裁剪 REINFORCE  |
| **CISPO**                | `--advantage-estimator cispo`                        | 保留梯度方向、需要更高精度  |
| **GSPO**                 | `--advantage-estimator gspo`                         | 序列级约束、稳定训练        |
| **SAPO**                 | `--advantage-estimator sapo`                         | 平滑优化、soft 信任域       |

## 选择建议

### PPO（Actor-Critic）

- 使用独立 Critic 预测 token 级 value，并通过 GAE 计算 advantage 与 return
- Actor 使用 PPO-Clip，Critic 使用裁剪 value loss
- 必须在 `--resource` 中配置 `critic` 与 `advantages`，不能只替换 `GRPO_ARGS`
- 暂不支持 fully-async PPO
- 8 卡 colocate 入门配置：`scripts/training/text/run-qwen35-9B-8xgpu-ppo.sh`

### GRPO（推荐首选）

- 默认算法，特性平衡，大多数场景可用
- 对超出信任域的 token 直接置零梯度
- 适合大多数强化学习场景

### REINFORCE++ 两个变体

- `reinforce_plus_plus` 使用 token 级 k1 KL reward shaping、KL-to-go return 和跨 DP rank 的有效 token 全局归一化
- `reinforce_plus_plus_baseline` 先减去包含自身的 prompt group mean，不除 group std，再做相同的全局归一化
- baseline 变体不把 token KL 放入 advantage，而是使用独立的 k2 KL loss
- 两个变体当前只支持同步 colocate、CP=1 和 response-mean reduction
- 完整公式、mask 与边界行为参见[REINFORCE++ 设计文档](../../docs/zh/guide/reinforce-plus-plus.md)

### RLOO（无偏 Leave-One-Out 基线）

- 用同一 prompt 的其他采样奖励构造 leave-one-out baseline，不按组内标准差归一化
- 使用非裁剪 REINFORCE loss，`train/pg_clipfrac` 始终为 `0`
- 仅支持同步固定批量，要求 `rollout_batch_size × n_samples_per_prompt = global_batch_size`
- 推荐从 `run-qwen3-0.6B-1xgpu-rloo.sh` 开始，并通过环境变量按硬件调整批量

### CISPO（保留梯度信号）

- 对超出信任域的 token 保留梯度方向（仅限幅）
- 梯度方差较大，需配合 `--kl-loss-coef 0.001` 稳定训练
- 适合需要更精细学习信号的任务

### GSPO（序列级 KL 约束）

- 使用序列级 KL 而非 token 级 KL
- 序列内所有 token 共享统一的约束强度
- 更稳定，适合长序列任务

### SAPO（平滑信任域）

- 用 sigmoid 门控替代硬裁剪
- 梯度流更平滑，避免梯度突变
- 适合对稳定性要求高的场景

## 快速开始

### 基础操作：修改算法参数

GRPO-like 脚本采用统一的参数块设计。要在 GRPO、CISPO、GSPO 与 SAPO 之间切换，只需修改相应 `*_ARGS` 数组：

```bash
# 例如：在任意训练脚本中，将 GRPO_ARGS 替换为 CISPO_ARGS

# 原始（GRPO）：
GRPO_ARGS=(
   --advantage-estimator grpo
   --eps-clip 0.2
)

# 改为 CISPO：
CISPO_ARGS=(
   --advantage-estimator cispo
   --use-kl-loss
   --kl-loss-coef 0.001
   --eps-clip 0.2
   --eps-clip-high 10
   --use-tis
)

# 启动时使用修改后的参数
python3 -m relax.entrypoints.train \
    "${MODEL_ARGS[@]}" \
    "${CISPO_ARGS[@]}" \  # 使用 CISPO 而非 GRPO
    "${OPTIMIZER_ARGS[@]}" \
    ...
```

### 示例：运行 RLOO（文本）

```bash
export MODEL_DIR=/path/to/models
export DATA_DIR=/path/to/data

NUM_ROLLOUT=60 \
ROLLOUT_BATCH_SIZE=4 \
N_SAMPLES=8 \
GLOBAL_BATCH_SIZE=32 \
bash examples/algorithms/run-qwen3-0.6B-1xgpu-rloo.sh
```

保持 `ROLLOUT_BATCH_SIZE × N_SAMPLES = GLOBAL_BATCH_SIZE`。设置 `ADVANTAGE_ESTIMATOR=grpo` 可使用相同 recipe 运行对照臂。脚本将清洗数据写入 artifact cache（可用 `RLOO_DATA_CACHE_DIR` 覆盖），并要求模型用 `\boxed{...}` 输出最终答案，以匹配 `math` reward parser。

### 示例：运行 CISPO（多模态）

```bash
# 1) 准备模型和数据
export MODEL_DIR=/path/to/Qwen3.5-9B
export DATA_DIR=/path/to/multimodal-open-r1-8k-verified
export EXP_DIR=/path/to/experiments

# 2) 运行 CISPO 异步训练（Fully Async 模式）
bash examples/algorithms/run-qwen35-9B-8xgpu-openr1mm-cispo-async.sh async

# 或运行同步训练（Colocate 模式）
bash examples/algorithms/run-qwen35-9B-8xgpu-openr1mm-cispo-async.sh sync
```

### 示例：运行 GSPO（文本）

编辑 `scripts/training/text/run-qwen3-4B-8xgpu.sh`，将 `GRPO_ARGS` 改为：

```bash
GSPO_ARGS=(
   --advantage-estimator gspo
   --eps-clip 0.2
)
```

然后启动：

```bash
export MODEL_DIR=/path/to/Qwen3-4B
export DATA_DIR=/path/to/aime-2024
export EXP_DIR=/path/to/experiments

bash scripts/training/text/run-qwen3-4B-8xgpu.sh
```

## 关键参数说明

### 通用参数

| 参数                    | 默认值               | 说明                                                                                                            |
| ----------------------- | -------------------- | --------------------------------------------------------------------------------------------------------------- |
| `--advantage-estimator` | `grpo`               | 算法类型：`grpo`, `cispo`, `gspo`, `sapo`, `ppo`, `rloo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline` |
| `--eps-clip`            | `0.2`                | 下方裁剪边距（ratio 下界 = `1 - eps_clip`）                                                                     |
| `--eps-clip-high`       | 与 `--eps-clip` 相同 | 上方裁剪边距（ratio 上界 = `1 + eps_clip_high`）                                                                |
| `--clip-grad`           | —                    | 梯度裁剪范数，CISPO 下推荐设为 `1.0`                                                                            |
| `--kl-coef`             | `0.0`                | KL 惩罚系数；当前同步 PPO 会将非零值重置为 `0.0`，REINFORCE++ 等算法可使用                                      |

### RLOO 专用约束与指标

| 项目                                         | 要求/含义                                             |
| -------------------------------------------- | ----------------------------------------------------- |
| `--n-samples-per-prompt`                     | 至少为 `2`                                            |
| `rollout_batch_size × n_samples_per_prompt`  | 必须等于 `global_batch_size`                          |
| `--num-steps-per-rollout`                    | 不设置或为 `1`                                        |
| `--calculate-per-token-loss`                 | 必须开启，按全局有效 response token 数归一化          |
| `--kl-coef`                                  | 必须为 `0`；配置 `--ref-load` 后使用直接 KL loss 路径 |
| `--max-staleness`                            | 必须为 `0`                                            |
| async、partial rollout、dynamic global batch | 不支持                                                |
| `--normalize-advantages`                     | 不支持                                                |
| `rollout/rloo/*`                             | 训练侧 baseline、advantage signal、空响应与异常组诊断 |

### CISPO 专用参数

| 参数             | 默认值 | 推荐值       | 说明                                |
| ---------------- | ------ | ------------ | ----------------------------------- |
| `--use-kl-loss`  | off    | on           | 启用 KL loss（CISPO 推荐必开）      |
| `--kl-loss-coef` | `0.0`  | `0.001`      | KL loss 系数，约束策略偏移          |
| `--kl-loss-type` | `k1`   | `low_var_kl` | KL 估计方式，`low_var_kl` 方差更低  |
| `--use-tis`      | off    | on           | Token Importance Sampling，推荐开启 |

### SAPO 专用参数

| 参数             | 默认值 | 说明                                             |
| ---------------- | ------ | ------------------------------------------------ |
| `--sapo-tau-pos` | `1.0`  | Positive advantage 的温度参数                    |
| `--sapo-tau-neg` | `1.05` | Negative advantage 的温度参数（更高 = 更强抑制） |

### PPO 专用参数

| 参数                       | 默认值         | 说明                        |
| -------------------------- | -------------- | --------------------------- |
| `--gamma`                  | `1.0`          | GAE 折扣因子                |
| `--lambd`                  | `1.0`          | GAE lambda                  |
| `--value-clip`             | `0.2`          | Critic value loss 裁剪范围  |
| `--num-critic-only-steps`  | `0`            | 初始 Critic-only 预热步数   |
| `--critic-lr`              | 与 `--lr` 相同 | Critic 学习率               |
| `--critic-lr-warmup-iters` | `0`            | Critic 学习率线性预热迭代数 |

PPO 还要求在 `--resource` 中提供 `critic` 与 `advantages`。当前 colocate 示例使用 `--use-rollout-logprobs`，并保持 `--kl-coef 0.0`。完整说明参见[双语 PPO 训练文档](../../docs/zh/guide/ppo-training.md)。

## 最佳实践

### 1. CISPO：启用 KL 约束与 TIS

```bash
CISPO_ARGS=(
   --advantage-estimator cispo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --eps-clip 0.2
   --eps-clip-high 10
   --use-tis
   --clip-grad 1.0
)
```

**为什么**：

- `--use-kl-loss --kl-loss-coef 0.001`：CISPO 保留超界梯度，易导致策略漂移；小 KL 惩罚约束偏移
- `--use-tis`：Token Importance Sampling 可增强样本有效性
- `--eps-clip-high 10`：近似取消上侧裁剪，对 positive advantage token 更宽松

### 2. GSPO：序列级约束，更稳定

```bash
GSPO_ARGS=(
   --advantage-estimator gspo
   --eps-clip 0.2
)
```

**为什么**：GSPO 使用序列级 KL，序列内 token 约束一致，减少长序列训练的振荡。

### 3. 异步 vs 同步模式选择

- **Fully Async**（`--fully-async`）：

  - 适合：GPU 充足、高吞吐要求
  - 风险：off-policy 数据，需配合 `--max-staleness` 和 `--use-health-check`

- **Colocate**（`--colocate`）：

  - 适合：GPU 有限、小规模实验
  - 优势：pure on-policy，梯度更稳定

## 文件组织

```
examples/algorithms/
├── README.md                              (本文件)
├── run-qwen35-9B-8xgpu-openr1mm-cispo-async.sh    (CISPO 多模态示例)
├── ... (其他算法脚本)
```

## 常见问题

### Q: 哪个算法性能最好？

**A**: 这取决于任务。一般规律：

- **GRPO**：all-round，快速尝试的首选
- **CISPO**：需要精细学习信号时更好，但需要 KL 约束
- **GSPO**：长序列任务，训练更稳定
- **PPO**：如果已有 Critic 资源，性能可能更好

### Q: CISPO 的梯度波动很大，正常吗？

**A**: 正常。CISPO 保留超界 token 的梯度，会导致梯度方差更大。配合 `--clip-grad 1.0` 限制实际参数更新，并配置 `--kl-loss-coef 0.001` 稳定策略。

### Q: 我应该用 async 还是 sync？

**A**:

- **Sync**（colocate）：GPU 有限（≤8）、学习率高、需要稳定梯度 → 推荐
- **Async**（fully-async）：GPU 充足（≥16）、追求高吞吐、可接受轻微 off-policy → 推荐

## 参考文献

- [Relax 算法文档](../../docs/en/examples/algorithms.md)
- [GRPO - DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [CISPO - MiniMax-M1](https://arxiv.org/abs/2506.13585)
- [PPO - Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [REINFORCE++ - Simple Efficient Alignment](https://arxiv.org/abs/2501.03262)
- [RLOO - Back to Basics (Ahmadian et al. 2024)](https://arxiv.org/abs/2402.14740)
- [RLOO - Buy 4 REINFORCE Samples, Get a Baseline for Free (Kool et al. 2019)](https://arxiv.org/abs/1905.12705)
