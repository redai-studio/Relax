# 算法参考

Relax 支持多种策略梯度算法，均通过 `--advantage-estimator` 参数选择。本文档覆盖 PPO 与主要 GRPO-family 算法（OPD 在线策略蒸馏请参阅[单独文档](./on-policy-distillation.md)）。

GRPO、RLOO、CISPO、GSPO、SAPO 与 M2PO 使用相同的 Actor/Rollout 服务拓扑；其中 RLOO 仅支持同步模式，并要求固定批量不变量。PPO 还需要 Critic 模型与 Advantages 服务，因此应从 [PPO 训练配置](../guide/ppo-training.md)开始，而不是只替换 `GRPO_ARGS`。

REINFORCE++ 与 REINFORCE++-baseline 同样复用 GRPO 服务拓扑，但 return、全局归一化和 KL 契约由算法单独定义。启用任一 estimator 前，请先阅读 [REINFORCE++ 训练文档](../guide/reinforce-plus-plus.md)。

---

## GRPO

GRPO（Group Relative Policy Optimization）是 Relax 的默认算法。将组内标量奖励广播到每个 token，使用 PPO 风格的裁剪目标函数。

参考论文：[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)。

### 算法原理

GRPO 目标函数为标准 PPO-Clip：

$$J_\text{GRPO}(\theta) = \mathbb{E} \left[ \min\!\left( r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\varepsilon,\ 1+\varepsilon)\hat{A}_t \right) \right]$$

其中 $r_t(\theta) = \pi_\theta / \pi_{\theta_\text{old}}$，$\hat{A}_t$ 是组相对 advantage（组内奖励减去组均值，按组标准差归一化）。当 ratio 超出裁剪边界时，梯度被直接置零。

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator grpo` | 默认 | 启用 GRPO |
| `--eps-clip` | `0.2` | 裁剪边距（ratio 范围 = `[1-ε, 1+ε]`） |
| `--eps-clip-high` | 与 `--eps-clip` 相同 | 上方裁剪边距，可设为不同值以构造非对称裁剪 |
| `--clip-grad` | — | 梯度裁剪范数 |

### 快速开始

GRPO 是默认算法，无需修改参数。直接使用训练脚本即可：

```bash
MODEL_DIR=/path/to/model \
DATA_DIR=/path/to/data \
EXP_DIR=/path/to/exp \
bash scripts/training/text/run-qwen3-4B-8xgpu.sh
```

---

## RLOO

RLOO（REINFORCE Leave-One-Out）使用同一 prompt 的其他采样结果作为无偏基线。Relax 实现的是同步 RLOO，并采用不裁剪的 REINFORCE 策略损失，不使用 PPO ratio 或 clipping。

参考论文：[Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740)。

### 算法原理

对一个 prompt 采样 $G$ 条 response、得到标量奖励 $r_i$ 时，leave-one-out 基线与 advantage 为：

$$b_i = \frac{1}{G-1}\sum_{j\ne i}r_j, \qquad A_i = r_i-b_i = \frac{G}{G-1}(r_i-\bar r)$$

与 GRPO 不同，RLOO 不除以组内标准差。逐 token 损失为：

$$L_{i,t} = -\operatorname{stopgrad}(A_i)\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})$$

每条样本的标量 advantage 会广播到 response token。Relax 会屏蔽 padding，并按全局有效 response token 数 $N_{\mathrm{eff}}$ 归一化 token loss 之和：

$$L_{\mathrm{RLOO}} = -\frac{1}{N_{\mathrm{eff}}}\sum_i\sum_t m_{i,t}\operatorname{stopgrad}(A_i)\log\pi_{i,t}$$

这种 global-token reduction 不会为每条 response 单独附加 $1/T_i$ 权重。RLOO 不使用 clipping，因此 `train/pg_clipfrac` 始终为 `0`。

### 约束与参数

| 参数 | 要求 | 说明 |
|------|------|------|
| `--advantage-estimator rloo` | 必填 | 启用 RLOO |
| `--n-samples-per-prompt` | 至少为 `2` | 组大小 $G$；更大的组可提高 LOO 基线稳定性，但增加 rollout 成本 |
| `--rollout-batch-size × --n-samples-per-prompt` | 等于 `--global-batch-size` | 每个 rollout 恰好执行一次 optimizer update |
| `--num-steps-per-rollout` | 不设置或为 `1` | 禁止对同一 rollout 重复执行未经 ratio 校正的更新 |
| `--calculate-per-token-loss` | 开启 | 使用全局有效 token 归一化；per-response token mean 会按 $1/T_i$ 重新加权不等长 response |
| `--kl-coef` | `0` | RLOO 尚未实现 reward-side KL shaping；提供有效的 `--ref-load <checkpoint>` 后，可使用受支持的 `--use-kl-loss --kl-loss-coef <value>` 直接 KL loss |
| `--max-staleness` | `0` | 非裁剪目标没有 importance-ratio correction，因此拒绝 stale rollout |
| reward normalization | 开启 | RLOO 的组变换位于 normalized-reward 路径 |
| `--normalize-advantages` | 关闭 | DP 切分后的再次白化会改变 RLOO 语义，并使结果依赖分区 |
| `--fully-async`、`--hybrid`、`--partial-rollout`、`--use-dynamic-global-batch-size` | 关闭 | RLOO 当前要求同步、固定大小的 rollout batch |

只要保持批量等式，便可根据硬件调整批量参数。例如 `ROLLOUT_BATCH_SIZE=4`、`N_SAMPLES=8`、`GLOBAL_BATCH_SIZE=32` 仍保持每 rollout 一次更新，同时比 `16 × 8 = 128` 降低每步显存压力。

### 诊断指标

训练 rollout 日志会发布以下最终指标名：

- `rollout/rloo/baseline_mean`：LOO baseline 均值（等于组奖励均值；保留为显式 baseline 轨迹）
- `rollout/rloo/adv_abs_mean`：RLOO advantage 的平均绝对值
- `rollout/rloo/no_signal_frac`：属于零 advantage 样本的有效 loss token 比例
- `rollout/rloo/empty_response_frac`：response 字面为空的样本比例
- `rollout/rloo/zero_adv_group_frac`：所有 advantage 都为零的完整组比例
- `rollout/rloo/dropped_group_frac`：因组大小不完整而未纳入诊断的已观察组比例

这些诊断只用于训练 rollout，属于纯观测统计，不影响训练路径。Eval 使用独立的采样组大小，不会记录具有误导性的 `eval/*/rloo/*` 全零指标。当自定义 reward post-processor 或 agentic custom-advantage hook 替换标准 RLOO 信号时，也会省略这些指标，因为此时无法仅靠 raw reward 重建 optimizer 的真实输入。

### 快速开始

使用 Qwen3-0.6B GSM8K 专用 recipe；批量与 rollout 数均可通过环境变量覆盖：

```bash
MODEL_DIR=/path/to/models \
DATA_DIR=/path/to/data \
NUM_ROLLOUT=60 \
ROLLOUT_BATCH_SIZE=4 \
N_SAMPLES=8 \
GLOBAL_BATCH_SIZE=32 \
bash examples/algorithms/run-qwen3-0.6B-1xgpu-rloo.sh
```

设置 `ADVANTAGE_ESTIMATOR=grpo` 可在相同 recipe 与 seed 下运行对照臂。该 recipe 会把清洗后的 GSM8K 写入可写的 artifact cache（可用 `RLOO_DATA_CACHE_DIR` 覆盖），并在问题后追加最终答案须使用 `\boxed{...}` 的指令，以匹配 `math` reward parser 的输入契约。

---

## PPO

PPO（Proximal Policy Optimization）是一种 Actor-Critic 算法。Relax 会训练独立的 Critic 来预测 token 级 value，计算 GAE advantages 与 returns，对 Actor 使用 PPO-Clip，并对 Critic 使用裁剪 value loss。

参考论文：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)。

### 算法原理

Temporal-difference residual 与 GAE 递推为：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\hat{A}_t = \delta_t + \gamma\lambda\hat{A}_{t+1}, \qquad \hat{R}_t = \hat{A}_t + V(s_t)$$

Actor 随后使用 GRPO 一节展示的裁剪策略目标，但 advantage 来自 Critic 的 token 级估计。Critic 则最小化裁剪与未裁剪 value 平方误差中的较大值。

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator ppo` | — | 启用 PPO 与 Critic 服务图 |
| `--gamma` | `1.0` | GAE 折扣因子 |
| `--lambd` | `1.0` | GAE lambda |
| `--eps-clip` | `0.2` | Actor 裁剪边距 |
| `--value-clip` | `0.2` | Critic value 裁剪范围 |
| `--num-critic-only-steps` | `0` | 初始 Critic-only 预热步数 |
| `--critic-lr` | 与 `--lr` 相同 | Critic 学习率 |

### 快速开始

PPO 的服务图要求 `critic` 与 `advantages` 资源，因此不能只修改算法参数来启用。暂不支持 fully-async PPO；请使用专用同步 colocate 配置：

```bash
MODEL_DIR=/path/to/models \
DATA_DIR=/path/to/data \
EXP_DIR=/path/to/experiments \
bash scripts/training/text/run-qwen35-9B-8xgpu-ppo.sh
```

资源拓扑、checkpoint 规则与 KL 约束详见 [PPO 训练](../guide/ppo-training.md)。

---

## CISPO

CISPO（Clipped Importance-ratio Soft Policy Optimization）对超出信任域的 token 保留梯度信号，而非将其清零。通过 stop-gradient 系数限制梯度幅度，同时保留梯度方向。

参考论文：[MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)。

### 算法原理

CISPO 目标函数为：

$$J_\text{CISPO}(\theta) = \mathbb{E}_{(q,a)\sim\mathcal{D},\ \{o_i\}_{i=1}^G \sim \pi_{\theta_\text{old}}(\cdot|q)} \left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \text{sg}\!\left(\hat{r}_{i,t}(\theta)\right) \hat{A}_{i,t} \log \pi_\theta(o_{i,t} \mid q, o_{i,<t}) \right]$$

其中 $\hat{r}_{i,t}(\theta)$ 是裁剪后的重要性采样权重：

$$\hat{r}_{i,t}(\theta) = \text{clip}\!\left(r_{i,t}(\theta),\ 1 - \varepsilon_\text{low}^\text{IS},\ 1 + \varepsilon_\text{high}^\text{IS}\right)$$

$r_{i,t}(\theta) = \pi_\theta(o_{i,t} \mid q, o_{i,<t}) / \pi_{\theta_\text{old}}(o_{i,t} \mid q, o_{i,<t})$。梯度**只**流过 $\log\pi_\theta$，$\hat{r}_{i,t}$ 和 $\hat{A}_{i,t}$ 均被 stop-gradient 处理。

### 关键参数

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `--advantage-estimator cispo` | — | — | 启用 CISPO |
| `--eps-clip` | `0.2` | `0.2` | 下方裁剪边距（ratio 下界 = `1 - eps_clip`） |
| `--eps-clip-high` | 与 `--eps-clip` 相同 | `10` | 上方裁剪边距（ratio 上界 = `1 + eps_clip_high`）。设为 `10` 可近似取消上侧裁剪 |
| `--kl-loss-coef` | `0.0` | `0.001` | KL 损失系数。推荐设为 `0.001`，添加小幅 KL 惩罚以约束策略偏移 |
| `--use-kl-loss` | 关闭 | 开启 | 启用 KL 损失计算（`--kl-loss-coef` 生效的前提） |
| `--use-tis` | 关闭 | 开启 | Token Importance Sampling，推荐与 CISPO 同时开启 |
| `--clip-grad` | — | `1.0` | 梯度裁剪范数 |

### 快速开始

使用任意 GRPO 训练脚本，将 `GRPO_ARGS` 替换为 `CISPO_ARGS`：

```bash
CISPO_ARGS=(
   --advantage-estimator cispo
   --use-kl-loss
   --kl-loss-coef 0.001
   --eps-clip 0.2
   --eps-clip-high 10
   --use-tis
)
```

---

## GSPO

GSPO（Group-wise Sequence-level Policy Optimization）与 GRPO 的区别在于 KL 散度的计算方式：GSPO 使用**序列级** KL 而非逐 token KL。每个 token 的 KL 值等于该序列所有 token KL 的均值，这为序列内所有 token 提供统一的约束强度。

### 算法原理

GSPO 使用与 GRPO 相同的 PPO-Clip 目标函数，但 ratio 的计算基于序列级 KL：

$$\text{KL}_\text{seq} = \frac{1}{|o|} \sum_{t=1}^{|o|} \left(\log\pi_{\theta_\text{old}}(o_t) - \log\pi_\theta(o_t)\right)$$

每个 token 的 ratio 均为 $r_t = \exp(-\text{KL}_\text{seq})$，而非各自独立的 token 级 ratio。

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator gspo` | — | 启用 GSPO |
| `--eps-clip` | `0.2` | 裁剪边距 |
| `--eps-clip-high` | 与 `--eps-clip` 相同 | 上方裁剪边距 |
| `--clip-grad` | — | 梯度裁剪范数 |

### 快速开始

```bash
GSPO_ARGS=(
   --advantage-estimator gspo
   --eps-clip 0.2
)
```

---

## SAPO

SAPO（Soft Adaptive Policy Optimization）用平滑的 sigmoid 门控替代硬裁剪。通过温度参数控制门控曲线的陡峭程度，实现可微的信任域约束。

### 算法原理

SAPO 的核心是一个以 ratio=1 为中心的 sigmoid 门控函数：

$$f(r) = \frac{4}{\tau} \cdot \sigma\!\left(\tau(r - 1)\right)$$

其中 $\sigma$ 是 sigmoid 函数，$\tau$ 根据 advantage 的符号选择不同的温度：

- $A > 0$: 使用 $\tau_\text{pos}$（默认 1.0）
- $A \leq 0$: 使用 $\tau_\text{neg}$（默认 1.05，对 negative token 更强的抑制）

SAPO 目标：$J_\text{SAPO}(\theta) = f(r) \cdot A$

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--advantage-estimator sapo` | — | 启用 SAPO |
| `--sapo-tau-pos` | `1.0` | positive advantage 的温度参数 |
| `--sapo-tau-neg` | `1.05` | negative advantage 的温度参数（更高 = 更强抑制） |
| `--clip-grad` | — | 梯度裁剪范数 |

### 快速开始

```bash
SAPO_ARGS=(
   --advantage-estimator sapo
   --sapo-tau-pos 1.0
   --sapo-tau-neg 1.05
)
```

---

## M2PO

M2PO（Second-Moment Trust Policy Optimization）用有害 token 上重要性比对数的**二阶矩**作为信任域约束：仅当二阶矩超过预算时才收紧裁剪，否则保留 token。相比固定裁剪，它在 off-policy（陈旧数据）场景下能保留更多有效梯度、缓解熵坍缩，是专为数据复用 / 异步训练设计的算法。

参考论文：[Prosperity before Collapse: How Far Can Off-Policy RL Reach with Reuse of Mini-Batch Data?](https://arxiv.org/abs/2510.01161)（NeurIPS 2025）。

### 算法原理

M2PO 只约束 PPO 会裁剪的"有害" token —— 即优势符号与 ratio 偏移方向一致、会造成过度更新的 token：

$$\mathcal{H} = \{t : \hat{A}_t > 0,\ r_t > 1\} \cup \{t : \hat{A}_t < 0,\ r_t < 1\}$$

其中 $r_t = \exp(-\text{KL}_t)$，$\text{KL}_t = \log\pi_{\theta_\text{old}}(o_t) - \log\pi_\theta(o_t)$。这些 token 上 log-ratio 的二阶矩为：

$$M_2 = \frac{1}{|\mathcal{H}|} \sum_{t \in \mathcal{H}} (\log r_t)^2$$

Relax 会在每个 Megatron microbatch 的本地 token 上独立求解该统计量。日志保留既有聚合方式：开启 `--calculate-per-token-loss` 时，`train/ppo_kl_m2_before` 与 `train/ppo_kl_m2_after` 按 microbatch 参与 loss 的 token 数加权；否则直接累加各 microbatch 的标量，再由框架除以 sample 数。后者不是按 sample 加权的均值，两种模式也都不是将整个 global batch 所有有害 token 合并计算的二阶矩。这些日志诊断值不会反馈到 loss。

- 若 $M_2 \le$ `kl2_budget`：不裁剪，token 全部保留；
- 否则用 water-filling 选择前一个已观测到的 breakpoint，得到保守的信任域半径 $\tau$；截顶后的总和不超过 $|\mathcal{H}| \cdot \text{kl2\_budget}$，并据此得到裁剪区间 $[e^{-\tau},\ e^{\tau}]$。

最终裁剪边距为 $\varepsilon = \max(\text{自适应值},\ \text{miniclip})$，因此不会比配置的地板更紧；若希望以 GRPO 的边距为下限，应把地板设成相应值。`ppo_kl_m2_after` 是应用地板前、所选 breakpoint 对应的旧版求解器诊断值；在第一个 breakpoint 的边界情形下，它保留未截顶的局部均值。它并不是最终 policy clip 后重新计算的值。策略损失沿用 GRPO 一节的 PPO-Clip 悲观形式，仅裁剪边界改为自适应求解。

### 关键参数

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `--advantage-estimator m2po` | — | — | 启用 M2PO |
| `--m2po-kl2-budget` | `0.01` | `0.01`~`0.04` | 每个有害 token 的二阶矩预算。越小裁剪越紧/越频繁，越大越容忍 off-policy（论文用 `0.04`） |
| `--m2po-miniclip-low` | `0.3` | `0.2` | 下侧裁剪边距地板（边距至少为 `miniclip_low`） |
| `--m2po-miniclip-high` | `0.5` | `0.28` | 上方裁剪边距地板 |
| `--use-rollout-logprobs` | 关闭 | 陈旧异步数据时开启 | 使用实际生成 token 的 behavior-policy log probabilities 作为 M2PO old policy |
| `--use-tis` | 关闭 | 使用 rollout log probs 时关闭 | TIS 是 policy loss 之后的修正，不会驱动 M2PO 自适应阈值；它与 `--use-rollout-logprobs` 互斥 |

> M2PO 自适应推导裁剪边界，因此**不使用** `--eps-clip` / `--eps-clip-high`。

::: warning 既有实现行为
注册表重构保留 M2PO 的既有计算方式：reward 处理直接传递原始样本奖励，阈值求解器不按 loss mask 过滤 token，也不跨 CP rank 汇总统计量；breakpoint 比较仍使用 host 标量。完成注册不代表与论文等价。由于裁剪统计量在本地计算，M2PO 声明 `supports_context_parallel=False`，启动时要求 `--context-parallel-size 1`，并关闭动态 context parallel。该校验保持算法的 reward、求解器和指标计算不变。
:::

### 推荐使用场景

M2PO 的收益随训练数据的 off-policy 程度增大而放大，因此在以下场景**优先**考虑开启：

- **较大 staleness 的异步训练**：fully-async 模式下 rollout 权重明显滞后于 actor（`--max-staleness` 取 32、256 甚至更高），陈旧样本会推高重要性比。固定裁剪此时要么把大量 token 直接置零、丢失梯度，要么放行造成过度更新；M2PO 用二阶矩自适应地只收紧真正"有害"的那部分，在保留学习信号的同时抑制崩溃。
- **mini-batch 数据复用 / 多轮采样**：同一批 rollout 被反复用于多步更新时，后几步实际上也在 off-policy 数据上训练，M2PO 能延长这批数据的可用寿命。
- **训练后期熵坍缩、reward 停滞**：当固定裁剪导致策略过快收窄、探索不足时，M2PO 更宽松的自适应边界有助于维持熵、延缓坍缩。

反过来，若是严格 on-policy（`--max-staleness 0`、每步同步权重）的同步训练，M2PO 相对 GRPO 的增量有限，可先用 GRPO 作为基线。

M2PO 在 true-on-policy 模式下仍是合法的，但此时重要性比恒为 1，自适应裁剪不会触发。要评估陈旧数据下的效果，请确保运行能提供 old-policy log probabilities，而不是自动开启 `--true-on-policy-mode`。对于 fully-async rollout staleness，应传入 `--use-rollout-logprobs`；`--use-tis` 只会在 M2PO 已经选完裁剪边界之后生效。

### 快速开始

使用任意 GRPO 训练脚本，将 `GRPO_ARGS` 替换为 `M2PO_ARGS`：

```bash
M2PO_ARGS=(
   --advantage-estimator m2po
   --m2po-kl2-budget 0.01
   --m2po-miniclip-low 0.2
   --m2po-miniclip-high 0.28
)
```

---

## 算法对比

| 算法 | Advantage 计算 | 策略损失 | KL 约束方式 |
|------|---------------|---------|-----------|
| **PPO** | Critic value + GAE | PPO-Clip（硬裁剪） | 当前同步拓扑中禁用 |
| **GRPO** | 组相对奖励 | PPO-Clip（硬裁剪） | 可选 KL loss |
| **REINFORCE++** | Token KL-to-go return + 全局 token 归一化 | PPO-Clip（硬裁剪） | shaped reward 中的 k1 KL |
| **REINFORCE++-baseline** | Inclusive group mean + 全局 token 归一化 | PPO-Clip（硬裁剪） | 独立 k2 KL loss |
| **CISPO** | 组相对奖励 | Stop-gradient 系数 | 推荐 KL loss |
| **GSPO** | 组相对奖励 | PPO-Clip + 序列级 KL | 序列级 ratio |
| **SAPO** | 组相对奖励 | Sigmoid 门控 | 温度控制 |
| **M2PO** | 原始样本奖励广播到 token | 二阶矩自适应裁剪 | 可选 KL loss（大 staleness / off-policy 场景优先） |
| **RLOO** | Leave-one-out 基线 | 非裁剪 REINFORCE | 可选 KL loss（同 GRPO） |

## 下一步

- [PPO 训练](../guide/ppo-training.md)
- [REINFORCE++ 训练](../guide/reinforce-plus-plus.md)
- [快速开始](../guide/quick-start.md)
- [在线策略蒸馏](./on-policy-distillation.md)
- [生成式奖励模型](./generative-reward-model.md)
