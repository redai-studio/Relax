# REINFORCE++ 与 REINFORCE++-baseline

Relax 提供两个独立的 estimator 名称，因为 baseline 变体并不只是使用了不同的 recipe：

- `reinforce_plus_plus`
- `reinforce_plus_plus_baseline`

本文冻结两个变体的 return、advantage、归一化、mask、KL 和 reduction 语义。实现遵循
[REINFORCE++ arXiv v9](https://arxiv.org/abs/2501.03262) 的主要公式，以及 OpenRLHF
commit [`bc71bb1`](https://github.com/OpenRLHF/OpenRLHF/tree/bc71bb19464aca306b33080b2d2bb45d154e2f49)
中的可执行归一化约定。

## 版本与命名说明

论文从 v1 到 v9 发生了变化，甚至 v9 的主要方法与附录 B.2 对 token 放置方式的描述也并不完全一致。下表明确列出版本和实现边界。

| 来源 | Return / baseline | 归一化与 KL | 本实现中的状态 |
|---|---|---|---|
| 论文 v1 | token k1 KL-to-go advantage 加 PPO clipping | 描述 reward 归一化/裁剪和 batch z-score advantage 归一化，但没有单独命名的 group-baseline-plus-k2 变体 | 仅作为历史 REINFORCE++；不是 baseline 定义 |
| 论文 v9 主要公式 | token KL-to-go return；inclusive group-mean baseline 变体 | 对 advantage token 做全局归一化 | 规范性的论文定义 |
| 论文 v9 附录 B.2 | 最后一个 token 之前 advantage 为零 | sample-level reward 归一化 | 已记录的冲突；未采用 |
| OpenRLHF `bc71bb1` | inclusive group mean 和全局有效 token population 归一化 | 固定的 baseline 训练脚本同时启用 token KL shaping 和独立 k2 loss | 仅作为归一化参考；有意不复制其组合 KL baseline |
| 本功能之前的 Relax | 部分 helper 名称和 return 代码 | 没有注册两个变体、冻结校验、专用 population moments、recipe 或完整数值契约 | 兼容性基线 |

Relax 采用 v9 主要公式的解释。本文所说的“与 OpenRLHF 对齐”仅指其可执行的 inclusive baseline 和 masked population normalization 约定。根据 Task 29 Proposal 冻结的定义，Relax baseline 明确不把 KL 放入 advantage，只应用独立 k2 loss，因此它并不是固定 OpenRLHF 训练脚本的完全复现。

## 符号与 mask 契约

对于 response $i$ 和 response 位置 $t$：

- $R_i$ 是标量终止奖励；
- $m_{i,t}\in\{0,1\}$ 是 response loss mask；
- $T_i$ 是最后一个有效 response 位置；
- $L_i=\sum_t m_{i,t}$ 是有效 response 长度。

Prompt token、padding 和 `mask=0` 的 response token 均不参与 reward shaping、return、归一化或 loss。生产环境中的 return 和 advantage 张量在 mask 外显式为零。选择操作使用布尔条件而非乘法，因此即使 mask 外存储位置包含 `NaN` 或 `Inf`，也不会污染有效 token 的结果。

## REINFORCE++

定义带符号的 k1 estimator：

$$
d_{i,t}=\log\pi_{old}(a_{i,t})-\log\pi_{ref}(a_{i,t}).
$$

shaped token reward 为：

$$
r_{i,t}=m_{i,t}\left[-\beta d_{i,t}+\mathbf{1}(t=T_i)R_i\right].
$$

终止奖励只加到最后一个有效 response token。Return 从后向前累计：

$$
G_{i,t}=m_{i,t}\sum_{u=t}^{T_i}\gamma^{u-t}r_{i,u}.
$$

正式 recipe 固定 `gamma=1`。原始 advantage 为 $G$，随后执行下文的全局 masked normalization。该变体使用 token KL reward shaping，不再增加第二个 KL loss。

```text
--advantage-estimator reinforce_plus_plus
--normalize-advantages
--gamma 1.0
--kl-coef 0.01
--kl-loss-type k1
```

## REINFORCE++-baseline

对于包含 $K$ 个采样 response 的 prompt group $g$：

$$
b_g=\frac{1}{K}\sum_{j\in g}R_j,\qquad C_i=R_i-b_g.
$$

group mean 包含当前 response，因此不是 leave-one-out baseline。Relax 不使用 group 标准差除以 $C_i$。

原始 token advantage 为：

$$
A^{raw}_{i,t}=m_{i,t}C_i.
$$

Token KL 不从该 advantage 中减去。Reference regularization 使用独立 k2 loss：

$$
D^{k2}_{i,t}=\frac{1}{2}
\left(\log\pi_\theta(a_{i,t})-\log\pi_{ref}(a_{i,t})\right)^2.
$$

```text
--advantage-estimator reinforce_plus_plus_baseline
--normalize-advantages
--n-samples-per-prompt 8
--kl-coef 0
--use-kl-loss
--kl-loss-type k2
--kl-loss-coef 0.01
```

baseline 变体要求每个 prompt 的采样数大于 1。group mean 包含样本自身，因此 `n_samples_per_prompt=1` 会使每个原始 advantage 都退化为零，配置会被拒绝。该 estimator 同样拒绝自定义 reward 后处理和 agentic 自定义 advantage hook，因为它们会绕过已经冻结的 inclusive group-mean 语义。

## 全局 masked normalization

统计总体包含闭合同步 global batch 中、跨全部 data-parallel rank 的所有有效 response token：

$$
S=\{(r,i,t)\mid m_{r,i,t}=1\},\qquad N=|S|.
$$

Relax 使用总体方差（`ddof=0`）：

$$
\mu=\frac{1}{N}\sum_{S}A,
\qquad
\sigma^2=\frac{1}{N}\sum_{S}(A-\mu)^2.
$$

归一化输出为：

$$
\hat A=m(A-\mu)\left[\max(\sigma^2,10^{-8})\right]^{-1/2}.
$$

epsilon 的语义是**方差下限**，与固定的 OpenRLHF 实现一致。它不同于 `sqrt(var) + epsilon`，也不同于 Relax 旧有的 `sqrt(unbiased_var + epsilon)` helper。两个 REINFORCE++ 变体使用专用 helper，既有算法保持原有归一化行为。

预期边界行为：

- 零方差和单个有效 token 产生有限的零 advantage；
- 全零 baseline reward group 产生有限的零；
- reward 全零但 KL 非零时，REINFORCE++ 可产生有限的 KL-shaped return；
- 本地 response 全部被 mask 时贡献零张量，但仍参与 data-parallel collective；
- 全局 mask 为空时，在所有参与 rank 上触发设备端异步断言。

由于 baseline 标量会广播到每个有效 token，更长的 response 在 token-level global moments 中权重更大。这是有意的设计。

## PPO 与 KL reduction

两个变体都使用普通的 token PPO clipped surrogate。其正式标量目标是 response mean：

$$
L_{PG}=\frac{1}{B}\sum_i\frac{1}{L_i}
\sum_t m_{i,t}\max\left(
-\rho_{i,t}\hat A_{i,t},
-\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_{i,t}
\right).
$$

baseline k2 loss 使用相同的 response-mean reduction。初始实现拒绝对这两个变体使用 `--calculate-per-token-loss`，因为该参数会把目标改为全局 token mean。

## 公式级对比

定义：

$$
\rho_{i,t}=\exp(\log\pi_\theta(a_{i,t})-\log\pi_{old}(a_{i,t})),
$$

并用 `clip-PPO` 表示上文的 token objective。Relax 的既有 group-relative 算法首先计算：

$$
A_i^{grp}=R_i-\bar R_g,
$$

并在启用 `--grpo-std-normalization` 时除以 `torch.std({R_j:j in g}) + 1e-6`。该 `torch.std` 使用 Bessel 样本校正（`ddof=1`），不同于新增的全局总体方差。

| 算法 | 原始 advantage 与统计轴 | Ratio / policy objective | Reference regularization |
|---|---|---|---|
| REINFORCE++ | token KL-to-go $G_{i,t}$；使用 `ddof=0` 在全部有效 token 和 DP rank 上归一化 | token $\rho_{i,t}$ 和 clip-PPO；response mean | k1 位于 token reward 内 |
| REINFORCE++-baseline | $R_i-\bar R_g$ 广播到有效 token，不除以 group std；随后使用相同的全局 token/DP 归一化 | token $\rho_{i,t}$ 和 clip-PPO；response mean | 独立 k2 loss，response mean |
| GRPO | $A_i^{grp}$，可选同 prompt sample-std 缩放；在 response 内广播 | token $\rho_{i,t}$ 和 clip-PPO | Relax 既有可配置 KL |
| GSPO | 与 GRPO 相同的 group advantage | sequence ratio $\rho_i=\exp[L_i^{-1}\sum_t m_{i,t}(\log\pi_\theta-\log\pi_{old})]$ 扩展到其 token，再使用 clip-PPO | Relax 既有可配置 KL |
| SAPO | 与 GRPO 相同的 group advantage | token ratio 和 $f_\tau(\rho)=4\,\sigma[\tau(\rho-1)]/\tau$；loss 为 $-f_\tau(\rho)A$，正负 advantage 使用不同 $\tau$ | Relax 既有可配置 KL |

对于这五条路径，mask 选择参与计算的 response token，Relax 既有 reducer 先对每个 response 求均值，再对 response 求均值。新增变体拒绝另一种 global-token reduction，避免分母被静默改变。

本功能不改变 GRPO、GSPO 或 SAPO 的默认行为。

## 支持的模式

首个实现支持：

- synchronous colocate training；
- data-parallel normalization；
- `context_parallel_size=1`；
- response-mean loss reduction。

它拒绝 fully-async、hybrid、context parallelism 大于 1，以及 per-token global loss reduction。fully-async 当前没有可用于计算这些 moments 的闭合 global batch；CP 大于 1 则需要单独验证 unique-token ownership 契约。

## 监控指标

Rollout 指标包括：

- 原始 global advantage mean 和 standard deviation；
- normalized advantage mean 和 standard deviation；
- valid-token count；
- zero-variance indicator；
- 常规 reward、return 和 advantage 汇总。

三个 KL 相关观测量具有明确不同的含义：

- `train/ppo_kl` 是用于构造 PPO importance ratio 的 response-reduced old-policy/current-policy log-prob 差值。它度量 policy-update drift，不是 reference-policy KL，也不能说明 k1 或 k2 regularization 是否生效。
- 对 REINFORCE++，reference-policy k1 shaping 已折入 `rollout/returns`。比较同一步的 `rollout/returns` 与 `rollout/raw_reward` 汇总可以观察其实际影响；该变体没有独立 `train/kl_loss`。
- 对 REINFORCE++-baseline，`train/kl_loss` 是单独 reduction 的 k2 reference-policy penalty。它不进入 advantage，而是通过 `--kl-loss-coef` 加到总 loss。

训练指标仍会报告 policy loss 和 clip fraction。

## 测试

数值测试使用独立 float64 参考实现，不调用生产环境的 return、advantage、归一化或 loss 函数。覆盖变长 response、padding、内部 mask hole、mask 外的有限与非有限 sentinel、全零 reward、零方差、单个有效 token、本地全 mask rank、PPO clipping，以及 response-reduced policy/k2 loss。Megatron backend 集成测试还会调用生产环境的 `compute_advantages_and_returns` dispatcher。在宿主环境缺少 Megatron 时，测试只注入该函数所需的最小 `mpu` 接口，因此仍执行真实的 Relax dispatch 和 normalization 代码，而不是 mock 这些代码。

分布式归一化使用两个真实 Gloo 进程和真实 `all_reduce` 测试，包括一个 rank 没有有效 token 的情况。输出与独立拼接得到的全局总体进行比较。

参数化 Qwen3-0.6B recipe 参见：
`examples/algorithms/run-qwen3-0.6B-1xgpu-reinforce-plus-plus.sh`。

相同预算的 Qwen3-0.6B 稳定性实验、数值证据、曲线及与 GRPO 的对比记录在
[训练与数值验证报告](./reinforce-plus-plus-training-report.md)中。
