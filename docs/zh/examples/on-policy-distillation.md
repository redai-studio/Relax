# 在线策略蒸馏 (OPD)

在线策略蒸馏 (OPD) 通过在学生模型自身的回滚数据上训练学生，同时匹配教师的词元级对数概率，实现从大型教师模型到小型学生模型的知识迁移。OPD 与优势估计器正交——它作为 KL 惩罚项，可以与任何估计器结合使用，包括 PPO、GRPO、GSPO、SAPO、CISPO 和 REINFORCE++。

## 关键参数

| 参数                      | 描述                                                                                                             |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `--use-opd`               | 启用在线策略蒸馏。使用 OPD 时需要此标志。                                                                        |
| `--opd-type`              | OPD 类型：`sglang` 或 `megatron`。启用 `--use-opd` 时必须设置。                                                  |
| `--opd-token-selection`   | Token 选择模式：`student_sampled`（默认）、`student_topk`、`teacher_topk`、`union`。                            |
| `--opd-kl-coef`           | Advantage 模式的 KL 系数（默认 1.0）。设置时 `--opd-loss-coef` 必须为 0。                                        |
| `--opd-loss-coef`         | Loss 模式的 KL 系数（默认 0.0）。设置时 `--opd-kl-coef` 必须为 0。                                               |
| `--opd-kl-type`           | KL 散度类型：`reverse_kl`（默认）、`forward_kl`、`low_var_kl`、`jsd`。                                           |
| `--opd-jsd-alpha`         | JSD 混合系数（默认 0.5）。0.0 等价于 reverse_kl，1.0 等价于 forward_kl。                                         |
| `--opd-log-prob-top-k`    | Top-K 候选集合大小（设为 `0` 可关闭，默认 `0`）。                                                               |
| `--opd-norm-mode`         | Top-K 尾部处理方式：`tail`（默认）、`norm`、`trunc`。                                                            |
| `--opd-per-token-clip`    | Per-token KL 的硬上界（可选）。                                                                                  |
| `--opd-is-clip`           | Importance sampling ratio 的硬上界（可选，仅 loss 模式）。                                                       |
| `--opd-teacher-load`      | 教师模型路径。当 `--opd-type=megatron` 时**必须**设置，当 `--opd-type=sglang` 时**不能**设置。                   |
| `--opd-teacher-ckpt-step` | 教师模型的可选检查点步骤。                                                                                       |
| `--opd-teacher-timeout-s` | SGLang 模式下 OPD teacher HTTP 请求超时（秒），默认 `30`。                                                      |
| `--opd-only-reward`       | 仅保留 OPD 奖励信号（将基础 RL reward 置零，只使用 OPD KL 项）。需配合 `--use-opd`。                            |

## 工作原理

OPD 通过计算教师与学生之间的 token 级 KL 散度，将蒸馏信号注入训练。Relax 支持两种注入方式：

- **Advantage 模式（adv）**：将 KL 从 advantage 中减去（通过 `--opd-kl-coef` 设置）
- **Loss 模式（loss）**：将 KL 作为额外 loss 项（通过 `--opd-loss-coef` 设置）

两种方式只能选其一，不能同时启用。OPD 与优势估计器正交，可以与任何估计器结合使用，包括 PPO、GRPO、GSPO、SAPO、CISPO 和 REINFORCE++。

## Token-Selection 模式

OPD 支持四种 token-selection 策略，决定在哪些 token 上计算 KL 散度：

| 模式 | 学生自 top-K | 教师自 top-K | 教师 @ 学生 top-K | 学生 @ 教师 top-K | 说明 |
| --- | --- | --- | --- | --- | --- |
| `student_sampled` | — | — | — | — | 仅在学生采样的 1D token 上计算 KL，开销最小 |
| `student_topk` | ✅ | — | ✅ | — | 在学生 top-K token 集合上计算 KL |
| `teacher_topk` | — | ✅ | — | ✅ | 在教师 top-K token 集合上计算 KL |
| `union` | ✅ | ✅ | ✅ | ✅ | 在学生和教师 top-K 的并集上计算 KL，覆盖最全面 |

通过 `--opd-token-selection` 指定模式，`--opd-log-prob-top-k` 指定 top-K 大小。除 `student_sampled` 外，其余模式需要设置环境变量 `RELAX_OPD_PER_POS_TOKEN_IDS=1`。

## 两种应用方式：adv 与 loss

### Advantage 模式（adv）

通过 `--opd-kl-coef` 设置（此时 `--opd-loss-coef` 必须为 0）。在 advantage 计算后，将 per-token KL 从 advantage 中减去：

$$\hat{A}_t = A_t - \lambda_{\text{opd}} \cdot D_{\text{KL}}(P_{\text{teacher}} \| P_{\text{student}})_t$$

特点：

- KL 项使用 `.detach()`，**不产生梯度**
- 仅影响 advantage 估计，不改变 loss 函数形式
- 与任何优势估计器（PPO、GRPO、GSPO、SAPO、CISPO 等）正交

架构流程：

```
Rollout 阶段:
  学生 Rollout → 学生 top-K token IDs / log-probs
  教师 Prefill → 教师 log-probs / 教师 top-K
  学生 Prefill → 学生 @ 教师 top-K log-probs（adv 模式独有）
      ↓
  组装训练数据 (opd_topk_token_ids, opd_topk_student_log_probs, opd_topk_teacher_log_probs)

Training 阶段:
  compute_advantages_and_returns()
    → apply_opd_to_advantages()
    → 修改 advantage: adv = adv - opd_kl_coef * kl_term.detach()
```

### Loss 模式（loss）

通过 `--opd-loss-coef` 设置（此时 `--opd-kl-coef` 必须为 0）。在 policy loss 计算中，将 per-token KL 作为额外的 loss 项：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PG}} + \lambda_{\text{loss}} \cdot \mathbb{E}_t[D_{\text{KL}}(P_{\text{teacher}} \| P_{\text{student}})_t]$$

特点：

- KL 项**产生梯度**，直接影响策略梯度方向
- 支持 per-token clipping（`--opd-per-token-clip`）和 importance ratio clipping（`--opd-is-clip`）
- 与 advantage 估计器无关

架构流程：

```
Rollout 阶段:
  学生 Rollout → 学生 top-K token IDs / log-probs
  教师 Prefill → 教师 log-probs / 教师 top-K
      ↓
  组装训练数据 (opd_topk_token_ids, opd_topk_teacher_log_probs)

Training 阶段:
  policy_loss_function()
    → get_log_probs_and_entropy()（收集学生 top-K log-probs）
    → compute_policy_opd_loss()
    → 计算 KL → clipping → reduce
    → loss = loss + opd_loss_coef * opd_loss
```

> **注意**：adv 和 loss 两种方式只能选其一，不能同时启用。
