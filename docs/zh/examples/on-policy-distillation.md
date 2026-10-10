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
| `--opd-teacher-url`       | 自行部署的 SGLang teacher 的 `/generate` 地址。见 [SGLang Teacher 的部署](#sglang-teacher-的部署)。             |
| `--teacher-hf-checkpoint` | 由 Relax 拉起并托管的 teacher 的 HF checkpoint。需要在 `--resource` 中配置 `teacher` 项。                       |
| `--teacher-num-gpus-per-engine` | 每个 teacher 引擎的 GPU 数（即 TP 大小）。默认使用全部 teacher GPU，即单副本。                            |
| `--opd-teacher-routes`    | 数据源到 teacher checkpoint 的 JSON 映射，用于托管多个 teacher（仅 colocate）。                                 |
| `--opd-teacher-key`       | 使用 `--opd-teacher-routes` 时用来选择 teacher 的样本 metadata 字段（默认 `data_source`）。                     |
| `--opd-teacher-defer`     | 等一批样本生成完再向托管 teacher 请求，而不是在生成过程中请求（仅 colocate）。                                  |
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

## SGLang Teacher 的部署

`--opd-type sglang` 时，teacher 是一个 SGLang 服务，为学生的 rollout 返回对数概率。可以让 Relax 访问你自己部署的服务，也可以让 Relax 来拉起 teacher。

**外部 teacher。** 传入 `--opd-teacher-url http://teacher-host:30001/generate`。Relax 只向它发请求，服务的启动、规格和停止都由你负责。

**托管 teacher。** 传入 `--teacher-hf-checkpoint`，并在 `--resource` 中加上 `teacher` 项。Relax 会在其他服务之前拉起 teacher 引擎，并自行填入 teacher 的地址。teacher 的 GPU 按 `--teacher-num-gpus-per-engine` 切成若干副本，请求在副本之间分摊，同一个 prompt 组的样本固定发往同一个副本。

```bash
python3 relax/entrypoints/train.py \
    --colocate \
    --resource '{"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4]}' \
    --rollout-num-gpus 4 \
    --use-opd --opd-type sglang \
    --teacher-hf-checkpoint /path/to/teacher \
    --teacher-num-gpus-per-engine 2 \
    ...
```

`--colocate` 下 teacher 位于 actor 的 placement group 内，紧接在 Rollout 之后的 bundle 上，因此 `rollout GPU 数 + teacher GPU 数` 必须等于 actor 的 GPU 数。Actor 训练时 teacher 被卸载，之后与 Rollout 的权重一起重新加载。不使用 `--colocate` 时，每个 teacher 副本各有一个自己的 placement group。`--opd-teacher-routes`（每个数据源一个 teacher）只能在 `--colocate` 下使用，teacher 的 GPU 在各个 teacher 之间平均分配。

teacher 引擎启动之前，Relax 会检查每个引擎的 GPU 是否位于同一个节点、GPU id 是否连续。不满足时训练以 "Physical placement mismatch" 报错退出，报错信息会指出是哪个引擎以及它的 bundle 实际落在哪里。

### 延迟 Teacher 打分

默认情况下，teacher 在 Rollout 生成的过程中被请求，两者同时占用显存，所以必须使用不同的 bundle。`--opd-teacher-defer` 把 teacher 挪到生成之后：Rollout 生成期间 teacher 保持休眠，一批样本生成完成后，Relax 卸载 Rollout、加载 teacher，为整批样本请求一次，再把 teacher 卸载。teacher 的对数概率写回样本之后，这批样本才会发布给训练；如果 teacher 请求全部失败，这批样本不会发布，训练会报错。

由于两者不再同时加载，它们可以共用 bundle。colocate 下接受两种布局：

| 布局   | 条件                                              | Teacher 使用的 bundle     |
| :----- | :------------------------------------------------ | :------------------------ |
| Split  | `rollout GPU 数 + teacher GPU 数 == actor GPU 数` | Rollout 之后的 bundle     |
| Shared | `rollout GPU 数 == teacher GPU 数 == actor GPU 数` | 与 Rollout 相同的 bundle  |

不加 `--opd-teacher-defer` 时 Shared 布局会被拒绝。如果还需要学生在 teacher 的 top-K token 上计算（advantage 模式下的 `teacher_topk` 或 `union`），Relax 会在 teacher 卸载后重新加载 Rollout，再完成这一步。评测批次不会发给 teacher。

`--opd-teacher-defer` 要求在 `--colocate` 下使用托管 teacher。使用外部 teacher、不使用 `--colocate`，或与 `--use-agentic-rollout` 同时使用时，启动时会被拒绝。

advantage 模式下，延迟评分的 `teacher_topk` 和 `union` 要求样本来自当前学生策略。请关闭 `--partial-rollout` 和 `--dynamic-sampling-filter-path`，并保持 `--over-sampling-batch-size` 与 `--rollout-batch-size` 相等（默认行为）。也可以同时启用 `--partial-rollout` 和 `--mask-offpolicy-in-partial-rollout`，让跨轮携带的响应 token 不参与训练。其他组合会在启动前报错，因为跨轮缓存的样本可能在学生 prefill 之前经历权重更新，混用不同策略的概率。这个限制不适用于 loss 模式、`student_sampled` 和 `student_topk`。

### Teacher 的接口

托管 teacher 可以通过 Ray Serve HTTP 端口上的 `/teacher` 路由访问，与 `/rollout`、`/genrm` 并列。没有配置托管 teacher 时不存在这个路由。

| 接口                                                              | 用途                                                    |
| :---------------------------------------------------------------- | :------------------------------------------------------ |
| `GET /teacher/health`                                             | 每个 teacher 模型的状态（`ready`、`sleeping` 等）       |
| `GET /teacher/engines`                                            | 拓扑：每个 teacher、它的副本、各副本的地址与状态        |
| `GET /teacher/v1/models`                                          | teacher 模型列表，OpenAI 模型列表格式                   |
| `POST /teacher/generate`                                          | SGLang 原生的 `/generate` 请求体，转发给一个就绪的副本  |
| `POST /teacher/v1/chat/completions`、`/teacher/chat/completions`  | OpenAI 风格的聊天请求，转发给一个就绪的副本             |

```bash
curl http://localhost:8000/teacher/health
```

使用 `--opd-teacher-routes` 时，用 `model`（或 `route_key`）填数据源来指定 teacher；只有一个 teacher 时两者都不需要。请求的 teacher 处于卸载状态时返回 `503` 并带 `Retry-After` 响应头，不会因此被唤醒；`model` 对不上任何 teacher 时返回 `400` 并列出可用的 teacher。Relax 自己也通过 `/teacher/engines` 查找副本，所以某个 teacher 引擎在另一个地址上重建之后，不需要重启训练就会被使用。
