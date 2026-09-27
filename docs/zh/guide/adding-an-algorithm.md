# 接入一个新算法

Relax 的算法通过 `relax/algorithms/` 下的注册表接入。一个算法名不再散落在各处的 `if/elif` 里——它由一条 `AlgorithmSpec` 描述，各个阶段按需查表。

## 注册表的结构

```
relax/algorithms/
├── spec.py        AlgorithmSpec 定义 + ALGORITHM_SPECS 注册表
├── rewards.py     reward 归一化策略 + REWARD_NORMALIZERS
├── advantages.py  advantage 估计器 + ADVANTAGE_FNS
└── policy.py      policy loss 适配器 + POLICY_LOSS_FNS
```

三条硬约束：

1. **`relax/algorithms/` 下禁止顶层 import 重依赖**——不能有 `megatron`、`ray`、`transfer_queue`、`tensordict`、`relax.components`、`relax.backends`。注册表会被参数解析和两个 worker 进程 import；一个重依赖会把整个训练栈拖进 `--help` 和只有 CPU 的 CI。确实需要时在函数内 import。
2. **spec 的字段存字符串标识符，不存函数引用**。advantage 计算跑在 Ray Serve 的 `Advantages` 进程，policy loss 跑在 Megatron worker 进程，两者 import 的模块子集不同。跨进程只传算法名，各进程本地查表。
3. **`ALGOS` 角色表不用手改**。它从注册表自动派生，新算法自动获得标准 RL 角色集合。

## 接入一个新算法要改多少

先说实话：**不是「加一条 dict entry」就完事**。

| 情况 | 要改的文件 |
|---|---|
| 复用现成的 reward 归一化 / advantage / policy loss，只是组合方式不同 | 1 个（`spec.py`） |
| 需要一种新的数学（如新的 advantage 公式） | 2–3 个（`spec.py` + 对应的实现模块） |
| 还需要新的命令行参数 | 4–6 个（上述 + `arguments.py` 的参数声明与校验 + 示例 + 文档） |

注册表消除的是「同一个算法名散落在 6 处 if/elif」，不是「新增算法零成本」。

`ALGOS` 角色表是唯一真正做到零改动的部分——它从注册表自动派生。

## 步骤

### 1. 加一条 spec

编辑 `relax/algorithms/spec.py` 的 `ALGORITHM_SPECS`：

```python
"my_algo": AlgorithmSpec(
    name="my_algo",
    reward_normalizer="group_mean_std",   # 复用现成的，或见第 2 步
    requires_complete_reward_groups=True,
    advantage_fn="grpo_broadcast",
    policy_loss_fn="ppo_clip",
),
```

如果新算法在某个阶段与已有算法完全一致，直接复用那个标识符即可。例如 GRPO / GSPO / SAPO / CISPO / M2PO / RLOO 在 advantage 层都广播标量 reward，因此共享 `"grpo_broadcast"`。它们的 reward 预处理不同：M2PO 保持 `reward_normalizer="none"`，以保留既有行为。

可用的能力字段：

| 字段 | 作用 |
|------|------|
| `requires_complete_reward_groups` | reward 处理依赖组级统计时，在 debug 子采样中保留完整 prompt 组；目前由 debug 数据选择逻辑消费 |
| `kl_level` | `"token"` 或 `"sequence"`（GSPO 用序列级） |
| `needs_full_log_probs` | loss 是否需要 CP all-gather 后的完整 log probs |
| `supports_context_parallel` | 算法的 advantage 和 policy 路径是否支持 CP 切分的 response；默认为 `True`。设为 `False` 会在启动时拒绝不等于 1 的静态 CP size，以及开启动态 CP 的配置 |
| `policy_scalar_metric_names` | policy adapter 额外返回的标量诊断项名称，顺序与返回值一致 |
| `advantage_normalization` | `--normalize-advantages` 的归一化方式：`"whiten"`（掩码白化）或 `"token_global"`（REINFORCE++ 的全局 token 级归一化，同时切换掩码安全的 loss reducer） |
| `needs_critic` | 是否需要 critic 服务，驱动 `args.use_critic` |
| `requires_normalize_advantages` | 强制要求 `--normalize-advantages` |
| `forbids_normalize_advantages` | 禁止 `--normalize-advantages`（算法刻意保留了 advantage 的尺度时） |
| `requires_rewards_normalization` | 禁止 `--disable-rewards-normalization` |
| `min_group_size` | `--n-samples-per-prompt` 的下限 |
| `forbids_reward_side_kl` | 要求 `--kl-coef 0`（reward 侧 KL 项无处可放；`--use-kl-loss` 不受影响） |
| `requires_global_token_loss` | 强制要求 `--calculate-per-token-loss`（否则按样本取 token 均值，会按 `1 / response_length` 重新加权） |
| `requires_on_policy_updates` | 一次性拒绝五项：`--fully-async` / `--hybrid`、`--max-staleness != 0`、`--num-steps-per-rollout != 1`、`rollout_batch_size * n_samples != global_batch_size`、`--partial-rollout` / `--use-dynamic-global-batch-size`。适用于没有重要性比值修正的目标函数 |

M2PO 和两个 REINFORCE++ 变体均声明 `supports_context_parallel=False`。它描述整个算法，包括 advantage 计算和 policy loss，与 `requires_complete_reward_groups` 相互独立：后者关注同一 prompt 下的样本组，不是一个 response 内的 token。M2PO 的 reward 阶段没有组归一化，因此仍保持 `requires_complete_reward_groups=False`。

`relax/utils/arguments.py` 的 `validate_*` 函数统一消费启动约束字段。其余能力由运行时消费者读取：reward dispatch 使用 `reward_normalizer`，debug 子采样使用 `requires_complete_reward_groups`，policy 路径使用 KL、归一化、完整 log-probability 与标量指标声明。复用已有能力时只需声明，不要再给这些消费者增加算法名 `if`；只有新增一种前所未有的枚举值或实现时，才需要为该值增加一个通用处理器。

### 2. 需要新公式时，写纯函数并登记

只有当新算法在某个阶段的数学与现有算法都不同时才需要这一步。

**Reward 归一化**（`relax/algorithms/rewards.py`），签名固定为 `fn(args, samples, raw_rewards) -> list[float]`：

```python
def normalize_my_strategy(args, samples, raw_rewards):
    positions_by_group = group_positions(samples, args.n_samples_per_prompt)
    ...
    return normalized  # 每个 sample 一个标量

REWARD_NORMALIZERS["my_strategy"] = normalize_my_strategy
```

产出必须是**每个 sample 一个标量**。这条约束让 TransferQueue 的 schema 保持不变——即使算法内部要看多个奖励分量，也要在这一层收敛成一个标量。

**Advantage 估计器**（`relax/algorithms/advantages.py`），签名 `fn(args, *, rewards, kl, loss_masks, response_lengths, total_lengths, values) -> (advantages, returns)`，两者都是 `list[Tensor]`：

```python
def advantage_my_algo(args, *, rewards, kl, **_unused):
    ...
    return advantages, returns

ADVANTAGE_FNS["my_algo"] = advantage_my_algo
```

**Policy loss**（`relax/algorithms/policy.py`），签名 `fn(args, *, log_probs, ppo_kl, advantages) -> (pg_loss, pg_clipfrac, *scalar_metrics)`。底层算子签名不一致，适配器负责统一。大多数 adapter 只返回前两个值；若要额外返回标量诊断项，应按相同顺序在 `policy_scalar_metric_names` 中声明名称。每个诊断项必须恰好包含一个值。标量日志保留 M2PO 的既有约定：sample 模式下直接累加每个 microbatch 的标量，再由框架除以 sample 数；token 模式下先乘以该 microbatch 的 token 数。因此它不是 microbatch 诊断值按 sample 加权的均值。实数诊断项会统一转换为 float32，避免单个 adapter 提升整条分布式日志向量的 dtype；复数会被拒绝。

### 3. 写单测

`tests/algorithms/` 下的测试使用 pytest、torch、NumPy 和 PyYAML，即可在 CPU 上运行。
测试 fixture 隔离了无关的训练依赖，无需安装 megatron、ray、tensordict 或 transfer_queue：

```bash
pytest tests/algorithms/ -v
```

至少覆盖：

- 注册与分发：算法名在 `ALGORITHM_SPECS` 里；能力字段与预期一致；未注册名报错。
- 数值：手算一个小例子做对照，别用全零或全相同的 reward——那种输入下任何公式都输出 0，测不出东西。
- 退化场景：组内 reward 全相同、`n_samples_per_prompt` 取边界值、缺字段、非数值输入。
- **改动已有算法时**：把旧实现冻结进测试文件当参照，逐位对拍（`view(torch.int32).equal`），不要用 `allclose`——它的默认容差足以吞掉无偏/有偏标准差的差异。`tests/algorithms/test_reward_normalizers.py` 是现成范例。

### 4. 加示例与文档

- `examples/<algo>/`：启动脚本，必要时附自定义 reward 函数。
- `docs/{zh,en}/examples/algorithms.md`：算法原理、关键参数表、快速开始，以及**已知偏差**——实现与论文不一致的地方要写出来，不要留给使用者去发现。

## 参数

新增算法专用参数时改 `relax/utils/arguments.py` 的 `add_algo_arguments`。`--advantage-estimator` 的 `choices` 由 `list_algorithm_names()` 生成，注册即可用，不需要手动维护名单。

跨参数的校验写进 `validate_algorithm_args`，并优先用 spec 字段表达而不是比较算法名——后者正是这套注册表要消除的东西。

## 参考

- [算法参考](../examples/algorithms.md)
