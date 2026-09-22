# 轨迹重放（Trajectory Replay）

> 状态：离线重放核心（PR A）与生产捕获（PR B，含 `loss.policy` 热路径 hook、异步 `CaptureManager` 与 GPU smoke）已实现；分布式布局（PR C）已落地。捕获拆分为两类 bundle：**rollout 级**（`reward.raw → reward.post_process → advantage.kl → advantage.estimate`，身份 = `rollout_id`）与 **step 级**（`loss.policy`，身份 = `(rollout_id, step_id)`）；`advantage.kl` 使用 rollout-vs-reference 语义，需 `ref_log_probs` payload。bundle 可由训练侧自动产出，也可通过 `BundleWriter` 以编程方式构造。

轨迹重放是面向异步训练的可验证离线复现工具：它把一次训练 step 实际消费的数据、阶段输出与身份信息打包成**自描述、可校验的 replay bundle**，在不启动 Ray Serve、SGLang、Rollout Worker 或 GPU 的普通 CPU 环境里，按同样的阶段语义重算 reward、advantage、return 与 loss，并报告**首个可验证的分歧位置**。

模型 forward、backward、gradient 与 optimizer step 不在重放边界内。系统保存 loss 实际消费的紧凑 post-forward 统计量（current/old log-probabilities、entropy、values、masks、advantages、returns），而不是完整 logits 或 checkpoint。

## 为什么需要它

在同步训练里 rollout step 与 Actor update 近似一一对应；在 fully-async / hybrid 模式下这个假设不成立：

- Rollout 可以领先 Actor 多个权重版本；
- 一个 rollout partition 可能被拆成多个 TransferQueue consumer batch；
- dynamic batching 会改变 micro-batch 边界；
- 一次 Actor update 可能消费来自多个 rollout partition 的样本。

因此 replay 把「一起生成」「一起归一化」「一起被消费」视为彼此独立的身份。历史故障 PR #65（streamed prompt group 被错误地按物理 batch 做 reward 归一化）就是「物理传输 batch ≠ 语义归一化 cohort」的典型案例：最终 loss 会变，但**首个错误阶段是 `reward.post_process`**。轨迹重放正是用来定位这类错误，而不是只比较最终 scalar。

## Bundle 结构

一个 bundle 是一个目录：

```text
<bundle>/
├── manifest.json    # 版本、producer、stage contract、payload 清单与 checksum、比较策略
├── index.json       # 身份 + per-sample 记录 + 重算配置
├── expected.json    # 各阶段的 JSON 期望输出（标量/列表）
├── payloads/        # tensor payload（输入与期望张量）
│   ├── old_log_probs.pt
│   ├── log_probs.pt
│   ├── entropy.pt
│   ├── kl.pt
│   └── advantages.pt
└── COMPLETE         # 完成哨兵（多 rank 时为 COMPLETE.<rank> + 最终 COMPLETE）
```

- **文本走 JSON**：`index.json` 只存 JSON 兼容元数据（sample_id、group_index、长度、loss_mask、raw_reward 等），不存 prompt/response 原文（按 redaction policy 以 hash/截断代替）。
- **张量走 `weights_only=True`**：`payloads/` 只允许 `torch.Tensor`，读取时 `weights_only=True` 并递归拒绝 Python object。
- **完整性**：writer 先写临时目录、再写 payload 与 manifest hash、最后原子 rename；缺少 `COMPLETE`、payload、rank shard 或 checksum 不一致时，在 stage 执行前失败。

## 身份模型

重放身份分两层：

- **逻辑身份**：sample、semantic group、normalization cohort、Actor step。
- **物理 provenance**：rollout partition、consumer batch、micro-batch、rank shard、weight lineage。

training step 的身份基准是 **Actor 实际消费的完整 cohort**。`actor_step_id` 用 `(rollout_id, step_id)` 二元组（对应 `train_one_step` 的入参坐标）；派生的 `accumulated_step_id` 标量在 dynamic batching / streaming schedule 下 `num_steps_per_rollout` 可变，因此**不作为持久化身份**。

选择器在缺失 membership 映射时**拒绝执行**，而不是用物理 batch 大小猜测语义 group（这正是 PR #65 故障的根因）。

## 阶段契约与 V1 capability

流水线拆成独立版本化的 stage：

```text
sample -> reward.raw -> reward.post_process -> advantage.kl -> advantage.estimate -> loss.policy
```

每个 stage 声明 capability：

| Capability | 含义 |
| --- | --- |
| `recompute` | 输入与实现完整，可离线重算并比较 |
| `recorded-only` | 只能查看生产输出，不能可靠重算 producer |
| `inspect-only` | 只能查看部分输入或摘要，contract 不完整 |
| `unsupported` | capture 或 reader 不支持该阶段 |

**冻结的 V1 capability matrix**：只有 **GRPO、`CP=1`** 的 `sample / reward.raw / reward.post_process / advantage.kl / advantage.estimate / loss.policy` 声明 `recompute`。其余拓扑（`CP>1`、PPO、SAPO/CISPO、OPD、Agentic flattened）一律 `unsupported`，在 runner 中显式失败，不做 best-effort 猜测或静默降级。

advantage/loss 适配器**复用 production kernel**（`relax/utils/training/ppo_utils.py` 的 `compute_approx_kl` / `get_grpo_returns` / `compute_policy_loss`），单一事实来源。`reward.post_process` 因生产函数位于顶层 `import ray` 的模块中，离线侧以重实现方式提供（manifest 标记 `implementation="reimplemented"`），并由 PR #65 fixture 钉住语义。

## CLI 用法

```bash
# 查看身份、capability、producer 与 payload 摘要（无需 COMPLETE，可查看不完整 bundle）
python -m relax.tools.trajectory_replay inspect <bundle>

# 校验格式、完整性、安全性与 dependency closure（不执行数值重放）
python -m relax.tools.trajectory_replay validate <bundle>

# 按 stage DAG 重算并比较期望输出；退出码 0 表示通过，1 表示存在分歧
python -m relax.tools.trajectory_replay replay <bundle> [--stage all]

# 选择单条 / 单 group / 单 micro-batch 重放（可重复），或断言 step 坐标
python -m relax.tools.trajectory_replay replay <bundle> \
    [--sample s-0] [--group g-1] [--batch mb-0007] [--step 120:0] [--rollout 120]
```

`replay` 的分歧报告给出：首个分歧 stage、sample、field、token offset、expected/actual 值与最大绝对误差。被跳过（`recorded-only` / `inspect-only` / `unsupported`）的 stage 不计入分歧判定。

**选择粒度**：`--sample`/`--group`/`--batch` 任选其一或多个。选择任何 sample 或 micro-batch 都会**展开到其完整 semantic-group closure**（因为 reward/advantage 归一化是 group 级），缺失 membership 时拒绝执行。部分选择下，per-sample/per-token 阶段（sample → advantage.estimate）正常重算，而 **cohort 级阶段（`loss.policy`）会跳过**——子集 scalar 无法与全 cohort 的期望值相比。`--step ROLLOUT_ID:STEP_ID` 只接受精确的 `actor_step_id=(rollout_id, step_id)` 匹配；rollout 级包用 `--rollout ROLLOUT_ID`。两者互斥。若路径是抓包目录（含多个 bundle 或 `rank-*` 子目录），则选出匹配的那一份。`--batch` 按 DataIterator 的 micro-batch 序号选择（`mb-0000`、`mb-0001`、…），不是 actor `step_id`。

示例（PR #65 历史故障）：

```text
bundle b-00001 — first divergent stage: reward.post_process
[   pass] sample
[   pass] reward.raw
[   fail] reward.post_process — normalized reward mismatch in 4 sample(s)
          reward s-0: expected=-5.5 actual=-1.0 abs_err=4.5
          ...
```

## 启用生产捕获

捕获默认关闭，不改变训练数值。通过环境变量打开（不改 CLI 参数解析）；Actor 在
`MegatronTrainRayActor.init` 里调用 `maybe_enable_for_actor`，只在 last pipeline
stage 的 rank 上启用捕获（这些 rank 持有 post-forward payload），其余 rank 保持沉默。
单 producer 写 `<DIR>/<bundle>/`；多 producer 写 `<DIR>/<bundle>/rank-<rank>/`，各
producer 的 writer 线程在写完 `COMPLETE.<rank>` 后 **try-finalize**（缺人就返回，
到齐才写最终 `COMPLETE`），训练线程不等待。rank-local 的 `COMPLETE` 会写入
`expected_ranks`；`validate` / `replay`（以及 `BundleReader`）在父目录最终
`COMPLETE` 出现前拒绝多 rank 的 `rank-*` 路径，直接传入 `rank-0/` 也不能绕过：

```text
<DIR>/
  replay-0-0/
    COMPLETE.0          # 该 rank 的身份、owned payloads、checksum
    COMPLETE.1
    COMPLETE            # 全部预期 rank 到齐后才出现
    rank-0/             # rank-local bundle；多 rank replay 需要父目录 COMPLETE
    rank-1/
```

```bash
export RELAX_REPLAY_CAPTURE=1
export RELAX_REPLAY_CAPTURE_DIR=/path/to/replay-bundles
# 可选：只抓指定 Actor step / rollout（逗号分隔）。未设置则抓全部。
export RELAX_REPLAY_CAPTURE_STEPS=0:0
export RELAX_REPLAY_CAPTURE_ROLLOUTS=0
```

打开后训练会写出两类 bundle：rollout 级（reward / advantage）和 step 级（`loss.policy`）。再用上面的 CLI 离线 `inspect` / `validate` / `replay`。

## 编程构造 bundle

除训练侧自动产出外，仍可用 `BundleWriter` 手工构造 bundle 用于离线验证（fixture、注入故障、跨版本校验）：

```python
import torch
from relax.utils.replay.bundle import BundleWriter
from relax.utils.replay.schema import (
    ActorStepId, BundleIndex, Identity, Manifest, ProducerInfo,
    RecomputeConfig, SampleRecord, StageCapability, StageContract, StageId,
)

index = BundleIndex(
    bundle_id="b-00001",
    identity=Identity(actor_step_id=ActorStepId(rollout_id=120, step_id=0), rank={"cp": 1}),
    samples=[SampleRecord(sample_id="s-0", group_index=0, response_length=2, total_length=3,
                          loss_mask=[1, 1], raw_reward=1.0, reward=1.0), ...],
    config=RecomputeConfig(advantage_estimator="grpo", n_samples_per_prompt=2),
)
manifest = Manifest(
    format_version="1.0.0", bundle_id="b-00001",
    producer=ProducerInfo(commit="...", torch_version=torch.__version__),
    stage_contracts={stage: StageContract(stage=stage, version="v1", capability=StageCapability.RECOMPUTE)
                     for stage in (StageId.SAMPLE, StageId.REWARD_RAW, StageId.REWARD_POST_PROCESS,
                                   StageId.ADVANTAGE_KL, StageId.ADVANTAGE_ESTIMATE, StageId.LOSS_POLICY)},
    payloads={}, comparison_policy=..., redaction={"prompt": "hash"},
)
expected = {"reward.raw": {"raw_rewards": [...]}, "reward.post_process": {"rewards": [...]},
            "loss.policy": {"loss": ...}}

writer = BundleWriter("<path>", manifest, index, expected)
writer.write_payload("old_log_probs", old_log_probs)
writer.write_payload("log_probs", log_probs)
writer.write_payload("entropy", entropy)
writer.write_payload("kl", kl)
writer.write_payload("advantages", advantages)
writer.finalize(ranks=[0])
```

## 安全与脱敏

- 公共 artifact 只允许 JSON 元数据 + tensor payload；tensor 用 `weights_only=True` 加载。
- prompt/response/label/tool output/路径/endpoint 按显式 redaction policy 处理（`hash` / `truncate` / `drop`）。
- 不保存完整 `Namespace`、环境变量、secret、checkpoint 或服务凭据。

## 当前限制

- **不重放** model forward/backward/gradient/optimizer；不保证重新采样得到相同 trajectory；不要求跨硬件/编译器 bitwise identical。
- **V1 仅支持 GRPO `CP=1`**；`CP>1`、PPO value loss、OPD、Agentic flattened 为 `unsupported`。
- 生产捕获拆成两类 bundle：rollout 级（reward/advantage，身份 `rollout_id`，在 `train_actor` 打点）与 step 级（loss，身份 `(rollout_id, step_id)`，在 `train_one_step` 打点）；两者尚未在同一 bundle 中打通「首个分歧阶段」的跨 bundle 传播。通过 `RELAX_REPLAY_CAPTURE=1` 与 `RELAX_REPLAY_CAPTURE_DIR` 打开。
- 远端 RM/GenRM 结果按 `recorded-only` 处理，不在离线侧重算。

相关设计讨论见 [Task 34 RFC #171](https://github.com/redai-studio/Relax/issues/171)。
