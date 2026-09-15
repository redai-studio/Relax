# 全异步训练流水线

## 概述

**全异步训练流水线（Fully Async）** 是一种以最大化 GPU 利用率为目标的高吞吐 RLHF/RL 训练模式。与 Colocate（同步）模式不同，Fully Async 将 **训练（Actor）**、**推理（Rollout）**、**前向计算（ActorFwd / Reference）** 和 **优势计算（Advantages）** 部署在独立的 GPU 集群上，服务间通过 TransferQueue 交换数据，通过 DCS（Distributed Checkpoint Service）异步同步权重。

### 模式对比

| 维度          | Colocate（同步）                                                  | Fully Async（全异步）                                                |
| ------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------- |
| **GPU 共享**  | Actor 与 Rollout 共享同一组 GPU                                   | Actor、Rollout、ActorFwd/Reference 各自拥有独立 GPU                  |
| **执行模型**  | 串行：Rollout 完成 → 切换到 Train → 更新权重                      | 全并行：Rollout、Train、前向计算同时进行                             |
| **权重同步**  | 进程内 tensor 拷贝（同机器）                                      | 跨节点 NCCL broadcast，通过 DCS（Checkpoint Engine）                 |
| **数据流**    | 同样走 TransferQueue，但同步：Rollout 写完整批数据后 Actor 才读取 | 走 TransferQueue + StreamingDataLoader 异步流式传输（生产消费重叠）  |
| **Staleness** | `max_staleness=0`（严格 On-Policy）                               | 可配置 `max_staleness`（允许一定程度 Off-Policy）                    |
| **角色**      | `actor`, `critic`, `rollout`                                      | `actor`, `critic`, `rollout`, `advantages`, `reference`, `actor_fwd` |

::: tip
两种模式都使用 TransferQueue 作为数据传输层。Colocate 模式下 Rollout 和 Actor 分时复用同一组 GPU——Rollout 写完整批数据到 TransferQueue 后释放 GPU，Actor 接管 GPU 进行训练。Fully Async 模式下各服务运行在独立 GPU 上，数据的生产和消费可以并行进行。
:::

### 核心优势

1. **消除 GPU 空闲时间** — Rollout 和 Training 同时运行，推理引擎在训练期间持续生成数据
2. **灵活的资源配比** — 训练和推理可以使用不同数量的 GPU，适应异构硬件
3. **可控的 On/Off-Policy 程度** — `max_staleness` 参数精确控制数据新鲜度
4. **流水线化的权重更新** — DCS 使权重分发与训练计算重叠

______________________________________________________________________

## 系统架构

### 架构图

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        Controller (Orchestrator)                          │
│                     relax/core/controller.py                              │
│                                                                           │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌─────────┐  │
│    │ Rollout  │  │  Actor   │  │ ActorFwd │  │ Reference  │  │  Adv    │  │
│    │ Service  │  │ Service  │  │ Service  │  │  Service   │  │ Service │  │
│    └──┬───────┘  └──┬───────┘  └──┬───────┘  └──┬─────────┘  └──┬──────┘  │
└───────┼─────────────┼─────────────┼─────────────┼────────────────┼────────┘
        │             │             │             │                │
        ▼             ▼             ▼             ▼                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      TransferQueue (Data Plane)                           │
│                                                                           │
│  ┌────────────────┐       ┌──────────────────────────────────┐            │
│  │ TQ Controller  │◄──────┤  SimpleStorageUnit × N           │            │
│  │ (Metadata Mgr) │       │  (Partitioned Data Storage)      │            │
│  └────────────────┘       └──────────────────────────────────┘            │
│                                    ▲                                      │
│                                    │                                      │
│                ┌───────────────────┼────────────────────┐                 │
│                │ StreamingDataset / StreamingDataLoader │                 │
│                │ (relax/utils/data/stream_dataloader.py)│                 │
│                └────────────────────────────────────────┘                 │
└───────────────────────────────────────────────────────────────────────────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
┌───────────────────────────────────────────────────────────────────────────┐
│              Distributed Checkpoint Service (DCS)                         │
│                                                                           │
│  ┌──────────────┐     ┌──────────────────────────────────┐                │
│  │  Coordinator │◄───┤  CheckpointEngineClient × N      │                 │
│  │  (HTTP REST) │    │  (Per-rank weight send/recv)     │                 │
│  └──────────────┘     └──────────────────────────────────┘                │
│                                                                           │
│  ┌───────────────────────────────────────────────┐                        │
│  │  DeviceDirectBackend (NCCL/GLOO)              │                        │
│  │  - Actor → Rollout: weight broadcast to SGLang│                        │
│  │  - Actor → ActorFwd/Ref: PP-aware broadcast   │                        │
│  └───────────────────────────────────────────────┘                        │
└───────────────────────────────────────────────────────────────────────────┘
```

### 服务角色

Fully Async 模式下系统部署 6 个角色（由 `relax/core/registry.py` 中的 `ROLES` StrEnum 定义）：

```python
class ROLES(StrEnum):
    actor: str = "actor"           # 策略模型训练
    critic: str = "critic"         # 价值模型训练（可选）
    rollout: str = "rollout"       # SGLang 推理引擎，生成样本
    advantages: str = "advantages" # 优势和回报计算
    reference: str = "reference"   # 参考模型前向（KL 散度）
    actor_fwd: str = "actor_fwd"   # 当前策略前向（log prob）
```

角色选择逻辑（`relax/core/registry.py`）：

```python
def process_role(config):
    if config.fully_async:
        return ROLES           # 全部 6 个角色
    else:
        return ROLES_COLOCATE  # 仅 actor, critic, rollout
```

______________________________________________________________________

## 数据流：TransferQueue 上的 StreamingDataLoader

### 两种模式下的 TransferQueue

Colocate 和 Fully Async 模式都使用 TransferQueue 进行数据传输。核心区别在于**生产和消费的时序关系**：

```
Colocate 模式（串行）：
  Rollout 完整写入分区 train_N ── 全部就绪 ──► Actor 一次性读取 train_N
  （同组 GPU 分时复用；Rollout offload 后 Actor 唤醒训练）
  （ref log prob、advantages 在 Actor 的 train_actor() 内串行计算）

Fully Async 模式（流式并行）：
  Rollout 增量写入分区 train_N ──► Actor 通过 StreamingDataLoader 消费
  Rollout 可同时开始 train_N+1   ──► ActorFwd/Reference/Advantages 并行消费 train_N
  （不同 GPU 集群完全并行；ref log prob、adv 独立计算并写回 TQ）
```

**分区机制**：

- **分区 ID 格式**：`train_{rollout_id}`，例如 `train_0`、`train_1`、`train_2`
- **生产者（Rollout）**：完成一次 rollout 后将数据写入 `train_{rollout_id}`
- **消费者（Actor/ActorFwd/Reference/Advantages）**：从对应分区读取数据，通过 `task_name` 追踪消费进度
- **分区清理**：Actor 训练完成后调用 `async_clear_partition()` 清理分区

**存储容量与 max_staleness**：

```python
# relax/core/controller.py
total_storage_size = (
    self.config.rollout_batch_size
    * (self.config.max_staleness + 1)
    * self.config.n_samples_per_prompt
)
```

TransferQueue 必须能同时缓存 `max_staleness + 1` 个 rollout batch 的数据。例如 `max_staleness=2`、`rollout_batch_size=8`、`n_samples_per_prompt=8` 时，需要 `8 × 3 × 8 = 192` 个样本的存储空间。

**Task names** 用于追踪不同消费者的消费进度：

| 消费者     | task_name                                          | 消费的数据字段                                                         |
| ---------- | -------------------------------------------------- | ---------------------------------------------------------------------- |
| Actor      | `actor_train`（StreamDataLoader）/ `train`（旧版） | tokens, loss_masks, log_probs, ref_log_probs, advantages, returns 等   |
| ActorFwd   | `actor_log_probs`                                  | tokens, total_lengths, response_lengths, loss_masks, rollout_log_probs |
| Reference  | `ref_log_probs`                                    | tokens, total_lengths, response_lengths, loss_masks, rollout_log_probs |
| Advantages | `compute_advantages_and_returns`                   | rollout_log_probs, log_probs, ref_log_probs, rewards 等                |

### StreamingDataLoader 与 StreamingDataset

在 Fully Async 模式下，Actor 使用 `StreamingDataLoader` 进行**流式数据消费**。与 Colocate 模式下 Actor 需要等待 Rollout 完全生成一个 batch 后再读取不同，StreamingDataLoader 可以在 TransferQueue 中的数据被增量写入的同时进行消费。这是实现"训练和推理并行"的核心机制。

#### StreamingDataset

```python
# TransferQueue (installed from https://github.com/redai-studio/TransferQueue)
class StreamingDataset(IterableDataset):
    """流式数据集，从 TransferQueue 动态获取数据"""

    def __init__(self, config, batch_size, micro_batch_size, data_fields,
                 partition_id, task_name, dp_rank, fetch_batch_fn, process_batch_fn):
        self.buffer = []       # 已拉取批次的缓存
        self.batch_index = 0   # 当前消费位置

    def __iter__(self):
        while not consumed:
            if self.batch_index <= len(self.buffer) - 1:
                # 从缓存中读取（支持多次遍历）
                yield from self.process_batch_fn(...)
            else:
                # 从 TransferQueue 拉取新数据
                batch_data, batch_meta = self.fetch_batch_fn(...)
                if batch_data is not None:
                    self.buffer.append((batch_data, batch_meta))
```

**核心特性**：

- **按需拉取**：每次拉取 `global_batch_size / num_iters_per_train_update` 大小的数据
- **缓存复用**：`buffer` 支持对同一批数据进行多次遍历（例如多轮训练）
- **分区切换**：`step(partition_id)` 清空缓存并切换到新的 rollout 数据分区

#### 数据拉取函数（fetch_batch_fn）

Fully Async 模式使用定制的 `get_data_from_transfer_queue()` 函数（`relax/utils/data/stream_dataloader.py`）：

```python
# broadcast_pp 是 fully_async 的反义
fetch_batch_fn = partial(get_data_from_transfer_queue,
                         broadcast_pp=not getattr(args, "fully_async", False))
```

**广播策略差异**：

| 模式        | `broadcast_pp` | 数据拉取节点                       | 广播范围                         |
| ----------- | -------------- | ---------------------------------- | -------------------------------- |
| Colocate    | `True`         | `tp_rank==0 && pp_rank==0`         | TP 组 + PP 组                    |
| Fully Async | `False`        | `tp_rank==0`（每个 PP stage 独立） | 仅 TP 组（各 PP stage 独立拉取） |

- **Colocate 模式**：Rollout 已经将完整 batch 写入 TransferQueue。Actor 在同一组 GPU 上启动，PP rank 0 从 TQ 拉取数据并广播到其他 PP stage。所有数据一次性就绪用于训练。
- **Fully Async 模式**：每个 PP stage 位于不同 rank，各自独立从 TransferQueue 拉取数据，避免跨 PP stage 通信开销。由于数据可能仍在增量写入，StreamingDataLoader 会在数据未就绪时自动重试。

#### create_stream_dataloader

```python
# relax/utils/data/stream_dataloader.py
def create_stream_dataloader(args, rollout_id, task_name, data_fields, dp_rank):
    dataset = StreamingDataset(
        config=args.tq_config,
        batch_size=args.micro_batch_size * args.n_samples_per_prompt,
        micro_batch_size=args.micro_batch_size,
        data_fields=data_fields,
        partition_id=f"train_{rollout_id}",
        task_name=task_name,
        dp_rank=dp_rank,
        fetch_batch_fn=fetch_batch_fn,
        process_batch_fn=split_dict,
    )
    dataloader = StreamingDataLoader(dataset)

    # 计算每个 rollout 的训练步数
    num_steps_per_rollout = (args.rollout_batch_size * args.n_samples_per_prompt
                            // args.global_batch_size)
    num_microbatches = [
        args.global_batch_size // dp_world_size // args.micro_batch_size
        for _ in range(num_steps_per_rollout)
    ]
    return [dataloader for _ in range(vpp_size)], num_microbatches
```

______________________________________________________________________

## 异步权重同步：分布式 Checkpoint 服务（DCS）

### DCS 在 Fully Async 中的作用

Actor 完成训练后，需要将权重分发到：

1. **Rollout（SGLang 引擎）** — 更新推理引擎权重
2. **ActorFwd** — 更新前向模型以计算当前策略 log prob
3. **Reference** — 更新参考模型（依据 `ref_update_interval`）

### DCS 架构

```
                          ┌──────────────────────┐
                          │   DCS Coordinator    │
                          │   (Ray Serve HTTP)   │
                          │                      │
                          │ - Node Registration  │
                          │ - Topology Discovery │
                          │ - Weight Meta Buffer │
                          │ - Group Rank Assign  │
                          └──────────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
    ┌─────────▼──────────┐ ┌─────────▼──────────┐ ┌─────────▼──────────┐
    │ CheckpointEngine   │ │ CheckpointEngine   │ │ CheckpointEngine   │
    │ Client (Actor)     │ │ Client (ActorFwd)  │ │ Client (Reference) │
    │                    │ │                    │ │                    │
    │ DeviceDirectBackend│ │ DeviceDirectBackend│ │ DeviceDirectBackend│
    │ (NCCL broadcast)   │ │ (NCCL recv)        │ │ (NCCL recv)        │
    └────────────────────┘ └────────────────────┘ └────────────────────┘
```

### 权重更新流程

#### Actor → Rollout

```python
# relax/backends/megatron/actor.py
def update_weights_fully_async(self, rollout_id, rollout_only=False, actor_fwd_only=False):
    dist.barrier(group=get_gloo_group())
    if not rollout_only:
        run(self.checkpoint_engine_client.init_process_groups_for_actor_fwd_ref(rollout_id))
    run(self.checkpoint_engine_client.update_weights_for_rollout(rollout_only, actor_fwd_only))
```

`update_weights_for_rollout` 内部流程（`DeviceDirectBackend`）：

1. **暂停 Rollout 推理**：HTTP 请求 SGLang 引擎 `/pause_generation`
2. **刷新 KV Cache**：HTTP 请求 `/flush_cache`
3. **分发权重**：
   - **非专家参数**：`all_gather` TP 分片 → 完整参数，PP 源 rank 广播到 Rollout（HF 格式）和 ActorFwd/Reference（原始格式）
   - **专家参数**：额外 EP `all_gather`，然后同上
4. **恢复 Rollout 推理**：HTTP 请求 `/continue_generation`

#### Actor → ActorFwd/Reference

ActorFwd 和 Reference 通过 DCS 的 PP 感知通信组接收权重：

- 每个 Actor PP stage 创建独立的 NCCL process group（`update_actor_pp_{pp_rank}`）
- ActorFwd/Reference rank 加入这些 group 接收对应 PP stage 的权重
- 接收端轮询 Coordinator 获取权重元数据，分配空 tensor，然后通过 `dist.broadcast` 接收

______________________________________________________________________

## max_staleness：On-Policy 与 Off-Policy 的控制

### 概念

**Staleness（陈旧度）** 衡量训练使用的 rollout 数据与当前模型权重之间的版本差距。

- **Staleness = 0**：训练数据必须来自当前模型版本
- **Staleness = N**：训练数据可来自当前或前 N 个版本的模型

```bash
--max-staleness 2    # 允许 Rollout 最多领先 Actor 2 步
```

### 对训练的影响

```
max_staleness = 0（On-Policy）：
  Rollout step 0 → Actor 训练 step 0 → Rollout step 1 → Actor 训练 step 1 → ...
  （Rollout 必须等待 Actor 消费完当前数据才能继续）

max_staleness = 2（部分 Off-Policy）：
  Rollout: step 0 → step 1 → step 2 → [等待] → step 3 → step 4 → step 5 → [等待] → ...
  Actor:   ........................step 0 → step 1 → step 2 → ...............step 3 → ...
  （Rollout 最多领先 2 步；超限时暂停等待 Actor 追赶）
```

### 实现机制

```python
# relax/components/rollout.py
def satisfy_staleness(partition_list, current_rollout_id, max_staleness):
    """检查当前 rollout 是否在允许的 staleness 范围内"""
    if not partition_list:
        return True
    oldest_step = min(int(p.split("_")[-1]) for p in partition_list)
    return current_rollout_id + 1 - oldest_step <= max_staleness
```

当 TransferQueue 中有 `max_staleness` 个或更多未消费的分区时，Rollout 将暂停等待 Actor 消费。

### 不同 max_staleness 值的效果

| `max_staleness` | 训练语义        | 吞吐量 | 稳定性 | 典型场景           |
| --------------- | --------------- | ------ | ------ | ------------------ |
| **0**           | 严格 On-Policy  | 低     | 最高   | 调试、小模型       |
| **1**           | 接近 On-Policy  | 中等   | 高     | 生产环境、中等模型 |
| **2-4**         | 轻度 Off-Policy | 高     | 中等   | 大模型、推理较慢   |
| **>4**          | 显著 Off-Policy | 最高   | 需验证 | 极致吞吐场景       |

::: tip
生产环境建议 `max_staleness=1~2`，兼顾吞吐量和训练稳定性。搭配 `--eps-clip` 和 `--eps-clip-high` 裁剪参数可缓解 Off-Policy 带来的训练不稳定问题。
:::

______________________________________________________________________

## 动态 Batch Size（流式 token 预算）

### 概念

样本长度差异很大时，固定 micro-batch 数会浪费 GPU。开启 `--use-dynamic-batch-size` 后，Actor 不再按固定数量切 micro-batch，而是按 **token 预算**（`--max-tokens-per-gpu`）打包：每个 micro-batch 累积样本直到接近预算上限。

在 fully-async 下，这一过程是**流式**的——Actor 边训练边从 TransferQueue 按需拉取下一个 micro-batch，无需等整批 rollout 数据就绪。

```bash
--use-dynamic-batch-size       # 启用动态 batch size
--max-tokens-per-gpu 20480     # 每张 GPU 每个 micro-batch 的 token 预算
```

### 工作流程

```
Rollout 产出样本 → TransferQueue
                      │
   ┌──────────────────┴────────────────────────┐
   │ StreamingTokenBudgetSampler（TQ 控制器侧）│
   │ · 按完整 GRPO 组（n_samples 个）攒样本    │
   │ · 按 token 总量在各 DP 间均衡             │
   │ · 凑够一个 token 预算 → 切出一个 mb       │
   └──────────────────┬────────────────────────┘
                      │ 按 batch_index 缓存（所有 DP 一次性算好）
   ┌──────────────────┴────────────────────┐
   │ StreamingTQIterator（Actor 侧）       │
   │ · 逐个拉 mb，直到 StopIteration       │
   │ · 拉到 mb N 后，后台预热 mb N+1 的缓存│
   └──────────────────┬────────────────────┘
                      │
        流式调度（streaming_schedules.py，支持 PP=1 / PP>1）
```

- **Sampler（`StreamingTokenBudgetSampler`）**：运行在 TransferQueue 控制器侧。它把就绪样本按完整 GRPO 组（每组 `n_samples_per_prompt` 个）攒起来，按 token 总量在各 DP 间均衡后，凑够一个 token 预算就切出一个 micro-batch。
- **按 `batch_index` 对齐缓存**：某个 `batch_index` 第一次被请求时，Sampler **一次性为所有 DP 算好各自的切片并缓存**。这样无论哪个 DP、哪个 PP stage、以什么顺序来取，拿到的数据都完全一致——保证 PP>1 时各 stage 步调一致、micro-batch 数对齐。
- **Iterator（`StreamingTQIterator`）**：运行在 Actor 侧，逐个拉取 micro-batch，直到分区数据消费完（`StopIteration`）。拉到 mb N 后会在后台**预热** mb N+1 的缓存，让下一次拉取直接命中、减少等待。

### 与流式调度配合

micro-batch 数事先未知，由 Iterator 拉到 `StopIteration` 为止。因此 fully-async + 动态 batch 使用专门的流式 forward/backward 调度（`relax/backends/megatron/streaming_schedules.py`），分别支持 PP=1 与 PP>1；所有 micro-batch 累积梯度，在末尾统一做一次梯度同步。

::: tip
动态 batch 主要收益是把变长样本打满 GPU、提升 MFU。注意 `max_staleness=0` 时，每步训练开始仍需等 Rollout 产出第一批完整 GRPO 组；要让 Rollout 提前跑、消除这段等待，配合 `max_staleness>=1` 使用。
:::

______________________________________________________________________

## 训练循环

### Actor 训练循环

```python
# relax/components/actor.py
def _background_run(self):
    while True:
        if self._stop_event.is_set():
            break
        with self._lock:
            local_step = self.step
        if local_step >= self.config.num_rollout:
            break
        self._execute_training()
        run(self.data_system_client.async_clear_partition(f"train_{local_step}"))
        with self._lock:
            self.step += 1

def _execute_training(self):
    if self.step < self.config.num_critic_only_steps:
        return  # 跳过仅训练 Critic 的阶段
    if self.config.fully_async:
        ray.get(self.actor_model.train_fully_async(self.step))
        self._maybe_save_model()
    else:
        ray.get(self.actor_model.async_train(self.step))
```

### ActorFwd 与 Reference 工作流

1. 从 TransferQueue 分批拉取数据（`_get_data_from_transfer_queue`）
2. 执行前向计算（`forward_only`）获取 log prob
3. 将结果写回 TransferQueue（`_put_data_to_transfer_queue`）
4. 消费完全部数据后，调用 `recv_weight_fully_async()` 接收新权重

### Advantages 服务

Advantages 服务等待 `ref_log_probs` 和 `log_probs` 都在 TransferQueue 中就绪后，计算优势和回报并写回。依赖关系由 TransferQueue 的 `get_meta` 自动处理——当所需字段未就绪时会阻塞等待。

______________________________________________________________________

## 数据流时序

```
Time ──────────────────────────────────────────────────────────────────────►

Rollout:  ┌──generate(step=N)──┐     ┌──generate(step=N+1)────┐    ...
          │ SGLang inference   │     │  (if staleness allows) │
          │ + reward scoring   │     │                        │
          └─────────┬──────────┘     └────────────────────────┘
                    │
                    ▼ Write to TransferQueue (partition=train_N)
                    │ Fields: tokens, loss_masks, rollout_log_probs,
                    │         rewards, total_lengths, response_lengths, ...
                    │
    ┌───────────────┼──────────────────────┐
    │               │                      │
    ▼               ▼                      ▼
  ActorFwd:      Reference:            Advantages:
read train_N    read train_N        wait for log_probs
  compute        compute            and ref_log_probs
 log_probs     ref_log_probs               │
 write to TQ    write to TQ                │
    │               │                      │
    └───────────────┼──────────────────────┘
                    │ All forward results ready
                    ▼
              Advantages Service:
                read rollout_log_probs + log_probs + ref_log_probs + rewards
                compute advantages + returns
                write to TransferQueue
                    │
                    ▼
              Actor (Training):
                StreamingDataLoader streams data
                 → Megatron forward + backward + optimizer step
                 → DCS distributes new weights to Rollout, ActorFwd, Reference
                 → Clear partition train_N

    ┌───────────────┼──────────────────────┐
    │               │                      │
    ▼               ▼                      ▼
 Rollout:         ActorFwd:             Reference:
 update weights   recv_weight            recv_weight (if needed)
 resume inference (NCCL broadcast)      (NCCL broadcast)
```

______________________________________________________________________

## 配置参数

### CLI 参数

| 参数                           | 默认值  | 说明                                                                    |
| ------------------------------ | ------- | ----------------------------------------------------------------------- |
| `--fully-async`                | `false` | 启用全异步训练流水线                                                    |
| `--max-staleness`              | `0`     | 最大允许 staleness（0=On-Policy，>0=部分 Off-Policy）                   |
| `--num-data-storage-units`     | `1`     | TransferQueue SimpleStorageUnit actor 数量                              |
| `--num-iters-per-train-update` | `1`     | 每个 global batch 的训练迭代次数                                        |
| `--checkpoint-engine-backend`  | `nccl`  | DCS 通信后端（`nccl` 或 `gloo`）                                        |
| `--polling-mode`               | `true`  | TransferQueue Controller 使用轮询模式获取元数据                         |
| `--ref-update-interval`        | `None`  | 参考模型更新周期（None=不更新）                                         |
| `--use-dynamic-batch-size`     | `false` | 按 token 预算流式打包 micro-batch（见「动态 Batch Size」）              |
| `--max-tokens-per-gpu`         | `None`  | 每张 GPU 每个 micro-batch 的 token 预算（启用动态 batch 时必填）        |
| `--resource`                   | -       | JSON 格式角色资源分配，如 `'{"actor": [1, 2], "rollout": [1, 4], ...}'` |

### 配置示例

```bash
# 8 GPU 全异步（来自 scripts/training/text/run-qwen3-4B-8xgpu-async.sh）
ray job submit -- python3 relax/entrypoints/train.py \
    --resource '{"actor": [1, 2], "rollout": [1, 4], "reference": [1, 1], "actor_fwd": [1, 1], "advantages": [1, 0]}' \
    --max-staleness 2 \
    --num-data-storage-units 1 \
    --num-iters-per-train-update 8 \
    --fully-async \
    --use-health-check \
    ...
```

**资源分配详解**：

- **Actor**：1 副本 × 2 GPU（TP=2 训练）
- **Rollout**：1 副本 × 4 GPU（4 个 SGLang 引擎）
- **Reference**：1 副本 × 1 GPU（单 GPU 前向）
- **ActorFwd**：1 副本 × 1 GPU（单 GPU 前向）
- **Advantages**：1 副本 × 0 GPU（仅 CPU 计算）

______________________________________________________________________

## 容错机制

### 重启策略

| 故障角色       | 策略     | 原因                                                |
| -------------- | -------- | --------------------------------------------------- |
| Actor          | 全局重启 | Actor 是核心训练服务，所有其他服务依赖于它          |
| Rollout        | 全局重启 | 引擎状态复杂，难以原地恢复                          |
| ActorFwd       | 全局重启 | 权重通信组状态难以恢复                              |
| Reference      | 原地重启 | 与 Advantages 类似，可安全重新部署                  |
| Advantages     | 原地重启 | 无状态服务，可安全重新部署                          |
| 任意角色 ≥3 次 | 全局重启 | 系统不稳定，需全量重新初始化                        |

### 权重更新期间的容错

```python
# relax/backends/megatron/actor.py — MegatronTrainRayActor.train_async()
rollout_only, actor_fwd_only = self._check_services_health()
# rollout_only=True：跳过 ActorFwd 权重更新（ActorFwd 不可用）
# actor_fwd_only=True：跳过 Rollout 权重更新（Rollout 不可用）
self.update_weights_fully_async(rollout_id, rollout_only, actor_fwd_only)
```

______________________________________________________________________

## 性能调优

### 关键调优参数

| 参数                           | 推荐值 | 影响                                 |
| ------------------------------ | ------ | ------------------------------------ |
| `--max-staleness`              | 1-2    | 平衡吞吐量与训练稳定性               |
| `--num-iters-per-train-update` | 4-8    | 值越大数据利用率越高，但单步延迟增加 |
| `--num-data-storage-units`     | 1-2    | 更多存储单元提升并行数据访问性能     |

### GPU 资源分配策略

```
总 GPU 数：N
├── Actor（训练）：~25-30%（需 TP/PP/CP 支持）
├── Rollout（推理）：~50-60%（推理吞吐是瓶颈）
├── ActorFwd：~5-10%（单 GPU 通常足够）
├── Reference：~5-10%（单 GPU 通常足够）
└── Advantages：0 GPU（仅 CPU 计算）
```

______________________________________________________________________

## Colocate 与 Fully Async 对比

```
                Colocate Mode                           Fully Async Mode
          (Same GPUs, time-shared)                 (Dedicated GPU clusters)
            ┌──────────────────┐                     ┌──────────────────────┐
  Time ──►  │   Rollout        │                     │  Rollout ──────────► │
            │ (SGLang infer)   │                     │  (continuous infer)  │
            │ write TQ train_N │                     │                      │
            ├──────────────────┤                     │  Actor  ──────────►  │
            │ offload rollout  │                     │  (StreamDataLoader   │
            │ wake up actor    │                     │   streaming + train) │
            ├──────────────────┤                     │                      │
            │   Actor Train    │                     │  ActorFwd ────────►  │
            │ (read TQ train_N)│                     │  (compute log prob)  │
            │ (incl ref/adv)   │                     │                      │
            ├──────────────────┤                     │  Reference ────────► │
            │   Weight Update  │                     │  (compute ref logp)  │
            │ (tensor copy)    │                     │                      │
            ├──────────────────┤                     │  Advantages ──────►  │
            │ offload actor    │                     │  (compute adv/ret)   │
            │ wake up rollout  │                     │                      │
            ├──────────────────┤                     │  DCS weight sync     │
            │   Rollout        │                     │  (overlaps training) │
            │   (continue)     │                     └──────────────────────┘
            └──────────────────┘
         GPU utilization: ~30-50%                    GPU utilization: ~70-90%
         All operations strictly serial              All operations parallel
         Data via TransferQueue, no overlap           Data via TransferQueue, streaming overlap
```

______________________________________________________________________

## 延伸阅读

- [系统架构](./architecture.md) — Relax 整体架构设计
- [分布式 Checkpoint](./distributed-checkpoint.md) — DCS 详细文档
- [健康检查管理器](./health-check-manager.md) — 健康监控与故障恢复
