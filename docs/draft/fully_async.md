# Fully Async Training Pipeline — 架构设计与实现详解

## 1. 概述

Relax 的 **Fully Async（全异步）训练流水线** 是一种高吞吐量的 RLHF/RL 训练模式，旨在最大化 GPU 利用率。与传统的 Colocate（同步协同）模式不同，Fully Async 模式将 **训练（Actor）**、**推理（Rollout）**、**前向计算（ActorFwd / Reference）** 和 **优势计算（Advantages）** 部署在独立的 GPU 集群上，各服务通过 TransferQueue 进行数据交换，并通过 Distributed Checkpoint Service (DCS) 进行异步权重同步。

### 1.1 核心设计理念

| 维度          | Colocate（同步协同）模式                                        | Fully Async（全异步）模式                                                 |
| ------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **GPU 共享**  | Actor 与 Rollout 共享同一组 GPU                                 | Actor、Rollout、ActorFwd/Reference 各自独立 GPU 集群                      |
| **执行模型**  | 串行交替：Rollout 完全生成 → 切换到 Train → 权重更新            | 完全并行：Rollout、Train、前向计算同时进行                                |
| **权重同步**  | 通过 Tensor 直接写入（colocate，同一进程内存拷贝）              | 通过 DCS（Checkpoint Engine）跨节点 NCCL broadcast                        |
| **数据流转**  | 同样经由 TransferQueue，但同步：Rollout 完整写入后 Actor 再读取 | 经由 TransferQueue + StreamingDataLoader 异步流式消费（生产与消费可重叠） |
| **Staleness** | `max_staleness=0`（严格 On-Policy）                             | `max_staleness` 可配置（允许一定程度的 Off-Policy）                       |
| **角色列表**  | `actor`, `critic`, `rollout`                                    | `actor`, `critic`, `rollout`, `advantages`, `reference`, `actor_fwd`      |

> **注意**：两种模式都使用 TransferQueue 作为数据传输层。区别在于 Colocate 模式下 Rollout 和 Actor 时分复用同一组 GPU，数据生成与训练是严格串行的——Rollout 将一整个 batch 的数据完全写入 TransferQueue 后才切换到 Actor 进行训练。而 Fully Async 模式下各服务在独立 GPU 上并行运行，数据可以边生产边消费。

### 1.2 关键优势

1. **消除 GPU 空闲时间**：Rollout 和 Training 可以同时运行，训练过程中 Rollout 引擎可以继续生成数据
2. **灵活的资源分配**：训练和推理可以使用不同数量的 GPU，适配异构硬件
3. **可控的 On/Off Policy 程度**：通过 `max_staleness` 参数精确控制训练数据的新鲜度
4. **流水线化的权重更新**：DCS 使得权重分发与训练计算可以流水线重叠

______________________________________________________________________

## 2. 系统架构

### 2.1 整体架构图

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Controller (Orchestrator)                           │
│                     relax/core/controller.py                               │
│                                                                            │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌─────────┐   │
│    │ Rollout  │  │  Actor   │  │ ActorFwd │  │ Reference  │  │  Adv    │   │
│    │ Service  │  │ Service  │  │ Service  │  │  Service   │  │ Service │   │
│    └──┬───────┘  └──┬───────┘  └──┬───────┘  └──┬─────────┘  └──┬──────┘   │
└───────┼─────────────┼─────────────┼─────────────┼────────────────┼─────────┘
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
│                ┌───────────────────┼───────────────────┐                  │
│                │ StreamingDataset / StreamingDataLoader│                  │
│                │ (relax/utils/stream_dataloader.py)    │                  │
│                └───────────────────────────────────────┘                  │
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
│  │  - Actor → Rollout: 权重 broadcast 到 SGLang  │                        │
│  │  - Actor → ActorFwd/Ref: PP-aware broadcast   │                        │
│  └───────────────────────────────────────────────┘                        │
└───────────────────────────────────────────────────────────────────────────┘
```

### 2.2 服务角色定义

在 Fully Async 模式下，系统部署 6 种角色（由 `relax/utils/const.py` 中的 `ROLES` StrEnum 定义）：

```python
class ROLES(StrEnum):
    actor: str = "actor"         # 策略模型训练
    critic: str = "critic"       # 价值模型训练（可选）
    rollout: str = "rollout"     # SGLang 推理引擎，生成采样数据
    advantages: str = "advantages"  # 优势和回报计算
    reference: str = "reference"   # 参考模型前向计算（KL 散度）
    actor_fwd: str = "actor_fwd"   # 当前策略前向计算（log prob）
```

**角色选择逻辑**（`relax/utils/const.py:process_role()`）：

```python
def process_role(config):
    if config.fully_async:
        return ROLES           # 全部 6 个角色
    else:
        return ROLES_COLOCATE  # 仅 actor, critic, rollout
```

______________________________________________________________________

## 3. 数据流详解：TransferQueue 上的 StreamingDataLoader

### 3.1 两种模式下的 TransferQueue 使用方式

无论是 Colocate 还是 Fully Async，数据都通过 TransferQueue 流转。核心区别在于 **生产与消费的时序关系**：

```
Colocate 模式（串行）:
  Rollout 完整写入 partition train_N ──全部就绪──► Actor 一次性读取 train_N
  （同一组 GPU 时分复用，Rollout offload 后 Actor wake up 再训练）
  （ref log prob、advantages 计算都在 Actor 内部 train_actor() 中串行完成）

Fully Async 模式（流式并行）:
  Rollout 逐批写入 partition train_N ──► Actor 通过 StreamingDataLoader 边写边读
  Rollout 可以同时开始 train_N+1     ──► ActorFwd/Reference/Advantages 并行消费 train_N
  （不同 GPU 集群上的服务完全并行，ref log prob、adv 各自独立计算并写回 TQ）
```

**数据分区（Partition）机制**：

- **Partition ID 格式**：`train_{rollout_id}`，例如 `train_0`, `train_1`, `train_2`
- **生产者（Rollout）**：完成一次 rollout 后，将数据写入 `train_{rollout_id}` 分区
- **消费者（Actor/ActorFwd/Reference/Advantages）**：从对应分区读取数据，通过 `task_name` 区分不同消费者
- **分区清理**：Actor 训练完成后调用 `async_clear_partition()` 释放已消费的数据

**存储容量与 max_staleness 的关系**：

SimpleStorage 不设固定行数上限（`total_storage_size=None`），这不代表存在自动内存限制。MooncakeStore 则按字节检查容量，通过 `RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB` 配置。

容量规划应考虑 `max_staleness + 1` 个 rollout 批次。固定 `rollout_batch_size=8`、`n_samples_per_prompt=8`、`max_staleness=2` 时，若每个生成样本只写入一行，基准估算为 `8 × 3 × 8 = 192` 行。Agent 或工具调用可能导出额外的物理行，因此该估算不是通用行数上限；实际内存需求还取决于每行的数据大小。

**任务名称**（task_name）用于跟踪不同消费者的消费进度：

| 消费者     | task_name                                            | 消费的数据字段                                                         |
| ---------- | ---------------------------------------------------- | ---------------------------------------------------------------------- |
| Actor      | `actor_train`（StreamDataLoader）/ `train`（legacy） | tokens, loss_masks, log_probs, ref_log_probs, advantages, returns 等   |
| ActorFwd   | `actor_log_probs`                                    | tokens, total_lengths, response_lengths, loss_masks, rollout_log_probs |
| Reference  | `ref_log_probs`                                      | tokens, total_lengths, response_lengths, loss_masks, rollout_log_probs |
| Advantages | `compute_advantages_and_returns`                     | rollout_log_probs, log_probs, ref_log_probs, rewards 等                |

### 3.2 StreamingDataLoader 与 StreamingDataset

在 Fully Async 模式下，Actor 使用 `StreamingDataLoader` 进行 **流式数据消费**。与 Colocate 模式中 Actor 等待 Rollout 完整生成后再一次性取回数据不同，StreamingDataLoader 可以在数据被逐批写入 TransferQueue 的过程中即时消费。这是 Fully Async 实现"训练与推理并行"的核心机制。

#### 3.2.1 StreamingDataset

```python
# transfer_queue/dataloader/streaming_dataset.py
class StreamingDataset(IterableDataset):
    """流式数据集，支持从 TransferQueue 动态拉取数据"""

    def __init__(self, config, batch_size, micro_batch_size, data_fields,
                 partition_id, task_name, dp_rank, fetch_batch_fn, process_batch_fn):
        self.buffer = []       # 缓存已获取的批次
        self.batch_index = 0   # 当前消费位置

    def __iter__(self):
        while not consumed:
            if self.batch_index <= len(self.buffer) - 1:
                # 从缓存中读取（支持多 pass 训练）
                yield from self.process_batch_fn(...)
            else:
                # 从 TransferQueue 拉取新数据
                batch_data, batch_meta = self.fetch_batch_fn(...)
                if batch_data is not None:
                    self.buffer.append((batch_data, batch_meta))
```

**关键特性**：

- **按需拉取**：每次只拉取一个 `global_batch_size / num_iters_per_train_update` 大小的批次
- **缓存复用**：`buffer` 支持对同一批数据进行多次迭代（例如 PPO 的多个 epoch）
- **分区切换**：`step(partition_id)` 方法清空缓存并切换到新的 rollout 数据分区

#### 3.2.2 数据获取函数（fetch_batch_fn）

Fully Async 模式使用定制的 `get_data_from_transfer_queue()` 函数（`relax/utils/stream_dataloader.py`）：

```python
# broadcast_pp 是 fully_async 的反向标记
fetch_batch_fn = partial(get_data_from_transfer_queue,
                         broadcast_pp=not getattr(args, "fully_async", False))
```

**广播策略差异**：

| 模式        | `broadcast_pp` | 数据获取节点                  | 广播范围                            |
| ----------- | -------------- | ----------------------------- | ----------------------------------- |
| Colocate    | `True`         | `tp_rank==0 && pp_rank==0`    | TP group + PP group                 |
| Fully Async | `False`        | `tp_rank==0`（每个 PP stage） | 仅 TP group（各 PP stage 独立获取） |

- **Colocate 模式**：Rollout 已将整个 batch 完整写入 TransferQueue，Actor 在同一组 GPU 上启动，PP rank 0 统一从 TQ 拉取数据并广播给其他 PP stage，所有数据在一次性获取后即可开始训练。
- **Fully Async 模式**：各 PP stage 部署在不同 rank 上，各自独立从 TransferQueue 获取数据，避免了跨 PP stage 的通信开销。由于数据可能正在被逐批写入，StreamingDataLoader 在数据未就绪时会自动等待重试。

#### 3.2.3 create_stream_dataloader

```python
# relax/utils/stream_dataloader.py
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

**`num_iters_per_train_update` 参数**：控制每个 global batch 被拆分为多少次训练迭代。在 Fully Async 模式下，由于 Actor 和 Rollout 独立运行，可以将一个大的 rollout batch 拆分为多次小批次训练，提高数据利用效率。

______________________________________________________________________

## 4. 异步权重同步：Distributed Checkpoint Service (DCS)

### 4.1 DCS 在 Fully Async 中的角色

在 Fully Async 模式下，权重需要从 Actor（训练完成后）分发到以下接收方：

1. **Rollout（SGLang 引擎）**：更新推理引擎的模型权重
2. **ActorFwd**：更新用于计算当前策略 log prob 的前向模型
3. **Reference**：更新参考模型（按 `ref_update_interval` 周期更新）

### 4.2 DCS 架构

```
                          ┌─────────────────────┐
                          │   DCS Coordinator   │
                          │   (Ray Serve HTTP)  │
                          │                     │
                          │ - Node Registration │
                          │ - Topology Discovery│
                          │ - Weight Meta Buffer│
                          │ - Group Rank Assign │
                          └──────────┬──────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
    ┌─────────▼──────────┐ ┌────────▼───────────┐ ┌────────▼───────────┐
    │ CheckpointEngine   │ │ CheckpointEngine   │ │ CheckpointEngine   │
    │ Client (Actor)     │ │ Client (ActorFwd)  │ │ Client (Reference) │
    │                    │ │                    │ │                    │
    │ DeviceDirectBackend│ │ DeviceDirectBackend│ │ DeviceDirectBackend│
    │ (NCCL broadcast)   │ │ (NCCL recv)        │ │ (NCCL recv)        │
    └────────────────────┘ └────────────────────┘ └────────────────────┘
```

### 4.3 初始化流程

当 `--fully-async` 启用时，`MegatronTrainRayActor._init()` 中会创建 DCS 客户端：

```python
# relax/backends/megatron/actor.py（init 方法的 fully_async 分支）
metadata = {
    "tp_size": mpu.get_tensor_model_parallel_world_size(),
    "dp_size": mpu.get_data_parallel_world_size(),
    "pp_size": mpu.get_pipeline_model_parallel_world_size(),
    "pp_rank": mpu.get_pipeline_model_parallel_rank(),
    "is_pp_src_rank": is_pp_src_rank,
    "master_address": master_address,
    "master_port": master_port,
}
self.checkpoint_engine_client = run(
    create_client(
        args=self.args,
        coordinator_url=self.args.coordinator_url,
        role=role,       # "actor", "actor_fwd", 或 "reference"
        rank=dist.get_rank(),
        model=self.model,
        model_name=...,
        backend_type=self.args.checkpoint_engine_backend,  # 默认 "nccl"
        metadata=metadata,
    )
)
```

**DCS Coordinator 部署**：在 Controller 初始化时由 `create_dcs_deployment()` 部署为 Ray Serve service，URL 存储在 `config.coordinator_url` 中。

### 4.4 权重更新流程

#### 4.4.1 Actor → Rollout 权重更新

```python
# relax/backends/megatron/actor.py
def update_weights_fully_async(self, rollout_id, rollout_only=False, actor_fwd_only=False):
    dist.barrier(group=get_gloo_group())
    # 1. 初始化 ActorFwd/Reference 的通信组（如果不是 rollout_only）
    if not rollout_only:
        run(self.checkpoint_engine_client.init_process_groups_for_actor_fwd_ref(rollout_id))
    # 2. 更新 Rollout 引擎权重（同时也向 ActorFwd/Ref 发送权重）
    run(self.checkpoint_engine_client.update_weights_for_rollout(rollout_only, actor_fwd_only))
```

**`update_weights_for_rollout` 的内部流程**（`DeviceDirectBackend`）：

1. **暂停 Rollout 推理**：通过 HTTP 请求 SGLang 引擎 `/pause_generation`
2. **清空 KV Cache**：通过 HTTP 请求 `/flush_cache`
3. **权重分发**：
   - **非专家参数（Non-Expert）**：
     a. 在训练 rank 上 `all_gather` TP 分片 → 完整参数
     b. PP source rank（`dp_rank==0 && tp_rank==0`）负责：
     - 转换为 HuggingFace 格式 → broadcast 到 Rollout 引擎
     - 保留原始格式 → broadcast 到 ActorFwd/Reference
   - **专家参数（Expert）**：额外进行 EP `all_gather`，然后同上
4. **恢复 Rollout 推理**：通过 HTTP 请求 `/continue_generation`

#### 4.4.2 Actor → ActorFwd/Reference 权重更新

ActorFwd 和 Reference 通过 DCS 的 PP-aware 通信组接收权重：

```python
# CheckpointEngineClient
async def init_process_groups_for_actor_fwd_ref(self, rollout_id):
    # 检查是否需要更新 Reference（按 ref_update_interval）
    self.need_update_ref = (
        self.args.ref_update_interval is not None
        and (rollout_id + 1) % self.args.ref_update_interval == 0
    )
    # 从 Coordinator 获取通信组信息
    data = await self._http_client.get(
        f"{self.coordinator_url}/get_model_update_group_ranks",
        params={"role": self.role, "rank": self.rank,
                "need_update_ref": self.need_update_ref},
    )
    # 初始化 NCCL process group
    self._backend.init_process_groups_for_actor_fwd_ref(data)
```

**通信组设计**：每个 Actor PP stage 建立一个独立的 NCCL process group（命名为 `update_actor_pp_{pp_rank}`），ActorFwd/Reference 的所有 rank 加入该组接收对应 PP stage 的权重。

**接收端（ActorFwd/Reference）**：

```python
# DeviceDirectBackend.recv_weight()
def recv_weight(self):
    """轮询 Coordinator 获取权重元数据，然后通过 NCCL broadcast 接收"""
    index = 0
    while True:
        data = self.http_client.get(f"{self.coordinator_url}/recv_weight_meta", ...)
        for metadata in data:
            if names[0] == "weight_updated_stop":
                return  # 接收完成标记
            # 根据元数据分配空 tensor，然后 dist.broadcast 接收
            for name, dtype, shape in zip(names, dtypes, shapes):
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                dist.broadcast(weight, src=0, group=self._model_update_groups_for_actor_fwd_ref[group_name])
            # 加载权重到模型
            load_weight(self.args, self.model, weights)
```

#### 4.4.3 Controller 层的编排

```python
# relax/core/controller.py - training_loop
async def run_all_services():
    if self.config.fully_async:
        # 初始化异步权重通道
        handle1 = self.serve_dict[ROLES.actor].update_weights_fully_async()
        handle2 = self.serve_dict[ROLES.actor_fwd].recv_weight_fully_async()
        handle3 = self.serve_dict[ROLES.reference].recv_weight_fully_async()
        handles = [handle1, handle2, handle3]
        [await handle for handle in handles]

    # 设置初始 step 并启动所有服务
    step = await self.serve_dict[ROLES.actor].get_step()
    for service in self.serve_dict.values():
        await service.set_step(step)

    # 并行启动所有服务
    task_refs = []
    for role, service in self.serve_dict.items():
        task_ref = service.run()
        if task_ref is not None:
            task_refs.append(task_ref)
    [await task_ref for task_ref in task_refs]
```

______________________________________________________________________

## 5. max_staleness：On-Policy 与 Off-Policy 的控制机制

### 5.1 概念定义

**Staleness（陈旧度）** 是指训练使用的 rollout 数据与当前模型权重之间的版本差距。

- **Staleness = 0**（严格 On-Policy）：训练数据必须由当前版本的模型生成
- **Staleness = 1**：训练数据可以来自当前版本或前一个版本的模型
- **Staleness = N**：训练数据可以来自最近 N 个版本中任何一个版本的模型

`max_staleness` 参数控制系统允许的最大陈旧度：

```bash
# 命令行参数
--max-staleness 2    # 允许 Rollout 最多领先 Actor 2 个 step
```

### 5.2 Staleness 对训练的影响

```
max_staleness = 0 (On-Policy):
  Rollout step 0 → Actor trains step 0 → Rollout step 1 → Actor trains step 1 → ...
  (Rollout 必须等待 Actor 消费完当前数据才能继续)

max_staleness = 2 (部分 Off-Policy):
  Rollout: step 0 → step 1 → step 2 → [等待] → step 3 → step 4 → step 5 → [等待] → ...
  Actor:   ........................step 0 → step 1 → step 2 → ...............step 3 → ...
  (Rollout 可以领先 Actor 最多 2 个 step，超过后需等待 Actor 追上)
```

### 5.3 实现机制

#### 5.3.1 Staleness 检查函数

```python
# relax/components/rollout.py
def satisfy_staleness(partition_list, current_rollout_id, max_staleness):
    """检查当前 rollout 是否在允许的 staleness 范围内

    Args:
        partition_list: TransferQueue 中当前存在的分区列表，如 ['train_5', 'train_6']
        current_rollout_id: 当前 rollout 的 step 编号
        max_staleness: 允许的最大 staleness

    Returns:
        True 表示可以继续 rollout，False 表示需要等待 Actor 消费数据
    """
    if not partition_list:
        return True

    # 找到最老的（未消费的）partition 的 step 编号
    oldest_step = min(int(p.split("_")[-1]) for p in partition_list)

    # 当前 rollout_id + 1 - oldest_step <= max_staleness 时才允许继续
    return current_rollout_id + 1 - oldest_step <= max_staleness
```

**直观理解**：如果 TransferQueue 中还有 `max_staleness` 个以上未被 Actor 消费的 partition，Rollout 就需要暂停等待。

#### 5.3.2 Rollout 中的 Staleness 控制

```python
# relax/components/rollout.py - _background_run()
while True:
    # ... 执行 rollout generation ...
    ray.get(self.rollout_manager.generate.remote(rollout_id=local_step))

    # Staleness 控制：检查 Actor 是否跟上进度
    wait_count = 0
    while True:
        partition_list = run(self.data_system_client.async_get_partition_list())
        rollout_done = local_step + 1 == self.config.num_rollout

        should_continue = rollout_done or satisfy_staleness(
            partition_list, local_step, self.config.max_staleness
        )

        if not should_continue:
            # Rollout 领先太多，等待 Actor 消费数据
            if wait_count % 10 == 0:
                self._logger.warning(
                    f"Rollout {local_step}: waiting for data system to catch up. "
                    f"Current partitions: {partition_list}, waited {wait_count}s"
                )
            wait_count += 1
            time.sleep(1)
            continue
        else:
            break  # 可以继续下一次 rollout
```

#### 5.3.3 存储容量与 Staleness 的关系

Rollout 可以领先 Actor `max_staleness` 个 step，容量规划需考虑 `max_staleness + 1` 个 rollout 批次的数据。该关系用于估算同时驻留的数据量，不是 SimpleStorage 的行数限制；还需考虑每批实际导出的物理行数和每行的数据大小。

### 5.4 不同 max_staleness 值的效果

| `max_staleness` | 训练语义        | 吞吐量 | 训练稳定性 | 典型场景                     |
| --------------- | --------------- | ------ | ---------- | ---------------------------- |
| **0**           | 严格 On-Policy  | 低     | 最高       | 初始调试、小模型训练         |
| **1**           | 近似 On-Policy  | 中     | 高         | 生产环境、中等模型           |
| **2-4**         | 轻度 Off-Policy | 高     | 中等       | 大模型、长序列推理较慢的场景 |
| **>4**          | 显著 Off-Policy | 最高   | 需验证     | 极端吞吐量优先的场景         |

**推荐实践**：

- 生产环境推荐 `max_staleness=1~2`，在吞吐量和训练稳定性之间取得平衡
- 配合 `--eps-clip` 和 `--eps-clip-high` 等 PPO/GRPO clipping 参数来缓解 Off-Policy 带来的训练不稳定性

______________________________________________________________________

## 6. Fully Async 训练循环详解

### 6.1 Actor 训练循环

```python
# relax/components/actor.py - _background_run()
def _background_run(self):
    while self.step < self.config.num_rollout:
        # Fully Async 模式不需要等待 Rollout 数据就绪
        # 数据由 StreamingDataLoader 在训练内部流式拉取
        self._execute_training()
        # 清除已消费的数据分区
        run(self.data_system_client.async_clear_partition(f"train_{self.step}"))
        self.step += 1

def _execute_training(self):
    if self.config.fully_async:
        ray.get(self.actor_model.train_fully_async(self.step))
        self._maybe_save_model()
    else:
        ray.get(self.actor_model.async_train(self.step))
```

### 6.2 train_fully_async（MegatronTrainRayActor.train_async）

```python
# relax/backends/megatron/actor.py
async def train_async(self, rollout_id):
    # 1. 创建 / 切换 StreamingDataLoader
    if self.data_iterator is None:
        self.data_iterator, self.num_microbatches = create_stream_dataloader(
            self.args,
            rollout_id=rollout_id,
            task_name="actor_train",
            data_fields=[...],
            dp_rank=mpu.get_data_parallel_rank(),
        )
    else:
        for di in self.data_iterator:
            di.step(f"train_{rollout_id}")

    # 2. 执行 Megatron 训练
    with timer("actor_train"):
        train(rollout_id, self.model, self.optimizer,
              self.opt_param_scheduler, self.data_iterator, self.num_microbatches)

    # 3. 检查服务健康状态
    rollout_only, actor_fwd_only = self._check_services_health()

    # 4. 异步权重更新（通过 DCS 分发到 Rollout 和 ActorFwd/Reference）
    self.update_weights_fully_async(rollout_id, rollout_only, actor_fwd_only)

    # 5. 通知 Rollout 恢复推理并触发评估
    dist.barrier(group=get_gloo_group())
    if dist.get_rank() == 0:
        rollout_serve_url = get_serve_url("rollout")
        requests.get(f"{rollout_serve_url}/evaluate", params={"train_step": rollout_id})
        requests.get(f"{rollout_serve_url}/end_update_weight")
```

### 6.3 权重更新前的健康检查

在 Fully Async 模式下，Actor 在进行权重更新前会检查 Rollout 和 ActorFwd 服务的健康状态：

```python
# relax/backends/megatron/actor.py
def _check_services_health(self):
    """检查 Rollout 和 ActorFwd 服务是否可用"""
    if dist.get_rank() == 0:
        # 轮询 Rollout 直到它准备好接收权重
        while True:
            response = requests.get(f"{rollout_url}/can_do_update_weight_for_async")
            if response.json():  # Rollout 已暂停生成，准备好接收
                requests.get(f"{rollout_url}/recover_rollout_engines")
                break
            time.sleep(1)
    # 通过 allreduce 广播结果确保所有 rank 一致
    flags = torch.tensor([int(rollout_only), int(actor_fwd_only)])
    dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=get_gloo_group())
    return bool(flags[0]), bool(flags[1])
```

**Rollout 侧的配合**（`relax/components/rollout.py`）：

```python
@app.get("/can_do_update_weight_for_async")
async def can_do_update_weight_for_async(self):
    """检查当前是否可以进行权重更新"""
    with self._lock:
        # 检查当前 partition 的数据是否已被消费（已写入 TransferQueue）
        if self.data_system_client.check_production_status(
            ["tokens"], f"train_{self.step - 1}"
        ) or self.data_system_client.check_production_status(
            ["tokens"], f"train_{self.step}"
        ):
            self.status = "paused"  # 暂停 rollout 循环
            ray.get(self.rollout_manager.health_monitoring_pause.remote())
            return 1
    return 0
```

### 6.4 ActorFwd 和 Reference 的工作流

```python
# relax/components/actor_fwd.py
def _background_run(self):
    while self.step < self.config.num_rollout:
        if self.role == "actor_fwd":
            self.compute_actor_log_prob(local_step)
        else:  # reference
            self.compute_ref_log_prob(local_step)
        self.step += 1
```

**在 MegatronTrainRayActor 中**，`compute_ref_log_prob` 和 `compute_actor_log_prob` 的流程类似：

1. 从 TransferQueue 分批获取数据（`_get_data_from_transfer_queue`）
2. 执行前向计算（`forward_only`）得到 log probs
3. 将结果写回 TransferQueue（`_put_data_to_transfer_queue`）
4. 全部消费完成后，调用 `recv_weight_fully_async()` 接收新权重

```python
def compute_ref_log_prob(self, rollout_id):
    while not self.all_consumed("ref_log_probs", rollout_id):
        data, batch_meta = self._get_data_from_transfer_queue("ref_log_probs", ...)
        output_dict = self.compute_log_prob(data_iterator, num_microbatches, store_prefix="ref_")
        self._put_data_to_transfer_queue(output_dict, batch_meta)
    # 接收新权重
    self.recv_weight_fully_async(rollout_id)
```

### 6.5 Advantages 计算服务

```python
# relax/components/advantages.py
async def run(self):
    while step < self.config.num_rollout:
        while not consumed("compute_advantages_and_returns", step):
            batch_meta = await self.data_system_client.async_get_meta(
                data_fields=["rollout_log_probs", "log_probs", "ref_log_probs", "rewards", ...],
                batch_size=self.config.global_batch_size // self.config.num_iters_per_train_update,
                partition_id=f"train_{step}",
                task_name="compute_advantages_and_returns",
            )
            rollout_data = await self.data_system_client.async_get_data(batch_meta)
            result = self.compute_advantages_and_returns(rollout_data)
            await self.data_system_client.async_put(data=result, metadata=batch_meta)
        step += 1
```

**数据依赖**：Advantages 服务需要等待 `ref_log_probs` 和 `log_probs`（由 Reference 和 ActorFwd 计算后写入 TransferQueue）都就绪后才能开始计算。这种依赖关系通过 TransferQueue 的 `get_meta` 自动处理——当所需字段尚未就绪时，`get_meta` 会阻塞等待。

______________________________________________________________________

## 7. 完整数据流时序

以下是 Fully Async 模式下一个完整训练 step 的数据流时序：

```
时间 ──────────────────────────────────────────────────────────────────────►

Rollout:  ┌──generate(step=N)───┐     ┌──generate(step=N+1)──┐    ...
          │ SGLang 推理引擎    ││  (若 staleness 允许)   │
          │ + reward 计算       │     │                       │
          └─────────┬───────────┘     └───────────────────────┘
                    │
                    ▼ 写入 TransferQueue (partition=train_N)
                    │ 字段: tokens, loss_masks, rollout_log_probs,
                    │       rewards, total_lengths, response_lengths, ...
                    │
    ┌───────────────┼──────────────────────┐
    │               │                      │
    ▼               ▼                      ▼
ActorFwd:     Reference:              Advantages:
  读取 train_N   读取 train_N            等待 log_probs
  计算 log_probs  计算 ref_log_probs      和 ref_log_probs
  写回 TQ        写回 TQ                     │
    │               │                      │
    └───────────────┼──────────────────────┘
                    │ 所有前向结果就绪
                    ▼
              Advantages 服务:
                读取 rollout_log_probs + log_probs + ref_log_probs + rewards
                计算 advantages + returns
                写回 TransferQueue
                    │
                    ▼
              Actor (Training):
                通过 StreamingDataLoader 流式读取
                 → Megatron forward + backward + optimizer step
                 → 通过 DCS 分发新权重给 Rollout, ActorFwd, Reference
                 → 清除 partition train_N

    ┌───────────────┼──────────────────────┐
    │               │                      │
    ▼               ▼                      ▼
 Rollout:       ActorFwd:             Reference:
 更新权重        recv_weight            recv_weight (如需)
 恢复推理        (NCCL broadcast)       (NCCL broadcast)
```

______________________________________________________________________

## 8. 关键配置参数

### 8.1 命令行参数一览

| 参数                           | 默认值  | 说明                                                                        |
| ------------------------------ | ------- | --------------------------------------------------------------------------- |
| `--fully-async`                | `false` | 启用 Fully Async 训练流水线                                                 |
| `--max-staleness`              | `0`     | 允许的最大 staleness（0=On-Policy, >0=部分 Off-Policy）                     |
| `--num-data-storage-units`     | `1`     | TransferQueue SimpleStorageUnit 的数量（并行度）                            |
| `--num-iters-per-train-update` | `1`     | 每个 global batch 拆分为多少次训练迭代                                      |
| `--checkpoint-engine-backend`  | `nccl`  | DCS 通信后端（`nccl` 或 `gloo`）                                            |
| `--polling-mode`               | `true`  | TransferQueue Controller 使用轮询模式获取元数据                             |
| `--ref-update-interval`        | `None`  | Reference 模型的更新周期（None=不更新）                                     |
| `--resource`                   | -       | JSON 格式的各角色资源分配，如 `'{"actor": [1, 2], "rollout": [1, 4], ...}'` |
| `--ref-actor-config`           | -       | ActorFwd/Reference 的并行配置覆盖                                           |

### 8.2 典型配置示例

```bash
# 8 GPU Fully Async 配置（来自 scripts/training/text/run-qwen3-4B-8xgpu-async.sh）
ray job submit -- python3 relax/entrypoints/train.py \
    --resource '{"actor": [1, 2], "rollout": [1, 4], "reference": [1, 1], "actor_fwd": [1, 1], "advantages": [1, 0]}' \
    --max-staleness 2 \
    --num-data-storage-units 1 \
    --num-iters-per-train-update 8 \
    --ref-actor-config '{"tensor_model_parallel_size": 1, "max_tokens_per_gpu": 16384, "sequence_parallel": false, "only_load_weight": true}' \
    --fully-async \
    --use-health-check \
    ...
```

**资源分配解析**：

- **Actor**: 1 个 replica × 2 GPU（TP=2 训练）
- **Rollout**: 1 个 replica × 4 GPU（4 个 SGLang 引擎）
- **Reference**: 1 个 replica × 1 GPU（单 GPU 前向计算）
- **ActorFwd**: 1 个 replica × 1 GPU（单 GPU 前向计算）
- **Advantages**: 1 个 replica × 0 GPU（纯 CPU 计算）

______________________________________________________________________

## 9. 服务创建与并行初始化

### 9.1 并行服务创建

Fully Async 模式下，所有服务使用 `ThreadPoolExecutor` 并行创建，而非串行：

```python
# relax/core/controller.py - register_all_serve()
if self.config.fully_async:
    # 并行创建所有服务
    with ThreadPoolExecutor(max_workers=len(roles_to_create)) as executor:
        futures = {
            executor.submit(self._create_service_task, role, cls, num_gpus, data_source, pgs): role
            for role, cls, num_gpus, data_source in roles_to_create
        }
        done, _ = concurrent.futures.wait(futures.keys(), return_when=ALL_COMPLETED)
```

这显著减少了初始化时间，因为 6 个服务（涉及不同的 Ray placement group 和模型加载）可以同时创建。

### 9.2 初始化时序

```
Controller.__init__()
├── _initialize_data_system()          # 创建 TransferQueue
├── create_dcs_deployment()            # 部署 DCS Coordinator
├── _deploy_metrics_service()          # 部署 Metrics Service (可选)
└── register_all_serve()               # 并行创建所有角色服务
    ├── Actor (创建 RayTrainGroup → MegatronTrainRayActor)
    │   └── init → CheckpointEngineClient.start() → 注册到 DCS
    ├── Rollout (创建 RolloutManager → SGLang EngineGroup)
    ├── ActorFwd (创建 RayTrainGroup → MegatronTrainRayActor[role=actor_fwd])
    │   └── init → CheckpointEngineClient.start() → 注册到 DCS
    ├── Reference (创建 RayTrainGroup → MegatronTrainRayActor[role=reference])
    │   └── init → CheckpointEngineClient.start() → 注册到 DCS
    └── Advantages (纯 CPU 服务)

training_loop()
├── update_weights_fully_async()       # 初始化权重：Actor → Rollout
├── recv_weight_fully_async() × 2      # 初始化权重：Actor → ActorFwd, Reference
├── set_step() → all services          # 同步 rollout_id
└── run() → all services               # 并行启动所有服务的主循环
```

______________________________________________________________________

## 10. 容错与健康检查

### 10.1 健康检查系统

Fully Async 模式通过 `--use-health-check` 启用健康检查，`HealthManager` 监控所有服务的心跳：

- 每个服务在 `_background_run()` 每个 step 完成后更新心跳
- `HealthChecker` 后台线程定期检查心跳超时
- 当服务不健康时，触发 `Controller._on_service_unhealthy(role)`

### 10.2 重启策略

| 失败的角色     | 重启策略                   | 原因                                   |
| -------------- | -------------------------- | -------------------------------------- |
| Actor          | 全局重启（Global Restart） | Actor 是核心训练服务，其他服务都依赖它 |
| Rollout        | 全局重启                   | Rollout 引擎状态复杂，难以原地恢复     |
| ActorFwd       | 全局重启                   | 权重通信组状态难以恢复                 |
| Advantages     | 局部重启（In-place）       | 无状态服务，可安全重新部署             |
| 任何角色 ≥3 次 | 全局重启                   | 系统不稳定，需要完全重新初始化         |

### 10.3 权重更新中的容错

```python
# Actor 训练循环中的容错
rollout_only, actor_fwd_only = self._check_services_health()
# rollout_only=True: 跳过 ActorFwd 权重更新（ActorFwd 服务不可用）
# actor_fwd_only=True: 跳过 Rollout 权重更新（Rollout 服务不可用）
self.update_weights_fully_async(rollout_id, rollout_only, actor_fwd_only)
```

______________________________________________________________________

## 11. 性能调优指南

### 11.1 关键调优参数

| 参数                           | 推荐值 | 影响                                     |
| ------------------------------ | ------ | ---------------------------------------- |
| `--max-staleness`              | 1-2    | 平衡吞吐量与训练稳定性                   |
| `--num-iters-per-train-update` | 4-8    | 更大值提高数据利用率，但增加单步延迟     |
| `--num-data-storage-units`     | 1-2    | 更多 storage unit 可提高并行数据访问性能 |
| `--micro-batch-size`           | 1-4    | 取决于模型大小和显存                     |
| `--sglang-mem-fraction-static` | 0.8    | SGLang 静态内存占比                      |

### 11.2 GPU 资源分配策略

```
总 GPU 数: N
├── Actor (训练): ~25-30% GPU（需要支持 TP/PP/CP）
├── Rollout (推理): ~50-60% GPU（推理吞吐量是瓶颈）
├── ActorFwd: ~5-10% GPU（单 GPU 通常足够）
├── Reference: ~5-10% GPU（单 GPU 通常足够）
└── Advantages: 0 GPU（纯 CPU 计算）
```

### 11.3 监控指标

通过 Metrics Service 监控以下关键指标：

- **Rollout 等待时间**：`Rollout {step}: waiting for data system to catch up` 日志频率
- **权重更新耗时**：DCS `update_weights_for_rollout` 计时
- **数据消费延迟**：各 task 的消费完成时间
- **GPU 利用率**：各角色的 GPU 使用率

______________________________________________________________________

## 12. 与 Colocate 模式的对比总结

```
                Colocate 模式                           Fully Async 模式
          (同一组 GPU 时分复用)                     (独立 GPU 集群并行)
            ┌─────────────────┐                     ┌──────────────────────┐
  时间 ──►  │   Rollout       │                     │  Rollout ──────────► │
            │ (SGLang 推理)    │                     │  (持续推理)          │
            │ 写入 TQ train_N  │                     │                      │
            ├─────────────────┤                     │  Actor  ──────────► │
            │ offload rollout  │                     │  (StreamDataLoader   │
            │ wake up actor    │                     │   流式消费 + 训练)   │
            ├─────────────────┤                     │                      │
            │   Actor Train   │                     │  ActorFwd ────────► │
            │ (读 TQ train_N)  │                     │  (计算 log prob)     │
            │ (含 ref/adv 计算)│                     │                      │
            ├─────────────────┤                     │  Reference ────────► │
            │   Weight Update │                     │  (计算 ref log prob) │
            │ (Tensor 直接写入)│                     │                      │
            ├─────────────────┤                     │  Advantages ──────► │
            │ offload actor    │                     │  (计算 adv & returns)│
            │ wake up rollout  │                     │                      │
            ├─────────────────┤                     │  DCS 权重同步        │
            │   Rollout       │                     │  (与训练流水线重叠)   │
            │   (继续生成)     │                     └──────────────────────┘
            └─────────────────┘
         GPU 利用率: ~30-50%                         GPU 利用率: ~70-90%
         所有操作严格串行                              所有操作并行执行
         数据经 TransferQueue 但无并行                 数据经 TransferQueue 流式并行
```

______________________________________________________________________

## 13. 代码入口索引

| 文件路径                                            | 关键功能                                            |
| --------------------------------------------------- | --------------------------------------------------- |
| `relax/core/controller.py`                    | Controller 编排、数据系统初始化、服务注册、训练循环 |
| `relax/components/actor.py`                               | Actor 服务：训练循环、StreamingDataLoader 创建      |
| `relax/components/rollout.py`                             | Rollout 服务：staleness 控制、异步权重更新协调      |
| `relax/components/actor_fwd.py`                           | ActorFwd/Reference 服务：前向计算、权重接收         |
| `relax/components/advantages.py`                          | Advantages 服务：优势和回报计算                     |
| `relax/utils/const.py`                              | 角色定义、算法注册表                                |
| `relax/utils/stream_dataloader.py`                  | StreamingDataLoader 创建、数据获取函数              |
| `relax/backends/megatron/actor.py`                  | MegatronTrainRayActor：train_async、权重更新        |
| `relax/distributed/checkpoint_service/client/engine.py`    | DCS 客户端：注册、权重发送/接收                     |
| `relax/distributed/checkpoint_service/coordinator/service.py` | DCS Coordinator：拓扑管理、通信组分配               |
| `relax/distributed/checkpoint_service/backends/device_direct.py` | DeviceDirectBackend：NCCL 权重 broadcast 实现       |
| `transfer_queue/controller.py`                      | TransferQueue Controller：元数据管理、采样          |
| `transfer_queue/dataloader/streaming_dataset.py`    | StreamingDataset：流式数据迭代器                    |
| `transfer_queue/dataloader/streaming_dataloader.py` | StreamingDataLoader：PyTorch DataLoader 封装        |
| `transfer_queue/sampler/grpo_group_n_sampler.py`    | GRPO 分组采样器                                     |
| `relax/utils/arguments.py`                          | 命令行参数定义（fully_async、max_staleness 等）     |
| `scripts/training/text/run-qwen3-4B-8xgpu-async.sh`               | 8 GPU Fully Async 启动脚本示例                      |
