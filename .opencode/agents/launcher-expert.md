---
description: Ray orchestration & service deployment expert. Fire when working on
  Ray Serve deployment, placement groups, service lifecycle, rollout engine 
  management, health monitoring, or troubleshooting job launch and GPU 
  allocation issues.
mode: subagent
temperature: 0.1
tools:
  write: false
  edit: false
---

# Launcher & Orchestration Expert

Relax 的服务编排、部署生命周期、资源分配和健康管理。For project-level rules see `AGENTS.md`. Ray 底层细节见 `ray-expert`.

**不用于**：RL 算法 (`algorithm-expert`)、Megatron (`megatron-expert`)、FSDP (`fsdp-expert`).

## 三层架构

| 层 | 类 | 位置 | 职责 |
|----|-----|------|------|
| Controller | `Controller` | `relax/core/controller.py` | 顶层编排、训练循环 |
| Service | `Service` | `relax/core/service.py` | 生命周期、placement groups |
| Implementation | `Actor`, `Rollout`, etc. | `relax/components/` | 具体训练/推理组件 |

## Controller 初始化

`Controller.__init__()`:
1. `_initialize_data_system()` — TransferQueue
2. 创建 DCS coordinator
3. 部署 Metrics Service（可选）
4. 注册所有 Ray Serve 服务
5. 启动健康监控

## Service 部署

每个 Service 创建 placement group → `serve.run()` 部署 → 返回 handle。

**服务角色**：`actor` · `critic` · `rollout` · `advantages` · `genrm` · `actor_fwd` · `agent_loop`

## 资源分配

```
--resource '{"actor": [1, 8], "rollout": [1, 8], ...}'   # [num_serves, num_gpus]
--colocate                                                  # Actor/Rollout 共享 GPU
```

Colocate 模式：共享 PG + sleep/wake 机制切换训练/推理，需 `--offload-train`.

## 推理控制面

- `InferenceManager`（`relax/distributed/ray/inference_manager.py`）：每个任务一个、钉在 head 节点的 CPU Ray Actor，
  持有 Rollout / GenRM / Teacher 的引擎池、服务发现快照、请求准入与生命周期（`activate` · `drain` · `deactivate` · `shutdown`）
- `RolloutEnginePool`（`relax/distributed/ray/rollout.py`）：Manager 内的 Rollout 引擎池
  - 引擎类型: `regular` · `prefill` · `decode` · `placeholder`
  - 生命周期: 启动 SGLang → 健康探测 → 权重更新 → 可选重启/缩放
- `RolloutWorker`（`relax/distributed/ray/rollout_worker.py`）：承载生成、评估与 rollout 数据流，通过 Manager 访问引擎
- `InferenceGateway`（`relax/components/inference_gateway.py`）：各角色的 `/<role>` 入口，`GET /engines` 默认返回 v2 服务发现格式

关联: `relax/distributed/ray/actor_group.py` (`RayTrainGroup`)

## 健康监控

位置: `relax/utils/health_system.py` → `HealthManager`

- 周期性 ping 所有已注册服务
- 不健康时触发 `on_unhealthy` 回调自动恢复
- `InferenceManager` 用 `concurrency_groups={"rollout": 8}` 隔离 rollout 操作与生命周期 RPC

## 数据管道

```
RolloutDataSource → RolloutWorker → (InferenceManager) SGLang → 奖励计算
  → TransferQueueController → SimpleStorageUnit
    → TransferQueueClient → TrainRayActor
```

存储后端: `ray_storage_client` (默认) · `mooncake_client` · `yuanrong_client`  
采样器: `grpo_group_n_sampler` · `rank_aware_sampler` · `sequential_sampler`

## 故障排除

| 症状 | 可能原因 | 首要步骤 |
|------|----------|----------|
| Job 启动失败 | Ray 集群未初始化 | `ray status` 检查 |
| GPU 分配错误 | GPU 不足或 PG 冲突 | 对比 GPU 总数 vs 请求量 |
| Service 超时 | 初始化慢或 OOM | 增大超时；检查 GPU 内存 |
| Rollout 引擎崩溃 | SGLang 服务失败 | 检查 SGLang 日志；验证模型路径 |
| 权重同步超时 | NCCL 通信失败 | 检查网络；尝试 `--colocate` |
| TransferQueue 空 | Rollout 未产出数据 | 验证 rollout 服务健康 |

## 关键文件

| 文件 | 用途 |
|------|------|
| `relax/entrypoints/train.py` | 训练入口 |
| `relax/core/controller.py` | Controller 编排 |
| `relax/core/service.py` | Service 生命周期 + PG |
| `relax/components/` | Actor / Rollout / GenRM 等实现 |
| `relax/utils/health_system.py` | 健康监控 |
| `relax/distributed/ray/inference_manager.py` | InferenceManager |
| `relax/distributed/ray/rollout.py` | RolloutEnginePool |
| `relax/distributed/ray/rollout_worker.py` | RolloutWorker |
| `relax/distributed/ray/actor_group.py` | RayTrainGroup |
| `relax/distributed/ray/placement_group.py` | PG 工具 |
| `transfer_queue/` | 分布式数据管道 |
