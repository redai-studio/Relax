# 架构设计

## 概述

Relax 是一个基于 Ray Serve 的大模型强化学习训练框架，支持 Megatron 训练后端、SGLang 推理引擎，以及 PPO/GRPO/GSPO/SAPO/CISPO 等算法族。框架采用分层架构设计，将编排、组件、引擎、后端和分布式能力解耦为独立模块。

## 分层架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    Entrypoints Layer                             │
│              train.py — signal handling, launch training         │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                   Orchestration Layer (Core)                     │
│  ┌──────────────────┐  ┌──────────┐  ┌────────────────────────┐  │
│  │   Controller     │  │ Service  │  │     Registry           │  │
│  │ (training loop,  │  │(lifecycle│  │  (roles/algo registry) │  │
│  │  deployment)     │  │ mgmt)    │  │                        │  │
│  └──────────────────┘  └──────────┘  └────────────────────────┘  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│               Components Layer (Ray Serve Deployments)           │
│  ┌────────┐ ┌──────────┐ ┌────────┐ ┌───────────┐ ┌──────────┐   │
│  │ Actor  │ │ Rollout  │ │ Critic │ │ ActorFwd  │ │Advantages│   │
│  └────────┘ └──────────┘ └────────┘ └───────────┘ └──────────┘   │
│  ┌────────┐                                                      │
│  │ GenRM  │                                                      │
│  └────────┘                                                      │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                   Engine Layer                                   │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐   │
│  │  Rollout Engine     │  │  Reward Functions                │   │
│  │  (SGLang rollout,   │  │  (deepscaler, math, genrm, ...)  │   │
│  │   on-policy distill)│  │                                  │   │
│  └─────────────────────┘  └──────────────────────────────────┘   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│               Backends Layer            Distributed Layer        │
│  ┌──────────────────┐ ┌───────────┐  ┌──────────┐ ┌───────────┐  │
│  │ Megatron Backend │ │  SGLang   │  │ Ray Actor│ │   DCS     │  │
│  │ (Actor, weight   │ │  Inference│  │  Groups  │ │ Checkpoint│  │
│  │  update, HF conv)│ │  Engine   │  │  Mgmt    │ │  Service  │  │
│  └──────────────────┘ └───────────┘  └──────────┘ └───────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 1. 入口层

[`relax/entrypoints/train.py`](../../../relax/entrypoints/train.py) 是主训练入口：

- 解析命令行参数（基于 Megatron-LM 扩展解析器）
- 初始化 Ray 集群连接
- 注册 SIGTERM/SIGINT 信号处理器，确保 SGLang 进程清理
- 创建 Controller 并启动训练循环

### 2. 编排层 (Core)

编排层协调整个训练工作流：

- [**Controller**](../../../relax/core/controller.py)：核心编排器，负责部署所有 RL 服务组件、管理训练循环、处理健康检查与全局重启
- [**Service**](../../../relax/core/service.py)：Ray Serve 部署的生命周期封装，管理 placement group、心跳上报、服务重启
- [**Registry**](../../../relax/core/registry.py)：全局常量与角色注册表，定义 ROLES、ALGOS 等映射

### 3. 组件层 (Components)

组件层包含所有 RL 服务组件，每个组件是一个 `@serve.deployment` 部署：

| 组件 | 文件 | 职责 |
|------|------|------|
| **Actor** | [`actor.py`](../../../relax/components/actor.py) | 策略训练（Megatron 后端） |
| **Rollout** | [`rollout.py`](../../../relax/components/rollout.py) | Rollout 服务编排，管理 RolloutManager |
| **Critic** | [`critic.py`](../../../relax/components/critic.py) | PPO value 估计与裁剪 value loss 训练 |
| **ActorFwd** | [`actor_fwd.py`](../../../relax/components/actor_fwd.py) | 前向推理 log-prob（fully-async 模式） |
| **Advantages** | [`advantages.py`](../../../relax/components/advantages.py) | 优势计算（PPO/GRPO/GSPO/SAPO/CISPO 等） |
| **GenRM** | [`genrm.py`](../../../relax/components/genrm.py) | 生成式奖励模型 |

实际部署的服务图取决于算法与运行模式。Colocate 的 GRPO-like 算法使用 Actor + Rollout；同步 colocate PPO 会增加 Critic + Advantages。对于非 PPO 算法，**全异步模式**（`--fully-async`）可根据 log-probability 与 KL 配置进一步部署 ActorFwd 和 Reference 服务。暂不支持 fully-async PPO；当前支持的数据流详见 [PPO 训练](./ppo-training.md)。

### 4. 引擎层 (Engine)

引擎层提供 Rollout 数据生成和奖励计算能力：

- [**Rollout 引擎**](../../../relax/engine/rollout/)：SGLang rollout 实现、on-policy distillation 等
- [**奖励函数**](../../../relax/engine/rewards/)：可插拔的奖励计算（deepscaler、math_utils、genrm 等）
- [**路由器**](../../../relax/engine/router/)：请求路由与负载均衡
- [**过滤器**](../../../relax/engine/filters/)：数据过滤策略

### 5. 后端层 (Backends)

后端层封装底层训练和推理引擎：

- [**Megatron 后端**](../../../relax/backends/megatron/)：基于 Megatron-LM 的分布式训练，支持 TP/PP/CP/EP 并行
- [**SGLang 引擎**](../../../relax/backends/sglang/)：高性能推理引擎管理、进程生命周期控制

### 6. 分布式层 (Distributed)

- [**Ray Actor 组**](../../../relax/distributed/ray/)：管理 RolloutManager、GenRMManager 等 Ray Actor 组
- [**DCS Checkpoint Service**](../../../relax/distributed/checkpoint_service/)：分布式权重同步服务，支持 NCCL/GLOO/TCP 通信后端

## 目录结构

```
relax/
├── core/                    编排层
│   ├── controller.py        Controller：部署编排、训练循环、健康管理
│   ├── service.py           Service：Ray Serve 部署生命周期封装
│   └── registry.py          全局常量与角色注册表
├── components/              组件层（Ray Serve Deployment）
│   ├── actor.py             策略训练
│   ├── actor_fwd.py         前向推理 log-prob
│   ├── critic.py            价值估计
│   ├── advantages.py        优势计算
│   ├── genrm.py             生成式奖励模型
│   └── rollout.py           Rollout 服务编排
├── engine/                  引擎层
│   ├── rollout/             Rollout 引擎实现
│   ├── rewards/             奖励函数
│   ├── router/              请求路由
│   └── filters/             数据过滤
├── backends/                后端层
│   ├── megatron/            Megatron 训练后端
│   └── sglang/              SGLang 推理引擎
├── distributed/             分布式层
│   ├── ray/                 Ray Actor 组管理
│   └── checkpoint_service/  DCS 分布式 Checkpoint 服务
├── entrypoints/             入口层
│   ├── train.py             主训练入口
│   └── deploy_metrics_service.py  独立 Metrics 服务部署
└── utils/                   基础设施
    ├── arguments.py         命令行参数解析
    ├── data/                数据集加载与流式处理
    ├── metrics/             指标采集与上报
    └── ...                  日志、定时器、健康监控等
```

## 数据流

### 同步模式

```
┌───────────────────────────────────────────────────────────────────────┐
│                       Sync Training Data Flow                         │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────┐   ┌──────────────────┐   ┌────────────────────────┐   │
│  │ JSONL/     │   │ Dataset /        │   │ RolloutDataSource      │   │
│  │ Parquet    │──>│ StreamingDataset │──>│ WithBuffer             │   │
│  │ Files      │   │                  │   │                        │   │
│  └────────────┘   └──────────────────┘   └───────────┬────────────┘   │
│                                                      │                │
│                                                      ▼                │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                  RolloutManager.generate()                       │ │
│  │  SGLang engine inference → reward computation → assemble data    │ │
│  └───────────────────────────┬──────────────────────────────────────┘ │
│                              │                                        │
│                              ▼                                        │
│  ┌───────────┐   ┌──────────────────┐   ┌──────────────────────────┐  │
│  │  Actor    │──>│  DCS Weight Sync │──>│  Rollout (update weights)│  │
│  │ (training)│   │  (NCCL AllGather)│   │                          │  │
│  └───────────┘   └──────────────────┘   └──────────────────────────┘  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### 全异步模式 (`--fully-async`)

```
┌───────────────────────────────────────────────────────────────────────┐
│                    Fully-Async Training Data Flow                     │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Rollout ──→ TransferQueue ──→ Advantages ──→ TransferQueue ──→ Actor │
│     ↑                              │                             │    │
│     │                              │                             │    │
│     └──── DCS Weight Sync ◄── ActorFwd ◄──── DCS Weight Sync ───┘     │
│                                                                       │
│  5 components run independently, passing data via TransferQueue       │
│  Actor updates weights to ActorFwd → Reference → Rollout every N steps│
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

## 通信模式

### 组件间通信

组件通过 Ray Serve HTTP 接口和 Ray Actor handle 通信：

```python
# Controller 通过 Service 封装部署组件
service = Service(role="actor", cls=Actor, num_gpus=8, ...)
service.run()  # 启动训练任务

# 组件通过 DCS Checkpoint Service 同步权重
client = CheckpointEngineClient(coordinator_url="http://...", role="actor", ...)
await client.update_weights_for_rollout(rollout_only=True)
```

### Controller 编排

Controller 通过 Ray Serve 管理所有组件的生命周期：

```python
# Controller 注册所有服务
ctrl = Controller(args, runtime_env)
ctrl.register_all_serve()   # 部署 Actor, Rollout, Critic, ...
ctrl.training_loop()        # 启动训练循环
ctrl.shutdown()             # 清理 SGLang 引擎
```

## 部署

通过启动脚本部署训练任务：

```bash
# 同步训练（8 GPU）
bash scripts/training/text/run-qwen3-4B-8xgpu.sh

# 全异步训练（8 GPU）
bash scripts/training/text/run-qwen3-4B-8xgpu-async.sh

# 或直接调用入口
python relax/entrypoints/train.py [ARGS...]
```

## 可扩展性

架构支持多维度扩展：

- **Rollout 组件**：通过 `--rollout-num-gpus` 和 `--rollout-num-gpus-per-engine` 配置 SGLang 推理引擎，支持弹性扩缩容
- **Actor 组件**：支持 Megatron TP/PP/CP/EP 多种并行策略
- **奖励函数**：可插拔的奖励计算模块，通过 `--rm-type`（内置）或 `--custom-rm-path`（自定义）指定

## 监控和可观测性

内置监控组件：

- **Metrics 服务**：收集和聚合训练指标（WandB、Prometheus）
- **健康检查管理器**：监控服务健康状况，支持自动重启和全局恢复
- **通知系统**：通过 Apprise 在故障时发出警报

参见：

- [Metrics 服务使用](./metrics-service-detailed.md)
- [健康检查管理器](./health-check-manager.md)
- [通知系统](./notification-system.md)

## 下一步

- [数据集设计](./dataset-design.md) - 了解数据加载
- [Distributed Checkpoint](./distributed-checkpoint.md) - 理解检查点机制
- [配置说明](./configuration.md) - 配置您的部署
