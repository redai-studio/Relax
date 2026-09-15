# Relax 介绍

## 什么是 Relax？

**Relax**（**R**einforcement **E**ngine **L**everaging **A**gentic **X**-modality）是一个面向多模态大模型的高性能强化学习后训练框架。Relax 基于 Ray Serve 构建面向服务的架构，以 Megatron-LM 为训练后端、SGLang 为推理引擎，通过 [TransferQueue](https://github.com/redai-studio/TransferQueue) 数据传输系统实现训练与推理的完全解耦，支持从文本到图像、视频、音频的全模态强化学习训练。

---

## 核心特性

### 🌐 全模态训练支持

Relax 原生支持文本、图像、视频、音频的全模态强化学习训练，是业界少数能够在统一框架内完成 Omni 模型 RL 训练的系统。

| 模态 | 能力 | 代表模型 |
|------|------|----------|
| **文本** | 数学推理、代码生成、多轮对话、工具调用 | Qwen3, GLM5 |
| **视觉** | 视觉问答、图像理解、多模态推理 | Qwen3-VL, Qwen3.5, Qwen3.6, Kimi K2.6 |
| **Omni** | 图文音频联合理解 | Qwen3-Omni |

多模态数据通过 `--multimodal-keys` 参数灵活配置，框架内置了完整的图像、视频、音频处理管线（`relax/utils/multimodal/`），支持图像 token 数量控制、视频帧率采样、音频采样率配置等精细调节。

### ⚙️ 面向服务的六层架构（Server-Based）

Relax 采用面向服务的六层架构设计，所有角色均部署为独立的 Ray Serve 服务，天然支持服务级别的弹性调度和故障恢复。详情见 [架构设计](./architecture.md)。

### ⚡ 基于 TransferQueue 的全异步训练（Fully Async）

开源地址 [TransferQueue](https://github.com/redai-studio/TransferQueue)，详细介绍见 [全异步训练](./fully-async-training.md)。

在全异步模式下，Rollout（推理）、Actor（训练）、ActorFwd（前向计算）、Reference（参考模型）和 Advantages（优势计算）五个角色运行在**独立的 GPU 集群**上，通过 TransferQueue 交换数据，通过 DCS（Distributed Checkpoint Service）异步同步权重。

**核心机制**：

- **StreamingDataLoader**：Actor 使用流式数据加载器，在 Rollout 增量写入数据的同时即可开始消费训练，消除等待时间
- **可配置 Staleness**：`--max-staleness` 参数精确控制数据新鲜度，在 on-policy 准确性和训练吞吐量之间灵活权衡
- **DCS 权重同步**：训练完成后通过 NCCL broadcast 将权重分发到 Rollout/ActorFwd/Reference，与训练计算重叠执行

### 🔀 弹性 Rollout 扩缩容

详情见 [弹性扩缩容](./elastic-rollout.md)。

Relax 支持在训练过程中通过 HTTP REST API **动态调整 Rollout 引擎数量**，无需中断训练。在 RL 训练中，60~70% 的时间消耗在 Rollout 阶段，弹性扩缩容可以根据实际负载灵活增减推理资源。

**两种扩容模式**：

| 模式 | 场景 | 说明 |
|------|------|------|
| **ray_native** | 同集群扩容 | 指定目标引擎数，自动在 Ray 集群内创建新引擎 |
| **external** | 跨集群联邦推理 | 接入外部已部署的 SGLang 引擎，跨集群弹性利用算力 |

### 🎯 Agentic RL

与传统 VLM 的单次问答不同，**Agentic VLM** 通过连续交互实现闭环迭代——"执行→观察→决策"。Relax 内建了对 Agentic 多轮训练的完整支持：

- **多轮采样与 loss mask**：通过自定义 generate 函数，模型在每轮根据当前上下文生成动作指令，环境返回 Observation 并增量注入上下文，直至任务完成或达到终止条件。得到完整 trajectory 后，通过 loss mask 精确区分模型输出（mask=1）与环境反馈（mask=0），确保只有模型动作参与训练
- **环境与 Rollout 解耦**：Relax 为环境定义了标准接口（`BaseInteractionEnv`），包括 `reset()`、`step(response) → (observation, done, info)` 和 `format_observation()`。环境如何解析 Action 等逻辑完全独立于采样与训练之外，便于复用和扩展
- **VLM 多模态上下文维护**：在视觉多轮交互中，每轮 Observation 可能携带新的图像/视频。Relax 同时维护 Rollout 侧的 `image_data`（逐轮 append 编码后的图像）和训练侧的 `multimodal_train_inputs`（逐轮合并 processor 产生的张量），实现多轮多模态数据的正确拼接
- **灵活的终止条件**：支持 `max_turns`（最大交互轮数）、token budget（可用 token 预算耗尽即截断）、`env done`（环境返回任务完成）三种终止机制的组合
- **典型应用**：DeepEyes 任务——使用 Qwen3-VL-30B-A3B 进行 Agentic Multi-Turn GRPO Training


## 项目结构

```
relax/                   核心框架：六层架构
examples/                示例（DeepEyes 等）
scripts/                 各种模型 & 规模的训练启动脚本
configs/                 运行时环境配置
```

---

## 下一步

- [安装指南](./installation.md) — 环境搭建与依赖安装
- [快速开始](./quick-start.md) — 运行你的第一个 RL 训练任务
- [配置参考](./configuration.md) — 完整的配置参数手册
- [架构设计](./architecture.md) — 深入理解 Relax 的六层架构
- [DeepEyes 示例](../examples/deepeyes.md) — 多模态视觉 RL 训练实战
