---
outline: deep
---

# Actor 服务 API

Actor 服务负责训练策略模型。它以 [Ray Serve](https://docs.ray.io/en/latest/serve/) 部署方式运行，通过 FastAPI ingress 暴露 HTTP 端点，用于生命周期管理和故障恢复。

## 概览

| 属性 | 值 |
|------|---|
| **模块** | `relax.components.actor` |
| **部署方式** | `@serve.deployment(max_ongoing_requests=10, max_queued_requests=20)` |
| **入口** | FastAPI |

### 执行模式

- **fully_async（全异步）** — 异步训练，不等待 rollout 数据。每步训练后将权重推送到 rollout 引擎。
- **sync（同步）** — 每步训练前等待 rollout 数据就绪。用于共置（colocated）模式。

### 生命周期

Actor 运行后台训练循环：

1. 从 `TransferQueueClient` 获取训练数据
2. 通过 `RayTrainGroup` 执行 forward/backward/optimizer step
3. 将更新后的权重推送到 rollout 引擎
4. 按配置的间隔保存 checkpoint

## HTTP 端点

<SwaggerUI specUrl="/Relax/openapi/actor.json" />

## 源码

- 实现：[`relax/components/actor.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/actor.py)
- 基类：[`relax/components/base.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/base.py)
