---
outline: deep
---

# ActorFwd 服务 API

ActorFwd 服务使用策略模型的前向推理副本计算 actor/reference log-probabilities。它以 Ray Serve 部署方式运行，通过 FastAPI ingress 暴露 HTTP 端点。

## 概览

| 属性 | 值 |
|------|---|
| **模块** | `relax.components.actor_fwd` |
| **部署方式** | `@serve.deployment` |
| **入口** | FastAPI |

### 用途

ActorFwd 运行策略模型的前向推理副本，用于计算 rollout 数据的 log-probabilities。用于以下场景：

- **KL 散度计算**：计算 `log π(a|s)` 用于 KL 惩罚
- **参考模型 log-probs**：计算 `log π_ref(a|s)` 用于参考模型
- **全异步模式**：通过 NCCL 接收来自 Actor 的权重更新

### 生命周期

ActorFwd 运行后台循环：

1. 等待 rollout 数据就绪
2. 计算 actor 或 reference log-probabilities（取决于角色配置）
3. 将结果发布回 `TransferQueue`
4. 在全异步模式下，通过 `/recv_weight_fully_async` 接收权重更新

## HTTP 端点

<SwaggerUI specUrl="/Relax/openapi/actor_fwd.json" />

## 源码

- 实现：[`relax/components/actor_fwd.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/actor_fwd.py)
- 基类：[`relax/components/base.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/base.py)
