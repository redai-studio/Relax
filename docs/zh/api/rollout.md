---
outline: deep
---

# Rollout 服务 API

Rollout 服务通过 SGLang 引擎生成训练样本。它以 Ray Serve 部署方式运行，通过 FastAPI ingress 暴露 HTTP 端点，用于生命周期管理、评估触发和异步权重更新协调。

## 概览

| 属性 | 值 |
|------|---|
| **模块** | `relax.components.rollout` |
| **部署方式** | `@serve.deployment` |
| **入口** | FastAPI |

### 生命周期

Rollout 运行后台循环：

1. 通过 `RolloutManager.generate()` 使用 SGLang 引擎生成样本
2. 通过可插拔的奖励函数（`rm_hub/`）计算奖励
3. 将数据发布到 `TransferQueue` 供 Actor 消费
4. 可选地按配置的间隔触发评估
5. 管理过期边界以避免数据漂移

### 异步权重协调

在全异步模式下，Rollout 服务与 Actor 协调权重更新：

1. Actor 调用 `/can_do_update_weight_for_async` 检查 rollout 是否可以暂停
2. 如果当前步的数据生产已完成，Rollout 暂停
3. Actor 推送新权重
4. Actor 调用 `/end_update_weight` 恢复 rollout

## HTTP 端点

<SwaggerUI specUrl="/Relax/openapi/rollout.json" />

## 源码

- 实现：[`relax/components/rollout.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/rollout.py)
- 基类：[`relax/components/base.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/base.py)
