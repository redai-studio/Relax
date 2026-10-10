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

### 扩缩状态清理契约

`ScaleOutStatusResponse` 与 `ScaleInStatusResponse` 均携带 `cleanup_required: bool` 字段，遵循与 Autoscaler 共享的三态清理契约：

| 取值 | 含义 |
|-------|---------|
| `true` | 物理清理（引擎拆除 / placement group 释放）仍未完成；携带该标志的终态请求必须通过 reconcile 等待清理完成。 |
| `false` | 权威清理已完成——所报状态背后不存在遗留的物理工作。 |
| 缺失（旧 schema） | 未知；调用方不得将缺失视为已清理。共享 Autoscaler 将缺失字段视为 unknown，并在观察到显式 `false` 前保持终态请求 pending。 |

Rollout 扩容/缩容的终态在报告前自行完成清理：扩容引擎要么继续服务（`ACTIVE`/`PARTIAL`），要么已回滚（`FAILED`/`CANCELLED`）；缩容 `COMPLETED` 表示引擎已移除、`FAILED` 表示移除已回滚——因此 rollout 终态响应报告 `cleanup_required: false`。显式声明该字段（而非省略）正是为了满足 Autoscaler 的「缺失 ≠ 已清理」契约。

## HTTP 端点

<SwaggerUI specUrl="/Relax/openapi/rollout.json" />

## 源码

- 实现：[`relax/components/rollout.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/rollout.py)
- 基类：[`relax/components/base.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/base.py)
