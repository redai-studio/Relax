---
outline: deep
---

# GenRM 服务 API

GenRM（生成式奖励模型）服务提供基于 LLM 的响应评估。它以 Ray Serve 部署方式运行，通过 FastAPI ingress 暴露 HTTP 端点。

## 概览

| 属性 | 值 |
|------|---|
| **模块** | `relax.components.genrm` |
| **部署方式** | `@serve.deployment(logging_config=...)` |
| **入口** | FastAPI |

### 架构

与 Actor 和 Rollout 不同，GenRM 没有自主训练循环。它通过 HTTP 处理生成与生命周期请求，并异步跟踪已受理的扩缩容操作。

服务使用 SGLang 引擎执行偏好评估：

1. 通过 `/generate` 接收 OpenAI 格式的聊天消息
2. 应用聊天模板并进行分词
3. 发送到 SGLang 引擎，使用可配置的采样参数
4. 返回原始模型响应文本

### 共置模式

当与 Actor 共置（共享 GPU 资源）时，GenRM 支持卸载/加载操作：

- **Offload（卸载）**：在 Actor 训练前释放 GPU 显存
- **Onload（加载）**：在 rollout 前将模型权重重新加载到 GPU

根据 GPU 分配，框架会自动识别两种 colocate 子模式：

- **Split**（`rollout_num_gpus + genrm_num_gpus == actor_total_gpus`）：GenRM 与 Rollout 占用不重叠的 bundle。
- **Shared**（`rollout_num_gpus == genrm_num_gpus == actor_total_gpus`）：GenRM 与 Rollout 占用相同的 bundle，通过 SGLang 的 `mem_fraction_static` 切分每张 GPU 的显存。GenRM 的 `mem_fraction_static` 从 `--genrm-engine-config` 读取。GenRM 不会从 Actor 同步权重，onload 仅恢复 KV cache 和 CUDA graph。

完整配置参见 [GenRM 示例](/zh/examples/generative-reward-model)。

### 多实例（`route_key`）

一次 GenRM 部署可以同时托管多个独立的评判模型（用 `--genrm-instances` 配置），由请求体里的 `route_key` 字段选择具体调用哪一个。不传 `route_key`（或服务仍使用旧的 `--genrm-model-path` 单实例配置）时，请求会落到唯一的 `"__default__"` 实例。配置格式、`/health` 与 `/metrics` 的按实例明细，以及 agentic 场景下按模块路由的写法，参见 [GenRM 示例 · 多实例 GenRM](/zh/examples/generative-reward-model#多实例-genrm一个服务托管多个评判模型)。

## 弹性扩缩容

扩缩容改变冻结模型的服务副本数，不改变模型权重。初始副本受保护，缩容只删除弹性副本。目前弹性创建仅支持每引擎一张 GPU。

### 请求与状态

以下路径相对于 GenRM 服务路由，通常为 `/genrm`。

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/scale_out` 或 `/scale_in` | 提交绝对目标副本数 |
| `GET` | `/scale_out/{request_id}` 或 `/scale_in/{request_id}` | 查询操作状态与清理状态 |
| `POST` | `/scale_out/{request_id}/reconcile` 或 `/scale_in/{request_id}/reconcile` | 重试原操作尚未完成的清理 |
| `GET` | `/engines` | 查询存活引擎与容量 |

```json
{
  "model_name": "default",
  "num_replicas": 2,
  "timeout_secs": 600,
  "idempotency_key": "judge-scale-out-001"
}
```

- `model_name` 选择已配置的实例。只有一个实例时，`default` 指向该实例；多个实例时使用其配置的 route key。
- `num_replicas` 是**绝对总数**，不是增量。必须是正 JSON 整数；布尔值、字符串和浮点数（包括 `2.0`）返回 `422`。
- `timeout_secs` 是整个操作的截止时限；省略时使用 600 秒。
- `idempotency_key` 可选。在内存历史记录仍保留时，相同方向、key 和 `(model_name, num_replicas, timeout_secs)` 复用原操作。同一 key 对应不同请求体时返回 `409`。历史记录有容量限制，不提供跨重启的持久幂等保证。

受理后返回 HTTP `200`、`status: PENDING` 和 `request_id`。应轮询该 ID；受理不等于完成。目标已满足对应方向条件时返回 `NOOP`，没有操作 ID。带 key 的 `NOOP` 重放最初的决定，即使容量后来发生变化。

| 方向 | 正常状态流转 | 终态 |
| --- | --- | --- |
| 扩容 | `PENDING → CREATING → HEALTH_CHECKING → READY → ACTIVE` | `ACTIVE`、`PARTIAL`、`FAILED` |
| 缩容 | `PENDING → DRAINING → REMOVING → COMPLETED` | `COMPLETED`、`FAILED` |

`PARTIAL` 表示扩容失败前已有部分新副本发布；应检查 `current`、`ready`、`created`、`failed` 和 `cleanup_required`，不能据此认定目标已达到。实例或目标范围无效返回 `400`，未知操作 ID 返回 `404`，冲突返回 `409`；Manager 发现、容量查询不可用或缺少 reconcile 证明时可能返回 `503`。

### 清理与容量

超时只请求中止，不证明物理操作已经停止。`cleanup_required: true` 会继续阻止该模型的新扩缩操作。Reconcile 保留原操作 ID、终态和原定 victim，只重试清理，不重新扩缩。物理线程或 victim 上的请求仍在运行时可能返回 `409`。因此清理成功后，原来的 `FAILED` 仍可保持 `FAILED`；应以清理标记判断是否已解除阻塞。

对于旧式默认实例，`/engines` 在顶层返回下列计数；多实例部署则放在 `instances[route_key]` 下：

| 字段 | 含义 |
| --- | --- |
| `current` | 已发布的服务容量，包括正在排空或尚未释放的失败缩容 victim |
| `ready` | 当前可路由的副本数 |
| `occupied` | 仍占用的副本／资源槽，包括尚未发布的候选副本 |
| `pending_cleanup` | 等待清理的失败 victim 或未发布候选副本槽数 |

每个发现的引擎还包含 `host`、`port`、`inflight` 和 `served`。这些请求计数仅属于当前 GenRM 组件，不覆盖所有直连客户端。`/metrics` 在 Manager 查询成功时报告动态容量，失败时保留启动计数并包含 `capacity_error`。`/engines` 容量查询失败时可能回退到可路由计数；降级快照不是资源已释放的证明。扩缩提交无法取得权威容量时会拒绝请求，不使用该回退值。

::: warning 范围限制
排空计数仅覆盖一个 GenRM Gateway。尚未实现直连引擎客户端、跨 Gateway admission lease 和 Manager 重启恢复；单 Gateway 缩容成功不能证明这些能力。
:::

## HTTP 端点

<SwaggerUI specUrl="/Relax/openapi/genrm.json" />

## 源码

- 实现：[`relax/components/genrm.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/genrm.py)
- 基类：[`relax/components/base.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/base.py)

## 下一步

- [GenRM 配置与示例](../examples/generative-reward-model.md)
- [Rollout 服务 API](./rollout.md)
