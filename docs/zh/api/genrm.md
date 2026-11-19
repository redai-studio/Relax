---
outline: deep
---

# GenRM 服务 API

GenRM（生成式奖励模型）服务提供基于 LLM 的响应评估。它以 Ray Serve 部署方式运行，通过 FastAPI ingress 暴露 HTTP 端点。

## 概览

| 属性 | 值 |
|------|---|
| **模块** | `relax.components.inference_gateway` |
| **部署方式** | `@serve.deployment(logging_config=...)` |
| **入口** | FastAPI |

### 架构

与 Actor 和 Rollout 不同，GenRM 是一个**被动 HTTP 服务** — 它不运行后台循环，仅响应传入的 `/generate` 请求。

服务使用 SGLang 引擎执行偏好评估：

1. 通过 `/generate` 接收 OpenAI 格式的聊天消息
2. 应用聊天模板并进行分词
3. 发送到 SGLang 引擎，使用可配置的采样参数
4. 返回原始模型响应文本

### 独立 GPU 资源

每个任务只有一个 CPU InferenceManager，统一持有 Rollout、GenRM 和 Teacher 的模型池、准入及生命周期状态。角色句柄仅转发调用，不创建独立 Manager；独立与共置布局使用同一请求执行器。每个 judge 保持自己的 checkpoint 和 GPU 切片，不接收 Actor 权重更新。请求读取当前副本地址，保留 `route_key` 与采样配置。`/genrm/engines` 默认返回 v2，旧投影需显式使用 `?schema_version=1`；消息经 Manager 准入后由 Gateway 本地适配，不依赖命名 GenRM actor。

### 共置 GPU 布局

`--colocate` 让 Actor 在训练阶段复用推理 GPU。同一个 placement group 不代表 Rollout 与 GenRM 必须占用相同 bundle：

- **Split**（`rollout_num_gpus + genrm_num_gpus == actor_total_gpus`）：Rollout 与 GenRM 使用不同 bundle，可以在线评分。八卡训练使用 `--resource '{"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4]}'` 可选择 4+4 切分。
- **Shared**（`rollout_num_gpus == genrm_num_gpus == actor_total_gpus`）：Rollout 与 GenRM 分时复用相同 bundle。需要设置 `--defer-reward-to-post-process` 及兼容的 `--custom-reward-post-process-path`。生成结束且请求排空后才能加载 judge，评分完成后才能发布数据；降低 `mem_fraction_static` 不能替代阶段切换。

托管共享路径由 phase coordinator 负责排空、卸载和加载。示例 `post_process_genrm_swap.py` 要求在该阶段内执行，只评分而不自行切换显存，judge 失败会向上传播，不再降为零奖励。只有整批所需奖励齐全且为有限数值时才发布；缺失发布确认会使本次 attempt 失败，不会重放写入。

自定义和外部引擎也使用同一任务控制面，必须满足其准入及确认释放契约。GenRM 保持自己的 checkpoint，不接收 Actor 权重。

完整配置参见 [GenRM 示例](/zh/examples/generative-reward-model)。

### 多实例（`route_key`）

一次 GenRM 部署可以同时托管多个独立的评判模型（用 `--genrm-instances` 配置），由请求体里的 `route_key` 字段选择具体调用哪一个。不传 `route_key`（或服务仍使用旧的 `--genrm-model-path` 单实例配置）时，请求会落到唯一的 `"__default__"` 实例。配置格式、`/health` 与 `/metrics` 的按实例明细，以及 agentic 场景下按模块路由的写法，参见 [GenRM 示例 · 多实例 GenRM](/zh/examples/generative-reward-model#多实例-genrm一个服务托管多个评判模型)。

## HTTP 端点

<SwaggerUI specUrl="/Relax/openapi/genrm.json" />

## 源码

- 实现：[`relax/components/inference_gateway.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/inference_gateway.py)
- 基类：[`relax/components/base.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/base.py)
