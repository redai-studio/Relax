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

与 Actor 和 Rollout 不同，GenRM 是一个**被动 HTTP 服务** — 它不运行后台循环，仅响应传入的 `/generate` 请求。

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

## HTTP 端点

<SwaggerUI specUrl="/Relax/openapi/genrm.json" />

## 源码

- 实现：[`relax/components/genrm.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/genrm.py)
- 基类：[`relax/components/base.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/base.py)
