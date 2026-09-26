---
outline: deep
---

# 统一推理服务

对应 [RFC #71](https://github.com/redai-studio/Relax/issues/71)。Rollout、GenRM、Teacher 保留各自业务入口，共用 `InferenceGateway`、`InferenceManager` 状态机和 `SGLangEngine`。Gateway 使用 CPU，不占用引擎的 GPU placement group。只有 Rollout 允许动态权重更新和 DCS 注册。

## API 与路由

每个角色分别部署在 `/rollout`、`/genrm`、`/teacher`，提供：

| 路径 | 功能 |
| --- | --- |
| `GET /engines` | 模型、逻辑引擎、状态、router URL 和 `topology_revision` |
| `GET /v1/models` | 可配置的模型 ID |
| `GET /health` | 控制面健康状态与模型状态 |
| `POST /generate` | SGLang 原始请求，保留 logprob/base64 等字段 |
| `POST /v1/chat/completions` | OpenAI chat 请求与 SSE 流式响应 |
| `POST /chat/completions` | chat 兼容别名 |

模型选择顺序为 `model`、`route_key`、默认模型。未知显式模型或 route key 返回 400；没有默认模型时必须指定模型。Gateway 和 direct client 使用同一套选择函数。PD 模型通过 router 请求，禁止直连 prefill/decode worker；普通多节点引擎只发布 head HTTP 端点，不发布 TP/PP follower。

`sleeping`、`draining`、`onloading`、`failed`、`dead` 或尚未完成权重同步的模型不接收新请求，Gateway 返回 503 和 `Retry-After`。请求不会隐式激活模型。`topology_revision` 会随状态和端点变化更新，已返回的 snapshot 不会被后续修改。

旧 GenRM `messages + sampling_params + route_key` 请求和 `{"response": ...}` 响应保留；旧 Teacher URL、Manager onload/offload 和 GenRMEngine 名称保留。启用新 placement/discovery 的 Teacher 恢复后允许更换端点；旧的原始 URL 模式保留恢复约束。

```python
from relax.utils.inference_client import InferenceClient

async with InferenceClient(service_url, direct=True) as client:
    result = await client.generate(
        {"input_ids": [1, 2, 3], "return_logprob": True}, model="default"
    )
    async for chunk in client.stream(
        {"messages": [{"role": "user", "content": "Hello"}]},
        path="v1/chat/completions", model="default",
    ):
        consume_sse_bytes(chunk)
```

`direct=False` 通过 Gateway 转发。direct 模式使用原始 SGLang payload；旧 GenRM 消息模板转换继续由 GenRM Gateway 完成。Client 每次请求读取 discovery，不在传输失败后自动重放生成请求。

## Placement 与阶段切换

- **Decoupled**：角色使用独立 GPU 池；Teacher 专用 PG 由 Manager 创建和回收。
- **Split**：`--colocate` 下，Rollout、非 deferred GenRM、非 deferred Teacher 按顺序占用 Actor 池内互不重叠的区间，训练阶段复用该池。
- **Defer**：`--opd-teacher-defer` 和 `--defer-reward-to-post-process` 分别启用 Teacher、GenRM 延后处理。延后角色从共享池起点分配，但分属不同执行阶段；同一角色的多个模型仍需互不重叠。

共享 Actor 池要求训练和推理双方 offload。规划器在创建引擎前检查总量、replica 完整性、TP×PP、节点宽度、局部跨度、配置覆盖和阶段重叠。decoupled 或 hybrid 的显式 Rollout GPU 数量超过独立资源预算时，在 Teacher 或 PG 创建前拒绝。PG 就绪后再检查实际节点与物理 GPU 连续性。首次 deferred 引擎按需启动，避免启动时共同驻留。

defer 流程为：完成生成并排空未保留的请求 → Rollout offload → Teacher onload、完整写回、offload → 必要的学生 top-k 补算 → GenRM onload、奖励计算、offload → 发布训练数据。按原 Sample 对象写回，不依赖 sample index 唯一。Teacher 失败或字段长度不完整时不发布数据。group RM 保留 prompt 分组；评测奖励不做训练用的归一化。deferred 生成和评测互斥。

生命周期支持重复调用和分批恢复 `weights`、`kv_cache`、`cuda_graph`，全部恢复前不发布 READY。失败时尝试清理所有候选；清理未确认时保留引擎句柄、PG ownership 和阶段锁，discovery 保持不可用，并阻止后续激活。Engine 在子进程启动后立即记录句柄，初始化中途失败也能重试清理。Manager 重试清理成功不会自动恢复失败 batch 或释放 Coordinator 保留的阶段锁。Manager 不删除借用的 PG。

一期拒绝同卡同阶段共同驻留。defer 只支持同步 colocate、框架 SGLang 完整 batch 流程；不支持 fully async、partial rollout、agentic/custom rollout，以及依赖尚未生成奖励的 dynamic sampling filter。

## 下一步

- [Rollout API](./rollout.md)
- [GenRM API](./genrm.md)
