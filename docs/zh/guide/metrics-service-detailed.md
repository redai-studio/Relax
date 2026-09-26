# Metrics Service 使用指南

## 概述

新的 Metrics Service 将 metrics 上报逻辑（包括 tensorboard、wandb、clearml）重构为一个独立的服务，类似 rollout service 的设计。该服务只在 step end 时记录一次，支持批量上报。

## 架构

```
┌─────────────────┐    HTTP/REST     ┌─────────────────┐
│    Client Code  │ ───────────────> │ Metrics Service │
│ (Training/Eval) │                  │  (Ray Serve)    │
└─────────────────┘    JSON API      └────────┬────────┘
                                               │
                                     ┌─────────┴─────────┐
                                     │  Metrics Buffer   │
                                     └─────────┬─────────┘
                                               │
                                ┌──────────────┼──────────────┐
                                ▼              ▼              ▼
                         ┌──────────┐   ┌──────────┐   ┌──────────┐
                         │ Tensor-  │   │   WandB  │   │ ClearML  │
                         │  Board   │   │          │   │          │
                         └──────────┘   └──────────┘   └──────────┘
```

## 代码结构

```
relax/utils/metrics/
├── __init__.py          # 包入口，导出 MetricsClient, get_metrics_client
├── service.py           # MetricsService (Ray Serve 部署) + MetricsBuffer
└── client.py            # MetricsClient (HTTP 客户端) + get_metrics_client

relax/utils/metrics/
├── metrics_service_adapter.py  # MetricsServiceAdapter (向后兼容适配器)

relax/utils/
└── tracking_utils.py           # 集成入口 (init_tracking, log, flush_metrics)
```

## 主要特性

1. **独立服务**：使用 Ray Serve 部署，与主应用解耦
2. **批量上报**：只在 step end 时记录一次，减少网络开销
3. **向后兼容**：保持与现有 `tracking_utils.log()` 相同的接口
4. **多后端支持**：同时支持 TensorBoard、WandB、ClearML
5. **异步处理**：metrics 收集和上报分离

## 配置选项

### 必需配置

```python
args.use_metrics_service = True  # 启用 metrics service
# 服务 URL 会自动通过 get_serve_url() 获取，无需手动配置
```

### 后端配置（与之前相同）

```python
# TensorBoard
args.use_tensorboard = True
args.tb_project_name = "my-project"
args.tb_experiment_name = "experiment-1"

# WandB
args.use_wandb = True
args.wandb_project = "my-project"
args.wandb_team = "my-team"
args.wandb_group = "my-group"

# ClearML
args.use_clearml = True
# ClearML 会自动从环境变量读取配置
```

## 迁移指南

### 从旧系统迁移

1. **无痛迁移**：如果希望保持现有代码不变，只需：

   - 在配置中添加 `use_metrics_service=True`
   - 在适当位置（如 step end）添加 `tracking_utils.flush_metrics(args, step)`

2. **逐步迁移**：可以同时运行新旧系统，通过配置控制：

   - 设置 `use_metrics_service=False` 使用旧系统
   - 设置 `use_metrics_service=True` 使用新系统

### 代码示例对比

**之前**：

```python
# 每次需要记录时直接调用
tracking_utils.log(args, metrics, "step")
```

**之后（批量模式）**：

```python
# 在训练循环中
for step in range(total_steps):
    # ... 训练代码 ...

    # 记录 metrics（缓冲，不立即发送）
    metrics = {
        "step": step,
        "train/loss": loss,
        "train/accuracy": accuracy,
    }
    tracking_utils.log(args, metrics, "step")

    # 在 step end 时上报所有缓冲的 metrics
    tracking_utils.flush_metrics(args, step)
```

## API 参考

### Metrics Service HTTP API

- `POST /metrics/log_metric` - 记录单个 metric
- `POST /metrics/log_metrics_batch` - 批量记录 metrics
- `POST /metrics/report_step` - 上报指定 step 的所有 metrics
- `GET /metrics/health` - 健康检查
- `GET /metrics/query_metrics` - 获取已记录的 metrics
- `POST /metrics/clear_metrics` - 清除 metrics

### Python Client API

```python
class MetricsClient:
    def __init__(self, service_url: str = "http://localhost:8000/metrics")
    def log_metric(step, metric_name, metric_value, tags=None, immediate=False)
    def log_metrics_batch(step, metrics, tags=None, immediate=False)
    def report_step(step)
    def health_check()
    def clear_buffer(step=None)
    def get_buffered_metrics_count(step=None)
```

### 向后兼容 Adapter

```python
class MetricsServiceAdapter:
    def __init__(args)  # 服务 URL 自动通过 get_serve_url() 获取
    def log(metrics, step_key="step")  # 与 tracking_utils.log 相同接口
    def flush()
    def direct_log(step, metrics)
```

## 性能考虑

1. **网络延迟**：Metrics Service 是独立服务，会有网络往返开销
2. **批量优势**：只在 step end 上报一次，减少总请求数
3. **缓冲机制**：Client 端缓冲 metrics，减少网络调用
4. **异步处理**：Service 内部异步处理上报，不阻塞客户端

## 故障排除

### 常见问题

1. **服务不可达**：检查 Ray Serve 是否正确部署，以及网络连接
2. **Metrics 未上报**：确保调用了 `flush_metrics()` 或 `report_step()`
3. **后端配置错误**：检查 TensorBoard/WandB/ClearML 配置

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查服务健康
from relax.utils.metrics.client import MetricsClient
from relax.utils.utils import get_serve_url

service_url = get_serve_url(route_prefix="/metrics")
client = MetricsClient(service_url)
health = client.health_check()
print(f"Service health: {health}")
```

## 示例

完整的示例请参考 `relax/entrypoints/deploy_metrics_service.py`。

运行示例：

```bash
python relax/entrypoints/deploy_metrics_service.py
```

## TimelineTrace

TimelineTrace 用于记录和可视化训练过程中的时间线事件，支持 [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview?tab=t.0#heading=h.yr4qxyxotyw)，可在 Chrome 浏览器的 `chrome://tracing` 中可视化。

### 配置

```bash
--timeline-dump-dir ./timeline_traces  # 为空代表关闭，为一个目录代表开启
```

### 使用示例

```python
import time
from relax.utils.metrics.client import MetricsClient

client = MetricsClient()

# 记录事件开始
event_begin = {
    "name": "forward_pass",
    "ph": "B",
    "ts": int(time.time() * 1e6),
    "pid": 0,
    "tid": 0,
    "args": {"step": 100}
}
client.log_metric(step=100, metric_name="timeline", metric_value=[event_begin])

# 执行操作
perform_forward_pass()

# 记录事件结束
event_end = {
    "name": "forward_pass",
    "ph": "E",
    "ts": int(time.time() * 1e6),
    "pid": 0,
    "tid": 0,
    "args": {"step": 100}
}
client.log_metric(step=100, metric_name="timeline", metric_value=[event_end])

# 上报 step，自动导出 timeline
client.report_step(step=100)
# 生成文件: ./timeline_traces/timeline_step_100.json
```

### 可视化

![Timeline Demo](/timeline_demo.png)

1. 打开 Chrome 浏览器
2. 访问 `chrome://tracing`
3. 点击 "Load" 按钮
4. 选择生成的 JSON 文件

## 总结

新的 Metrics Service 提供了：

1. **更好的架构**：服务化设计，与主应用解耦
2. **性能优化**：批量上报，减少网络开销
3. **易于维护**：集中管理所有 metrics 上报逻辑
4. **向后兼容**：现有代码无需修改即可迁移
5. **扩展性**：易于添加新的 metrics 后端

## 投机解码指标

启用投机解码后，`rollout/spec_accept_rate` 为 accepted 总量除以 proposed 总量，`rollout/spec_accept_length` 为 backend completion token 总量除以 verify 总量，替代原来的样本比例算术平均。评估复用相同计算，沿用数据集指标前缀。

Agentic 导出在 `Sample.spec_generations` 中携带已提交请求的 session ID、request ID 和四项计数。每个日志批次按 `(session_id, request_id)` 去重，只统计导出轨迹上的状态，不包含未选择分支和始终未提交的请求。中断续跑的多个 backend attempt 在同一个逻辑请求内累计，终态提交后才导出。相同文本的独立请求各自计数。现有导出选择 canonical 消息状态，因此包含该状态上保留的全部独立请求记录，不支持从相同状态中单独选择某个请求。去重集合不跨日志批次保留。

后端别名按 `spec_num_correct_drafts` / `spec_num_proposed_drafts`、`spec_accepted_drafts` / `spec_proposed_drafts`、`spec_accept_token_num` / `spec_draft_token_num` 的顺序选择首个出现的字段组，不跨组拼接。另两项来自 `spec_verify_ct` 和 `completion_tokens`。completion 采用 backend 计数，不以导出 response_length、工具文本或 loss mask 重建。

每个比值仅使用所有 attempt 中分子和分母都完整的记录。显式零是有效观测，缺失、null 或非法计数不可用。总分母为零时省略比值键，不输出伪造的 0% 或 NaN；accepted=0、proposed>0 时仍输出真实零接受率。

以下新指标沿用 `rollout/` 或既有 eval 前缀：

| 指标 | 含义 |
| --- | --- |
| `spec/accepted_tokens`、`spec/proposed_tokens` | 接受率配对完整记录的总量 |
| `spec/completion_tokens`、`spec/verify_count` | length 配对完整记录的总量 |
| `spec/generation_count` | 去重后的已提交生成身份数量 |
| `spec/sample_block_count` | 新版普通生成样本累计块数量，也参与主指标 |
| `spec/acceptance_observed_count`、`spec/length_observed_count` | 各比值的完整记录数；与 generation_count + sample_block_count 比较可判断覆盖 |
| `spec/legacy_sample_count` | 缺少生成记录和字段可用性信息的旧样本数 |

`Sample.spec_info` 保留原计数字段名，新增 `missing_fields`。旧序列化数据仍可读取，但旧零值不能证明覆盖，旧共享前缀无法可靠去重，因此不混入新指标，只报告 `spec/legacy_sample_count`；只有旧样本时不输出比值。新版普通生成样本由 `SpecInfo.add` 保存覆盖信息，即使与 Agentic 样本混合也参与主指标。无需新增 CLI 参数。

### 可复现 CPU 样例

| 生成 | Accepted | Proposed | Verify | Completion |
| --- | ---: | ---: | ---: | ---: |
| A | 1 | 2 | 1 | 2 |
| B | 9 | 10 | 2 | 10 |
| C | 2 | 4 | 1 | 3 |
| D（未导出） | 100 | 100 | 1 | 100 |

仅 A、B 两个独立请求：接受率 **10/12 ≈ 83.33%**，length **12/3 = 4**。导出 A→B、A→C：去重后为 A、B、C，接受率 **12/16 = 0.75**，length **15/4 = 3.75**，覆盖完整。D 不计入，重复导出不会改变结果。

`tests/test_agentic_speculative_metrics.py` 贯通 backend 结果累计、终态提交、显式轨迹导出、transport 和 JSON 往返，再将生产聚合结果与 `tests/fixtures/speculative_metrics.json` 对照。运行：`python -m pytest tests/utils/test_speculative_metrics.py tests/test_agentic_speculative_metrics.py -q`。测试使用本地 tokenizer 和可控 backend 结果，不需要 GPU、模型权重或运行中的 Ray 集群，不证明真实推理加速。
