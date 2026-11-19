# API 概览

本节提供 Relax 组件的全面 API 文档。

## 核心模块

### Controller（控制器）

Controller 模块管理整体实验生命周期和服务协调。

```python
from relax.core.controller import Controller

controller = Controller(config)
controller.register_all_serve()
controller.training_loop()
```

### Services（服务）

Service 类封装了每个 RL 组件的 Ray Serve 部署。

```python
from relax.core.service import Service

service = Service(cls=Actor, role="actor", healthy=health_handle, config=config, num_gpus=2)
```

### 服务 HTTP API

Implementation 模块将具体的 RL 组件部署为 Ray Serve 服务，通过 FastAPI HTTP 端点提供生命周期管理、故障恢复和协调功能。

| 服务 | 描述 | API 文档 |
|------|------|----------|
| **Actor** | 策略模型训练 | [Actor API](./actor.md) |
| **Rollout** | 通过 SGLang 生成样本 | [Rollout API](./rollout.md) |
| **GenRM** | 生成式奖励模型（LLM-as-judge）| [GenRM API](./genrm.md) |
| **ActorFwd** | 前向推理 log-prob 计算 | [ActorFwd API](./actor-fwd.md) |

::: tip OpenAPI 规范
每个服务页面都包含从 OpenAPI 规范生成的交互式 Swagger UI。
规范文件通过 `python scripts/tools/generate_openapi.py` 离线生成。
:::

## 工具模块

### Checkpoint Engine（检查点引擎）

Distributed Checkpoint 管理。

```python
from relax.distributed.checkpoint_service.client.engine import CheckpointEngineClient

engine = CheckpointEngineClient(config)
```

### Metrics Service（Metrics 服务）

指标收集和报告。

```python
from relax.utils.metrics.client import MetricsClient

metrics = MetricsClient(service_url="http://localhost:8000/metrics")
metrics.log_metric(step=100, metric_name="reward", metric_value=0.75)
metrics.log_metrics_batch(step=100, metrics={"reward": 0.75, "loss": 0.3})
```

### Health System（健康系统）

服务健康监控。

```python
from relax.utils.health_system import HealthManager

health_manager = HealthManager(check_interval=1.0)
health_manager.start(on_unhealthy=callback_fn)
```

## 数据结构

### Sample（样本）

```python
from relax.utils.types import Sample

sample = Sample(
    prompt="法国的首都是什么？",
    response="巴黎",
    reward=1.0,
    metadata={"source": "dataset"}
)
```

### Episode（回合）

```python
from relax.utils.types import Episode

episode = Episode(
    samples=[sample1, sample2, sample3],
    total_reward=2.5,
    length=3
)
```

## 配置

### 配置

Relax 使用命令行参数进行配置。详见[配置指南](../guide/configuration.md)。

```python
# 配置通过 argparse 完成
from relax.utils.arguments import parse_args

args = parse_args()
# args 包含所有配置参数
```

## 快速参考

| 模块                      | 描述           | 文档                                        |
| ------------------------- | -------------- | ------------------------------------------- |
| `relax.core`                           | 实验协调       | [Controller API](./controller.md)           |
| `relax.components.actor`               | 策略模型训练   | [Actor API](./actor.md)                     |
| `relax.components.rollout`             | 样本生成       | [Rollout API](./rollout.md)                 |
| `relax.components.inference_gateway`              | 生成式奖励模型 | [GenRM API](./genrm.md)                     |
| `relax.components.actor_fwd`          | log-prob 计算  | [ActorFwd API](./actor-fwd.md)              |
| `relax.distributed.checkpoint_service` | 检查点管理     | [Checkpoint API](./checkpoint.md)           |
| `relax.utils.metrics`                 | 指标收集       | [Metrics API](./metrics.md)                 |
| `relax.utils`                          | 工具函数       | [Utils API](./utils.md)                     |

## 类型提示

Relax 在整个代码库中使用类型提示：

```python
from typing import Dict, List, Optional
from relax.components import Actor

def train_actor(
    actor: Actor,
    episodes: List[Episode],
    learning_rate: float = 1e-5
) -> Dict[str, float]:
    """在 episodes 上训练 actor。"""
    ...
```

## 错误处理

常见异常：

```python
from relax.exceptions import (
    ConfigurationError,
    ServiceDeploymentError,
    CheckpointError
)

try:
    controller.deploy_services()
except ServiceDeploymentError as e:
    logger.error(f"部署服务失败: {e}")
```

## 下一步

- [Actor API](./actor.md) - Actor（策略训练）HTTP API
- [Rollout API](./rollout.md) - Rollout（样本生成）HTTP API
- [GenRM API](./genrm.md) - GenRM（生成式奖励模型）HTTP API
- [ActorFwd API](./actor-fwd.md) - ActorFwd（log-prob 计算）HTTP API
