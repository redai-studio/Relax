# API Overview

This section provides comprehensive API documentation for Relax components.

## Core Modules

### Controller

The Controller module manages the overall experiment lifecycle and service coordination.

```python
from relax.core.controller import Controller

controller = Controller(config)
controller.register_all_serve()
controller.training_loop()
```

### Services

The Service class wraps Ray Serve deployments for each RL component.

```python
from relax.core.service import Service

service = Service(cls=Actor, role="actor", healthy=health_handle, config=config, num_gpus=2)
```

### Service HTTP APIs

The Implementation module deploys concrete RL components as Ray Serve services with FastAPI HTTP endpoints for lifecycle management, recovery, and coordination.

| Service | Description | API Docs |
|---------|-------------|----------|
| **Actor** | Policy model training | [Actor API](/en/api/actor) |
| **Rollout** | Sample generation via SGLang | [Rollout API](/en/api/rollout) |
| **GenRM** | Generative reward model (LLM-as-judge) | [GenRM API](/en/api/genrm) |
| **ActorFwd** | Forward-only log-prob computation | [ActorFwd API](/en/api/actor-fwd) |

::: tip OpenAPI Specification
Each service page includes an interactive Swagger UI generated from the OpenAPI specification.
The specs are generated offline via `python scripts/tools/generate_openapi.py`.
:::

## Utility Modules

### Checkpoint Engine

Distributed checkpoint management.

```python
from relax.distributed.checkpoint_service.client.engine import CheckpointEngineClient

engine = CheckpointEngineClient(config)
```

### Metrics Service

Metrics collection and reporting.

```python
from relax.utils.metrics.client import MetricsClient

metrics = MetricsClient(service_url="http://localhost:8000/metrics")
metrics.log_metric(step=100, metric_name="reward", metric_value=0.75)
metrics.log_metrics_batch(step=100, metrics={"reward": 0.75, "loss": 0.3})
```

### Health System

Service health monitoring.

```python
from relax.utils.health_system import HealthManager

health_manager = HealthManager(check_interval=1.0)
health_manager.start(on_unhealthy=callback_fn)
```

## Data Structures

### Sample

```python
from relax.utils.types import Sample

sample = Sample(
    prompt="What is the capital of France?",
    response="Paris",
    reward=1.0,
    metadata={"source": "dataset"}
)
```

### Episode

```python
from relax.utils.types import Episode

episode = Episode(
    samples=[sample1, sample2, sample3],
    total_reward=2.5,
    length=3
)
```

## Configuration

### Configuration

Relax uses command-line arguments for configuration. See the [Configuration Guide](/en/guide/configuration) for details.

```python
# Configuration is done via argparse
from relax.utils.arguments import parse_args

args = parse_args()
# args contains all configuration parameters
```

## Quick Reference

| Module                    | Description             | Documentation                                           |
| ------------------------- | ----------------------- | ------------------------------------------------------- |
| `relax.core`                           | Experiment coordination | [Controller API](/en/api/controller)                    |
| `relax.components.actor`               | Policy training         | [Actor API](/en/api/actor)                              |
| `relax.components.rollout`             | Sample generation       | [Rollout API](/en/api/rollout)                          |
| `relax.components.inference_gateway`              | Generative reward model | [GenRM API](/en/api/genrm)                              |
| `relax.components.actor_fwd`          | Log-prob computation    | [ActorFwd API](/en/api/actor-fwd)                       |
| `relax.distributed.checkpoint_service` | Checkpoint management   | [Checkpoint API](/en/api/checkpoint)                    |
| `relax.utils.metrics`                 | Metrics collection      | [Metrics API](/en/api/metrics)                          |
| `relax.utils`                          | Utilities               | [Utils API](/en/api/utils)                              |

## Type Hints

Relax uses type hints throughout the codebase:

```python
from typing import Dict, List, Optional
from relax.components import Actor

def train_actor(
    actor: Actor,
    episodes: List[Episode],
    learning_rate: float = 1e-5
) -> Dict[str, float]:
    """Train actor on episodes."""
    ...
```

## Error Handling

Common exceptions:

```python
from relax.exceptions import (
    ConfigurationError,
    ServiceDeploymentError,
    CheckpointError
)

try:
    controller.deploy_services()
except ServiceDeploymentError as e:
    logger.error(f"Failed to deploy services: {e}")
```

## Next Steps

- [Actor API](/en/api/actor) - Actor (policy training) HTTP API
- [Rollout API](/en/api/rollout) - Rollout (sample generation) HTTP API
- [GenRM API](/en/api/genrm) - GenRM (generative reward model) HTTP API
- [ActorFwd API](/en/api/actor-fwd) - ActorFwd (log-prob computation) HTTP API
