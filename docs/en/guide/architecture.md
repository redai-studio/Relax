# Architecture

## Overview

Relax is a reinforcement learning training framework for large language models, built on Ray Serve. It supports the Megatron training backend, SGLang inference engine, and algorithm families including PPO/GRPO/GSPO/SAPO/CISPO. The framework uses a layered architecture that decouples orchestration, components, engines, backends, and distributed capabilities into independent modules.

## Layered Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Entrypoints Layer                             │
│              train.py — signal handling, launch training         │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                   Orchestration Layer (Core)                     │
│  ┌──────────────────┐  ┌──────────┐  ┌────────────────────────┐  │
│  │   Controller     │  │ Service  │  │     Registry           │  │
│  │ (training loop,  │  │(lifecycle│  │  (roles/algo registry) │  │
│  │  deployment)     │  │ mgmt)    │  │                        │  │
│  └──────────────────┘  └──────────┘  └────────────────────────┘  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│               Components Layer (Ray Serve Deployments)           │
│  ┌────────┐ ┌──────────┐ ┌────────┐ ┌───────────┐ ┌──────────┐   │
│  │ Actor  │ │ Rollout  │ │ Critic │ │ ActorFwd  │ │Advantages│   │
│  └────────┘ └──────────┘ └────────┘ └───────────┘ └──────────┘   │
│  ┌────────┐                                                      │
│  │ GenRM  │                                                      │
│  └────────┘                                                      │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                   Engine Layer                                   │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐   │
│  │  Rollout Engine     │  │  Reward Functions                │   │
│  │  (SGLang rollout,   │  │  (deepscaler, math, genrm, ...)  │   │
│  │   on-policy distill)│  │                                  │   │
│  └─────────────────────┘  └──────────────────────────────────┘   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│               Backends Layer            Distributed Layer        │
│  ┌──────────────────┐ ┌───────────┐  ┌──────────┐ ┌───────────┐  │
│  │ Megatron Backend │ │  SGLang   │  │ Ray Actor│ │   DCS     │  │
│  │ (Actor, weight   │ │  Inference│  │  Groups  │ │ Checkpoint│  │
│  │  update, HF conv)│ │  Engine   │  │  Mgmt    │ │  Service  │  │
│  └──────────────────┘ └───────────┘  └──────────┘ └───────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 1. Entrypoints Layer

[`relax/entrypoints/train.py`](../../../relax/entrypoints/train.py) is the main training entry point:

- Parses command-line arguments (extended from Megatron-LM argument parser)
- Initializes Ray cluster connection
- Registers SIGTERM/SIGINT signal handlers for SGLang process cleanup
- Creates Controller and starts training loop

### 2. Orchestration Layer (Core)

The orchestration layer coordinates the entire training workflow:

- [**Controller**](../../../relax/core/controller.py): Core orchestrator — deploys all RL service components, manages the training loop, handles health checking and global restart
- [**Service**](../../../relax/core/service.py): Lifecycle wrapper for Ray Serve deployments — manages placement groups, heartbeat reporting, service restart
- [**Registry**](../../../relax/core/registry.py): Global constants and role registry — defines ROLES, ALGOS mappings

### 3. Components Layer

The components layer contains all RL service components, each deployed as a `@serve.deployment`:

| Component | File | Responsibility |
|-----------|------|----------------|
| **Actor** | [`actor.py`](../../../relax/components/actor.py) | Policy training (Megatron backend) |
| **Rollout** | [`rollout.py`](../../../relax/components/rollout.py) | Rollout service orchestration, manages RolloutManager |
| **Critic** | [`critic.py`](../../../relax/components/critic.py) | PPO value estimation and clipped value-loss training |
| **ActorFwd** | [`actor_fwd.py`](../../../relax/components/actor_fwd.py) | Forward inference log-prob (fully-async mode) |
| **Advantages** | [`advantages.py`](../../../relax/components/advantages.py) | Advantage computation (PPO/GRPO/GSPO/SAPO/CISPO etc.) |
| **GenRM** | [`genrm.py`](../../../relax/components/genrm.py) | Generative reward model |

The deployed service graph depends on the algorithm and execution mode. Colocate GRPO-like algorithms use Actor + Rollout; synchronous colocate PPO adds Critic + Advantages. For non-PPO algorithms, **fully-async mode** (`--fully-async`) can additionally deploy ActorFwd and Reference services according to the log-probability and KL configuration. Fully-async PPO is not currently supported; see [PPO Training](./ppo-training.md) for the supported data flow.

### 4. Engine Layer

The engine layer provides rollout data generation and reward computation:

- [**Rollout Engine**](../../../relax/engine/rollout/): SGLang rollout implementation, on-policy distillation, etc.
- [**Reward Functions**](../../../relax/engine/rewards/): Pluggable reward computation (deepscaler, math_utils, genrm, etc.)
- [**Router**](../../../relax/engine/router/): Request routing and load balancing
- [**Filters**](../../../relax/engine/filters/): Data filtering strategies

### 5. Backends Layer

The backends layer wraps underlying training and inference engines:

- [**Megatron Backend**](../../../relax/backends/megatron/): Distributed training based on Megatron-LM, supporting TP/PP/CP/EP parallelism
- [**SGLang Engine**](../../../relax/backends/sglang/): High-performance inference engine management and process lifecycle control

### 6. Distributed Layer

- [**Ray Actor Groups**](../../../relax/distributed/ray/): Manages RolloutManager, GenRMManager, and other Ray Actor groups
- [**DCS Checkpoint Service**](../../../relax/distributed/checkpoint_service/): Distributed weight synchronization service supporting NCCL/GLOO/TCP communication backends

## Directory Structure

```
relax/
├── core/                    Orchestration layer
│   ├── controller.py        Controller: deployment, training loop, health mgmt
│   ├── service.py           Service: Ray Serve deployment lifecycle wrapper
│   └── registry.py          Global constants and role registry
├── components/              Components layer (Ray Serve Deployments)
│   ├── actor.py             Policy training
│   ├── actor_fwd.py         Forward inference log-prob
│   ├── critic.py            Value estimation
│   ├── advantages.py        Advantage computation
│   ├── genrm.py             Generative reward model
│   └── rollout.py           Rollout service orchestration
├── engine/                  Engine layer
│   ├── rollout/             Rollout engine implementations
│   ├── rewards/             Reward functions
│   ├── router/              Request routing
│   └── filters/             Data filtering
├── backends/                Backends layer
│   ├── megatron/            Megatron training backend
│   └── sglang/              SGLang inference engine
├── distributed/             Distributed layer
│   ├── ray/                 Ray Actor group management
│   └── checkpoint_service/  DCS distributed checkpoint service
├── entrypoints/             Entrypoints layer
│   ├── train.py             Main training entry point
│   └── deploy_metrics_service.py  Standalone metrics service deployment
└── utils/                   Infrastructure
    ├── arguments.py         Command-line argument parsing
    ├── data/                Dataset loading and streaming
    ├── metrics/             Metrics collection and reporting
    └── ...                  Logging, timers, health monitoring, etc.
```

## Data Flow

### Sync Mode

```
┌───────────────────────────────────────────────────────────────────────┐
│                       Sync Training Data Flow                         │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────┐   ┌──────────────────┐   ┌────────────────────────┐   │
│  │ JSONL/     │   │ Dataset /        │   │ RolloutDataSource      │   │
│  │ Parquet    │──>│ StreamingDataset │──>│ WithBuffer             │   │
│  │ Files      │   │                  │   │                        │   │
│  └────────────┘   └──────────────────┘   └───────────┬────────────┘   │
│                                                      │                │
│                                                      ▼                │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                  RolloutManager.generate()                       │ │
│  │  SGLang engine inference → reward computation → assemble data    │ │
│  └───────────────────────────┬──────────────────────────────────────┘ │
│                              │                                        │
│                              ▼                                        │
│  ┌───────────┐   ┌──────────────────┐   ┌──────────────────────────┐  │
│  │  Actor    │──>│  DCS Weight Sync │──>│  Rollout (update weights)│  │
│  │ (training)│   │  (NCCL AllGather)│   │                          │  │
│  └───────────┘   └──────────────────┘   └──────────────────────────┘  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### Fully-Async Mode (`--fully-async`)

```
┌───────────────────────────────────────────────────────────────────────┐
│                    Fully-Async Training Data Flow                     │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Rollout ──→ TransferQueue ──→ Advantages ──→ TransferQueue ──→ Actor │
│     ↑                              │                             │    │
│     │                              │                             │    │
│     └──── DCS Weight Sync ◄── ActorFwd ◄──── DCS Weight Sync ───┘     │
│                                                                       │
│  5 components run independently, passing data via TransferQueue       │
│  Actor updates weights to ActorFwd → Reference → Rollout every N steps│
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

## Communication Patterns

### Inter-Component Communication

Components communicate via Ray Serve HTTP interfaces and Ray Actor handles:

```python
# Controller deploys components via Service wrapper
service = Service(role="actor", cls=Actor, num_gpus=8, ...)
service.run()  # Start training task

# Components sync weights via DCS Checkpoint Service
client = CheckpointEngineClient(coordinator_url="http://...", role="actor", ...)
await client.update_weights_for_rollout(rollout_only=True)
```

### Network Port Allocation

Services use fixed port ranges to prevent conflicts during distributed process group initialization:

| Service | Port Range |
|---------|-----------|
| DCS weight sync (Actor → Rollout) | 11000 - 11999 |
| Rollout (SGLang engine) | 15000 - 15999 |
| GenRM (SGLang engine) | 16000 - 16999 |
| OS ephemeral (Megatron NCCL) | 32768 - 50000 (recommended) |

See [Distributed Checkpoint - Network Port Allocation](/en/guide/distributed-checkpoint#network-port-allocation) for details.

### Controller Orchestration

The Controller manages the lifecycle of all components via Ray Serve:

```python
# Controller registers all services
ctrl = Controller(args, runtime_env)
ctrl.register_all_serve()   # Deploy Actor, Rollout, Critic, ...
ctrl.training_loop()        # Start training loop
ctrl.shutdown()             # Clean up SGLang engines
```

## Deployment

Deploy training jobs via launch scripts:

```bash
# Sync training (8 GPUs)
bash scripts/training/text/run-qwen3-4B-8xgpu.sh

# Fully-async training (8 GPUs)
bash scripts/training/text/run-qwen3-4B-8xgpu-async.sh

# Or call entry point directly
python relax/entrypoints/train.py [ARGS...]
```

## Scalability

The architecture supports multi-dimensional scaling:

- **Rollout Component**: Configure SGLang inference engines via `--rollout-num-gpus` and `--rollout-num-gpus-per-engine`, with elastic scaling support
- **Actor Component**: Supports Megatron TP/PP/CP/EP parallelism strategies
- **Reward Functions**: Pluggable reward computation modules, specified via `--rm-type` (built-in) or `--custom-rm-path` (custom)

## Monitoring and Observability

Built-in monitoring components:

- **Metrics Service**: Collects and aggregates training metrics (WandB, Prometheus)
- **Health Check Manager**: Monitors service health with automatic restart and global recovery
- **Notification System**: Alerts on failures via Apprise

See:

- [Metrics Service Usage](/en/guide/metrics-service-detailed)
- [Health Check Manager](/en/guide/health-check-manager)
- [Notification System](/en/guide/notification-system)

## Next Steps

- [Dataset Design](/en/guide/dataset-design) - Learn about data loading
- [Distributed Checkpoint](/en/guide/distributed-checkpoint) - Understand checkpointing
- [Configuration](/en/guide/configuration) - Configure your deployment
