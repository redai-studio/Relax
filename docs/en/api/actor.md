---
outline: deep
---

# Actor Service API

The Actor service trains the policy model. It is deployed as a [Ray Serve](https://docs.ray.io/en/latest/serve/) deployment with a FastAPI ingress, exposing HTTP endpoints for lifecycle management and recovery.

## Overview

| Property | Value |
|----------|-------|
| **Module** | `relax.components.actor` |
| **Deployment** | `@serve.deployment(max_ongoing_requests=10, max_queued_requests=20)` |
| **Ingress** | FastAPI |

### Execution Modes

- **fully_async** — Asynchronous training without waiting for rollout data. Weights are pushed to rollout engines after each step.
- **sync** — Waits for rollout data before each training step. Used in colocated mode.

### Lifecycle

The Actor runs a background training loop that:

1. Fetches data from `TransferQueueClient`
2. Executes forward/backward/optimizer step via `RayTrainGroup`
3. Pushes updated weights to rollout engines
4. Saves checkpoints at configured intervals

## HTTP Endpoints

<SwaggerUI specUrl="/Relax/openapi/actor.json" />

## Source

- Implementation: [`relax/components/actor.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/actor.py)
- Base class: [`relax/components/base.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/base.py)
