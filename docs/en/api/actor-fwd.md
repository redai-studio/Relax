---
outline: deep
---

# ActorFwd Service API

The ActorFwd service computes actor/reference log-probabilities using a forward-only copy of the policy model. It is deployed as a Ray Serve deployment with a FastAPI ingress.

## Overview

| Property | Value |
|----------|-------|
| **Module** | `relax.components.actor_fwd` |
| **Deployment** | `@serve.deployment` |
| **Ingress** | FastAPI |

### Purpose

ActorFwd runs a forward-only replica of the policy model to compute log-probabilities for rollout data. This is used in:

- **KL divergence computation**: Computing `log π(a|s)` for KL penalty
- **Reference log-probs**: Computing `log π_ref(a|s)` for the reference model
- **Fully-async mode**: Receives weight updates from the Actor via NCCL

### Lifecycle

The ActorFwd runs a background loop that:

1. Waits for rollout data to become available
2. Computes actor or reference log-probabilities (depending on role)
3. Publishes results back to `TransferQueue`
4. In fully-async mode, receives weight updates via `/recv_weight_fully_async`

## HTTP Endpoints

<SwaggerUI specUrl="/Relax/openapi/actor_fwd.json" />

## Source

- Implementation: [`relax/components/actor_fwd.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/actor_fwd.py)
- Base class: [`relax/components/base.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/base.py)
