---
outline: deep
---

# Rollout Service API

The Rollout service generates training samples using SGLang engines. It is deployed as a Ray Serve deployment with a FastAPI ingress, exposing HTTP endpoints for lifecycle management, evaluation, and async weight-update coordination.

## Overview

| Property | Value |
|----------|-------|
| **Module** | `relax.components.rollout` |
| **Deployment** | `@serve.deployment` |
| **Ingress** | FastAPI |

### Lifecycle

The Rollout runs a background loop that:

1. Generates samples via `RolloutManager.generate()` using SGLang engines
2. Computes rewards via pluggable reward functions (`rm_hub/`)
3. Publishes data to `TransferQueue` for the Actor to consume
4. Optionally triggers evaluation at configured intervals
5. Manages staleness bounds to avoid data drift

### Async Weight Coordination

In fully-async mode, the Rollout service coordinates with the Actor for weight updates:

1. Actor calls `/can_do_update_weight_for_async` to check if rollout can pause
2. Rollout pauses if data production is complete for current step
3. Actor pushes new weights
4. Actor calls `/end_update_weight` to resume rollout

## HTTP Endpoints

<SwaggerUI specUrl="/Relax/openapi/rollout.json" />

## Source

- Implementation: [`relax/components/rollout.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/rollout.py)
- Base class: [`relax/components/base.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/base.py)
