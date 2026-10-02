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

### Scale Status Cleanup Contract

Both `ScaleOutStatusResponse` and `ScaleInStatusResponse` carry a `cleanup_required: bool` field that follows the three-state cleanup contract shared with the Autoscaler:

| Value | Meaning |
|-------|---------|
| `true` | Physical cleanup (engine teardown / placement-group release) is still pending; a terminal request with this flag must be reconciled until cleanup completes. |
| `false` | Authoritative cleanup complete — no deferred physical work remains behind the reported status. |
| absent (legacy schema) | Unknown; callers must not treat absence as clean. The shared Autoscaler treats a missing flag as unknown and keeps a terminal request pending until an explicit `false` is observed. |

Rollout scale-out and scale-in terminal states own their cleanup before reporting: scale-out engines either serve (`ACTIVE`/`PARTIAL`) or were rolled back (`FAILED`/`CANCELLED`), and scale-in `COMPLETED` reports the engines removed while `FAILED` reports the removal rolled back — so terminal rollout responses report `cleanup_required: false`. The field is declared explicitly (rather than omitted) precisely because the Autoscaler's contract is "missing ≠ clean".

## HTTP Endpoints

<SwaggerUI specUrl="/Relax/openapi/rollout.json" />

## Source

- Implementation: [`relax/components/rollout.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/rollout.py)
- Base class: [`relax/components/base.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/base.py)
