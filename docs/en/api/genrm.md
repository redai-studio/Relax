---
outline: deep
---

# GenRM Service API

The GenRM (Generative Reward Model) service provides LLM-based response evaluation. It is deployed as a Ray Serve deployment with a FastAPI ingress.

## Overview

| Property | Value |
|----------|-------|
| **Module** | `relax.components.genrm` |
| **Deployment** | `@serve.deployment(logging_config=...)` |
| **Ingress** | FastAPI |

### Architecture

Unlike Actor and Rollout, GenRM has no autonomous training loop. It serves generation and lifecycle requests over HTTP; accepted scaling operations are monitored asynchronously.

The service uses SGLang engines to perform preference evaluation:

1. Receives OpenAI-format chat messages via `/generate`
2. Applies chat template and tokenizes the prompt
3. Sends to SGLang engine with configurable sampling parameters
4. Returns raw model response text

### Colocated Mode

When colocated with the Actor (sharing GPU resources), GenRM supports offload/onload operations:

- **Offload**: Releases GPU memory before Actor training
- **Onload**: Loads model weights back to GPU before rollout

Two colocate sub-modes are auto-detected from the GPU allocation:

- **Split** (`rollout_num_gpus + genrm_num_gpus == actor_total_gpus`): GenRM and Rollout occupy disjoint bundles.
- **Shared** (`rollout_num_gpus == genrm_num_gpus == actor_total_gpus`): GenRM and Rollout occupy the same bundles, splitting each GPU's memory via SGLang `mem_fraction_static`. GenRM reads its `mem_fraction_static` from `--genrm-engine-config`. GenRM never sources weights from the Actor; onload only resumes its KV cache and CUDA graphs.

See [GenRM example](/en/examples/generative-reward-model) for full configuration.

### Multiple Instances (`route_key`)

A single GenRM deployment can host several independent judge models at once (configured via `--genrm-instances`), selected per request via the `route_key` field in the request body. Omitting `route_key` (or running the legacy single-instance `--genrm-model-path` config) routes to the sole `"__default__"` instance. For the config format, per-instance `/health` / `/metrics` detail, and an agentic module-routing example, see [GenRM Example · Multi-Instance GenRM](/en/examples/generative-reward-model#multi-instance-genrm-multiple-judge-models-behind-one-service).

## Elastic Scaling

Scaling changes the number of frozen-model serving replicas, not their weights.
Initial replicas are protected; scale-in removes only elastic replicas. Elastic
creation currently supports one GPU per engine.

### Request and Status

Paths below are relative to the GenRM service route, normally `/genrm`.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/scale_out` or `/scale_in` | Submit an absolute target replica count |
| `GET` | `/scale_out/{request_id}` or `/scale_in/{request_id}` | Read operation status and cleanup state |
| `POST` | `/scale_out/{request_id}/reconcile` or `/scale_in/{request_id}/reconcile` | Retry unfinished cleanup of the original operation |
| `GET` | `/engines` | Read live engine discovery and capacity |

```json
{
  "model_name": "default",
  "num_replicas": 2,
  "timeout_secs": 600,
  "idempotency_key": "judge-scale-out-001"
}
```

- `model_name` selects a configured instance. `default` selects the sole instance;
  with multiple instances, use its configured route key.
- `num_replicas` is the **absolute total**, not a delta. It must be a positive
  JSON integer; booleans, strings and floats (including `2.0`) are rejected with `422`.
- `timeout_secs` is the operation's total deadline; omission uses 600 seconds.
- `idempotency_key` is optional. Within the retained in-memory history, the same
  direction/key and `(model_name, num_replicas, timeout_secs)` reuse the original
  operation. A changed body with the same key returns `409`. History is bounded
  and is not a durable, cross-restart idempotency guarantee.

Accepted submissions return HTTP `200` with `status: PENDING` and `request_id`.
Poll that ID; admission is not completion. A direction whose target is already
satisfied returns `NOOP` without an operation ID. A keyed `NOOP` replays its
original decision even if capacity later changes.

| Direction | Normal progression | Terminal states |
| --- | --- | --- |
| Scale-out | `PENDING → CREATING → HEALTH_CHECKING → READY → ACTIVE` | `ACTIVE`, `PARTIAL`, `FAILED` |
| Scale-in | `PENDING → DRAINING → REMOVING → COMPLETED` | `COMPLETED`, `FAILED` |

`PARTIAL` means some new replicas were published before expansion failed; inspect
`current`, `ready`, `created`, `failed` and `cleanup_required`, rather than treating
it as target attainment. Invalid instance/target ranges return `400`, unknown
operation IDs return `404`, conflicts return `409`, and unavailable manager
discovery/capacity or missing reconciliation proof can return `503`.

### Cleanup and Capacity

A timeout requests abort; it does not prove that the physical operation stopped.
`cleanup_required: true` keeps that model blocked against new scale operations.
Reconcile preserves the operation ID and terminal result, uses the original
victim, and retries cleanup rather than scaling again. It may return `409` while
the physical thread or victim requests are still active. A terminal `FAILED`
can therefore remain `FAILED` after successful cleanup; check the cleanup flag.

`/engines` exposes these counts at the top level for the legacy default instance,
or under `instances[route_key]` for multi-instance deployments:

| Field | Meaning |
| --- | --- |
| `current` | Published service capacity, including a draining or unreleased failed scale-in victim |
| `ready` | Currently routable replicas |
| `occupied` | Held replica/resource slots, including unpublished candidates |
| `pending_cleanup` | Failed-victim or unpublished-candidate slots awaiting cleanup |

Each discovered engine also reports `host`, `port`, `inflight` and `served`.
These request counters belong to this GenRM component, not to all direct clients.
`/metrics` reports dynamic capacity when the manager query succeeds and includes
`capacity_error` on failure, retaining the startup count. `/engines` can fall
back to the routable count if its capacity query fails; a degraded snapshot is
not resource-release proof. Scale submission
rejects an unavailable authoritative capacity query instead of using that fallback.

::: warning Scope
Drain accounting covers one GenRM Gateway. Direct engine clients, cross-Gateway
admission leases and manager-restart recovery are not implemented. Do not infer
those guarantees from a successful single-Gateway scale-in.
:::

## HTTP Endpoints

<SwaggerUI specUrl="/Relax/openapi/genrm.json" />

## Source

- Implementation: [`relax/components/genrm.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/genrm.py)
- Base class: [`relax/components/base.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/base.py)

## Next Steps

- [GenRM configuration and examples](../examples/generative-reward-model.md)
- [Rollout service API](./rollout.md)
