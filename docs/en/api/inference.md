---
outline: deep
---

# Unified inference service

[RFC #71](https://github.com/redai-studio/Relax/issues/71) shares `InferenceGateway`, the `InferenceManager` state machine, and `SGLangEngine` across Rollout, GenRM, and Teacher while preserving their workloads. Each role has a CPU gateway outside its GPU placement group. Static GenRM and Teacher engines reject DCS registration and dynamic weight updates.

## API and routing

The role prefixes `/rollout`, `/genrm`, and `/teacher` expose `/engines`, `/health`, `/v1/models`, `/generate`, `/v1/chat/completions`, and the `/chat/completions` alias. Raw generation preserves logprob/base64 fields; chat supports SSE.

| Path | Purpose |
| --- | --- |
| `GET /engines` | Models, logical engines, states, router URL, and `topology_revision` |
| `GET /v1/models` | Configured model IDs |
| `GET /health` | Control-plane health and model states |
| `POST /generate` | Raw SGLang requests, including logprob/base64 fields |
| `POST /v1/chat/completions` | OpenAI chat requests and SSE |
| `POST /chat/completions` | Compatible chat alias |

Gateway and `relax.utils.inference_client.InferenceClient` share model selection: explicit `model`, then `route_key`, then a configured default. Unknown explicit selections return 400. Unavailable models return 503 with `Retry-After`; requests never wake a sleeping model. Discovery publishes immutable snapshots and a changing `topology_revision`. Only logical HTTP heads are exposed. PD workloads always use the router.

Models in `sleeping`, `draining`, `onloading`, `failed`, or `dead` states, or awaiting weight synchronization, reject new requests. When no default is configured, clients must select a model explicitly.

Use `InferenceClient(service_url, direct=True)` for raw SGLang requests to discovered endpoints, or `direct=False` for gateway requests. `generate()` returns JSON; `stream()` yields SSE bytes. The client refreshes discovery per request and does not replay generation after a transport failure.

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

Legacy GenRM message-template requests and `{"response": ...}` responses remain available through its gateway. Existing Teacher URLs, Manager lifecycle methods, and the `GenRMEngine` name remain supported. Planned Teacher deployments can publish replacement endpoints after recovery; legacy raw-URL deployments retain their original recovery restrictions.

## Placement and lifecycle

- **Decoupled:** separate role GPU pools; dedicated Teacher PGs belong to their manager.
- **Split:** synchronous colocate partitions the Actor pool between Rollout, inline GenRM, and inline Teacher. Training reuses this pool after inference offloads.
- **Defer:** `--opd-teacher-defer` and `--defer-reward-to-post-process` reuse the shared pool in separate Teacher and GenRM phases. Models within a phase still occupy disjoint slices.

The planner checks capacity, complete replicas, TP×PP, local node spans, overrides, and overlap before engine creation. An explicit decoupled or hybrid Rollout GPU request cannot exceed its independent resource budget; this is rejected before Teacher or PG creation. Actual PG node/device mappings are checked before actors launch. Shared Actor/inference pools require both training and inference offload. Deferred engines are created lazily to avoid startup co-residency.

Deferred batches finish generation and drain surplus requests, offload Rollout, run Teacher, optionally restore the original student weights for top-k queries, run GenRM, offload all overlapping models, and only then publish training samples. Teacher failures or incomplete fields prevent publication. Prompt groups are preserved for group rewards; evaluation does not apply training normalization. Deferred generation and evaluation are serialized.

Onload/offload/shutdown are idempotent. Partial restoration of `weights`, `kv_cache`, and `cuda_graph` remains unavailable until complete. Failed activation triggers cleanup. Unconfirmed cleanup keeps the engine handle, PG ownership, and phase lease; discovery stays unavailable and further activation is blocked. Engines record their child process immediately after spawning, so failed initialization can still retry cleanup. Successful manager cleanup alone does not resume a failed batch or release its retained coordinator lease. Managers never remove borrowed PGs.

Same-GPU co-residency within one phase is unsupported. Defer requires synchronous colocate and the framework's complete SGLang batch pipeline; fully async, partial rollout, agentic/custom rollout, and reward-dependent dynamic filtering are rejected.

## Next steps

- [Rollout API](./rollout.md)
- [GenRM API](./genrm.md)
