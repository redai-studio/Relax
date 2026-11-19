---
outline: deep
---

# GenRM Service API

The GenRM (Generative Reward Model) service provides LLM-based response evaluation through the role-neutral `InferenceGateway` Serve ingress.

## Overview

| Property | Value |
|----------|-------|
| **Module** | `relax.components.inference_gateway` |
| **Deployment** | `@serve.deployment(logging_config=...)` |
| **Ingress** | FastAPI |

### Architecture

Unlike Actor and Rollout, GenRM is a **passive HTTP service** — it does not run a background loop. It only responds to incoming `/generate` requests.

The service uses SGLang engines to perform preference evaluation:

1. Receives OpenAI-format chat messages via `/generate`
2. Applies chat template and tokenizes the prompt
3. Sends to SGLang engine with configurable sampling parameters
4. Returns raw model response text

### Dedicated GPU pools

One task-level CPU InferenceManager owns Rollout, GenRM and Teacher model pools, admission and lifecycle state. Role handles only forward calls; they do not create independent Managers. Dedicated and colocated GenRM use the same request executor. Each judge keeps its own checkpoint and GPU slice; Actor weight updates do not change judge weights. Requests resolve current replica addresses, so an engine replacement can change its port without leaving the service with a stale address cache. The existing `route_key` and sampling configuration remain applicable. `/genrm/engines` defaults to schema v2; use `?schema_version=1` for the old projection. Messages are adapted locally after Manager admission; no named GenRM actor is required.

### Colocated GPU layouts

`--colocate` lets Actor reuse inference GPUs during training. A shared placement group does not require Rollout and GenRM to occupy the same bundles:

- **Split** (`rollout_num_gpus + genrm_num_gpus == actor_total_gpus`): Rollout and GenRM use disjoint bundles and can score inline. For eight training GPUs, `--resource '{"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4]}'` selects a 4+4 split.
- **Shared** (`rollout_num_gpus == genrm_num_gpus == actor_total_gpus`): Rollout and GenRM reuse the same bundles at different times. Set `--defer-reward-to-post-process` and a compatible `--custom-reward-post-process-path`. Generation must finish and drain before the judge onloads; scoring must finish before publication. Lowering `mem_fraction_static` is not a replacement for this phase ordering.

In the managed shared path, the phase coordinator owns drain/offload/onload. The example `post_process_genrm_swap.py` requires that phase and performs scoring without its own memory swaps. Judge failures propagate instead of becoming zero rewards. A batch is published only after all required rewards are present and finite; a missing publication acknowledgement fails the attempt without replaying the write.

Custom and external engines use the same task control plane and must meet its admission and confirmed-release contracts. GenRM keeps its own checkpoint and does not receive Actor weights.

See [GenRM example](/en/examples/generative-reward-model) for full configuration.

### Multiple Instances (`route_key`)

A single GenRM deployment can host several independent judge models at once (configured via `--genrm-instances`), selected per request via the `route_key` field in the request body. Omitting `route_key` (or running the legacy single-instance `--genrm-model-path` config) routes to the sole `"__default__"` instance. For the config format, per-instance `/health` / `/metrics` detail, and an agentic module-routing example, see [GenRM Example · Multi-Instance GenRM](/en/examples/generative-reward-model#multi-instance-genrm-multiple-judge-models-behind-one-service).

## HTTP Endpoints

<SwaggerUI specUrl="/Relax/openapi/genrm.json" />

## Source

- Implementation: [`relax/components/inference_gateway.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/inference_gateway.py)
- Base class: [`relax/components/base.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/base.py)
