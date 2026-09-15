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

Unlike Actor and Rollout, GenRM is a **passive HTTP service** — it does not run a background loop. It only responds to incoming `/generate` requests.

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

## HTTP Endpoints

<SwaggerUI specUrl="/Relax/openapi/genrm.json" />

## Source

- Implementation: [`relax/components/genrm.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/genrm.py)
- Base class: [`relax/components/base.py`](https://github.com/redai-studio/Relax/blob/main/relax/components/base.py)
