# Generative Reward Model (GenRM) Example

Training examples that use a **Generative Reward Model** (GenRM) — an LLM-as-judge approach — to score rollout responses, replacing traditional trained reward models.

## Overview

GenRM (Generative Reward Model) leverages a pre-trained LLM (e.g., Qwen3-VL-30B-A3B-Instruct) to evaluate whether model responses are consistent with ground-truth labels. Instead of training a separate reward model, GenRM performs inference-time evaluation via an [SGLang](https://github.com/sgl-project/sglang) engine deployed as an independent Ray Serve service.

Key benefits:

- **Zero reward-model training** — uses an off-the-shelf LLM directly, no additional reward model training required
- **Strong generalization** — leverages LLM reasoning capabilities for effective evaluation on unseen tasks
- **Flexible criteria** — evaluation behavior is controlled via prompt templates

Both scripts in this example train **Qwen3-4B** with **GRPO** on the `dapo-math-17k` dataset, using GenRM (`--rm-type dapo-genrm`) for reward scoring and AIME-2024 for evaluation.

## Architecture

Relax supports two top-level deployment modes for GenRM:

- **Colocate** (`--colocate`, recommended) — all roles (Actor / Rollout / GenRM) share one placement group. The Actor reclaims all GPUs during training, so GenRM GPUs are never wasted.
- **Fully Async** (`--fully-async`) — each role has its own dedicated GPU pool. Rollout and training run fully in parallel; GenRM GPUs are dedicated and idle during training.

Under `--colocate`, how Rollout and GenRM cohabit the shared bundles has three flavors — pick based on GenRM size and rollout length variance:

| Sub-mode                    | Bundles                | Concurrency during inference     | Inline reward | Trigger                                                                                                        | Best for                                                                                       |
| :-------------------------- | :--------------------- | :------------------------------- | :------------ | :------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------- |
| **Split**                   | Disjoint bundles       | Parallel (dedicated slice each)  | ✅ per sample | Auto — `rollout_num_gpus + genrm_num_gpus == actor_total`                                                      | Small GenRM; long-tail rollout (agentic, high-variance response length)                        |
| **Shared / Co-resident**    | Same bundles           | Parallel (mem_fraction split)    | ✅ per sample | Auto — `rollout_num_gpus == genrm_num_gpus == actor_total`                                                     | Medium GenRM that benefits from full-cluster TP but still fits alongside Rollout               |
| **Shared / Defer-swap**     | Same bundles           | Serialized (sleep-wake sequence) | ❌ deferred   | Opt-in — Shared bundles + `--rm-type dummy` + `--defer-reward-to-post-process` + `--custom-reward-post-process-path` | GenRM much larger than policy; short-response RLVR / math (no rollout tail to hide GenRM behind) |

```
                 8-GPU Colocate (Split)

 ┌──────────── Placement Group (8 GPU) ────────────┐
 │                                                 │
 │  Inference phase:                               │
 │  ┌───────────────────┐   ┌───────────────────┐  │
 │  │  Rollout  (4 GPU) │──►│  GenRM   (4 GPU)  │  │
 │  │  bundles 0..3     │◄──│  bundles 4..7     │  │
 │  └───────────────────┘   └───────────────────┘  │
 │                                                 │
 │  Training phase (offload inference weights):    │
 │  ┌─────────────────────────────────────────┐    │
 │  │         Actor  (8 GPU)                  │    │
 │  │         Megatron Training               │    │
 │  └─────────────────────────────────────────┘    │
 └─────────────────────────────────────────────────┘


         8-GPU Colocate (Shared / Co-resident)

 ┌──────────── Placement Group (8 GPU) ────────────┐
 │                                                 │
 │  Inference phase (same bundles 0..7):           │
 │  ┌─────────────────────────────────────────┐    │
 │  │  Rollout: mem_fraction_static = 0.6     │    │
 │  │  GenRM  : mem_fraction_static = 0.3     │    │
 │  │  ~0.1 reserved for cuda / activations   │    │
 │  └─────────────────────────────────────────┘    │
 │                                                 │
 │  Training phase (offload inference weights):    │
 │  ┌─────────────────────────────────────────┐    │
 │  │         Actor  (8 GPU)                  │    │
 │  │         Megatron Training               │    │
 │  └─────────────────────────────────────────┘    │
 └─────────────────────────────────────────────────┘


        16-GPU Colocate (Shared / Defer-swap)

 ┌──────────── Placement Group (16 GPU) ───────────┐
 │                                                 │
 │  Phase A — rollout (all 16 GPU):                │
 │  ┌─────────────────────────────────────────┐    │
 │  │  Rollout awake  (mem_fraction ≈ 0.85)   │    │
 │  │  GenRM asleep   (release_memory_occ.)   │    │
 │  │  --rm-type dummy  → inline reward = 0   │    │
 │  └─────────────────────────────────────────┘    │
 │                    │                            │
 │        post_process_genrm_swap.py               │
 │  offload rollout ─►│─► onload GenRM             │
 │                    ▼                            │
 │  Phase B — score (all 16 GPU):                  │
 │  ┌─────────────────────────────────────────┐    │
 │  │  Rollout asleep                         │    │
 │  │  GenRM awake  (batch score every sample)│    │
 │  └─────────────────────────────────────────┘    │
 │                    │                            │
 │              offload GenRM                      │
 │                    ▼                            │
 │  Phase C — train (all 16 GPU):                  │
 │  ┌─────────────────────────────────────────┐    │
 │  │        Actor  (Megatron Training)       │    │
 │  │  GenRM stays offloaded via              │    │
 │  │  --defer-reward-to-post-process         │    │
 │  └─────────────────────────────────────────┘    │
 └─────────────────────────────────────────────────┘
```

All three colocate sub-modes reclaim every GPU for the Actor during training. Rollout produces candidate responses and (for Split and Shared / Co-resident) sends each one over HTTP to GenRM inline. In Shared / Defer-swap the HTTP call is batched once per rollout step from a userland `custom_reward_post_process` function; see [`examples/generate_reward_model/README.md`](https://github.com/redai-studio/Relax/blob/main/examples/generate_reward_model/README.md) for the split-vs-defer trade-off matrix.

## Scripts

| Script                                          | Colocate sub-mode          | Description                                                                    |
| :---------------------------------------------- | :------------------------- | :----------------------------------------------------------------------------- |
| `run-qwen3-4B-8xgpu-colocated.sh`               | Split (small GenRM)        | Qwen3-4B policy + small GenRM on 8 GPU; disjoint bundles, inline reward        |
| `run-qwen35-35B-A3B-16xgpu-genrm-397B-split.sh` | Split (large GenRM)        | 35B-A3B policy + 397B FP8 GenRM on 16 GPU; 8+8 disjoint shards, inline reward  |
| `run-qwen35-35B-A3B-16xgpu-genrm-397B-defer.sh` | Shared / Defer-swap        | 35B-A3B policy + 397B FP8 GenRM on 16 GPU; shared bundles, two-phase sleep-wake swap, batched reward via [`post_process_genrm_swap.py`](https://github.com/redai-studio/Relax/blob/main/examples/generate_reward_model/post_process_genrm_swap.py) |
| `run-qwen3-4B-8xgpu-async.sh`                   | (Fully Async)              | Independent GPU pools per role; rollout & training fully overlapped             |

### Resource Layout

Under `--colocate`, all three sub-modes reclaim every GPU for the Actor during training. They differ only in how Rollout / GenRM cohabit during the inference phase:

**Split** (disjoint bundles):

```
Actor:      8 GPU  (all)
Rollout:    4 GPU  (bundles 0..3, mem_fraction default)
GenRM:      4 GPU  (bundles 4..7, mem_fraction default)
```

**Shared / Co-resident** (same bundles, both resident, mem_fraction split):

```
Actor:      8 GPU  (all)
Rollout:    8 GPU  (bundles 0..7, mem_fraction_static = 0.6)
GenRM:      8 GPU  (bundles 0..7, mem_fraction_static = 0.3)
                    └── sum ≤ 0.9; remainder for cuda / activations
```

**Shared / Defer-swap** (same bundles, sequenced):

```
Actor:     16 GPU  (all)
Rollout:   16 GPU  (bundles 0..15, mem_fraction_static ≈ 0.85; asleep during scoring)
GenRM:     16 GPU  (bundles 0..15, mem_fraction_static ≈ 0.85; asleep during rollout / train)
```

**Async mode** (`--fully-async`, dedicated pools):

```
Actor (training):  2 GPU  (dedicated)
Rollout:           3 GPU  (dedicated)
Reference:         1 GPU
Actor Forward:     1 GPU
GenRM:             1 GPU  (dedicated)
```

## Quick Start

### Prerequisites

1. **Model weights** — Download Qwen3-4B (policy model) and Qwen3-VL-30B-A3B-Instruct (GenRM judge model):

   ```bash
   # Place under exps/ (or set EXP_DIR / MODEL_DIR)
   exps/Qwen3-4B/
   exps/Qwen3-VL-30B-A3B-Instruct/
   ```

2. **Dataset** — Prepare `dapo-math-17k` for training and `aime-2024` for evaluation:

   ```bash
   exps/dapo-math-17k/dapo-math-17k.jsonl
   exps/aime-2024/aime-2024.jsonl
   ```

3. **Ray cluster** — A running Ray cluster reachable at `http://127.0.0.1:8265`.

### Run Training

```bash
# Colocate mode (recommended, 8 GPU minimum)
bash examples/generate_reward_model/run-qwen3-4B-8xgpu-colocated.sh

# Fully async mode (8 GPU minimum)
bash examples/generate_reward_model/run-qwen3-4B-8xgpu-async.sh
```

### Verify Service Health

Once the training job is running, check that the GenRM service is healthy:

```bash
curl http://localhost:8000/genrm/health
```

Expected response:

```json
{
  "status": "healthy",
  "service": "genrm"
}
```

## Configuration

### GenRM-Specific CLI Arguments

| Argument                      | Type   | Default | Description                                                                         |
| :---------------------------- | :----- | :------ | :---------------------------------------------------------------------------------- |
| `--genrm-model-path`          | `str`  | `None`  | GenRM model path. Setting this enables GenRM                                        |
| `--genrm-num-gpus`            | `int`  | `1`     | Total number of GPUs for GenRM                                                      |
| `--genrm-num-gpus-per-engine` | `int`  | `1`     | Number of GPUs per GenRM engine instance                                            |
| `--genrm-engine-config`       | `JSON` | `None`  | JSON dict for engine initialization (e.g., `max_context_len`, `dp_size`, `pp_size`) |
| `--genrm-sampling-config`     | `JSON` | `None`  | JSON dict for sampling parameters                                                   |

### Engine Config Keys

| Key                   | Type    | Default          | Description                                                                                  |
| :-------------------- | :------ | :--------------- | :------------------------------------------------------------------------------------------- |
| `max_context_len`     | `int`   | `8192`           | Maximum context length                                                                       |
| `dp_size`             | `int`   | `1`              | Data parallelism size                                                                        |
| `pp_size`             | `int`   | `1`              | Pipeline parallelism size                                                                    |
| `ep_size`             | `int`   | `1`              | Expert parallelism size                                                                      |
| `mem_fraction_static` | `float` | SGLang default   | Per-engine SGLang static memory fraction. Set this in shared-GPU colocate mode (see below).  |

### Sampling Config Keys

| Key                | Type    | Default | Description                  |
| :----------------- | :------ | :------ | :--------------------------- |
| `temperature`      | `float` | `0.1`   | Sampling temperature         |
| `top_p`            | `float` | `1.0`   | Nucleus sampling probability |
| `top_k`            | `int`   | `-1`    | Top-k sampling (-1 disables) |
| `max_response_len` | `int`   | `4096`  | Maximum response length      |

### Resource Allocation

GenRM is included in the `--resource` JSON as a `"genrm"` role. The format is `[num_groups, num_gpus_per_group]`.

**Colocated / Split** (small GenRM, default):

```bash
python3 relax/entrypoints/train.py \
    --genrm-model-path /path/to/genrm/model \
    --genrm-num-gpus-per-engine 4 \
    --genrm-engine-config '{"max_context_len": 10240}' \
    --genrm-sampling-config '{"temperature": 0.1, "top_p": 1.0, "top_k": -1, "max_response_len": 1024}' \
    --resource '{"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4]}' \
    --colocate \
    --rm-type dapo-genrm
```

**Colocated / Shared** (large GenRM, NEW): set rollout and genrm to the full actor allocation; the framework auto-detects shared mode and lets the two engines split each GPU's memory via `mem_fraction_static`:

```bash
python3 relax/entrypoints/train.py \
    --genrm-model-path /path/to/genrm/model \
    --genrm-num-gpus-per-engine 8 \
    --genrm-engine-config '{"max_context_len": 10240, "mem_fraction_static": 0.3}' \
    --genrm-sampling-config '{"temperature": 0.1, "top_p": 1.0, "top_k": -1, "max_response_len": 1024}' \
    --rollout-num-gpus-per-engine 1 \
    --sglang-mem-fraction-static 0.6 \
    --resource '{"actor": [1, 8], "rollout": [1, 8], "genrm": [1, 8]}' \
    --colocate \
    --rm-type dapo-genrm
```

::: tip Auto-detected colocate sub-mode
On `--colocate` with GenRM, the GPU layout picks Split vs Shared automatically:

| Allocation                                           | Sub-mode                            |
| :--------------------------------------------------- | :---------------------------------- |
| `rollout_num_gpus + genrm_num_gpus == actor_total`   | **Split** (disjoint bundles) |
| `rollout_num_gpus == genrm_num_gpus == actor_total`  | **Shared** (same bundles) |
| Anything else                                        | Rejected at startup with a clear error |

Within Shared, the default is **Co-resident** (both engines held via `mem_fraction_static` split). Adding `--rm-type dummy` + `--defer-reward-to-post-process` + `--custom-reward-post-process-path` switches it to **Defer-swap** — sequenced sleep-wake, one engine holds full memory at a time. See the [example README](https://github.com/redai-studio/Relax/blob/main/examples/generate_reward_model/README.md) for when to prefer defer-swap.
:::

::: warning Set `mem_fraction_static` in Shared / Co-resident
In Shared / Co-resident mode the two SGLang engines live on the same GPUs concurrently. You **must** size their `mem_fraction_static` so that the sum is < 1.0 (≤ 0.9 recommended; the rest covers cuda graphs and activations). Rollout reads `--sglang-mem-fraction-static` (or YAML overrides via `--sglang-config`); GenRM reads `mem_fraction_static` inside `--genrm-engine-config`. Shared / Defer-swap does not need the split — each engine can take ≈ 0.85 alone since they are never resident together.
:::

**Fully-Async mode**:

```bash
python3 relax/entrypoints/train.py \
    --genrm-model-path /path/to/genrm/model \
    --genrm-num-gpus-per-engine 1 \
    --genrm-engine-config '{"max_context_len": 10240}' \
    --genrm-sampling-config '{"temperature": 0.1, "top_p": 1.0, "top_k": -1, "max_response_len": 1024}' \
    --resource '{"actor": [1, 2], "rollout": [1, 3], "reference": [1, 1], "actor_fwd": [1, 1], "advantages": [1, 0], "genrm": [1, 1]}' \
    --fully-async \
    --rm-type dapo-genrm
```

## Multi-Instance GenRM (multiple judge models behind one service)

Everything above deploys a single judge model. `--genrm-instances` lets **one GenRM Serve deployment host several independent judge models at once** — different sizes, different checkpoints, different scoring criteria — each addressed by a `route_key` string the caller supplies per request. The Serve deployment, HTTP route (`/genrm`), and health/metrics endpoints stay the same; only the request payload gains one field.

This is the natural fit for:

- **Multi-objective reward** — e.g. one instance scores answer correctness, another scores safety/harmlessness, and your reward function combines both into a single training signal.
- **Agentic pipelines** — different modules along an agent's trajectory (planner, tool-caller, final-answer judge, ...) can each be scored by a judge suited to that module, without standing up a separate GenRM deployment (and a separate GPU carve-out) per module.

### `--genrm-instances` CLI Argument

| Argument            | Type   | Default | Description                                                                                   |
| :------------------- | :----- | :------ | :---------------------------------------------------------------------------------------------- |
| `--genrm-instances` | `JSON` | `None`  | JSON dict of `{route_key: instance_spec}`. Setting this takes **priority** over `--genrm-model-path` (the legacy single-instance flags are ignored with a warning if both are set). |

Each `instance_spec` is a dict with these keys:

| Key                   | Type    | Required | Description                                                                 |
| :--------------------- | :------ | :------- | :----------------------------------------------------------------------------- |
| `model_path`           | `str`   | ✅ yes   | Path to this instance's judge model                                            |
| `num_gpus`             | `int`   | ✅ yes   | GPU budget for this instance. **No implicit even split** across instances — every instance must state its own budget explicitly. |
| `num_gpus_per_engine`  | `int`   | no       | GPUs per SGLang engine for this instance. Defaults to the global `--genrm-num-gpus-per-engine`. |
| `engine_config`        | `dict`  | no       | Per-instance engine config (e.g. `max_context_len`, `mem_fraction_static`). Defaults to the global `--genrm-engine-config`. |
| `sampling_config`      | `dict`  | no       | Per-instance sampling params. Defaults to the global `--genrm-sampling-config`. |

The legacy `--genrm-model-path` config still works unchanged — internally it is normalized into a single-instance `--genrm-instances` config under a reserved `"__default__"` key, so requests that omit `route_key` (or scripts that never adopt `--genrm-instances`) keep working exactly as before.

### Example: Two Judges, Split Bundles

The following mirrors [`run-qwen3-4B-8xgpu-dual-genrm-split.sh`](https://github.com/xhs-tech/Relax/blob/main/examples/generate_reward_model/run-qwen3-4B-8xgpu-dual-genrm-split.sh): an 8-GPU Split layout where rollout gets 4 GPU and two GenRM instances share the other 4 GPU (2 GPU each):

```bash
python3 relax/entrypoints/train.py \
    --genrm-instances '{
        "quality": {"model_path": "/path/to/quality-judge", "num_gpus": 2, "num_gpus_per_engine": 2,
                     "sampling_config": {"temperature": 0.1, "max_response_len": 64}},
        "safety":  {"model_path": "/path/to/safety-judge",  "num_gpus": 2, "num_gpus_per_engine": 2,
                     "sampling_config": {"temperature": 0.1, "max_response_len": 32,
                                          "chat_template_kwargs": {"enable_thinking": false}}}
    }' \
    --rollout-num-gpus 4 \
    --resource '{"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4]}' \
    --colocate \
    --custom-rm-path examples.generate_reward_model.reward_dual_genrm_quality_safety.reward_func \
    --reward-key score
```

`--resource`'s `"genrm"` entry must equal the **sum** of every instance's `num_gpus` (`2 + 2 = 4` here); the Split-vs-Shared bundle math from [Configuration](#configuration) above otherwise applies unchanged to the combined total.

::: tip Disable "thinking" mode for judges that must answer tersely
If a judge model defaults to emitting a long reasoning trace before its verdict (common for reasoning-tuned models), the reward function's loose parser may fail to find a clean `1`/`0` and silently score everything `0`. Pass `"chat_template_kwargs": {"enable_thinking": false}` inside that instance's `sampling_config` to force a direct answer, as shown for `safety` above.
:::

### Calling a Specific Instance: `route_key`

From a reward function (or any code holding a `GenRMClient`), pass `route_key` to select which instance answers the call:

```python
from relax.utils.genrm_client import get_genrm_client

genrm_client = get_genrm_client()

quality_response = await genrm_client.generate(
    messages=[{"role": "user", "content": "Judge correctness..."}],
    route_key="quality",
)
safety_response = await genrm_client.generate(
    messages=[{"role": "user", "content": "Judge safety..."}],
    route_key="safety",
)
```

Over HTTP, `route_key` is just one more field in the `/generate` request body:

```bash
curl -X POST http://localhost:8000/genrm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Judge safety..."}],
    "route_key": "safety"
  }'
```

Omitting `route_key` routes to the sole `"__default__"` instance — this is exactly what happens when the service was configured with the legacy `--genrm-model-path` flag instead of `--genrm-instances`. Passing a `route_key` that was never declared in `--genrm-instances` fails the request with a clear "no GenRM instance registered" error rather than a silent misroute.

The `/health` and `/metrics` endpoints report **per-instance** detail once `--genrm-instances` is used:

```bash
curl http://localhost:8000/genrm/health
```

```json
{
  "status": "healthy",
  "service": "genrm",
  "instances": {
    "quality": {"status": "healthy"},
    "safety": {"status": "healthy"}
  }
}
```

### Agentic Example: Routing by Module

In an agentic rollout, different trajectory steps typically already carry `metadata` identifying which module produced them (planner step, tool call, final answer, ...). A custom reward function can read that field and map it directly to a `route_key`, instead of hardcoding two fixed calls the way `reward_dual_genrm_quality_safety.py` does:

```python
# --genrm-instances '{"planner_judge": {...}, "tool_call_judge": {...}, "final_answer_judge": {...}}'

MODULE_TO_ROUTE_KEY = {
    "plan": "planner_judge",
    "tool_call": "tool_call_judge",
    "final_answer": "final_answer_judge",
}

async def agentic_reward_func(args, sample, **kwargs) -> dict:
    genrm_client = get_genrm_client()
    per_step_scores = []
    for step in sample.metadata["trajectory_steps"]:
        route_key = MODULE_TO_ROUTE_KEY[step["module"]]
        judge_response = await genrm_client.generate(
            messages=_format_step_messages(step),
            route_key=route_key,
        )
        per_step_scores.append(_parse_judgement(judge_response))

    # Aggregation is entirely up to you: weighted average, min() for a
    # "weakest-link" penalty, or scoring only the final step and treating
    # earlier ones as process reward — Relax does not prescribe one.
    return {"score": sum(per_step_scores) / len(per_step_scores)}
```

Size each judge's GPU budget for its role in the pipeline (a fast small model for high-frequency tool-call checks, a larger model for the final-answer judge that runs once per trajectory), and make sure the sum of every instance's `num_gpus` still satisfies the Split/Shared bundle equation from [Configuration](#configuration).

## Script Walkthrough

Both scripts share the same structure. Here is a breakdown of the key configuration groups:

### Reward Configuration

The critical setting that enables GenRM is `--rm-type dapo-genrm`, which routes reward computation through `async_compute_score_genrm()` in `relax/engine/rewards/dapo_genrm.py`. The core implementation is straightforward:

```python
DAPO_GENRM_PROMPT_TEMPLATE = """Below are two answers to a question. ...
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""

def _format_messages(question, ground_truth, predict_str):
    # Extract answer after "Answer:" marker, or take last 300 chars
    if "Answer:" in predict_str:
        predict_str = predict_str.split("Answer:")[-1]
    else:
        predict_str = predict_str[-300:]
    prompt = DAPO_GENRM_PROMPT_TEMPLATE.format(
        question=question, ground_truth=ground_truth, predict_str=predict_str,
    )
    return [{"role": "user", "content": prompt}]

async def async_compute_score_genrm(args, sample) -> dict:
    genrm_client = get_genrm_client()          # singleton HTTP client
    question = sample.metadata.get("question", "")
    ground_truth = sample.metadata.get("label", "")
    messages = _format_messages(question, ground_truth, sample.response)

    response = await genrm_client.generate(messages)  # call GenRM service
    prediction = response.strip()

    # Strict equality: only exact "1" yields a positive score
    score = 1.0 if prediction == "1" else 0.0
    return {"score": score, "acc": int(score), "pred": prediction}
```

```bash
ROLLOUT_ARGS=(
   --rm-type dapo-genrm        # Use GenRM for reward scoring
   --reward-key score           # Key for reward in output dict
   --n-samples-per-prompt 8     # Generate 8 responses per prompt
   --rollout-max-response-len 8192
   --rollout-temperature 1
)
```

### Training Configuration

Both scripts use GRPO with the following hyperparameters:

```bash
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis                    # Truncated importance sampling
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
)
```

### GenRM Service Configuration

The GenRM model and engine are configured at the `ray job submit` level:

```bash
--genrm-model-path ${MODEL_DIR}/Qwen3-VL-30B-A3B-Instruct/ \
--genrm-num-gpus-per-engine 1 \
--genrm-engine-config '{"max_context_len": 10240}' \
--genrm-sampling-config '{"temperature": 0.1, "top_p": 1.0, "top_k": -1, "max_response_len": 1024}'
```

::: tip
A low temperature (e.g., 0.1) is recommended for GenRM to produce deterministic evaluation results. Higher temperatures introduce evaluation variance.
:::

## Usage Examples

### Call GenRM API Directly

```bash
curl -X POST http://localhost:8000/genrm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Evaluate the answer consistency..."}
    ]
  }'
```

Response:

```json
{
  "response": "1"
}
```

### Use GenRMClient in Python

```python
from relax.utils.genrm_client import get_genrm_client

# Get singleton client (avoids per-request client creation overhead)
client = get_genrm_client()

# Async generate
response = await client.generate(
    messages=[{"role": "user", "content": "Evaluate..."}],
    sampling_params={"temperature": 0.2},
)
print(response)  # "1" or "0"
```

## Best Practices

1. **Prefer colocate mode**: In colocate mode, GenRM GPUs are offloaded back to training when not evaluating, so all GPUs participate in gradient computation. This gives better GPU utilization than async mode, where GenRM GPUs sit idle during training.
2. **Choose the right colocate sub-mode**:
   - Use **split** when GenRM is small enough to run on a partial slice (e.g., a 4B reward model on 4 GPUs).
   - Use **shared** when GenRM is large and benefits from cluster-wide TP (e.g., a 30B MoE on TP=8). Shared mode also avoids the "GenRM is too small to use a dedicated bundle and Rollout is starved for GPUs" tradeoff.
3. **Size `mem_fraction_static` carefully in shared mode**: Sum across engines should be ≤ 0.9. Common starting point: rollout 0.6, genrm 0.3.
4. **Set appropriate context length**: `max_context_len` in engine config should accommodate your longest prompt + response combination.
5. **Use low sampling temperature**: A temperature of 0.1 produces deterministic evaluations; increase only if evaluation diversity is desired.
6. **Monitor health**: Periodically check the `/health` endpoint to ensure GenRM engines are running properly.
7. **Match GPU allocation to model size**: For large GenRM models (e.g., 30B), use shared mode with `--genrm-num-gpus-per-engine` set to the full cluster size.
8. **State every instance's `num_gpus` explicitly under `--genrm-instances`**: there is no automatic even split across instances, so undersized or oversized instances are a config mistake, not a framework default — size each one to the judge model it runs.
9. **Disable "thinking" mode for judges expected to answer tersely**: reasoning-tuned models often emit a long trace before a short verdict by default; set `"chat_template_kwargs": {"enable_thinking": false}` in that instance's `sampling_config` if your parser expects a bare `1`/`0`.

## Troubleshooting

### GenRM Not Enabled

Ensure `--genrm-model-path` (or `--genrm-instances`) is set. GenRM is only activated when at least one instance is configured.

### Resource Allocation Error in Colocated Mode

In colocated mode with GenRM, GPU allocation must satisfy **exactly one** of:

- **Split**: `rollout_num_gpus + genrm_num_gpus == actor_total_gpus`
- **Shared**: `rollout_num_gpus == genrm_num_gpus == actor_total_gpus`

Anything else (e.g., `rollout + genrm < actor_total`, or `rollout < actor_total < rollout + genrm`) is rejected at startup. Adjust `rollout` and/or `genrm` in `--resource` to one of the two valid layouts.

### Shared Mode: OOM or Engine Init Fails

If shared mode hits OOM during engine startup or cuda graph capture, lower `mem_fraction_static` for one or both engines so that the per-GPU sum stays ≤ 0.9. With large MoE GenRM models, you may also need to disable cuda graphs or reduce `max_context_len`.

### Engine Initialization Timeout

If GenRM engines fail to initialize:

1. Check that the model path is accessible from all nodes
2. Verify sufficient GPU memory is available
3. Review Ray logs for SGLang engine startup errors

### GenRM Always Returns 0

The DAPO-GenRM reward function uses strict equality to parse responses — only an exact `"1"` string yields a positive score. If the GenRM model outputs anything else (e.g., `"1."`, `"Yes"`, or multi-line text), the score will be 0. Verify that the GenRM model and prompt template produce clean `"1"` / `"0"` outputs. If the judge is a reasoning-tuned model, check whether it is emitting a `<think>` trace before its verdict — see the "thinking mode" tip under [Multi-Instance GenRM](#multi-instance-genrm-multiple-judge-models-behind-one-service).

### `--genrm-instances`: "missing required key 'num_gpus'"

Every instance in `--genrm-instances` must state its own `num_gpus` explicitly — there is no implicit even split across instances (unlike some multi-teacher scheduling elsewhere in Relax). Add `"num_gpus": <n>` to the instance's config.

### "No GenRM instance registered for route_key=..."

The reward function (or direct HTTP caller) passed a `route_key` that doesn't match any key in `--genrm-instances`. Check for a typo, or confirm the service was actually started with `--genrm-instances` rather than the legacy `--genrm-model-path` (which only exposes the `"__default__"` key — passing any other `route_key` against a single-instance deployment fails the same way).

## File Structure

```
examples/generate_reward_model/
├── README.md                                              # Example overview + split-vs-defer decision guide
├── post_process_genrm_swap.py                             # Custom hook used by the defer script (sleep-wake swap + batch score)
├── run-qwen3-4B-8xgpu-colocated.sh                        # 4B colocate mode
├── run-qwen3-4B-8xgpu-async.sh                            # 4B fully async mode
├── run-qwen35-35B-A3B-16xgpu-genrm-397B-split.sh          # 35B + 397B, split-bundle inline reward
├── run-qwen35-35B-A3B-16xgpu-genrm-397B-defer.sh          # 35B + 397B, shared-bundle two-phase swap
├── run-qwen3-4B-8xgpu-dual-genrm-split.sh                 # 4B policy + two GenRM instances (--genrm-instances), split-bundle
└── reward_dual_genrm_quality_safety.py                    # Custom reward routing to two GenRM instances via route_key
```

## Further Reading

- [Architecture](/en/guide/architecture) — Understand the overall Relax architecture
- [Fully Async Training](/en/guide/fully-async-training) — How async mode works in Relax
- [Configuration](/en/guide/configuration) — Complete configuration reference
- [GenRM API](/en/api/genrm) — HTTP API reference for the GenRM service
