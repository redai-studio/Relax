# Customize Training

## Prerequisites

Make sure you have completed the [Installation](./installation.md) steps.

## Model Preparation

### Download

You can download models and datasets from platforms like Hugging Face and ModelScope. Below are example commands using `huggingface_hub` to download sample resources:

```bash
# Download model weights (Qwen3-VL-4B)
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /root/Qwen3-VL-4B-Instruct
```

### Megatron Weights to HF Weights

::: tip No Manual Conversion Needed with Megatron Bridge
Relax uses [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) as the weight bridging layer for its training backend, automatically handling bidirectional HF ↔ Megatron weight conversion during training — **no manual conversion steps required**. Simply specify the following option in your launch script:
:::

```bash
--megatron-to-hf-mode bridge
```

### HF Weights to Megatron Weights

See [Quick Start — Export Model](./quick-start.md#export-model).

### Adding New Models

Adding a new model requires two parts:

#### 1. Model Configuration Script

Model configuration files are located in `scripts/models/`. Extract the corresponding Megatron architecture parameters from the HF config. For example, `scripts/models/qwen3-4B.sh`:

```bash
MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 1000000
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
)
```

After adding the file, source the corresponding model configuration in your training launch script.

#### 2. Megatron Bridge Model Adaptation

Relax uses [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) for automatic HF ↔ Megatron weight conversion. If your model is not yet supported by Megatron Bridge, you need to add support on the Megatron Bridge side first — see its project documentation for details.

::: tip AI-Assisted Integration
This project provides a Codewiz skill `model-integration` (located at `.codewiz/skills/model-integration/`), covering the complete integration workflow for Bridge / Raw / FSDP backends, weight converter specifications, TP sharding logic, and common pitfalls. Invoke it in Codewiz via `invoke skill model-integration` for step-by-step guidance.
:::

## Data Preparation

Relax supports loading `.jsonl` and `.parquet` format files. Using `.jsonl` as an example, each line is a JSON object:

```json
{
  "prompt": [
    {
      "content": "<image><audio><video>What happened in the video?\nOptions:\nA. a sunny day\nB. It's Hailing\nC. a furious storm\nD. Flood",
      "role": "user"
    }
  ],
  "image_key": ["path to your image"],
  "audio_key": ["path to your audio"],
  "video_key": ["path to your video"],
  "label": "<answer>B</answer>"
}
```

For multimodal data, each modality should have a corresponding placeholder in the content field, such as `<image><audio><video>` above, for correct message formatting. Multimodal data supports local file paths, URLs, and binary files.

The corresponding configuration in the training script is:

```bash
--input-key prompt
--label-key label
--apply-chat-template
# Each multimodal data type must be explicitly configured to be loaded
--multimodal-keys '{"image":"image_key","audio":"audio_key","video":"video_key"}'
```

We provide conversion scripts for OpenR1 and AVQA datasets in `scripts/tools/`:

```bash
python scripts/tools/process_openr1.py \
  --input-dir /root/multimodal-open-r1-8k-verified/data/train-00000-of-00001.parquet \
  --output-dir /root/multimodal-open-r1-8k-verified/data/train-00000-of-00001-test.parquet

# --md-dir points to the directory containing image and audio files,
# used to join relative paths into absolute paths.
# If not provided, relative paths are used.
python scripts/tools/process_avqa.py \
  --input-dir /root/AVQA-R1-6K/AVQA_R1/train/omni_rl_format_train.json \
  --output-dir /root/AVQA-R1-6K/AVQA_R1/train/omni_rl_format_train_test.jsonl \
  --md-dir /root/AVQA-R1-6K/AVQA_R1/train
```

## Custom Reward Methods

Define either a single-sample reward,
`reward_func(args, sample: Sample, **kwargs) -> int | float | dict`, or a batched/group reward,
`reward_func(args, samples: list[Sample], **kwargs) -> list[int | float | dict]`, in your own `.py` file. Then add
its import path to the launch script. See [DeepEyes](../examples/deepeyes.md) for an example.

```bash
--custom-rm-path examples.deepeyes.reward_deepeyes.reward_func
# Custom reward_func may return a dict; if so, specify which key corresponds to the actual reward score
--reward-key score
```

Synchronous functions run in process-isolated Ray `RewardWorker` actors. Functions declared with `async def` are
awaited in the caller process and do not use the worker pool. `--reward-num-workers` sets the synchronous worker
count, while `--reward-max-concurrency` limits concurrent reward calls in each caller process.

A batched/group custom reward is called once with the complete sample list, and that call counts as one concurrent
call. Custom functions are loaded lazily and cached until `custom_rm` is explicitly reloaded.

Inputs and return values of synchronous custom rewards must be serializable by Ray.

> **Warning: Set timeouts for external I/O**
>
> Cancelling the caller or its Ray task is best-effort. It does **not** guarantee termination of a Python function
> that is already running in a `RewardWorker`. Every potentially blocking external I/O operation, such as an HTTP,
> RPC, or database call, must set an explicit timeout. Otherwise, the worker may remain occupied until the function
> returns, and later reward calls assigned to that worker must wait.

### Format-aware reward routing

Built-in rewards live in a registry (`relax/engine/rewards/registry.py`). Adding one is a single `register_reward` call — no dispatch branch to edit:

```python
register_reward("my_reward", "my_pkg.my_module:my_reward_fn")  # def my_reward_fn(response, label) -> float
```

Each sample resolves its reward type as: `metadata["rm_type"]` in the sample > `--rm-type` > label inference (opt-in) > fallback (opt-in). A batch can therefore mix task formats (e.g. math + multiple-choice) by carrying `"metadata": {"rm_type": "..."}` per data row (column name set by `--metadata-key`, default `metadata`).

Two optional flags control degraded samples; their defaults keep today's behavior exactly:

- `--rm-type-fallback`: `zero` scores unknown/missing-type samples `0.0` (reward-key aware) with a warning; a registered reward name routes them there; unset keeps the current error.
- `--rm-type-infer`: infers the type from the label via conservative registered matchers (strict numeric answer → `math`, `<answer>X</answer>` single letter → `multiple_choice`) when no explicit type exists. When the label disagrees with an explicit type, a conflict warning is logged and the explicit type wins.

Degradation warnings carry the sample index, offending value, and reason; each `(reason, value)` identity is logged once and counted exactly. Notes: `--custom-rm-path` still bypasses routing entirely; rewards returning dicts (e.g. `dapo`) cannot be mixed with scalar rewards in one batch because `--reward-key` is global; runtime `register_reward` calls in the driver are not visible to already-running Ray workers for sync rewards — register at import time (as `registry.py` does) or use `--custom-rm-path`.

## Custom Generate Function

For multi-turn dialogue, tool calling, or agentic rollout, define a custom `generate` function to replace the default single-turn logic. The function signature is:

```python
from relax.utils.types import Sample
# Required signature
async def generate(args: Any, sample: Sample, sampling_params: dict) -> Sample: ...
# Optional: add evaluation param — framework auto-passes True during eval
async def generate(args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample: ...
```

The function must populate these `sample` fields before returning: `tokens` (full prompt+response token IDs), `response` (decoded string), `response_length`, `loss_mask` (per-token: `1`=trainable, `0`=skip), `rollout_log_probs`, and `status` (`Sample.Status.COMPLETED` / `TRUNCATED` etc.).

**Example** — simplified from [`examples/deepeyes/rollout.py`](../examples/deepeyes.md) (multi-turn tool-use rollout):

```python
from relax.engine.rollout.sglang_rollout import GenerateState

async def generate(args, sample: Sample, sampling_params) -> Sample:
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    env = build_env(sample=sample, args=args); env.reset()
    prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)
    sample.tokens, sample.loss_mask, sample.rollout_log_probs, response_tokens = list(prompt_ids), [], [], []
    for turn in range(args.max_turns):
        # One permit per turn: post_generate acquires/releases it internally; the
        # permit is not held during env/tool execution.
        output = await state.post_generate(url, {"input_ids": sample.tokens, "sampling_params": sampling_params, "return_logprob": True})
        new_tokens = [t[1] for t in output["meta_info"]["output_token_logprobs"]]
        new_probs = [t[0] for t in output["meta_info"]["output_token_logprobs"]]
        sample.tokens.extend(new_tokens); response_tokens.extend(new_tokens)                 # model output
        sample.loss_mask.extend([1] * len(new_tokens)); sample.rollout_log_probs.extend(new_probs)
        observation, done, info = env.step(output["text"])
        if done: break
        obs_ids = state.tokenizer.encode(observation, add_special_tokens=False)
        sample.tokens.extend(obs_ids); response_tokens.extend(obs_ids)                       # env observation
        sample.loss_mask.extend([0] * len(obs_ids)); sample.rollout_log_probs.extend([0.0] * len(obs_ids))
    sample.response = state.tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    sample.status = Sample.Status.COMPLETED
    return sample


# Opt in to per-request permit management (without this flag the function falls
# back to the session-level lock, i.e. the legacy behavior).
generate.manages_inference_permit = True
```

Specify via launch script (`--custom-generate-function-path examples.deepeyes.rollout.generate`), or per eval dataset via `custom_generate_function_path` in eval config.

### Per-request concurrency scheduling for multi-turn rollout

By default `generate_and_rm` holds one session-level concurrency permit (`GenerateState.semaphore`) for the entire custom `generate` call — a multi-turn rollout keeps the slot even while running env/tool steps, which hurts engine utilization.

To scope concurrency down to a single model request, declare the opt-in flag on your function and issue each turn's request via `state.post_generate`:

- `generate.manages_inference_permit = True`: once declared, the function is **no longer** wrapped in the session-level lock; it acquires a per-request permit itself.
- `await state.post_generate(url, payload)`: acquire one permit → send the request → release immediately on return. One permit == one in-flight request (including up to 6 internal retries); keep env interaction and observation encoding **outside** `post_generate` so they do not hold a permit.
- **abort**: if the rollout has already been aborted by the time the permit is acquired, `post_generate` raises `GenerationAborted`. You may catch it to write fine-grained resume metadata; **not catching it will not crash the step** — the framework contains it and marks the sample `ABORTED`.

**Usage contract**: only functions that declare `manages_inference_permit = True` may call `state.post_generate` / `state.inference_permit()`. A function without the flag still runs under the session-level lock; acquiring a permit from there nests an acquire on the **same non-reentrant semaphore** and deadlocks when the concurrency limit is 1 (a runtime guard detects the misuse and raises a clear `RuntimeError` instead of hanging).

**Note**: once the session-level lock is lifted, the number of concurrently *active* sessions is no longer bounded by this semaphore (only in-flight requests are); throttle concurrent env/tool resource usage inside your function if needed.

## Training Script and Key Parameters

For complete parameter reference, see [Configuration](./configuration.md).

After completing the preparation steps, you can run the training script. Using Qwen3 VL 4B as an example:

```bash
cd /root/Relax && \
export MODEL_CONFIG_DIR=$(pwd)/scripts/models && \
bash scripts/training/multimodal/run-qwen3-vl-4B-8xgpu.sh
```

### Model Configuration Parameters

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${MODEL_CONFIG_DIR}/qwen3-vl-4B.sh"
```

This section provides Megatron with the required hyperparameters. Since Megatron cannot read model configurations directly from checkpoints, they must be specified manually. We provide configuration examples for common models in the `scripts/models/` directory. To add a new model, create a configuration file there and source it in your task launch script.

### Checkpoint and Path Parameters

```bash
CKPT_ARGS=(
  # Used to load tokenizer and other info; model weights from this HF path are not actually used
  --hf-checkpoint ${MODEL_DIR}/Qwen3-VL-4B-Instruct/
  # Reference model checkpoint
  # When --load is not set, this will be used as the initial checkpoint for training
  --ref-load ${MODEL_DIR}/Qwen3-VL-4B-Instruct/
  # Enable megatron bridge automatic weight conversion
  --megatron-to-hf-mode bridge
  # Actor model load path. If empty or no valid checkpoint exists, loads from --ref-load
  # For resuming training, point this to the checkpoint path
  --load /path/checkpoint/
  # Save path for model during training
  --save /path/checkpoint/
  # Model save interval (in steps)
  --save-interval 20
)
```

::: tip Path variable convention
The launcher scripts define three path variables at the top:

- `MODEL_DIR` — model paths such as HF weights and `--ref-load`
- `DATA_DIR` — dataset paths such as `PROMPT_SET` and `--eval-prompt-data`
- `EXP_DIR` — training output paths such as `--load` / `--save`

Each one can be overridden independently via environment variable. `MODEL_DIR` and `DATA_DIR` fall back to `EXP_DIR` when unset, so a single `export EXP_DIR=/root` still drives all three paths.
:::

### Data Generation and Training Parameters

```bash
# Dataset path
--prompt-data ${PROMPT_SET}
# Number of prompts to sample per round
--rollout-batch-size 32
# Number of responses to generate per prompt
# Multiplied with --rollout-batch-size to determine total samples per round
--n-samples-per-prompt 8
# Number of samples required for one parameter update (optimizer.step)
--global-batch-size 256
# Total number of "sample → train" loop iterations
--num-rollout ${NUM_ROLLOUT}
```

### Message Processing Parameters

```bash
# Dataset input key
--input-key prompt
# Dataset label key
--label-key label
# Apply Chat Template if the prompt's input_key is in OpenAI message format
--apply-chat-template
# Reward computation method; this option only supports built-in reward methods
# For custom reward, use --custom-rm-path
--rm-type openr1mm
# Multimodal data extraction keys
--multimodal-keys '{"image":"image"}'
# Custom SYSTEM_PROMPT; inserts a new message at the head of the prompt
--system-prompt ${SYSTEM_PROMPT}
```

### Evaluation Parameters

You can add eval datasets for evaluation. Note that each eval call processes the entire dataset, so keep eval datasets small.

```bash
VAL_ARGS=(
  # Evaluation interval (in rollout count)
  --eval-interval 5
  # Evaluation prompt dataset
  --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
  # Number of samples per evaluation prompt
  --n-samples-per-eval-prompt 16
  # Maximum response length during evaluation
  --eval-max-response-len 16384
  # Sampling parameters during evaluation
  --eval-top-p 0.7
)
```

### Monitoring and Dump

```bash
# Enable ClearML
--use-clearml
# Enable TensorBoard
--use-tensorboard
# Enable centralized metrics collection and reporting service
--use-metrics-service
# TensorBoard/ClearML storage path
--tb-project-name ${PROJECT_NAME}
# TensorBoard/ClearML storage name
--tb-experiment-name name
# Dump per-step rollout and training details to a specified directory
--dump-details /path
```

### Parallelism and Performance Tuning

```bash
# Training parallelism
--tensor-model-parallel-size 2
--sequence-parallel
--pipeline-model-parallel-size 1
--expert-model-parallel-size 8
# Recomputation
--recompute-granularity full
--recompute-method uniform
--recompute-num-layers 1
# CPU offload optimizer
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
# Inference
--rollout-num-gpus-per-engine 2 # sglang tp
--sglang-mem-fraction-static 0.8
# Enables dynamic batching. When enabled, --micro-batch-size is ignored.
--use-dynamic-batch-size
# Maximum number of tokens processed per GPU. 
# When dynamic batching (use_dynamic_batch_size) is enabled, the system intelligently packs samples of varying lengths
# so that each micro-batch's total token count approaches this limit, improving training efficiency.
# If a single sample exceeds this value, it will form its own batch.
--max-tokens-per-gpu 9216
```

### Ray Launch Command

```bash
ray job submit --address="http://127.0.0.1:8265" \
  -- python3 relax/entrypoints/train.py \
  # [1, 8] represent replica count and total GPU count respectively; set replicas to 1
  # Resources are partitioned via _derive_cluster_args_from_resource
  --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
  --max-staleness 0 \
  --num-data-storage-units 1 \
  --colocate \
  # Other parameters expanded below
```

### Multi-Node Launch

Relax provides two multi-node launch methods: **SPMD Multi-Node Mode** (self-built Ray cluster) and **Ray Job Mode** (existing Ray cluster).

#### Method 1: SPMD Multi-Node Mode

Suitable for launching a Ray cluster from scratch on bare-metal or container environments and running training. The script automatically distinguishes between Head and Worker nodes, forms a cluster, and submits the training task on the Head node.

**Required environment variables** (must be set on each machine):

| Variable | Description | Example |
|---|---|---|
| `MASTER_ADDR` | Hostname of the Head node | `node-0` |
| `POD_NAME` | Hostname of the current node | `node-0` / `node-1` |
| `HOST_IP` | IP address of the current node | `<node-ip>` |
| `WORLD_SIZE` | Total number of nodes (default 2) | `2` |
| `NUM_GPUS` | GPUs per node (default 8) | `8` |

**Run the same command on every machine**:

```bash
bash scripts/entrypoint/spmd-multinode.sh scripts/training/multimodal/run-qwen3-30B-A3B-omni-16xgpu.sh
```

The script determines roles automatically based on `MASTER_ADDR == POD_NAME`:
- **Head node**: Starts Ray Head → waits for all Workers to join → executes training script
- **Worker node**: Joins the Ray cluster → blocks until training completes

#### Method 2: Ray Job Mode

Suitable when the Ray cluster is already managed by an external platform (e.g., KubeRay). The script does not start or stop Ray; it only cleans up residual processes and runs training directly.

**Prerequisites**:
- Ray cluster is running and the current node can connect via `ray status`
- The script automatically obtains `MASTER_ADDR` from the Ray cluster

```bash
bash scripts/entrypoint/ray-job.sh scripts/training/multimodal/run-qwen35-9B-8xgpu-async.sh
```

#### Comparison of the Two Methods

| | SPMD Multi-Node | Ray Job |
|---|---|---|
| Ray Cluster Management | Self-built by script (Head + Worker) | Externally managed (KubeRay, etc.) |
| Must run on each machine | Yes | No (submit node only) |
| Use Case | Bare-metal / container SPMD scheduling | Existing Ray cluster |
| Entry Script | `scripts/entrypoint/spmd-multinode.sh` | `scripts/entrypoint/ray-job.sh` |

## Next Steps

### Custom Experiments

1. **Modify launch scripts**: Edit shell scripts in `examples/` or `scripts/`
2. **Switch models**: Update the `--hf-checkpoint` parameter to point to your model
3. **Tune training**: Modify optimizer, learning rate, and batch size parameters
4. **Custom rewards**: Implement custom reward functions (see DeepEyes example)

### Explore Examples

- [DeepEyes](../examples/deepeyes.md) — Multimodal vision-language reinforcement learning
- [On-Policy Distillation](../examples/on-policy-distillation.md) — Knowledge distillation

### Learn Core Concepts

- [Architecture](./architecture.md) — Understand the system design
- [Dataset Design](./dataset-design.md) — Learn about data loading
- [Distributed Checkpoint](./distributed-checkpoint.md) — Checkpoint management

## Getting Help

- [GitHub Issues](https://github.com/redai-studio/Relax/issues)
- [Discussions](https://github.com/redai-studio/Relax/discussions)
- [Introduction](../guide/introduction.md)
