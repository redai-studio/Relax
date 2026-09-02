# SFT Training

This guide shows the end-to-end supervised fine-tuning (SFT) workflow in Relax. It covers generative SFT with the current scripts under [`scripts/training/sft/`](../../../scripts/training/sft/), native sequence-classification SFT under [`examples/seq_cls_sft/`](../../../examples/seq_cls_sft/), model and data preparation, launch commands, and practical tuning.

Make sure you have completed [Installation](./installation.md) before running the commands below.

## Overview

SFT is enabled with `--loss-type sft`. In this mode, Relax starts an SFT producer that reads `--prompt-data`, renders samples through the model chat template, writes packed samples into TransferQueue, and trains the Megatron actor. The default `--task-type causal_lm` trains the language-model head, runs PPL evaluation when `--eval-interval` is set, and can use the Rollout role for periodic generative prediction through `--sft-predict-interval`. Native classification uses `--task-type seq_cls`, replaces the vocabulary head with a classification head, and evaluates classification loss and metrics without a Rollout role.

The current launch scripts use `ray job submit` and auto-source [`scripts/entrypoint/local.sh`](../../../scripts/entrypoint/local.sh) when no external entrypoint has already prepared the Ray environment.

## Scripts

| Script | Dataset | Model | Default resources | Notes |
| --- | --- | --- | --- | --- |
| [`run-qwen3.5-9B-math-8xgpu.sh`](../../../scripts/training/sft/run-qwen3.5-9B-math-8xgpu.sh) | `OpenMathReasoning-mini` | `Qwen3.5-9B` | 8 GPU actor plus SFT producer and Rollout | Text SFT with `problem` -> `generated_solution`, PPL eval, and predict. |
| [`run-qwen3-vl-4B-math-8xgpu.sh`](../../../scripts/training/sft/run-qwen3-vl-4B-math-8xgpu.sh) | `OpenMathReasoning-mini` | `Qwen3-VL-4B-Instruct` | 8 GPU actor plus SFT producer and Rollout | Text-only math SFT using a VL checkpoint. |
| [`run-qwen3-vl-4B-pokemon-8xgpu.sh`](../../../scripts/training/sft/run-qwen3-vl-4B-pokemon-8xgpu.sh) | `pokemon-gpt4o-captions` | `Qwen3-VL-4B-Instruct` | 8 GPU actor plus SFT producer and Rollout | Multimodal image SFT with two parquet files and prefetch enabled. |
| [`run-qwen3-vl-4B-pokemon-1xgpu.sh`](../../../scripts/training/sft/run-qwen3-vl-4B-pokemon-1xgpu.sh) | `pokemon-gpt4o-captions` | `Qwen3-VL-4B-Instruct` | 1 GPU actor plus SFT producer | Low-resource Pokemon SFT with CPU optimizer offload. |
| [`run-qwen3.5-35B-A3B-mtp-sft-16xgpu.sh`](../../../scripts/training/sft/run-qwen3.5-35B-A3B-mtp-sft-16xgpu.sh) | `OpenMathReasoning-mini` | `Qwen3.5-35B-A3B` | 16 GPU actor plus SFT producer | Advanced MTP SFT. Many knobs are exposed as environment variables. |
| [`run-qwen3.5-9B-classification-sft-8xgpu.sh`](../../../examples/seq_cls_sft/run-qwen3.5-9B-classification-sft-8xgpu.sh) | SST-2, AG News, or GoEmotions | `Qwen3.5-9B` | 8 GPU actor plus SFT producer | Full-parameter binary, multiclass, or multilabel sequence-classification SFT. |
| [`run-qwen3.5-35B-A3B-classification-lora-sft-8xgpu.sh`](../../../examples/seq_cls_sft/run-qwen3.5-35B-A3B-classification-lora-sft-8xgpu.sh) | SST-2, AG News, or GoEmotions | `Qwen3.5-35B-A3B` | 8 GPU actor plus SFT producer | LoRA sequence-classification SFT; defaults to GoEmotions. |

## Data Preparation

The SFT scripts default to `DATA_DIR=${SCRIPT_DIR}/data` unless you override `DATA_DIR`. For reusable jobs, put datasets on persistent storage and export `DATA_DIR` explicitly.

```bash
cd /root/Relax
export DATA_DIR=/root
mkdir -p "${DATA_DIR}/sft/data"
```

### Math: OpenMathReasoning-mini

The math scripts expect this exact file:

```text
${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet
```

Download it from [unsloth/OpenMathReasoning-mini](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini):

```bash
hf download --repo-type dataset unsloth/OpenMathReasoning-mini \
  data/cot-00000-of-00001.parquet \
  --local-dir "${DATA_DIR}/sft/data/OpenMathReasoning-mini"
```

The current math scripts configure:

```bash
--prompt-data "${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet"
--input-key problem
--label-key generated_solution
```

With `--label-key` set, each row must contain a prompt string under `problem` and a target response under `generated_solution`. Relax builds a user message from `problem` and appends an assistant message from `generated_solution`.

Minimal row shape:

```json
{
  "problem": "Find the value of x.",
  "generated_solution": "We solve the equation step by step..."
}
```

### Pokemon: pokemon-gpt4o-captions

The Pokemon scripts expect these files:

```text
${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_en.parquet
${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_zh.parquet
```

Download them from [llamafactory/pokemon-gpt4o-captions](https://huggingface.co/datasets/llamafactory/pokemon-gpt4o-captions):

```bash
hf download --repo-type dataset llamafactory/pokemon-gpt4o-captions \
  pokemon_gpt4o_en.parquet pokemon_gpt4o_zh.parquet \
  --local-dir "${DATA_DIR}/sft/data/pokemon-gpt4o-captions"
```

The current Pokemon scripts build a list-valued `PROMPT_DATA` from both files:

```bash
TRAIN_FILES=(
    "'${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_en.parquet'"
    "'${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_zh.parquet'"
)
PROMPT_DATA="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"
```

They then configure:

```bash
--prompt-data "${PROMPT_DATA}"
--input-key conversations
--multimodal-keys '{"image":"images"}'
--conversation-key-map '{"from":"role","value":"content","human":"user","gpt":"assistant"}'
```

Because `--label-key` is not set, `conversations` must be a complete message list. The `--conversation-key-map` rewrites ShareGPT-style message fields and role values into the OpenAI-style `{role, content}` format expected by SFT. The `--multimodal-keys` mapping tells Relax to load image paths from the row's `images` column. The text should contain an `<image>` placeholder for each image item.

Minimal row shape:

```json
{
  "conversations": [
    {"from": "human", "value": "Identify the object of this image.<image>"},
    {"from": "gpt", "value": "A round, pink Pokemon with a gentle expression."}
  ],
  "images": ["/path/to/pokemon.png"]
}
```

### Verify Data Files

Run this before launching to catch missing files or wrong columns:

```bash
python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq

checks = {
    "math": (
        Path("/root/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet"),
        {"problem", "generated_solution"},
    ),
    "pokemon_en": (
        Path("/root/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_en.parquet"),
        {"conversations", "images"},
    ),
    "pokemon_zh": (
        Path("/root/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_zh.parquet"),
        {"conversations", "images"},
    ),
}

for name, (path, required) in checks.items():
    if not path.exists():
        print(f"{name}: missing {path}")
        continue
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = required - names
    print(f"{name}: rows={pq.ParquetFile(path).metadata.num_rows}, missing={sorted(missing)}")
PY
```

Change `/root` in the check if you used a different `DATA_DIR`.

## Model Preparation

Download model weights to persistent storage. The math and Pokemon scripts read model paths from `MODEL_DIR` rather than from `EXP_DIR`, so set `MODEL_DIR` explicitly for these scripts.

```bash
# Math script: run-qwen3.5-9B-math-8xgpu.sh
hf download Qwen/Qwen3.5-9B --local-dir /root/Qwen3.5-9B

# Pokemon scripts: run-qwen3-vl-4B-pokemon-*.sh
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /root/Qwen3-VL-4B-Instruct
```

For the MTP script, the default convention is `EXP_DIR=/root`, then `MODEL_DIR=${EXP_DIR}` and `DATA_DIR=${EXP_DIR}` unless you override them.

## Sequence Classification SFT

Sequence-classification SFT trains a native `K`-logit classification head instead of generating label text. The example supports three objective shapes:

| Task | Input label | Output and loss |
| --- | --- | --- |
| Single-label binary classification | Integer `0` or `1` | 2 logits with CrossEntropy |
| Single-label multiclass classification | Integer in `[0, K)` | `K` logits with CrossEntropy |
| Multilabel classification | List of class indices, such as `[0, 2, 5]` | `K` logits with BCEWithLogits |

The classification threshold is used only to turn multilabel probabilities into discrete eval or inference predictions. It is not used to compute the training or eval loss.

### 1. Configure Paths

Run the following commands from the Relax repository root. The scripts expect the Qwen3.5 model directories directly under `MODEL_DIR` and write prepared datasets under `${DATA_DIR}/sft/seq_cls`.

```bash
export DATA_DIR=/path/to/relax-workspace
export EXP_DIR=/path/to/relax-workspace/exp
export MODEL_DIR=/path/to/relax-workspace/models
```

For the 9B script, place the original Hugging Face model at `${MODEL_DIR}/Qwen3.5-9B`. The 35B-A3B LoRA script uses `${MODEL_DIR}/Qwen3.5-35B-A3B`.

### 2. Prepare Datasets

The data preparation tool converts pinned revisions of SST-2, AG News, and the simplified GoEmotions dataset into the JSONL format consumed by the training scripts:

```bash
python examples/seq_cls_sft/tools/prepare_classification_sft_data.py \
  --dataset all \
  --subset full \
  --global-batch-size 64 \
  --output-dir "${DATA_DIR}/sft/seq_cls"
```

Use `--dataset sst2`, `--dataset ag_news`, or `--dataset go_emotions` to prepare only one task. The launch scripts default to the `full` subset; `DATA_SUBSET=smoke` and `DATA_SUBSET=extended` select the smaller deterministic subsets produced by the same tool.

Custom single-label rows use an integer `label`:

```json
{"messages": [{"role": "user", "content": "This movie is excellent."}], "label": 1}
```

Custom multilabel rows use a list of class indices:

```json
{"messages": [{"role": "user", "content": "I feel relieved and excited."}], "label": [4, 17]}
```

Set `TRAIN_DATA` and `EVAL_DATA` to use custom JSONL files. For a custom label space, also set `NUM_LABELS` and `PROBLEM_TYPE` to `single_label_classification` or `multi_label_classification`.

### 3. Launch Training

The Qwen3.5-9B full-parameter script defaults to SST-2. Select AG News or GoEmotions with `CLASSIFICATION_DATASET`:

```bash
bash examples/seq_cls_sft/run-qwen3.5-9B-classification-sft-8xgpu.sh

CLASSIFICATION_DATASET=ag_news \
  bash examples/seq_cls_sft/run-qwen3.5-9B-classification-sft-8xgpu.sh

CLASSIFICATION_DATASET=go_emotions \
  bash examples/seq_cls_sft/run-qwen3.5-9B-classification-sft-8xgpu.sh
```

The Qwen3.5-35B-A3B LoRA script defaults to GoEmotions:

```bash
bash examples/seq_cls_sft/run-qwen3.5-35B-A3B-classification-lora-sft-8xgpu.sh
```

Both scripts use dynamic batching and eight training GPUs, train for 10 epochs by default, and save a Megatron native checkpoint every 100 steps. Single-label eval reports loss and accuracy. Multilabel eval reports loss, micro precision, micro recall, micro F1, and subset accuracy.

::: warning Current Scope
Sequence-classification SFT currently supports text-only synchronous SFT. It does not support fully async, hybrid, SFT generative prediction, MTP, chunked SFT logits, or online `--save-hf`. Export the native checkpoint offline before serving it. The provided SGLang adapter supports tensor parallelism, not pipeline parallelism.
:::

::: tip Complete Export and Serving Workflow
See the [`examples/seq_cls_sft` README](../../../examples/seq_cls_sft/README.md) for every environment override, offline Megatron-to-Hugging-Face export, SGLang startup, and single-label or multilabel request example.
:::

## Configuration Walkthrough

The scripts are organized into argument blocks. Tune by editing the selected script, or by using environment variables where the script already exposes them.

### Paths and Checkpoints

For math and Pokemon scripts:

```bash
export MODEL_DIR=/root
export DATA_DIR=/root
```

The `CKPT_ARGS` block sets:

| Flag | Purpose |
| --- | --- |
| `--hf-checkpoint` | HF checkpoint used for tokenizer, config, and SGLang initialization when predict is enabled. |
| `--ref-load` | Initial reference checkpoint path. |
| `--load` | Training load path. If this is an existing Megatron checkpoint, training resumes from it. Otherwise bridge mode starts from HF weights. |
| `--megatron-to-hf-mode bridge` | Uses Megatron Bridge for HF <-> Megatron conversion. |
| `--save` | Megatron checkpoint output directory. |
| `--save-interval` | Save every N training steps. |
| `--num-epoch` | Number of dataset epochs. Relax resolves the actual training steps from dataset size and global batch size. |

For a fresh run, keep `--load` and `--save` pointing at the same experiment directory as the current scripts do. For resume, keep the same `--save` and make sure it contains a valid checkpoint with `latest_checkpointed_iteration.txt`.

### MTP Arguments

The MTP SFT script enables `--mtp-num-layers ${MTP_NUM_LAYERS:-1}`, `--enable-mtp-training`, and `--mtp-loss-scaling-factor ${MTP_LOSS_SCALING_FACTOR:-0.2}`. Increase `MTP_NUM_LAYERS` only for models/checkpoints with matching MTP layers; tune `MTP_LOSS_SCALING_FACTOR` as an auxiliary-loss weight, starting from `0.2`.

See [MTP Training](./mtp-rl-training.md) for joint training, MTP-only two-stage training, and Pokémon measurements.

### SFT Data Arguments

Required SFT flags:

```bash
--loss-type sft
--prompt-data "${PROMPT_DATA}"
--use-dynamic-batch-size
--max-tokens-per-gpu <tokens>
```

SFT requires dynamic batching. If `--use-dynamic-batch-size` is missing, argument validation fails. If `--balance-data` is not set, SFT validation auto-enables it so the SFT producer and Megatron data path stay consistent.

Use one of two row formats:

| Format | Flags | When to use |
| --- | --- | --- |
| Prompt plus label | `--input-key problem --label-key generated_solution` | Text rows where prompt and supervised answer are separate columns. |
| Full messages | `--input-key conversations` and no `--label-key` | Chat or multimodal rows that already contain user and assistant turns. Add `--conversation-key-map` for ShareGPT-style fields. |

Do not add `--apply-chat-template` for SFT. The SFT dataset renders samples with the tokenizer chat template internally and builds the assistant loss mask from that render.

### Chat Template Compatibility Patches

#### Dispatch and Scope

Before rendering an SFT sample, Relax resolves the effective template from an explicit `chat_template` override or `tokenizer.chat_template`, then runs the static model-specific patch registry. Global `--apply-chat-template-kwargs` are merged with `sample.metadata["apply_chat_template_kwargs"]`; the per-sample values win. Both the tokenized `render_with_loss_mask` path and the multimodal `render_to_text` path use the same result.

Patches are applied per sample and per call. They never mutate `tokenizer.chat_template`. If no patcher recognizes the effective template, the dispatcher is a strict no-op: the original template and explicitly supplied kwargs are forwarded unchanged. Model detection is based on template content rather than checkpoint names.

The generic dispatcher is implemented in [`chat_template_patch.py`](../../../relax/engine/sft/dataset/chat_template_patch.py), while model-specific behavior lives in separate modules such as [`qwen_chat_template_patch.py`](../../../relax/engine/sft/dataset/qwen_chat_template_patch.py).

::: tip Scope
These patches only affect SFT sample rendering. They do not change Rollout, Agentic Session, SGLang, or other inference-side chat-template behavior.
:::

#### Qwen Historical Thinking

Some Qwen templates keep reasoning only for assistant messages after the last ordinary user query. In multi-turn tool data, this can remove `<think>...</think>` from an earlier supervised assistant while leaving its tool call in place. Relax recognizes the exact legacy Qwen3.5 history gate and backports the compatible `preserve_thinking` gate for the current render. Templates that already contain the native gate remain unchanged.

`preserve_thinking` has three states:

| Setting | Behavior |
| --- | --- |
| Omitted or `null` | Default, sample-aware auto mode. Relax sets it to `true` when an assistant before the last ordinary user has `learn=true` and its text contains `</think>`; otherwise Qwen's native behavior is preserved. |
| `true` | Force preservation of historical thinking for the whole sample render. `learn` still controls which rendered tokens receive loss. |
| `false` | Disable auto-preservation and use Qwen's native historical-reasoning compression. Non-reasoning content, including answers and tool calls, is still rendered. |

Assistant messages default to `learn=true` when a full-message SFT row does not provide `learn`. A user message whose complete string content is wrapped in `<tool_response>...</tool_response>` is not treated as a new ordinary user boundary. If auto mode is triggered by one learnable historical assistant, preservation applies to the whole render; messages with `learn=false` may remain as context but still receive no loss.

Because per-sample kwargs take precedence, a sample-level `preserve_thinking: null` overrides a global `false` and re-enters auto mode for that sample.

Use JSON booleans, not strings:

```bash
# Force preservation
--apply-chat-template-kwargs '{"preserve_thinking": true}'

# Disable automatic preservation
--apply-chat-template-kwargs '{"preserve_thinking": false}'

# Re-enable sample-aware auto mode explicitly
--apply-chat-template-kwargs '{"preserve_thinking": null}'
```

::: warning `false` Does Not Disable All Thinking
`preserve_thinking=false` only restores Qwen's native compression for assistant turns before the last ordinary user. Assistant turns in the current tool episode still follow the native Qwen template and may contain an empty `<think>\n\n</think>` wrapper. This setting is unrelated to inference-side `enable_thinking`.

When historical reasoning is compressed, those reasoning tokens are absent from the rendered training sequence and cannot receive loss; the associated answer or tool call remains trainable.
:::

Values other than a JSON boolean or `null`, such as `"false"`, raise `ValueError`. A Qwen-like template with a duplicated, conflicting, or unrecognized history gate raises `RuntimeError` instead of being patched heuristically. The SFT log reports `source=qwen_history_thinking (patched|native)` and the resolved `preserve_thinking` value for verification.

### Evaluation and Predict

PPL evaluation is controlled by:

```bash
--eval-size 0.01
--eval-interval 10
```

`--eval-size` randomly splits rows from `--prompt-data` once using `--seed`, keeps that eval subset fixed, and removes it from every shuffled training epoch. A value below 1 is a fraction; a value of 1 or higher is an absolute sample count. This is a row-level split; related rows such as multilingual captions for the same image are not grouped automatically. You may use `--eval-prompt-data name path` instead, but in SFT mode do not use `--eval-config`.

Generative prediction is controlled by:

```bash
--sft-predict-interval 10
--eval-temperature 0.0
--eval-max-response-len 10240
```

When `--sft-predict-interval` is set, Relax spins up the Rollout role automatically, and predictions are written under:

```text
<save>/predict/predictions_step_<rollout_id>.jsonl
```

Use a longer `--eval-max-response-len` for math reasoning and a shorter value for captioning or image description tasks.

### Parallelism and Resources

The script-level `PERF_ARGS` and `--resource` must agree with the cluster size.

Math 8 GPU script:

```bash
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 2
--context-parallel-size 1
--resource '{"sft": [1, 0], "actor": [1, 8], "rollout": [1, 8]}'
```

Pokemon 8 GPU script:

```bash
--tensor-model-parallel-size 2
--pipeline-model-parallel-size 1
--context-parallel-size 1
--per-rank-fetch
--num-data-storage-units 8
--resource '{"sft": [1, 0], "actor": [1, 8], "rollout": [1, 8]}'
```

Pokemon 1 GPU script:

```bash
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
--resource '{"sft": [1, 0], "actor": [1, 1], "rollout": [1, 1]}'
```

`"sft": [1, 0]` means the SFT producer is CPU-only. The Actor owns training GPUs. Rollout GPUs are needed when periodic predict is enabled.

### Sharded SFT Producers

`RELAX_SFT_TQ_SHARDS` is an experimental throughput knob for SFT async prepacking. It splits each global SFT batch across multiple TransferQueue partitions. Configure it in the Ray runtime environment:

```yaml
# configs/env.yaml
env_vars:
  RELAX_SFT_TQ_SHARDS: "2"
```

Enable async prepacking in the training arguments as well:

```bash
--per-rank-fetch
--sft-async-prepack
--sft-max-in-flight-steps 4
```

`--sft-max-in-flight-steps 4` is an example; async prepacking requires at least 2, or equivalently `--max-staleness >= 1`. The environment variable does not enable prepacking by itself and is ignored without `--loss-type sft --sft-async-prepack`.

For `N` shards, both `--global-batch-size` and the per-DP-rank local batch (`global_batch_size / data_parallel_size`) must be divisible by `N`. For example, with global batch size 32 and DP size 8, the local batch is 4, so 2 or 4 shards are valid but 3 is not.

When the remote path is eligible, `N` shards launch `N` Ray producer actors for train batches. Eval can be enabled at the same time: train shards are produced remotely, while the coordinator renders the eval split or eval prompt data and pushes the usual `sft_eval_<step>_n<N>_<i>` partitions at eval intervals. Some configurations, including sequence classification, custom datasets (`--custom-dataset-class` / `--custom-dataset-class-path`), and skip-capable sample filtering, use one local producer that still writes `N` partitions. Therefore, `RELAX_SFT_TQ_SHARDS=2` does not always mean two producer actors. Confirm the remote path from this log:

```text
SFT remote shard producer enabled: ... shards=2 ...
```

When using `--eval-size`, every remote train producer applies the same seed-based held-out split so train and eval rows do not overlap. When launching Python directly, `export RELAX_SFT_TQ_SHARDS=2` is also supported. For a Ray Job, putting the value in `configs/env.yaml` ensures that producers and consumers receive the same value. See [Sharded TransferQueue Producers](./configuration.md#sharded-transferqueue-producers) for partition naming, fallback conditions, and worker allocation.

## Launch

### Single Node

Use the script directly. It sources `local.sh`, starts a local Ray head node, and submits the job:

```bash
cd /root/Relax
export MODEL_DIR=/root
export DATA_DIR=/root

bash scripts/training/sft/run-qwen3.5-9B-math-8xgpu.sh
bash scripts/training/sft/run-qwen3-vl-4B-pokemon-8xgpu.sh
```

For the 1 GPU Pokemon script:

```bash
cd /root/Relax
export MODEL_DIR=/root
export DATA_DIR=/root
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1

bash scripts/training/sft/run-qwen3-vl-4B-pokemon-1xgpu.sh
```

### Existing Ray Cluster

If a Ray cluster is already running, use [`scripts/entrypoint/ray-job.sh`](../../../scripts/entrypoint/ray-job.sh). It prepares the runtime environment, avoids stopping Ray, and then delegates to the run script:

```bash
cd /root/Relax
export MODEL_DIR=/root
export DATA_DIR=/root
export RAY_ADDRESS=http://127.0.0.1:8265

bash scripts/entrypoint/ray-job.sh scripts/training/sft/run-qwen3-vl-4B-pokemon-8xgpu.sh
```

### Multi-Node

Use [`scripts/entrypoint/spmd-multinode.sh`](../../../scripts/entrypoint/spmd-multinode.sh). Required environment variables are `MASTER_ADDR`, `POD_NAME`, `HOST_IP`, and `WORLD_SIZE`; `NUM_GPUS` defaults to 8 per node.

```bash
cd /root/Relax
export MODEL_DIR=/root
export DATA_DIR=/root
export WORLD_SIZE=2
export NUM_GPUS=8

bash scripts/entrypoint/spmd-multinode.sh \
  scripts/training/sft/run-qwen3.5-35B-A3B-mtp-sft-16xgpu.sh
```

## Tuning Workflow

Tune in this order so each change has a clear purpose.

### 1. Fit in Memory

If the job OOMs during training:

| Knob | Direction | Effect |
| --- | --- | --- |
| `--max-tokens-per-gpu` | Decrease | Lowers dynamic micro-batch token capacity. This is the first SFT memory knob. |
| `--global-batch-size` | Decrease | Reduces samples per optimizer update, but changes optimization dynamics. |
| `--recompute-num-layers` | Increase | Saves activation memory at the cost of compute. |
| `--optimizer-cpu-offload` | Enable | Saves GPU memory, useful for 1 GPU or tight VL runs. |
| `--sft-oversize-strategy skip` | Enable carefully | Drops samples longer than `max_tokens_per_gpu * context_parallel_size`. |

For context parallelism, capacity is `--max-tokens-per-gpu * --context-parallel-size`. Increase CP only when the model and Megatron configuration support it.

### 2. Improve Throughput

If GPUs wait on SFT data:

| Knob | Direction | Effect |
| --- | --- | --- |
| `--sft-prefetch-buffer-size` | Increase from 256 | Keeps more rendered samples ready. |
| `--sft-prefetch-num-workers` | Increase | Improves image decode and multimodal I/O parallelism. |
| `--sft-prefetch-chunk-size` | Increase | Dispatches larger prefetch chunks, with higher memory pressure. |
| `--per-rank-fetch` | Enable for multi-GPU | Lets TP/PP ranks pull from TransferQueue directly. Pair with enough `--num-data-storage-units`. |
| `--max-staleness` | Increase for I/O-heavy SFT | Lets the producer run ahead. The Pokemon 8 GPU script uses `--max-staleness 4`. |
| `RELAX_SFT_TQ_SHARDS` | Start from 2 | Parallelizes eligible async-prepack producers. Increase only when data preparation is the bottleneck and batch divisibility constraints are met. |

For text-only math, prefetch usually matters less than sequence length and model parallelism. For Pokemon, image loading and processor work are common bottlenecks.

### 3. Preserve Quality

Start with the script defaults, then change one optimization knob at a time:

| Knob | Math default | Pokemon 8 GPU default | Notes |
| --- | --- | --- | --- |
| `--lr` | `1e-5` | `1e-5` | Lower it if eval loss rises quickly or resume from a strong checkpoint. |
| `--lr-decay-style` | `cosine` | `cosine` | The 1 GPU Pokemon script uses `constant` with `3e-5`, which is more aggressive. |
| `--weight-decay` | `0.1` | `0.1` | Keep stable unless you are doing a controlled sweep. |
| `--clip-grad` | `1.0` | `1.0` | Lower only if gradients spike. |
| `--num-epoch` | `10` | `10` | Reduce for smoke tests or increase only with eval monitoring. |

### 4. Control Evaluation Cost

| Knob | Direction | Effect |
| --- | --- | --- |
| `--eval-size` | Lower | Less held-out data and cheaper PPL eval. |
| `--eval-interval` | Increase | Fewer eval rounds. |
| `--sft-predict-interval` | Increase or disable | Reduces SGLang predict overhead. |
| `--eval-max-response-len` | Lower | Caps generation cost for predict. |

For fast bring-up, disable predict by commenting out `PREDICT_ARGS` and use a small `--eval-size`. Re-enable predict once the training loop is stable.

## Troubleshooting

### `--loss-type sft requires --use-dynamic-batch-size`

SFT intentionally rejects static micro-batching. Add `--use-dynamic-batch-size` and set `--max-tokens-per-gpu`.

### `Under --loss-type sft with --eval-interval set...`

`--eval-interval` requires exactly one eval source: either `--eval-size` or `--eval-prompt-data name path`. Do not set both.

### `SFT row missing prompt key`

The value of `--input-key` does not match the dataset column. Math uses `problem`; Pokemon uses `conversations`.

### `--label-key is not set ... expects ... messages list`

You passed a string prompt without `--label-key`. Either add `--label-key` for prompt-plus-label rows, or convert the row to a full message list.

### Multimodal Placeholder Mismatch

For image SFT, each image referenced by `--multimodal-keys '{"image":"images"}'` must have a matching `<image>` placeholder in the conversation content. Extra images or missing placeholders cause data processing errors.

### Predict Does Not Write Files

Check that `--sft-predict-interval` is set, `--save` is set, and an eval source exists. Prediction files are written under `<save>/predict/`.

## Extending Chat Template Patches

Add compatibility for another model family through a dedicated pure-function patcher. Keep model knowledge out of [`chat_template.py`](../../../relax/engine/sft/dataset/chat_template.py); create the new module beside the existing [`qwen_chat_template_patch.py`](../../../relax/engine/sft/dataset/qwen_chat_template_patch.py).

### Patcher Contract

Use the `TemplatePatcher` signature from [`chat_template_patch.py`](../../../relax/engine/sft/dataset/chat_template_patch.py):

The private `_matches_exact_foo_template` and `_patch_exact_foo_template` names below are placeholders. Implement the corresponding model-specific recognition and exact replacement helpers in the new module.

```python
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from collections.abc import Mapping
from typing import Any

from relax.engine.sft.dataset.chat_template_patch import TemplatePatchResult
from relax.engine.sft.dataset.sample import CanonicalSample


def try_patch_foo_chat_template(
    sample: CanonicalSample,
    template: str | None,
    kwargs: Mapping[str, Any],
) -> TemplatePatchResult | None:
    if not isinstance(template, str):
        return None

    # Match exact, model-specific legacy or native template fragments.
    if not _matches_exact_foo_template(template):
        return None

    resolved_kwargs = dict(kwargs)
    patched_template = _patch_exact_foo_template(template)
    return TemplatePatchResult(
        template=patched_template,
        kwargs=resolved_kwargs,
        patch_name="foo_history_compatibility",
        changed=patched_template != template,
    )
```

Follow these rules:

1. Return `None` for every unrelated template. Use narrow, stable template signatures rather than model paths or tokenizer class names.
2. Recognize both the legacy template that needs a patch and the already-fixed native template so repeated rendering is idempotent.
3. If a template resembles the supported family but its critical fragment has drifted, fail with a clear error instead of guessing a replacement.
4. Copy `kwargs` before changing them. Do not mutate the sample, tokenizer, input mapping, or global state.
5. Return the effective template and kwargs. When the template changes, the dispatcher injects it as a per-call `chat_template` kwarg and keeps the template/kwargs pair consistent.

### Registration

Import the patcher in [`chat_template.py`](../../../relax/engine/sft/dataset/chat_template.py) and append it to the static registry:

```python
_CHAT_TEMPLATE_PATCHERS = (
    try_patch_qwen_chat_template,
    try_patch_foo_chat_template,
)
```

Patch signatures must be mutually exclusive. The dispatcher evaluates all registered patchers and raises `RuntimeError` if more than one matches; registration order is not a priority mechanism.

### Tests

Follow [`test_qwen_chat_template_patch.py`](../../../tests/engine/sft/dataset/test_qwen_chat_template_patch.py) and cover at least:

1. Unrelated-template strict no-op behavior.
2. Correct legacy-template patching and native-template idempotence.
3. Fail-fast behavior for drifted, duplicated, or conflicting signatures.
4. Explicit kwargs and default policy behavior without mutating the input mapping.
5. Consistent behavior through both `render_with_loss_mask` and `render_to_text`.
6. Preservation of the bound `tokenizer.chat_template`.
7. No overlapping match with existing patchers.

## Next Steps

- Follow the complete [sequence-classification SFT example](../../../examples/seq_cls_sft/README.md) for checkpoint export and SGLang serving.
- Read [Configuration Reference](./configuration.md) for the full SFT parameter table.
- Read [Performance Tuning](./performance-tuning.md) for broader throughput tuning.
- Read [OOM Troubleshooting](./oom-troubleshooting.md) if the job fails during model load or training.
