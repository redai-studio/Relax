# Quick Start

This guide provides three end-to-end examples covering **text**, **vision-language**, and **omni-modal** training tasks. Each example includes data preparation, model download, and training launch commands.

Make sure you have completed the [Installation](./installation.md) steps before proceeding.

## Task 1: DAPO Math (Text)

Train Qwen3-4B on the [dapo-math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k) math reasoning dataset using GRPO with 8 GPUs.

### Data Preparation

Download the training dataset and (optionally) the evaluation dataset:

```bash
# Download training dataset (dapo-math-17k)
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# Download evaluation dataset (aime-2024)
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024

# Add math instruction prefix to the aime evaluation dataset (in-place)
python scripts/tools/process_aime.py --input /root/aime-2024/aime-2024.jsonl
```

The training dataset is in `.jsonl` format and can be used directly — no conversion required. The evaluation dataset needs the above script to prepend an instruction prefix that guides the model to produce answers in `\boxed{}` format.

### Model Download

```bash
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
```

### Launch Training

Since the model and dataset are both downloaded to the `/root` directory, you only need to set `EXP_DIR=/root` and the script will automatically locate them — no manual script editing required.

::: tip Persistent Storage
Make sure the `/root` directory is mounted to persistent storage on the host, otherwise data will be lost when the container is destroyed. See the Docker mount instructions in the [Installation Guide](./installation.md).
:::

```bash
cd /root/Relax
export EXP_DIR=/root

# Single node
bash scripts/training/text/run-qwen3-4B-8xgpu.sh

# Multi-node
bash -x scripts/entrypoint/spmd-multinode.sh scripts/training/text/run-qwen3-4B-8xgpu.sh
```

::: tip Reward Method
This task uses the built-in `dapo` reward type (see `relax/engine/rewards/math_dapo_utils.py`). It employs rule-based answer extraction and symbolic math verification to evaluate correctness, assigning **1.0** for correct answers and **0.0** for incorrect ones.
:::

---

## Task 2: Open-R1 (Vision-Language)

Train Qwen3-VL-4B on the [multimodal-open-r1-8k-verified](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) image + text dataset using GRPO with 8 GPUs.

### Data Preparation

Download the dataset and convert it to the Relax format:

```bash
# Download dataset
hf download --repo-type dataset lmms-lab/multimodal-open-r1-8k-verified \
  --local-dir /root/multimodal-open-r1-8k-verified

# Convert to Relax format
python scripts/tools/process_openr1.py \
  --input-dir /root/multimodal-open-r1-8k-verified/data/train-00000-of-00001.parquet \
  --output-dir /root/multimodal-open-r1-8k-verified/data/train-00000-of-00001-converted.parquet
```

The conversion script reads the raw parquet, extracts `problem`, `image`, and `solution` fields, and produces a new parquet with `prompt`, `image`, and `label` columns in the format expected by Relax.

### Model Download

```bash
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /root/Qwen3-VL-4B-Instruct
```

### Launch Training

Since the model and dataset are both downloaded to the `/root` directory, you only need to set `EXP_DIR=/root` and the script will automatically locate them — no manual script editing required.

```bash
cd /root/Relax
export EXP_DIR=/root

# Single node
bash scripts/training/multimodal/run-qwen3-vl-4B-8xgpu.sh

# Multi-node
bash -x scripts/entrypoint/spmd-multinode.sh scripts/training/multimodal/run-qwen3-vl-4B-8xgpu.sh
```

::: tip Reward Method
This task uses the built-in `openr1mm` reward type (see `relax/engine/rewards/openr1mm.py`). It extracts the final answer from `<answer>...</answer>` tags in both the ground-truth and model output. Symbolic verification via `math_verify` is attempted first; string matching serves as a fallback.
:::

---

## Task 3: AVQA (Omni-Modal: Image + Audio)

Train Qwen3-Omni-30B-A3B on the [AVQA-R1-6K](https://huggingface.co/datasets/harryhsing/AVQA-R1-6K) image + audio question answering dataset using GRPO with 16 GPUs (2 nodes).

### Data Preparation

Download the dataset and convert it to the Relax format:

```bash
# Download dataset
hf download --repo-type dataset harryhsing/AVQA-R1-6K \
  --local-dir /root/AVQA-R1-6K

# Convert to Relax format
# --md-dir points to the directory containing image and audio files,
# used to join relative paths into absolute paths (Optional, default to use relative paths).
python scripts/tools/process_avqa.py \
  --input-dir /root/AVQA-R1-6K/AVQA_R1/train/omni_rl_format_train.json \
  --output-dir /root/AVQA-R1-6K/AVQA_R1/train/omni_rl_format_train_convert.jsonl \
  --md-dir /root/AVQA-R1-6K/AVQA_R1/train

python scripts/tools/process_avqa.py \
  --input-dir /root/AVQA-R1-6K/AVQA_R1/valid/omni_rl_format_valid.json \
  --output-dir /root/AVQA-R1-6K/AVQA_R1/valid/small_valid.jsonl \
  --md-dir /root/AVQA-R1-6K/AVQA_R1/valid
```

The conversion script reads the raw JSON, extracts problem, options, image, and audio fields, and produces a `.jsonl` file with `prompt`, `image`, `audio`, and `label` columns.

### Model Download

```bash
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir /root/Qwen3-Omni-30B-A3B-Instruct

# Qwen3-Omni ships its chat_template in a standalone chat_template.json that
# AutoTokenizer does not auto-load. Merge it into tokenizer_config.json (skipped if already present).
python -c "import json,sys; m=sys.argv[1]; p=f'{m}/tokenizer_config.json'; tc=json.load(open(p)); ('chat_template' in tc) or (tc.update(chat_template=json.load(open(f'{m}/chat_template.json'))['chat_template']) or json.dump(tc, open(p,'w'), indent=2, ensure_ascii=False))" /root/Qwen3-Omni-30B-A3B-Instruct
```

### Launch Training

Since the model and dataset are both downloaded to the `/root` directory, you only need to set `EXP_DIR=/root` and the script will automatically locate them — no manual script editing required.

```bash
cd /root/Relax
export EXP_DIR=/root

# Single node (requires 16 GPUs on a single machine)
bash scripts/training/multimodal/run-qwen3-30B-A3B-omni-16xgpu.sh

# Multi-node (recommended: 2 nodes × 8 GPUs)
bash -x scripts/entrypoint/spmd-multinode.sh scripts/training/multimodal/run-qwen3-30B-A3B-omni-16xgpu.sh
```

::: tip Reward Method
This task uses the built-in `multiple_choice` reward type (see `relax/engine/rewards/multiple_choice.py`). It extracts the answer from `<answer>...</answer>` tags and performs exact string matching against the ground truth, assigning **1.0** for correct and **0.0** for incorrect.
:::

---

## Task 4: NextQA (Video)

Train the Qwen3-Omni-30B-A3B model using the GRPO algorithm on the [TinyLLaVA-Video-R1-NextQA](https://huggingface.co/datasets/Zhang199/TinyLLaVA-Video-R1-training-data) video question-answering dataset with 16 GPUs (2 nodes).

### Data Preparation

Download the dataset and convert it to Relax format:

```bash
# Download the dataset
hf download --repo-type dataset Zhang199/TinyLLaVA-Video-R1-training-data \
  --local-dir /root/NextQA

# Unzip video files
unzip /root/NextQA/NextQA.zip -d /root/NextQA

# Convert to Relax format
python scripts/tools/process_nextqa.py \
  --input-dir /root/NextQA
```

The conversion script reads the original JSON file, extracts the question, options, and video fields, and generates a `.jsonl` file containing `prompt`, `video`, and `label` columns.

### Model Download

```bash
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir /root/Qwen3-Omni-30B-A3B-Instruct

# Qwen3-Omni ships its chat_template in a standalone chat_template.json that
# AutoTokenizer does not auto-load. Merge it into tokenizer_config.json (skipped if already present).
python -c "import json,sys; m=sys.argv[1]; p=f'{m}/tokenizer_config.json'; tc=json.load(open(p)); ('chat_template' in tc) or (tc.update(chat_template=json.load(open(f'{m}/chat_template.json'))['chat_template']) or json.dump(tc, open(p,'w'), indent=2, ensure_ascii=False))" /root/Qwen3-Omni-30B-A3B-Instruct
```

### Launch Training

Since both the model and dataset are downloaded to the `/root` directory, simply set `EXP_DIR=/root` and the script will automatically locate the corresponding paths without manually editing the script.

```bash
cd /root/Relax
export EXP_DIR=/root

# Single node (requires 16 GPUs on a single machine)
bash scripts/training/multimodal/run-qwen3-30B-A3B-omni-16xgpu-video.sh

# Multi-node (recommended: 2 nodes × 8 GPUs)
bash -x scripts/entrypoint/spmd-multinode.sh scripts/training/multimodal/run-qwen3-30B-A3B-omni-16xgpu-video.sh
```

::: tip Reward Method
This task uses the built-in `multiple_choice` reward type (see `relax/engine/rewards/multiple_choice.py`). It extracts the answer from `<answer>...</answer>` tags and performs exact string matching against the ground truth, assigning **1.0** for correct and **0.0** for incorrect.
:::

---

## Verifying Training Progress

After launching any of the above tasks, you should see logs like:

```text
Finish rollout 0/200
training step 0/200
```

This indicates training is running successfully.

## Export Model

Checkpoints saved by Relax are in Megatron DCP format. To convert them to Hugging Face weight format, use the [`convert_torch_dist_to_hf_bridge.py`](../../../scripts/tools/convert_torch_dist_to_hf_bridge.py) script:

```bash
python scripts/tools/convert_torch_dist_to_hf_bridge.py \
  --input-dir /path/to/dcp_checkpoint \
  --output-dir /path/to/hf_output \
  --origin-hf-dir /path/to/original_hf_model
```

The script automatically prepends the Relax repository root to `PYTHONPATH`. Megatron-LM and Megatron Bridge must still be available in the current environment.

To export directly to FP8 without first writing an intermediate BF16 HF checkpoint, enable streaming FP8 conversion:

```bash
python scripts/tools/convert_torch_dist_to_hf_bridge.py \
  --input-dir /path/to/dcp_checkpoint \
  --origin-hf-dir /path/to/original_hf_model \
  --output-dir /path/to/hf_output_fp8 \
  --fp8 \
  --fp8-strategy block \
  --fp8-block-size 128 128 \
  --fp8-device cuda \
  --fp8-max-shard-size-mb 4096
```

Parameter descriptions:

| Parameter | Description |
|---|---|
| `--input-dir` | Megatron DCP format checkpoint directory |
| `--output-dir` | Output directory for converted HF weights |
| `--origin-hf-dir` | Original HF safetensors directory, used for architecture, expected weight keys, and tokenizer files |
| `--force` | Optional, force overwrite if output directory already exists |

> **Note (MTP-free RL checkpoints):** When the source checkpoint has no MTP weights (e.g. an RL/SFT checkpoint trained without MTP) while the reference model's config enables MTP, the converter automatically reconciles the exported `model.safetensors.index.json` with the tensors actually written and supplements the missing MTP weights from `--origin-hf-dir`, so the exported model loads cleanly (including EAGLE speculative decoding). The same reconciliation is applied to online `--save-hf` exports.

For the FP8 strategy options, memory behavior, output format, and an 8-GPU SGLang launch command, see [Model Checkpoint Conversion](./model-conversion.md).

## Next Steps

- [Customize Training](./customize-training.md) — Learn how to customize training scripts, parameters, reward functions, and multi-node launch
- [Model Checkpoint Conversion](./model-conversion.md) — Export and serve trained checkpoints
- [Configuration](./configuration.md) — Full parameter reference
- [Architecture](./architecture.md) — Understand the system design
