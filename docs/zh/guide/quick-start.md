# 快速上手

本指南提供三个端到端的训练示例，分别覆盖**纯文本**、**视觉-语言**和**全模态**训练任务。每个示例包含数据准备、模型下载和训练启动命令。

开始之前，请确保您已完成[安装](./installation.md)步骤。

## 任务 1：DAPO Math（纯文本）

使用 GRPO 算法，在 [dapo-math-17k](https://huggingface.co/datasets/zhuzilin/dapo-math-17k) 数学推理数据集上，以 8 张 GPU 训练 Qwen3-4B 模型。

### 数据准备

下载训练数据集和（可选的）评估数据集：

```bash
# 下载训练数据集 (dapo-math-17k)
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# 下载评估数据集 (aime-2024)
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024

# 为 aime 评估数据集添加数学指令前缀（原地处理）
python scripts/tools/process_aime.py --input /root/aime-2024/aime-2024.jsonl
```

训练数据集为 `.jsonl` 格式，可直接使用，无需额外转换。评估数据集需通过上述脚本添加指令前缀，引导模型以 `\boxed{}` 格式输出答案。

### 模型下载

```bash
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
```

### 启动训练

由于模型和数据集均下载到 `/root` 目录下，只需统一设置 `EXP_DIR=/root`，脚本即可自动找到对应路径，无需手动编辑脚本。

::: tip 持久化存储
请确保 `/root` 目录已挂载到宿主机持久化存储，否则容器销毁后数据将丢失。参考[安装指南](./installation.md)中的 Docker 挂载说明。
:::

```bash
cd /root/Relax
export EXP_DIR=/root

# 单机
bash scripts/training/text/run-qwen3-4B-8xgpu.sh

# 多机
bash -x scripts/entrypoint/spmd-multinode.sh scripts/training/text/run-qwen3-4B-8xgpu.sh
```

::: tip Reward 方法
本任务使用内置的 `dapo` reward 类型（参见 `relax/engine/rewards/math_dapo_utils.py`）。它采用基于规则的答案提取和符号数学验证来评估正确性，正确答案得 **1.0** 分，错误答案得 **0.0** 分。
:::

---

## 任务 2：Open-R1（视觉-语言）

使用 GRPO 算法，在 [multimodal-open-r1-8k-verified](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) 图文数据集上，以 8 张 GPU 训练 Qwen3-VL-4B 模型。

### 数据准备

下载数据集并转换为 Relax 格式：

```bash
# 下载数据集
hf download --repo-type dataset lmms-lab/multimodal-open-r1-8k-verified \
  --local-dir /root/multimodal-open-r1-8k-verified

# 转换为 Relax 格式
python scripts/tools/process_openr1.py \
  --input-dir /root/multimodal-open-r1-8k-verified/data/train-00000-of-00001.parquet \
  --output-dir /root/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet
```

转换脚本读取原始 parquet 文件，提取 `problem`、`image` 和 `solution` 字段，生成包含 `prompt`、`image` 和 `label` 列的新 parquet 文件，即 Relax 所需的标准格式。

### 模型下载

```bash
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /root/Qwen3-VL-4B-Instruct
```

### 启动训练

由于模型和数据集均下载到 `/root` 目录下，只需统一设置 `EXP_DIR=/root`，脚本即可自动找到对应路径，无需手动编辑脚本。

```bash
cd /root/Relax
export EXP_DIR=/root

# 单机
bash scripts/training/multimodal/run-qwen3-vl-4B-8xgpu.sh

# 多机
bash -x scripts/entrypoint/spmd-multinode.sh scripts/training/multimodal/run-qwen3-vl-4B-8xgpu.sh
```

::: tip Reward 方法
本任务使用内置的 `openr1mm` reward 类型（参见 `relax/engine/rewards/openr1mm.py`）。它通过正则表达式从 `<answer>...</answer>` 标签中提取最终答案。系统会首先尝试通过 `math_verify` 进行符号验证，字符串匹配作为兜底方案。
:::

---

## 任务 3：AVQA（全模态：图片 + 音频）

使用 GRPO 算法，在 [AVQA-R1-6K](https://huggingface.co/datasets/harryhsing/AVQA-R1-6K) 图文音频问答数据集上，以 16 张 GPU（2 节点）训练 Qwen3-Omni-30B-A3B 模型。

### 数据准备

下载数据集并转换为 Relax 格式：

```bash
# 下载数据集
hf download --repo-type dataset harryhsing/AVQA-R1-6K \
  --local-dir /root/AVQA-R1-6K

# 转换为 Relax 格式
# --md-dir 指向 image 和 audio 文件目录所在路径，
# 用于将相对路径拼接为绝对路径（可选，默认用相对路径）。
python scripts/tools/process_avqa.py \
  --input-dir /root/AVQA-R1-6K/AVQA_R1/train/omni_rl_format_train.json \
  --output-dir /root/AVQA-R1-6K/AVQA_R1/train/omni_rl_format_train_convert.jsonl \
  --md-dir /root/AVQA-R1-6K/AVQA_R1/train

python scripts/tools/process_avqa.py \
  --input-dir /root/AVQA-R1-6K/AVQA_R1/valid/omni_rl_format_valid.json \
  --output-dir /root/AVQA-R1-6K/AVQA_R1/valid/small_valid.jsonl \
  --md-dir /root/AVQA-R1-6K/AVQA_R1/valid
```

转换脚本读取原始 JSON 文件，提取问题、选项、图片和音频字段，生成包含 `prompt`、`image`、`audio` 和 `label` 列的 `.jsonl` 文件。

### 模型下载

```bash
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir /root/Qwen3-Omni-30B-A3B-Instruct

# Qwen3-Omni 的 chat_template 单独存放在 chat_template.json 中，
# AutoTokenizer 不会自动加载，需要合并到 tokenizer_config.json（已存在则跳过）
python -c "import json,sys; m=sys.argv[1]; p=f'{m}/tokenizer_config.json'; tc=json.load(open(p)); ('chat_template' in tc) or (tc.update(chat_template=json.load(open(f'{m}/chat_template.json'))['chat_template']) or json.dump(tc, open(p,'w'), indent=2, ensure_ascii=False))" /root/Qwen3-Omni-30B-A3B-Instruct
```

### 启动训练

由于模型和数据集均下载到 `/root` 目录下，只需统一设置 `EXP_DIR=/root`，脚本即可自动找到对应路径，无需手动编辑脚本。

```bash
cd /root/Relax
export EXP_DIR=/root

# 单机（需要单台机器上有 16 张 GPU）
bash scripts/training/multimodal/run-qwen3-30B-A3B-omni-16xgpu.sh

# 多机（推荐：2 节点 × 8 GPU）
bash -x scripts/entrypoint/spmd-multinode.sh scripts/training/multimodal/run-qwen3-30B-A3B-omni-16xgpu.sh
```

::: tip Reward 方法
本任务使用内置的 `multiple_choice` reward 类型（参见 `relax/engine/rewards/multiple_choice.py`）。它从 `<answer>...</answer>` 标签中提取答案，与标准答案进行精确字符串匹配，正确得 **1.0** 分，错误得 **0.0** 分。
:::

---

## 任务 4：NextQA（video）

使用 GRPO 算法，在 [TinyLLaVA-Video-R1-NextQA](https://huggingface.co/datasets/Zhang199/TinyLLaVA-Video-R1-training-data) 视频问答数据集上，以 16 张 GPU（2 节点）训练 Qwen3-Omni-30B-A3B 模型。

### 数据准备

下载数据集并转换为 Relax 格式：

```bash
# 下载数据集
hf download --repo-type dataset Zhang199/TinyLLaVA-Video-R1-training-data \
  --local-dir /root/NextQA

# 解压视频文件
unzip /root/NextQA/NextQA.zip -d /root/NextQA

# 转换为 Relax 格式
python scripts/tools/process_nextqa.py \
  --input-dir /root/NextQA
```

转换脚本读取原始 JSON 文件，提取问题、选项、视频字段，生成包含 `prompt`、`video` 和 `label` 列的 `.jsonl` 文件。

### 模型下载

```bash
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir /root/Qwen3-Omni-30B-A3B-Instruct

# Qwen3-Omni 的 chat_template 单独存放在 chat_template.json 中，
# AutoTokenizer 不会自动加载，需要合并到 tokenizer_config.json（已存在则跳过）
python -c "import json,sys; m=sys.argv[1]; p=f'{m}/tokenizer_config.json'; tc=json.load(open(p)); ('chat_template' in tc) or (tc.update(chat_template=json.load(open(f'{m}/chat_template.json'))['chat_template']) or json.dump(tc, open(p,'w'), indent=2, ensure_ascii=False))" /root/Qwen3-Omni-30B-A3B-Instruct
```

### 启动训练

由于模型和数据集均下载到 `/root` 目录下，只需统一设置 `EXP_DIR=/root`，脚本即可自动找到对应路径，无需手动编辑脚本。

```bash
cd /root/Relax
export EXP_DIR=/root

# 单机（需要单台机器上有 16 张 GPU）
bash scripts/training/multimodal/run-qwen3-30B-A3B-omni-16xgpu-video.sh

# 多机（推荐：2 节点 × 8 GPU）
bash -x scripts/entrypoint/spmd-multinode.sh scripts/training/multimodal/run-qwen3-30B-A3B-omni-16xgpu-video.sh
```

::: tip Reward 方法
本任务使用内置的 `multiple_choice` reward 类型（参见 `relax/engine/rewards/multiple_choice.py`）。它从 `<answer>...</answer>` 标签中提取答案，与标准答案进行精确字符串匹配，正确得 **1.0** 分，错误得 **0.0** 分。
:::

---

## 验证训练进度

启动以上任意任务后，您应该会看到如下日志：

```text
Finish rollout 0/200
training step 0/200
```

这表明训练正在正常运行。

## 导出模型

Relax 保存的 checkpoint 为 Megatron DCP 格式，如需转换为 Hugging Face 权重格式，可使用 [`convert_torch_dist_to_hf_bridge.py`](../../../scripts/tools/convert_torch_dist_to_hf_bridge.py) 脚本：

```bash
python scripts/tools/convert_torch_dist_to_hf_bridge.py \
  --input-dir /path/to/dcp_checkpoint \
  --output-dir /path/to/hf_output \
  --origin-hf-dir /path/to/original_hf_model
```

脚本会自动把 Relax 仓库根目录添加到 `PYTHONPATH`；当前环境仍需能够导入 Megatron-LM 和 Megatron Bridge。

如果希望在导出过程中直接转成 FP8，而不先写一份中间 BF16 HF checkpoint，可开启流式 FP8 转换：

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

参数说明：

| 参数 | 说明 |
|---|---|
| `--input-dir` | Megatron DCP 格式的 checkpoint 目录 |
| `--output-dir` | 转换后 HF 权重的输出目录 |
| `--origin-hf-dir` | 原始 HF safetensors 目录，用于读取模型结构、预期权重 key 和 tokenizer 文件 |
| `--force` | 可选，若输出目录已存在则强制覆盖 |

> **注意（无 MTP 的 RL checkpoint）**：当源 checkpoint 不含 MTP 权重（如未训练 MTP 的 RL/SFT checkpoint），而参考模型 config 启用了 MTP 时，转换器会自动把导出的 `model.safetensors.index.json` 与实际写盘的 tensor 对齐，并从 `--origin-hf-dir` 补齐缺失的 MTP 权重，保证导出的模型可正常加载（含 EAGLE 投机解码）。在线 `--save-hf` 导出同样适用。

FP8 策略参数、显存行为、输出格式和 SGLang 8 卡 TP8 启动命令见[模型 Checkpoint 转换](./model-conversion.md)。

## 下一步

- [自定义训练](./customize-training.md) — 了解如何自定义训练脚本、参数、Reward 函数以及多机启动
- [模型 Checkpoint 转换](./model-conversion.md) — 导出并启动训练后的 checkpoint
- [配置说明](./configuration.md) — 完整参数参考
- [架构设计](./architecture.md) — 理解系统设计
