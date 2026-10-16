# 快速上手

本指南提供四个端到端的训练示例，覆盖**纯文本**、**视觉-语言**、**全模态**和**视频**训练任务。每个示例包含数据准备、模型下载和训练启动命令。

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


### 概述

`openr1mm_accuracy_format` 将答案准确率奖励和格式奖励独立相加，两项各为 `[0, 1]`，总奖励为 `[0, 2]`。实现位于 `relax/engine/rewards/openr1mm.py`，保留实验所用的 XML／预填 think 评分行为；原有 `openr1mm` reward 及其原生通道支持保持不变。

准确率比较回答与参考答案的最终答案，不使用中间推理。显式选择题标签先校验选项，再用 `math_verify` 比较数学等价性。格式要求一个非空 `<think>...</think>` 后跟一个非空 `<answer>...</answer>`，支持 prompt 预填的 `<think>` 和可选的末尾 `<|im_end|>`。准确率和格式独立计分：格式正确但答案错误可得 1 分，答案正确但格式不符合要求也可得 1 分。这不是通用语义判分器，数学提取和文本归一化都有局限。

评测对 MathVista-testmini 和 MMMU validation 单独计算本地 0/1 accuracy，不调用外部模型、不加格式分、不经过训练的 dynamic sampling filter。[上游 Open-R1-Multimodal 评测](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal/blob/main/local_scripts/lmms_eval_qwen2vl.sh) 使用不同的 lmms-eval/GPT-4o 流程，并包含 MMMU-Pro，不能与这里的本地分数直接横比。

### 准备评测数据

在仓库根目录、已有 Relax 环境中执行，使用 `huggingface_hub`、`pyarrow` 和 Pillow。图片和转换后的数据应放在训练 worker 可访问的共享存储，`DATA_DIR` 指向数据集根目录。

如尚未转换 OpenR1MM 训练 parquet，先执行以下命令；重叠检查要求训练数据的 `image` 列为编码后图片字节的列表：

```bash
python scripts/tools/process_openr1.py \
  --input-dir "$DATA_DIR/multimodal-open-r1-8k-verified/data/train-00000-of-00001.parquet" \
  --output-dir "$DATA_DIR/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet"
```

仅下载所需评测 split，再针对实际使用的训练 parquet 转换并检查重叠：

```bash
python scripts/tools/process_openr1.py --mode download-eval --output-dir "$DATA_DIR/openr1mm-eval"
python scripts/tools/process_openr1.py --mode prepare-eval \
  --source "$DATA_DIR/openr1mm-eval/source" \
  --train "$DATA_DIR/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet" \
  --output-dir "$DATA_DIR/openr1mm-eval"
```

下载脚本默认固定 MathVista revision 为 `2b6ad69445fbb5695c9b165475e8decdbeb97747`，MMMU 为 `876ce5cb130f7f7e290ce4d9984357737d4db5cf`。需要时可用 `--mathvista-revision`／`--mmmu-revision` 覆盖；`sources.json` 记录解析后的 revision。下载脚本拒绝复用 revision 未知或不一致的非空源目录；切换 revision 时请使用新的 `--output-dir` 目录，同版本中断下载可以续传。转换产物为：

- `mathvista_testmini.parquet`、`mmmu_validation.parquet`：完整转换的 split。
- `mathvista_testmini_disjoint.parquet`、`mmmu_validation_disjoint.parquet`：排除与训练图片完全重叠的样本。
- `overlap-report.json`：完整／去重后数量和被排除的源样本 ID。

每行包含 `prompt`、`image`、JSON 编码的 `label` 和源 `id`。MMMU 题干和选项中的图片引用按出现顺序展开。重叠依据是解码后的 RGB 像素和**图片尺寸**完全相同，图片编码格式可以不同；任一引用图片重叠就排除整道题。这不排除缩放图、近似图或语义重叠问题。

一次针对 verified 8k 训练集的准备结果为 MathVista 保留 997/1000 题、MMMU 保留 900/900 题，具体数量取决于训练文件。processor/tokenizer 的 prompt 长度过滤还可能进一步减少题数（该次实验实际评测了 992 道 MathVista）。请检查实际评测数量，不要只看 parquet 行数。

### 训练与评测配置

在模型对应的训练配置中使用 `--rm-type openr1mm_accuracy_format`。添加评测时，将以下参数追加到训练命令，保留现有并行、优化器和数据集配置：

```bash
--rm-type openr1mm_accuracy_format \
--custom-rm-path relax.engine.rewards.openr1mm.get_openr1mm_benchmark_reward \
--eval-config "$DATA_DIR/openr1mm-eval/eval.yaml" \
--eval-interval 20 \
--eval-temperature 0 \
--n-samples-per-eval-prompt 1 \
--eval-max-prompt-len 8192 \
--eval-max-response-len 8192 \
--eval-max-context-len 16384 \
--sglang-context-length 16384
```

不设置 `--skip-eval-before-train`，即可在第一次更新前评测。YAML 选择两个去重后的 parquet，每题贪心生成一个回答。推理引擎上下文上限必须容纳评测输入输出；提高该上限不会提高训练 rollout 的长度限制。

转换脚本还会生成使用数据集绝对路径的 `eval.yaml`，通过 `--eval-config` 加载；所有 worker 必须能访问这些路径。保留模型的 chat template 和多模态映射（`--input-key prompt --label-key label --multimodal-keys '{"image":"image"}' --apply-chat-template`）。原有 `process_openr1.py --input-dir ... --output-dir ...` 调用保持兼容，默认模式为 `--mode train`。

custom reward 将 YAML 中带 `eval_benchmark` 标记的评测样本交给本地 accuracy；普通训练样本仍调用 `get_openr1mm_accuracy_format_reward`。不要向训练样本添加这个保留 metadata。训练仍可通过 `--dynamic-sampling-filter-path relax.engine.filters.dynamic_sampling_filters.check_reward_nonzero_std` 开启 DAPO 过滤，它与评测相互独立。

### 判分与故障排除

评测评分器先隔离最终答案，接受选项字母或完整选项文本，也接受 `B. 选项内容`，但**字母和选项内容必须同时匹配**。字母与内容冲突、多选项罗列都不应得分。这修复了模型从 `B` 改为输出 `B. $7` 时出现的虚假骤降。开放题使用归一化文本精确匹配、候选别名、按 MathVista 精度比较数值和列表比较；通常不做单位、自然语言或符号等价归一化。

分数下降时，应先比较相同的 prompt/label、实际题数、截断和最终答案形式，再判断模型是否退化。历史回答也应使用相同版本的评分器重算，只更新最新评测点会使曲线不可比。训练 reward 包含格式分，因此可能超过 1；这里的 eval 指标不会超过 1。

CPU 回归测试命令：

```bash
python -m pytest tests/engine/rewards/test_openr1mm.py tests/tools/test_process_openr1.py
```

---

## 任务 3：AVQA（全模态：图片 + 音频）

使用 GRPO 算法，在 [AVQA-R1-6K](https://huggingface.co/datasets/harryhsing/AVQA-R1-6K) 图文音频问答数据集上，以 16 张 GPU（2 节点）训练 Qwen3-Omni-30B-A3B 模型。

### 数据准备

下载数据集并转换为 Relax 格式：

```bash
# 下载数据集
hf download --repo-type dataset harryhsing/AVQA-R1-6K \
  --local-dir /root/AVQA

# 转换为 Relax 格式
# --md-dir 指向 image 和 audio 文件目录所在路径，
# 用于将相对路径拼接为绝对路径（可选，默认用相对路径）。
python scripts/tools/process_avqa.py \
  --input-dir /root/AVQA/AVQA_R1/train/omni_rl_format_train.json \
  --output-dir /root/AVQA/AVQA_R1/train/omni_rl_format_train_convert.jsonl \
  --md-dir /root/AVQA/AVQA_R1/train

python scripts/tools/process_avqa.py \
  --input-dir /root/AVQA/AVQA_R1/valid/omni_rl_format_valid.json \
  --output-dir /root/AVQA/AVQA_R1/valid/small_valid.jsonl \
  --md-dir /root/AVQA/AVQA_R1/valid
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
