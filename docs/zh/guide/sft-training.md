# SFT 训练

本指南展示 Relax 中监督微调（SFT）的完整流程，覆盖 [`scripts/training/sft/`](../../../scripts/training/sft/) 下的生成式 SFT、[`examples/seq_cls_sft/`](../../../examples/seq_cls_sft/) 下的原生序列分类 SFT、模型与数据准备、启动命令以及常用调参方法。

开始之前，请先完成[安装](./installation.md)。

## 概述

SFT 通过 `--loss-type sft` 启用。该模式下，Relax 会启动一个 SFT producer，从 `--prompt-data` 读取数据，用模型的 chat template 渲染样本，将 packed 样本写入 TransferQueue，然后由 Megatron Actor 训练。默认的 `--task-type causal_lm` 训练语言模型头；设置 `--eval-interval` 时运行 PPL 评估，并可通过 `--sft-predict-interval` 使用 Rollout 角色周期性做生成式 predict。原生分类使用 `--task-type seq_cls`，将词表输出头替换为分类头，并在不启动 Rollout 的情况下计算分类 loss 和指标。

当前启动脚本使用 `ray job submit`，并且在没有外部 entrypoint 预先准备 Ray 环境时自动 source [`scripts/entrypoint/local.sh`](../../../scripts/entrypoint/local.sh)。

## 脚本

| 脚本 | 数据集 | 模型 | 默认资源 | 说明 |
| --- | --- | --- | --- | --- |
| [`run-qwen3.5-9B-math-8xgpu.sh`](../../../scripts/training/sft/run-qwen3.5-9B-math-8xgpu.sh) | `OpenMathReasoning-mini` | `Qwen3.5-9B` | 8 GPU Actor，加 SFT producer 和 Rollout | 文本 SFT，使用 `problem` -> `generated_solution`，开启 PPL eval 和 predict。 |
| [`run-qwen3-vl-4B-math-8xgpu.sh`](../../../scripts/training/sft/run-qwen3-vl-4B-math-8xgpu.sh) | `OpenMathReasoning-mini` | `Qwen3-VL-4B-Instruct` | 8 GPU Actor，加 SFT producer 和 Rollout | 使用 VL checkpoint 训练纯文本 math 数据。 |
| [`run-qwen3-vl-4B-pokemon-8xgpu.sh`](../../../scripts/training/sft/run-qwen3-vl-4B-pokemon-8xgpu.sh) | `pokemon-gpt4o-captions` | `Qwen3-VL-4B-Instruct` | 8 GPU Actor，加 SFT producer 和 Rollout | 图像多模态 SFT，使用两个 parquet 文件，并开启预取。 |
| [`run-qwen3-vl-4B-pokemon-1xgpu.sh`](../../../scripts/training/sft/run-qwen3-vl-4B-pokemon-1xgpu.sh) | `pokemon-gpt4o-captions` | `Qwen3-VL-4B-Instruct` | 1 GPU Actor，加 SFT producer | 低资源 Pokemon SFT，开启 CPU optimizer offload。 |
| [`run-qwen3.5-35B-A3B-mtp-sft-16xgpu.sh`](../../../scripts/training/sft/run-qwen3.5-35B-A3B-mtp-sft-16xgpu.sh) | `OpenMathReasoning-mini` | `Qwen3.5-35B-A3B` | 16 GPU Actor，加 SFT producer | 进阶 MTP SFT。多个参数已暴露为环境变量。 |
| [`run-qwen3.5-9B-classification-sft-8xgpu.sh`](../../../examples/seq_cls_sft/run-qwen3.5-9B-classification-sft-8xgpu.sh) | SST-2、AG News 或 GoEmotions | `Qwen3.5-9B` | 8 GPU Actor，加 SFT producer | 全参数二分类、多分类或多标签序列分类 SFT。 |
| [`run-qwen3.5-35B-A3B-classification-lora-sft-8xgpu.sh`](../../../examples/seq_cls_sft/run-qwen3.5-35B-A3B-classification-lora-sft-8xgpu.sh) | SST-2、AG News 或 GoEmotions | `Qwen3.5-35B-A3B` | 8 GPU Actor，加 SFT producer | LoRA 序列分类 SFT，默认使用 GoEmotions。 |

## 数据准备

SFT 脚本默认使用 `DATA_DIR=${SCRIPT_DIR}/data`，除非显式覆盖 `DATA_DIR`。对于可复用训练任务，建议把数据放在持久化存储中，并显式设置 `DATA_DIR`。

```bash
cd /root/Relax
export DATA_DIR=/root
mkdir -p "${DATA_DIR}/sft/data"
```

### Math: OpenMathReasoning-mini

math 脚本期望以下文件：

```text
${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet
```

从 [unsloth/OpenMathReasoning-mini](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini) 下载：

```bash
hf download --repo-type dataset unsloth/OpenMathReasoning-mini \
  data/cot-00000-of-00001.parquet \
  --local-dir "${DATA_DIR}/sft/data/OpenMathReasoning-mini"
```

当前 math 脚本配置为：

```bash
--prompt-data "${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet"
--input-key problem
--label-key generated_solution
```

设置 `--label-key` 后，每行数据必须在 `problem` 字段中提供 prompt 字符串，并在 `generated_solution` 字段中提供目标回复。Relax 会用 `problem` 构造 user message，并把 `generated_solution` 追加为 assistant message。

最小行结构：

```json
{
  "problem": "Find the value of x.",
  "generated_solution": "We solve the equation step by step..."
}
```

### Pokemon: pokemon-gpt4o-captions

Pokemon 脚本期望以下文件：

```text
${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_en.parquet
${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_zh.parquet
```

从 [llamafactory/pokemon-gpt4o-captions](https://huggingface.co/datasets/llamafactory/pokemon-gpt4o-captions) 下载：

```bash
hf download --repo-type dataset llamafactory/pokemon-gpt4o-captions \
  pokemon_gpt4o_en.parquet pokemon_gpt4o_zh.parquet \
  --local-dir "${DATA_DIR}/sft/data/pokemon-gpt4o-captions"
```

当前 Pokemon 脚本会从两个文件构造 list 形式的 `PROMPT_DATA`：

```bash
TRAIN_FILES=(
    "'${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_en.parquet'"
    "'${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_zh.parquet'"
)
PROMPT_DATA="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"
```

然后配置：

```bash
--prompt-data "${PROMPT_DATA}"
--input-key conversations
--multimodal-keys '{"image":"images"}'
--conversation-key-map '{"from":"role","value":"content","human":"user","gpt":"assistant"}'
```

由于没有设置 `--label-key`，`conversations` 必须是一列完整的消息列表。`--conversation-key-map` 会把 ShareGPT 风格的消息字段和值改写为 SFT 期望的 OpenAI 风格 `{role, content}`。`--multimodal-keys` 告诉 Relax 从每行的 `images` 字段加载图片路径。文本中每个图片条目都应该有对应的 `<image>` 占位符。

最小行结构：

```json
{
  "conversations": [
    {"from": "human", "value": "Identify the object of this image.<image>"},
    {"from": "gpt", "value": "A round, pink Pokemon with a gentle expression."}
  ],
  "images": ["/path/to/pokemon.png"]
}
```

### 验证数据文件

启动前运行下面的检查，提前发现缺文件或列名不匹配：

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

如果你使用的不是 `/root` 作为 `DATA_DIR`，请同步修改检查脚本中的路径。

## 模型准备

将模型权重下载到持久化存储。math 和 Pokemon 脚本从 `MODEL_DIR` 读取模型路径，而不是从 `EXP_DIR` 读取，因此这几份脚本建议显式设置 `MODEL_DIR`。

```bash
# Math 脚本：run-qwen3.5-9B-math-8xgpu.sh
hf download Qwen/Qwen3.5-9B --local-dir /root/Qwen3.5-9B

# Pokemon 脚本：run-qwen3-vl-4B-pokemon-*.sh
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir /root/Qwen3-VL-4B-Instruct
```

对于 MTP 脚本，默认约定是 `EXP_DIR=/root`，然后 `MODEL_DIR=${EXP_DIR}`、`DATA_DIR=${EXP_DIR}`，除非你单独覆盖它们。

## 序列分类 SFT

序列分类 SFT 训练原生的 `K`-logit 分类头，而不是生成标签文本。示例支持三种任务形式：

| 任务 | 输入标签 | 输出与 loss |
| --- | --- | --- |
| 单标签二分类 | 整数 `0` 或 `1` | 2 logits，使用 CrossEntropy |
| 单标签多分类 | `[0, K)` 范围内的整数 | `K` logits，使用 CrossEntropy |
| 多标签分类 | 类别索引列表，例如 `[0, 2, 5]` | `K` logits，使用 BCEWithLogits |

分类阈值只用于把多标签概率转换成离散的 eval 或推理结果，不参与训练或 eval loss 的计算。

### 1. 配置路径

从 Relax 仓库根目录运行以下命令。脚本期望 Qwen3.5 模型目录直接位于 `MODEL_DIR` 下，并将准备好的数据写入 `${DATA_DIR}/sft/seq_cls`。

```bash
export DATA_DIR=/path/to/relax-workspace
export EXP_DIR=/path/to/relax-workspace/exp
export MODEL_DIR=/path/to/relax-workspace/models
```

9B 脚本使用 `${MODEL_DIR}/Qwen3.5-9B` 下的原始 Hugging Face 模型；35B-A3B LoRA 脚本使用 `${MODEL_DIR}/Qwen3.5-35B-A3B`。

### 2. 准备数据集

数据准备工具会将固定 revision 的 SST-2、AG News 和 GoEmotions simplified 转换成训练脚本使用的 JSONL 格式：

```bash
python examples/seq_cls_sft/tools/prepare_classification_sft_data.py \
  --dataset all \
  --subset full \
  --global-batch-size 64 \
  --output-dir "${DATA_DIR}/sft/seq_cls"
```

使用 `--dataset sst2`、`--dataset ag_news` 或 `--dataset go_emotions` 可以只准备一个任务。启动脚本默认读取 `full` 子集；`DATA_SUBSET=smoke` 和 `DATA_SUBSET=extended` 会选择同一工具生成的较小确定性子集。

自定义单标签数据的 `label` 是整数：

```json
{"messages": [{"role": "user", "content": "This movie is excellent."}], "label": 1}
```

自定义多标签数据的 `label` 是类别索引列表：

```json
{"messages": [{"role": "user", "content": "I feel relieved and excited."}], "label": [4, 17]}
```

设置 `TRAIN_DATA` 和 `EVAL_DATA` 可使用自定义 JSONL 文件。自定义标签空间时，还需设置 `NUM_LABELS`，并将 `PROBLEM_TYPE` 设为 `single_label_classification` 或 `multi_label_classification`。

### 3. 启动训练

Qwen3.5-9B 全参数脚本默认使用 SST-2。通过 `CLASSIFICATION_DATASET` 可切换到 AG News 或 GoEmotions：

```bash
bash examples/seq_cls_sft/run-qwen3.5-9B-classification-sft-8xgpu.sh

CLASSIFICATION_DATASET=ag_news \
  bash examples/seq_cls_sft/run-qwen3.5-9B-classification-sft-8xgpu.sh

CLASSIFICATION_DATASET=go_emotions \
  bash examples/seq_cls_sft/run-qwen3.5-9B-classification-sft-8xgpu.sh
```

Qwen3.5-35B-A3B LoRA 脚本默认使用 GoEmotions：

```bash
bash examples/seq_cls_sft/run-qwen3.5-35B-A3B-classification-lora-sft-8xgpu.sh
```

两个脚本均启用 dynamic batching，使用 8 张训练 GPU，默认训练 10 个 epoch，并每 100 step 保存一次 Megatron native checkpoint。单标签 eval 输出 loss 和 accuracy；多标签 eval 输出 loss、micro precision、micro recall、micro F1 和 subset accuracy。

::: warning 当前范围
序列分类 SFT 当前只支持纯文本同步 SFT，不支持 fully async、hybrid、SFT 生成式 predict、MTP、SFT chunked logits 或在线 `--save-hf`。部署前需要离线导出 native checkpoint。提供的 SGLang adapter 仅支持 tensor parallel，不支持 pipeline parallel。
:::

::: tip 完整导出与部署流程
完整的环境变量覆盖、Megatron 到 Hugging Face 离线导出、SGLang 启动以及单标签和多标签请求示例，请参阅 [`examples/seq_cls_sft` README](../../../examples/seq_cls_sft/README.md)。
:::

## 配置详解

启动脚本按参数块组织。调参时可以直接修改选中的脚本；如果脚本已经暴露了环境变量，也可以通过环境变量覆盖。

### 路径与 checkpoint

对于 math 和 Pokemon 脚本：

```bash
export MODEL_DIR=/root
export DATA_DIR=/root
```

`CKPT_ARGS` 参数块设置：

| 参数 | 作用 |
| --- | --- |
| `--hf-checkpoint` | HF checkpoint，用于 tokenizer、config，以及开启 predict 时初始化 SGLang。 |
| `--ref-load` | 初始参考 checkpoint 路径。 |
| `--load` | 训练加载路径。若这是已有 Megatron checkpoint，则从它恢复训练；否则 bridge 模式从 HF 权重开始。 |
| `--megatron-to-hf-mode bridge` | 使用 Megatron Bridge 完成 HF <-> Megatron 权重转换。 |
| `--save` | Megatron checkpoint 输出目录。 |
| `--save-interval` | 每 N 个训练 step 保存一次。 |
| `--num-epoch` | 数据集 epoch 数。Relax 会根据数据集大小和 global batch size 推导实际训练步数。 |

新训练时，可以保持当前脚本里 `--load` 和 `--save` 指向同一个实验目录。断点续训时，保持同一个 `--save`，并确保目录中存在有效 checkpoint 和 `latest_checkpointed_iteration.txt`。

### MTP 参数

MTP SFT 脚本开启 `--mtp-num-layers ${MTP_NUM_LAYERS:-1}`、`--enable-mtp-training` 和 `--mtp-loss-scaling-factor ${MTP_LOSS_SCALING_FACTOR:-0.2}`。只有模型和 checkpoint 确实包含对应 MTP 层时才调大 `MTP_NUM_LAYERS`；`MTP_LOSS_SCALING_FACTOR` 是辅助 loss 权重，建议从 `0.2` 起调。

联合训练、MTP-only 两阶段训练及 Pokémon 实测结果见 [MTP 训练](./mtp-rl-training.md)。

### SFT 数据参数

SFT 必需参数：

```bash
--loss-type sft
--prompt-data "${PROMPT_DATA}"
--use-dynamic-batch-size
--max-tokens-per-gpu <tokens>
```

SFT 必须使用动态 batching。缺少 `--use-dynamic-batch-size` 时，参数校验会失败。如果没有设置 `--balance-data`，SFT 校验会自动开启它，以保持 SFT producer 和 Megatron 数据路径一致。

数据行格式二选一：

| 格式 | 参数 | 适用场景 |
| --- | --- | --- |
| Prompt 加 label | `--input-key problem --label-key generated_solution` | prompt 和监督答案分别在两列的文本数据。 |
| 完整 messages | `--input-key conversations`，且不设置 `--label-key` | 已包含 user 和 assistant 轮次的对话或多模态数据。ShareGPT 风格字段需加 `--conversation-key-map`。 |

SFT 不需要额外添加 `--apply-chat-template`。SFT 数据集会在内部使用 tokenizer chat template 渲染样本，并从渲染结果构造 assistant loss mask。

### Chat Template 兼容补丁

#### 分发与作用域

渲染 SFT 样本前，Relax 会从显式 `chat_template` override 或 `tokenizer.chat_template` 中确定 effective template，再执行静态的模型专用 patch 注册表。全局 `--apply-chat-template-kwargs` 会与 `sample.metadata["apply_chat_template_kwargs"]` 合并，样本级配置优先。tokenize 的 `render_with_loss_mask` 路径和多模态使用的 `render_to_text` 路径共用同一份结果。

patch 按样本、按调用生效，绝不会修改 `tokenizer.chat_template`。如果没有 patcher 识别 effective template，dispatcher 会严格 no-op：原模板和用户显式提供的 kwargs 均原样透传。模型识别依据是模板内容，而不是 checkpoint 名称。

通用 dispatcher 实现在 [`chat_template_patch.py`](../../../relax/engine/sft/dataset/chat_template_patch.py)，模型专用行为则放在独立模块中，例如 [`qwen_chat_template_patch.py`](../../../relax/engine/sft/dataset/qwen_chat_template_patch.py)。

::: tip 作用域
这些 patch 只影响 SFT 样本渲染，不会修改 Rollout、Agentic Session、SGLang 或其他推理侧的 chat template 行为。
:::

#### Qwen 历史 Thinking

部分 Qwen 模板只保留最后一个普通 user query 之后的 assistant reasoning。在多轮工具数据中，这可能删除更早的监督 assistant 中的 `<think>...</think>`，但仍留下它的 tool call。Relax 会识别 Qwen3.5 的精确旧 history gate，并在当前渲染调用中回填兼容的 `preserve_thinking` gate；已经包含原生 gate 的模板保持不变。

`preserve_thinking` 支持三种状态：

| 配置 | 行为 |
| --- | --- |
| 未配置或 `null` | 默认的按样本自动模式。最后一个普通 user 之前存在 `learn=true` 且文本含 `</think>` 的 assistant 时，Relax 将其设为 `true`；否则保持 Qwen 原生行为。 |
| `true` | 在整个样本渲染中强制保留历史 thinking。哪些渲染 token 计算 loss 仍由 `learn` 控制。 |
| `false` | 关闭自动保留，使用 Qwen 原生的历史 reasoning 压缩。answer、tool call 等非 reasoning 内容仍会正常渲染。 |

完整 messages 格式的 SFT 数据如果没有显式提供 `learn`，assistant message 默认 `learn=true`。如果一个 user message 的完整字符串内容被 `<tool_response>...</tool_response>` 包裹，它不会被视为新的普通 user 边界。自动模式只要被一个可学习的历史 assistant 触发，保留策略就会作用于整个渲染；`learn=false` 的 message 可以保留为上下文，但仍不计算 loss。

由于样本级 kwargs 优先，样本中的 `preserve_thinking: null` 会覆盖全局 `false`，使该样本重新进入自动模式。

请使用 JSON 布尔值，不要使用字符串：

```bash
# 强制保留
--apply-chat-template-kwargs '{"preserve_thinking": true}'

# 关闭自动保留
--apply-chat-template-kwargs '{"preserve_thinking": false}'

# 显式恢复按样本自动模式
--apply-chat-template-kwargs '{"preserve_thinking": null}'
```

::: warning `false` 不会关闭所有 Thinking
`preserve_thinking=false` 只会恢复 Qwen 对最后一个普通 user 之前 assistant 轮次的原生压缩。当前 tool episode 中的 assistant 仍遵循 Qwen 原生模板，可能包含空的 `<think>\n\n</think>` 外壳。这个配置与推理侧的 `enable_thinking` 无关。

历史 reasoning 被压缩后，这些 reasoning token 不会出现在渲染后的训练序列中，因此无法计算 loss；关联的 answer 或 tool call 仍可正常训练。
:::

除了 JSON 布尔值和 `null` 之外的值（例如 `"false"`）会触发 `ValueError`。如果疑似 Qwen 的模板包含重复、冲突或无法识别的 history gate，Relax 会触发 `RuntimeError`，而不是猜测如何 patch。可以通过 SFT 日志中的 `source=qwen_history_thinking (patched|native)` 和最终 `preserve_thinking` 值确认实际分支。

### 评估与 predict

PPL 评估由以下参数控制：

```bash
--eval-size 0.01
--eval-interval 10
```

`--eval-size` 会使用 `--seed` 对 `--prompt-data` 的行做一次随机切分，固定 eval 子集，并将其从每个 shuffle 后的训练 epoch 中移除。小于 1 的值表示比例；大于等于 1 的值表示绝对样本数。这是 row-level 切分；同一图片的多语言 caption 等关联行不会自动分组。也可以使用 `--eval-prompt-data name path` 提供独立评估集，但 SFT 模式下不要使用 `--eval-config`。

生成式 predict 由以下参数控制：

```bash
--sft-predict-interval 10
--eval-temperature 0.0
--eval-max-response-len 10240
```

设置 `--sft-predict-interval` 后，Relax 会自动拉起 Rollout 角色，并把预测结果写到：

```text
<save>/predict/predictions_step_<rollout_id>.jsonl
```

math 推理任务通常需要更长的 `--eval-max-response-len`；caption 或图像描述任务可以设短一些。

### 并行与资源

脚本中的 `PERF_ARGS` 和 `--resource` 需要与集群 GPU 数一致。

Math 8 GPU 脚本：

```bash
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 2
--context-parallel-size 1
--resource '{"sft": [1, 0], "actor": [1, 8], "rollout": [1, 8]}'
```

Pokemon 8 GPU 脚本：

```bash
--tensor-model-parallel-size 2
--pipeline-model-parallel-size 1
--context-parallel-size 1
--per-rank-fetch
--num-data-storage-units 8
--resource '{"sft": [1, 0], "actor": [1, 8], "rollout": [1, 8]}'
```

Pokemon 1 GPU 脚本：

```bash
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
--resource '{"sft": [1, 0], "actor": [1, 1], "rollout": [1, 1]}'
```

`"sft": [1, 0]` 表示 SFT producer 是 CPU-only。Actor 使用训练 GPU。开启周期性 predict 时，Rollout 也需要 GPU 资源。

### SFT 分片 Producer

`RELAX_SFT_TQ_SHARDS` 是 SFT async prepack 的实验性吞吐调优开关，用于把每个全局 SFT batch 拆到多个 TransferQueue 分区。推荐在 Ray 运行时环境中配置：

```yaml
# configs/env.yaml
env_vars:
  RELAX_SFT_TQ_SHARDS: "2"
```

同时在训练参数中开启 async prepack：

```bash
--per-rank-fetch
--sft-async-prepack
--sft-max-in-flight-steps 4
```

`--sft-max-in-flight-steps 4` 只是示例；async prepack 要求该值至少为 2，或者等价地设置 `--max-staleness >= 1`。这个环境变量本身不会开启 prepack；未使用 `--loss-type sft --sft-async-prepack` 时会被忽略。

使用 `N` 个 shard 时，`--global-batch-size` 和每个 DP rank 的本地 batch（`global_batch_size / data_parallel_size`）都必须能被 `N` 整除。例如 global batch size 为 32、DP size 为 8 时，本地 batch 为 4，因此可以设置 2 或 4 个 shard，不能设置 3 个。

满足 remote 路径条件时，`N` 个 shard 会为 train batch 启动 `N` 个 Ray producer actor。Eval 可以同时开启：train shard 由远端 producer 生产，coordinator 负责渲染 eval split 或 eval prompt data，并在 eval interval 推送原有的 `sft_eval_<step>_n<N>_<i>` 分区。部分配置（包括序列分类、自定义数据集和允许跳过样本的过滤策略）会使用单个本地 producer，但仍然写入 `N` 个分区。因此，`RELAX_SFT_TQ_SHARDS=2` 并不一定表示有两个 producer actor。可以通过下面的日志确认 remote 路径是否真正启用：

```text
SFT remote shard producer enabled: ... shards=2 ...
```

使用 `--eval-size` 时，每个远端 train producer 都会应用同一份基于 `--seed` 的 holdout split，保证 train 和 eval 行不重叠。直接启动 Python 时也可以使用 `export RELAX_SFT_TQ_SHARDS=2`。通过 Ray Job 启动时，将它写入 `configs/env.yaml` 可以确保 producer 和 consumer 收到相同的值。分区命名、fallback 条件和 worker 分配方式见 [TransferQueue 分片 Producer](./configuration.md#transferqueue-分片-producer)。

## 启动

### 单机

直接运行脚本即可。脚本会 source `local.sh`、启动本地 Ray head node，并提交 Ray job：

```bash
cd /root/Relax
export MODEL_DIR=/root
export DATA_DIR=/root

bash scripts/training/sft/run-qwen3.5-9B-math-8xgpu.sh
bash scripts/training/sft/run-qwen3-vl-4B-pokemon-8xgpu.sh
```

运行 1 GPU Pokemon 脚本：

```bash
cd /root/Relax
export MODEL_DIR=/root
export DATA_DIR=/root
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1

bash scripts/training/sft/run-qwen3-vl-4B-pokemon-1xgpu.sh
```

### 已有 Ray 集群

如果 Ray 集群已经启动，使用 [`scripts/entrypoint/ray-job.sh`](../../../scripts/entrypoint/ray-job.sh)。它会准备运行环境，不会停止 Ray，然后委托给训练脚本：

```bash
cd /root/Relax
export MODEL_DIR=/root
export DATA_DIR=/root
export RAY_ADDRESS=http://127.0.0.1:8265

bash scripts/entrypoint/ray-job.sh scripts/training/sft/run-qwen3-vl-4B-pokemon-8xgpu.sh
```

### 多机

使用 [`scripts/entrypoint/spmd-multinode.sh`](../../../scripts/entrypoint/spmd-multinode.sh)。必需环境变量为 `MASTER_ADDR`、`POD_NAME`、`HOST_IP`、`WORLD_SIZE`；`NUM_GPUS` 默认每节点 8 张。

```bash
cd /root/Relax
export MODEL_DIR=/root
export DATA_DIR=/root
export WORLD_SIZE=2
export NUM_GPUS=8

bash scripts/entrypoint/spmd-multinode.sh \
  scripts/training/sft/run-qwen3.5-35B-A3B-mtp-sft-16xgpu.sh
```

## 调参流程

建议按下面顺序调参，让每次改动都有明确目标。

### 1. 先让显存放得下

如果训练阶段 OOM：

| 参数 | 调整方向 | 作用 |
| --- | --- | --- |
| `--max-tokens-per-gpu` | 调小 | 降低动态 micro-batch 的 token 容量。这是 SFT 首选显存参数。 |
| `--global-batch-size` | 调小 | 降低每次 optimizer update 的样本数，但会改变优化动态。 |
| `--recompute-num-layers` | 调大 | 用更多计算换 activation 显存。 |
| `--optimizer-cpu-offload` | 开启 | 节省 GPU 显存，适合 1 GPU 或显存紧张的 VL 训练。 |
| `--sft-oversize-strategy skip` | 谨慎开启 | 丢弃超过 `max_tokens_per_gpu * context_parallel_size` 的样本。 |

使用 context parallelism 时，容量为 `--max-tokens-per-gpu * --context-parallel-size`。只有模型和 Megatron 配置支持时才建议增加 CP。

### 2. 提升吞吐

如果 GPU 在等 SFT 数据：

| 参数 | 调整方向 | 作用 |
| --- | --- | --- |
| `--sft-prefetch-buffer-size` | 从 256 往上调 | 缓存更多已渲染样本。 |
| `--sft-prefetch-num-workers` | 调大 | 提高图片解码和多模态 I/O 并行度。 |
| `--sft-prefetch-chunk-size` | 调大 | 一次派发更多预取样本，但会增加内存压力。 |
| `--per-rank-fetch` | 多 GPU 时开启 | 让 TP/PP rank 直接从 TransferQueue 拉数据，需配足 `--num-data-storage-units`。 |
| `--max-staleness` | I/O 重时调大 | 允许 producer 提前生产。Pokemon 8 GPU 脚本使用 `--max-staleness 4`。 |
| `RELAX_SFT_TQ_SHARDS` | 从 2 开始 | 并行化满足条件的 async-prepack producer。仅当数据准备是瓶颈且 batch 满足整除约束时再增加。 |

纯文本 math 任务通常更受序列长度和模型并行影响；Pokemon 任务更容易被图片读取和 processor 工作拖慢。

### 3. 保持效果稳定

先使用脚本默认值，再一次只改一个优化参数：

| 参数 | Math 默认值 | Pokemon 8 GPU 默认值 | 说明 |
| --- | --- | --- | --- |
| `--lr` | `1e-5` | `1e-5` | eval loss 快速上升或从强 checkpoint 恢复时可调低。 |
| `--lr-decay-style` | `cosine` | `cosine` | 1 GPU Pokemon 脚本使用 `constant` 和 `3e-5`，更激进。 |
| `--weight-decay` | `0.1` | `0.1` | 没有系统 sweep 时建议保持不变。 |
| `--clip-grad` | `1.0` | `1.0` | 只有梯度尖峰明显时再调低。 |
| `--num-epoch` | `10` | `10` | smoke test 可调小；想调大必须配合 eval 监控。 |

### 4. 控制评估成本

| 参数 | 调整方向 | 作用 |
| --- | --- | --- |
| `--eval-size` | 调小 | 减少 holdout 数据和 PPL eval 成本。 |
| `--eval-interval` | 调大 | 减少评估轮次。 |
| `--sft-predict-interval` | 调大或关闭 | 减少 SGLang predict 开销。 |
| `--eval-max-response-len` | 调小 | 限制 predict 生成成本。 |

快速 bring-up 时，可以先注释掉 `PREDICT_ARGS`，并使用较小 `--eval-size`。训练循环稳定后再恢复 predict。

## 故障排除

### `--loss-type sft requires --use-dynamic-batch-size`

SFT 会拒绝静态 micro-batch。添加 `--use-dynamic-batch-size`，并设置 `--max-tokens-per-gpu`。

### `Under --loss-type sft with --eval-interval set...`

`--eval-interval` 需要且只能有一个 eval 来源：`--eval-size` 或 `--eval-prompt-data name path`。不要同时设置两者。

### `SFT row missing prompt key`

`--input-key` 和数据列名不匹配。math 使用 `problem`；Pokemon 使用 `conversations`。

### `--label-key is not set ... expects ... messages list`

你传入的是字符串 prompt，但没有设置 `--label-key`。要么为 prompt-plus-label 数据添加 `--label-key`，要么把行数据转换成完整 message list。

### 多模态占位符不匹配

图像 SFT 中，`--multimodal-keys '{"image":"images"}'` 引用的每张图片都必须在 conversation content 中有对应 `<image>` 占位符。图片数量多于或少于占位符都会导致数据处理错误。

### Predict 没有写文件

检查是否设置了 `--sft-predict-interval`、`--save`，并且至少存在一个 eval 来源。预测文件会写到 `<save>/predict/`。

## 扩展 Chat Template 补丁

如需兼容其他模型族，请新增独立的纯函数 patcher。不要把模型专用知识堆回 [`chat_template.py`](../../../relax/engine/sft/dataset/chat_template.py)；新模块应与现有的 [`qwen_chat_template_patch.py`](../../../relax/engine/sft/dataset/qwen_chat_template_patch.py) 放在同一目录。

### Patcher 约定

使用 [`chat_template_patch.py`](../../../relax/engine/sft/dataset/chat_template_patch.py) 中的 `TemplatePatcher` 签名：

下面的私有函数名 `_matches_exact_foo_template` 和 `_patch_exact_foo_template` 只是占位示例。请在新模块中自行实现对应模型的识别逻辑和精确替换逻辑。

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

    # 匹配精确且属于该模型的 legacy 或 native 模板片段。
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

请遵循以下规则：

1. 所有无关模板必须返回 `None`。使用足够窄且稳定的模板特征，不要依赖模型路径或 tokenizer 类名。
2. 同时识别需要 patch 的 legacy 模板和已经修复的 native 模板，保证重复渲染幂等。
3. 如果模板疑似属于受支持的模型族，但关键片段已经漂移，应给出清晰错误并 fail-fast，不要猜测替换方式。
4. 修改前复制 `kwargs`。不要修改 sample、tokenizer、输入 mapping 或任何全局状态。
5. 返回 effective template 和 kwargs。模板变化时，dispatcher 会把它作为当前调用的 `chat_template` kwarg 注入，并维持 template/kwargs 一致。

### 注册

在 [`chat_template.py`](../../../relax/engine/sft/dataset/chat_template.py) 中导入 patcher，并追加到静态注册表：

```python
_CHAT_TEMPLATE_PATCHERS = (
    try_patch_qwen_chat_template,
    try_patch_foo_chat_template,
)
```

不同 patch 的识别条件必须互斥。dispatcher 会执行所有已注册 patcher；如果同一模板匹配多个 patcher，会触发 `RuntimeError`，注册顺序不是优先级机制。

### 测试

参考 [`test_qwen_chat_template_patch.py`](../../../tests/engine/sft/dataset/test_qwen_chat_template_patch.py)，至少覆盖：

1. 无关模板严格 no-op。
2. legacy 模板正确 patch，native 模板保持幂等。
3. 模板漂移、重复或冲突时 fail-fast。
4. 显式 kwargs 与默认策略正确，且不修改输入 mapping。
5. `render_with_loss_mask` 与 `render_to_text` 两条路径行为一致。
6. 不修改绑定的 `tokenizer.chat_template`。
7. 不与已有 patcher 发生重叠匹配。

## 下一步

- 按照完整的[序列分类 SFT 示例](../../../examples/seq_cls_sft/README.md)完成 checkpoint 导出和 SGLang 部署。
- 阅读[配置参考手册](./configuration.md)，查看完整 SFT 参数表。
- 阅读[性能调优](./performance-tuning.md)，了解更完整的吞吐优化方法。
- 如果任务在模型加载或训练中 OOM，阅读[OOM 排查](./oom-troubleshooting.md)。
