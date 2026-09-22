# Qwen3.5 Sequence Classification SFT

本示例展示如何使用 Relax 的 Megatron 后端训练 Qwen3.5 原生 sequence-classification 模型，并将
Megatron native checkpoint 导出后交给 SGLang 部署。

支持以下任务：

| 任务         | 标签格式                       | 模型输出 | 训练 loss     |
| ------------ | ------------------------------ | -------- | ------------- |
| 单标签二分类 | `int`，取值为 `0` 或 `1`       | 2 logits | CrossEntropy  |
| 单标签多分类 | `int`，范围为 `[0, K)`         | K logits | CrossEntropy  |
| 多标签分类   | 类别索引列表，例如 `[0, 2, 5]` | K logits | BCEWithLogits |

二分类统一使用 2-logit CrossEntropy。多标签输入在数据边界转换为 multi-hot；训练 loss 直接消费
logits，不使用分类阈值。`--classification-threshold` 仅用于多标签 eval 的离散指标，客户端的
`--threshold` 仅用于推理时从 sigmoid 概率中选择最终标签。

## 目录结构

```text
examples/seq_cls_sft/
├── README.md
├── run-qwen3.5-9B-classification-sft-8xgpu.sh
├── run-qwen3.5-35B-A3B-classification-lora-sft-8xgpu.sh
├── models/
│   └── sglang/
│       └── model.py
└── tools/
    ├── prepare_classification_sft_data.py
    ├── convert_sequence_classification_checkpoint.py
    ├── export_sequence_classification_sglang.sh
    ├── launch_sglang_text_classification.py
    ├── serve_sequence_classification_sglang.sh
    └── request_sequence_classification.py
```

## 实现范围

- Qwen3.5-9B dense 全参数训练。
- Qwen3.5-35B-A3B MoE LoRA 训练。
- Qwen3-VL 多模态分类训练，通过 `--multimodal-keys` 声明图片、视频或音频字段。
- 单标签二分类、单标签多分类和多标签分类。
- Hugging Face CausalLM backbone 初始化，分类头随机初始化。
- Megatron native checkpoint 保存与 resume。
- Megatron Bridge 离线导出；LoRA 导出时合并到完整 backbone。
- Qwen3.5 dense/MoE 与 Qwen3-VL dense 的 SGLang 外部模型适配器。
- 同步 SFT 拓扑，只部署 `sft` 和 `actor` Ray Serve role。

当前不支持回归、fully async、hybrid、SFT predict、MTP、训练期间直接
`--save-hf`，以及 SGLang pipeline parallel。SGLang 服务仅使用 tensor parallel。

## 数据格式

单标签样本：

```json
{"messages": [{"role": "user", "content": "This movie is excellent."}], "label": 1}
```

多标签样本：

```json
{"messages": [{"role": "user", "content": "I feel relieved and excited."}], "label": [4, 17]}
```

多模态单标签样本：

```json
{"messages": [{"role": "user", "content": [{"type": "text", "text": "Classify this image."}, {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}}]}], "label": 2}
```

多模态训练需同时配置 `--multimodal-keys`，processor 展开媒体 token 后，Relax 会在序列末尾追加分类
sentinel，并把该位置的 hidden state 送入分类头。

单标签必须是范围 `[0, K)` 内的整数。多标签必须是类别索引列表，允许空列表，但不允许重复或
越界索引。

## 环境变量

以下命令均从 Relax 仓库根目录执行：

```bash
export DATA_DIR=/path/to/relax-workspace
export EXP_DIR=/path/to/relax-workspace/exp
export MODEL_DIR=/path/to/relax-workspace/models
```

模型目录约定：

```text
${MODEL_DIR}/Qwen3.5-9B
${MODEL_DIR}/Qwen3.5-35B-A3B
```

训练数据默认写入并读取 `${DATA_DIR}/sft/seq_cls`。`EXP_DIR`、`MODEL_DIR` 和 `DATA_DIR` 均可按
部署环境覆盖，脚本不硬编码集群路径。

## 准备验证数据集

数据准备工具使用固定 revision 下载并转换以下公开数据集：

| 名称                  | 任务         | 分类数 | 训练/验证 split    |
| --------------------- | ------------ | -----: | ------------------ |
| SST-2                 | 单标签二分类 |      2 | train / validation |
| AG News               | 单标签多分类 |      4 | train / test       |
| GoEmotions simplified | 多标签分类   |     28 | train / validation |

准备完整数据：

```bash
python examples/seq_cls_sft/tools/prepare_classification_sft_data.py \
  --dataset all \
  --subset full \
  --global-batch-size 64 \
  --output-dir "${DATA_DIR}/sft/seq_cls"
```

也可用 `--dataset sst2`、`--dataset ag_news` 或 `--dataset go_emotions` 只准备一个任务。
`--subset smoke` 生成两步训练及非整批 eval 所需的最小数据，`--subset extended` 生成确定性中型
子集，`--subset full` 使用完整 split。正式训练脚本默认读取 `full`。

## 正式训练

### Qwen3.5-9B 全参数训练

默认任务是 SST-2：

```bash
bash examples/seq_cls_sft/run-qwen3.5-9B-classification-sft-8xgpu.sh
```

切换到 AG News 或 GoEmotions：

```bash
CLASSIFICATION_DATASET=ag_news \
bash examples/seq_cls_sft/run-qwen3.5-9B-classification-sft-8xgpu.sh
```

### Qwen3.5-35B-A3B LoRA 训练

默认任务是 GoEmotions：

```bash
bash examples/seq_cls_sft/run-qwen3.5-35B-A3B-classification-lora-sft-8xgpu.sh
```

两个脚本都启用 dynamic batch size，使用 8 张 GPU，默认训练 10 epoch，并每 100 step 保存一次
Megatron native checkpoint。9B 默认每 10 step eval，35B-A3B 默认每 20 step eval。

常用覆盖项：

| 环境变量                   | 作用                               |
| -------------------------- | ---------------------------------- |
| `CLASSIFICATION_DATASET`   | `sst2`、`ag_news` 或 `go_emotions` |
| `DATA_SUBSET`              | `smoke`、`extended` 或 `full`      |
| `TRAIN_DATA` / `EVAL_DATA` | 覆盖训练和验证 JSONL               |
| `NUM_EPOCH`                | 训练 epoch 数                      |
| `SAVE_INTERVAL`            | checkpoint 保存间隔                |
| `GLOBAL_BATCH_SIZE`        | global batch size                  |
| `MAX_TOKENS_PER_GPU`       | dynamic batch 的单卡 token 上限    |
| `SAVE_DIR` / `LOAD_DIR`    | native checkpoint 保存和恢复路径   |
| `RAY_ADDRESS`              | Ray Dashboard 地址                 |

## 导出 SGLang 模型

训练 checkpoint 不能直接由 SGLang 加载。导出工具会重建训练时的分类头和 LoRA wrapper，通过
Megatron Bridge 聚合 TP/PP/EP 分片，将 LoRA 合并进 backbone，把任务头改名为
`score.weight`，并更新 `config.json` 的 architecture、problem type 和标签映射。
对于 tied embedding 的 Qwen3-VL，Bridge 会跳过 `output_layer`；导出工具会从已加载的 native
checkpoint 中直接提取训练后的分类头，并写入独立的 `score.weight` shard。

导出 SST-2 模型：

```bash
MODEL_VARIANT=9B \
CLASSIFICATION_DATASET=sst2 \
bash examples/seq_cls_sft/tools/export_sequence_classification_sglang.sh \
  --label-names negative positive
```

导出 AG News 模型：

```bash
MODEL_VARIANT=9B \
CLASSIFICATION_DATASET=ag_news \
bash examples/seq_cls_sft/tools/export_sequence_classification_sglang.sh \
  --label-names world sports business sci_tech
```

导出 35B-A3B GoEmotions LoRA checkpoint：

```bash
MODEL_VARIANT=35B-A3B \
CLASSIFICATION_DATASET=go_emotions \
bash examples/seq_cls_sft/tools/export_sequence_classification_sglang.sh
```

默认导出最近一次 checkpoint。可通过 `CHECKPOINT_DIR` 指定 checkpoint 根目录或
`iter_XXXXXXX` 目录，通过 `ORIGIN_MODEL_DIR` 指定原始 Hugging Face 模型，通过
`CLASSIFICATION_MODEL_DIR` 指定输出目录。输出目录已存在时需显式追加 `--force`。

成功日志应包含：

```text
[seq-cls-export] done: architecture=Qwen3_5ForSequenceClassification, problem_type=single_label_classification, score.weight=(2, hidden_size)
```

Megatron Bridge 可能在索引中产生未实际导出的 MTP ghost keys；导出器会校验 shard 并删除确认不存在
的 ghost entries。

## 启动 SGLang

9B 模型默认使用 TP=4、端口 30000：

```bash
export CLASSIFICATION_MODEL_DIR=/path/to/exported-classifier
export TP_SIZE=4
export PORT=30000

bash examples/seq_cls_sft/tools/serve_sequence_classification_sglang.sh
```

启动脚本会设置：

- `SGLANG_EXTERNAL_MODEL_PACKAGE=examples.seq_cls_sft.models.sglang`；
- `SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=examples.seq_cls_sft.models.sglang`，让 Qwen3-VL 分类模型复用原生图片/视频 processor；
- `--is-embedding`，使 SGLang 走 pooling/classification 路径；
- Qwen3.5 自动设置 `enable_multimodal=False`，避免文本分类触发多模态 processor 初始化；
- Qwen3-VL 保持多模态 processor 开启。

文本分类 launcher 已在 SGLang `0.5.12.post1` 上验证，并兼容保留相同 ServerArgs 接口的
`0.5.15`。当前 adapter 只支持 TP，不要配置 PP。

## 发起分类请求

客户端从 `--model-dir` 读取 tokenizer、chat template、problem type 和标签映射，以保证请求 token
与训练预处理一致。

单标签请求：

```bash
python examples/seq_cls_sft/tools/request_sequence_classification.py \
  --model-dir "${CLASSIFICATION_MODEL_DIR}" \
  --text "This movie was wonderful, engaging, and beautifully acted."
```

单标签任务调用 `/v1/classify`，SGLang 返回 softmax 概率和 argmax 标签。

多标签请求：

```bash
python examples/seq_cls_sft/tools/request_sequence_classification.py \
  --model-dir "${CLASSIFICATION_MODEL_DIR}" \
  --text "I am relieved, grateful, and excited about the result." \
  --threshold 0.5
```

多标签任务调用 `/v1/score` 获取 raw logits，客户端逐类计算 sigmoid，再使用 `--threshold` 选择
标签。该阈值不参与训练或 eval loss。

## 评估指标

单标签 eval 输出 loss 和 accuracy。多标签 eval 输出 loss、micro precision、micro recall、micro
F1 和 subset accuracy。eval 尾批通过零权重 padding 补齐，padding 样本不进入 loss 或指标。

## 测试说明

分类核心测试位于对应的 `tests/backends`、`tests/components`、`tests/engine` 和 `tests/utils` 目录；
本示例专属的导出与 SGLang adapter 测试位于 `tests/examples/seq_cls_sft/`。

多节点 TP/PP/CP 集成测试和完整训练需要目标 Megatron/SGLang/CUDA 环境；没有相应硬件时应明确
记录跳过，不能视为已通过。
