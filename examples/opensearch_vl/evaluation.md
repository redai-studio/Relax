# 训练后模型评测

当前默认采用 **Direct Answer：关闭全部工具，模型直接看图回答，由裁判模型判分**。不需要搜索、图搜、OCR、图片上传或超分服务。

工具Agent保留为可选模式，区别与服务准备方式见[评测方案对比与工具准备](evaluation-setup.md)。

## 评什么

对每个模型分别对比**初始权重**与**Relax SFT 权重**。模型脚本和实验历史见[汇总](README.md)：

| 数据集    | 范围              | 指标   |
| --------- | ----------------- | ------ |
| SimpleVQA | test 中全部英文题 | Pass@1 |

每题仅一次模型生成，温度0，按模型原生模板保留thinking（Qwen3-VL-8B-Instruct没有该开关），输入和输出合计最多32K；不提供工具定义或历史轨迹，不执行工具。复用OpenSearch/VDR裁判prompt，通过裁判接口判断最终答案的语义正确性，Pass@1为答对题数/全部题数。参考答案只给裁判。超限、无答案、错误工具调用和异常题均保留在分母，另报与SFT train去重后的分数。

模型服务容量允许时，可设置 `BENCHMARK_WORKERS=4` 并发生成答案；裁判可并行调用；`JUDGE_QPS`控制请求发起速率，默认1 QPS，设为0表示不限速。初始/SFT两组使用相同设置，结果记录评测代码分支、commit与源码hash。

这是公开题目上的无工具问答评测，不是训练轨迹PPL或给定标准工具结果的下一步预测。

## 准备

先完成[数据准备](data-preparation.md)，生成完整七域训练集及去重索引，再配置模型和裁判接口：

```bash
git clone https://github.com/shawn0728/OpenSearch-VL.git /path/to/OpenSearch-VL
git -C /path/to/OpenSearch-VL checkout 1ac26e2c35a3e813e022813c88888db457d88f7d
cp examples/opensearch_vl/configs/benchmark.env.example /path/to/benchmark.env
# 填入实际服务配置后：
source /path/to/benchmark.env
```

| 配置                                           | 用途                                  |
| ---------------------------------------------- | ------------------------------------- |
| `OPENSEARCH_ROOT`                              | 上述工具代码目录                      |
| `AGENT_BASE_URL / AGENT_MODEL`                 | 待评模型的 OpenAI 兼容接口            |
| `JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY` | 裁判模型，使用 `JUDGE_MODEL` 明确指定 |

裁判配置：`JUDGE_BASE_URL` 填接口地址去掉 `/chat/completions` 的部分，`JUDGE_MODEL` 填实际可调用的模型名称，`JUDGE_API_KEY_HEADER=api-key`，密钥填入本地环境的 `JUDGE_API_KEY`。保留原评分 prompt，结果记录实际裁判型号；更换论文所用裁判后，结果应与论文分数分开报告。

客户端需具备Pillow、HF processor、PyArrow、OpenAI SDK和httpx。按服务实际限制设置 `JUDGE_QPS`；设为0表示不在客户端限速。裁判瞬时HTTP/连接故障最多重试2次，复用同一次生成的答案；最终失败仍保留记录和分母。结果记录实际裁判模型。

## 评测数据与去重

当前仅评测 SimpleVQA 全部1,013道英文题；训练后自动评测、基座评测与默认比较均不要求VDR。下载固定版本，运行时读取 `language == "EN"`：

```python
from huggingface_hub import hf_hub_download

benchmark_data_root = "/path/to/benchmark-datasets"
hf_hub_download("m-a-p/SimpleVQA", repo_type="dataset",
    revision="037cf89fb6f1212691756b66d1ecde6c2ce89e54", filename="test.parquet",
    local_dir=f"{benchmark_data_root}/simplevqa")
```

将 `BENCHMARK_DATA_ROOT` 设为相同下载目录。`BENCHMARK_OVERLAP_INDEX` 指向从实际 `train.jsonl` 生成的 JSON，包含 `questions` 和 `images` 两个 hash 列表。问题 hash 对首条用户消息去掉 `<image>`，再 casefold、合并空白并计算 UTF-8 SHA256；图片使用训练记录的 `metadata.image_hashes`，即原图 RGB 尺寸与像素的 SHA256。问题或任一图片命中都视为与训练重叠；去重分数仅统计未命中题目。

`AGENT_PROVENANCE_FILE` 指向保存在运行产物目录的模型来源 JSON，记录模型名称与 revision、`served_model_name`、完整训练 branch/commit、`kind`、实际训练预算和 `hashes_verified=true`。验证模型文件与可信校验清单一致后再设置此标记。本地最终导出应检查权重索引和分片、计算文件 SHA256，并记录 `origin=local-final-export`。`relax` 标签只接受 `kind=trained-final`；基座与短程训练分别记录。该文件及逐题结果不提交到 Git。

## 运行

Qwen3-VL-8B-Instruct使用原生模板，不强行添加thinking开关；Qwen3.5/3.6仍要求保留完整thinking历史的补丁成功。结果记录实际chat模板参数。

本地计数和模型服务使用相同模板，以核对包含图片展开的实际输入token数。生成模板后供两边共用：

```bash
export HF_CHECKPOINT="${MODEL_DIR}/Qwen3.5-4B"
export BENCHMARK_CHAT_TEMPLATE=/path/to/preserve-thinking.jinja
PYTHONPATH=examples/opensearch_vl/tools:. python3 - <<'PY'
import os
from pathlib import Path
from transformers import AutoProcessor
from benchmark import preserve_history_template
processor = AutoProcessor.from_pretrained(os.environ["HF_CHECKPOINT"], local_files_only=True)
Path(os.environ["BENCHMARK_CHAT_TEMPLATE"]).write_text(preserve_history_template(processor))
PY
```

在 GPU 节点部署初始模型；沿用数据准备中的 `MODEL_DIR / DATA_DIR / SAVE_DIR`：

```bash
export EXP_NAME=qwen3.5-4b-opensearch-vl-direct
export BENCHMARK_WORKFLOW=direct
python3 -m sglang.launch_server \
  --model-path "${MODEL_DIR}/Qwen3.5-4B" \
  --host 127.0.0.1 --port 30000 --tp 2 --context-length 32768 \
  --chat-template "${BENCHMARK_CHAT_TEMPLATE}" --served-model-name Qwen3.5-4B
```

另一个终端加载环境后执行：

```bash
export AGENT_BASE_URL=http://127.0.0.1:30000/v1
export AGENT_MODEL=Qwen3.5-4B
SCRIPT=examples/opensearch_vl/scripts/run-qwen3.5-4B-8xgpu.sh
bash "$SCRIPT" benchmark simplevqa base
```

初始模型评测完成后，停止对应服务，将 `--model-path` 换成选定的 Relax HF 目录并重新生成其共用模板。训练脚本默认导出至 `${SAVE_DIR}/${EXP_NAME}/final-hf-<step>`，可以直接用于推理。更新 `AGENT_PROVENANCE_FILE` 为该模型已经校验的来源文件；保持数据、图片处理、提示、生成预算、裁判和去重方式一致。

再执行：

```bash
export AGENT_MODEL=your-served-relax-model-name
bash "$SCRIPT" benchmark simplevqa relax
bash "$SCRIPT" compare
```

## 看结果

- 总表：`${SAVE_DIR}/${EXP_NAME}/benchmark-comparison.json`，包含SimpleVQA训练前后Pass@1、差值和去重集合分数；不读取或要求VDR结果。
- 逐题记录：`benchmark/<base|relax>/simplevqa/trajectories.jsonl`，包含问题、完整模型输出、最终答案、token用量和裁判判词。
- 汇总：同目录 `summary.json`，包含题数、失败数、工具错误和截断次数。命令异常退出时先排错，不能只统计成功题。

最终分数填写到 `results/<model>.md` 的本次实验段，最新实验插在前方，并同步[汇总表](README.md#评测结果汇总)。逐题 JSONL、summary 和模型来源文件留在仓库外的产物目录。记录实际裁判型号、模型版本和训练来源；未完成或限题测试不填完整分数。

## 与论文分数比较

完整参考见[公开基准](benchmark-references.md)：保留 Table 2 的带工具对照、Table 3 的纯 SFT 消融，并补充无工具分数及 Qwen3.5-4B 官方 SimpleVQA 43.4%。Table 3 的纯 SFT 仍使用工具。

[论文Table 2](https://arxiv.org/html/2605.05185v1#S5.T2)的Direct Reasoning参考分数（%）：

| 模型             | SimpleVQA | VDR |
| ---------------- | --------- | --- |
| Qwen3-VL-8B      | 47.1      | 2.8 |
| Qwen3-VL-30B-A3B | 53.2      | 3.8 |
| Qwen3-VL-32B     | 58.0      | 4.1 |

这些分数可作背景参考，不能与当前结果宣称严格同口径：论文使用GPT-4o裁判，本例使用实际配置的裁判模型；题集范围及生成预算也需对齐。当前实验的主要比较是同一设置下初始模型与SFT模型的差值。Agentic Workflow和Table 3中的SFT工具评测应另列。

## 可选的完整工具Agent

显式设置 `BENCHMARK_WORKFLOW=agent` 才会启用工具环境；需配置[工具准备文档](evaluation-setup.md)中的全部服务，共用工具缓存，最多50回合。为它使用独立的 `EXP_NAME` 和结果目录。比较脚本会拒绝不同workflow、提示或裁判配置的结果混比。
