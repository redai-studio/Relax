# 训练后模型评测

[English](evaluation.en.md)

当前默认采用 **Direct Answer：关闭全部工具，模型直接看图回答，由MaaS判分**。不需要搜索、图搜、OCR、图片上传或超分服务。

工具Agent保留为可选模式，区别与服务准备方式见[评测方案对比与工具准备](evaluation-setup.md)。

## 评什么

对比**初始 Qwen3.5-4B**与**Relax SFT 模型**：

| 数据集             | 范围              | 指标   |
| ------------------ | ----------------- | ------ |
| SimpleVQA          | test 中全部英文题 | Pass@1 |
| VDR-Bench-testmini | 全部 500 题       | Pass@1 |

每题仅一次模型生成，温度0，保留thinking，输入和输出合计最多32K；不提供工具定义或历史轨迹，不执行工具。复用OpenSearch/VDR裁判prompt，通过MaaS判断最终答案的语义正确性，Pass@1为答对题数/全部题数。参考答案只给裁判。超限、无答案、错误工具调用和异常题均保留在分母，另报与SFT train去重后的分数。

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

| 配置                                           | 用途                                       |
| ---------------------------------------------- | ------------------------------------------ |
| `OPENSEARCH_ROOT`                              | 上述工具代码目录                           |
| `AGENT_BASE_URL / AGENT_MODEL`                 | 待评模型的 OpenAI 兼容接口                 |
| `JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY` | MaaS 裁判模型，使用 `JUDGE_MODEL` 明确指定 |

MaaS 裁判配置：`JUDGE_BASE_URL` 填接口地址去掉 `/chat/completions` 的部分，`JUDGE_MODEL` 填实际可调用的模型名称，`JUDGE_API_KEY_HEADER=api-key`，密钥填入本地环境的 `JUDGE_API_KEY`。保留原评分 prompt，结果记录实际裁判型号；这是内部裁判分数，与论文分数分开报告。

客户端需具备Pillow、HF processor、PyArrow、OpenAI SDK和httpx。当前MaaS域名限制1 QPS，运行器对请求发起限速、允许裁判请求并行且关闭自动重试；结果记录实际裁判模型。

## 评测数据与去重

下载固定版本，SimpleVQA 仅读取 `language == "EN"` 的 1,013 题，VDR 读取全部 500 题：

```python
from huggingface_hub import hf_hub_download

benchmark_data_root = "/path/to/benchmark-datasets"
hf_hub_download("m-a-p/SimpleVQA", repo_type="dataset",
    revision="037cf89fb6f1212691756b66d1ecde6c2ce89e54", filename="test.parquet",
    local_dir=f"{benchmark_data_root}/simplevqa")
hf_hub_download("Osilly/Vision-DeepResearch-Eval", repo_type="dataset",
    revision="deeaf45779a3bbd407d8f0ccb9b4831fc78e81c9", filename="data/testmini500_v1.parquet",
    local_dir=f"{benchmark_data_root}/vdr")
```

将 `BENCHMARK_DATA_ROOT` 设为相同下载目录。`BENCHMARK_OVERLAP_INDEX` 指向从实际 `train.jsonl` 生成的 JSON，包含 `questions` 和 `images` 两个 hash 列表。问题 hash 对首条用户消息去掉 `<image>`，再 casefold、合并空白并计算 UTF-8 SHA256；图片使用训练记录的 `metadata.image_hashes`，即原图 RGB 尺寸与像素的 SHA256。问题或任一图片命中都视为与训练重叠；去重分数仅统计未命中题目。

`AGENT_PROVENANCE_FILE` 指向模型来源 JSON，包含实际 QS model/version/subversion、`served_model_name`、完整 branch/commit、训练 Job/Trial、`kind`、`epochs` 和 `hashes_verified=true`。先按上传收据校验下载文件，再写此标记；它不是自动校验的替代品。`relax` 标签只接受 `kind=trained-final` 且至少 2 epoch 的版本，上传验证版本只能作为 baseline。

## 运行

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
bash "$SCRIPT" benchmark vdr base
```

停止初始模型服务，将 `--model-path` 换成选定的 Relax HF 模型目录，重启服务。当前训练脚本导出至 `${SAVE_DIR}/${EXP_NAME}/final-hf-<step>`；自动上传流程会将目录改名，加入完整分支、commit 和训练 Job/Trial。若临时目录已经清理，先从 QS 模型库下载对应的正式训练版本，再使用下载后的 HF 目录。

确认所选版本的 `relax-artifact.json` 中 `kind=trained-final`，并记录实际 branch、commit、Job/Trial 和模型版本；`kind=upload-validation` 的版本仅用于验证上传流程，不作为训练后模型评测。

再执行：

```bash
export AGENT_MODEL=your-served-relax-model-name
bash "$SCRIPT" benchmark simplevqa relax
bash "$SCRIPT" benchmark vdr relax
bash "$SCRIPT" compare
```

## 看结果

- 总表：`${SAVE_DIR}/${EXP_NAME}/benchmark-comparison.json`，包含训练前后 Pass@1、差值和去重集合分数。
- 逐题记录：`benchmark/<base|relax>/<simplevqa|vdr>/trajectories.jsonl`，包含问题、完整模型输出、最终答案、token用量和裁判判词。
- 汇总：同目录 `summary.json`，包含题数、失败数、工具错误和截断次数。命令异常退出时先排错，不能只统计成功题。

结果另存于实验记录，注明实际裁判型号、模型版本和训练来源；不要将联调结果写成完整 benchmark 分数。

## 与论文分数比较

[论文Table 2](https://arxiv.org/html/2605.05185v1#S5.T2)的Direct Reasoning参考分数（%）：

| 模型             | SimpleVQA | VDR |
| ---------------- | --------- | --- |
| Qwen3-VL-8B      | 47.1      | 2.8 |
| Qwen3-VL-30B-A3B | 53.2      | 3.8 |
| Qwen3-VL-32B     | 58.0      | 4.1 |

这些分数可作背景参考，不能与当前结果宣称严格同口径：论文使用GPT-4o裁判，本例使用实际配置的MaaS模型；题集范围及生成预算也需对齐。当前实验的主要比较是同一设置下初始模型与SFT模型的差值。Agentic Workflow和Table 3中的SFT工具评测应另列。

## 可选的完整工具Agent

显式设置 `BENCHMARK_WORKFLOW=agent` 才会启用工具环境；需配置[工具准备文档](evaluation-setup.md)中的全部服务，共用工具缓存，最多50回合。为它使用独立的 `EXP_NAME` 和结果目录。比较脚本会拒绝不同workflow、提示或裁判配置的结果混比。
