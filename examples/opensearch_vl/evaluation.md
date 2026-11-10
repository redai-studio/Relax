# 训练后模型评测

[English](evaluation.en.md)

正式评测需要先配置全部工具服务。仅设置 QS 和 MaaS 密钥不足以运行本文的工具评测；入口会列出缺失项。显式限制题数并允许缺失工具的运行只用于联调，不计作完整 benchmark。

这里的工具要求对应完整OpenSearch-VL Agent，不是数据集本身的硬性要求。其他项目的直接问答、简化搜索Agent以及逐项准备方式见[评测方案对比与工具准备](evaluation-setup.md)。

## 评什么

对比**初始 Qwen3.5-4B**与**Relax SFT 模型**：

| 数据集             | 范围              | 指标   |
| ------------------ | ----------------- | ------ |
| SimpleVQA          | test 中全部英文题 | Pass@1 |
| VDR-Bench-testmini | 全部 500 题       | Pass@1 |

[原 benchmark](https://arxiv.org/html/2605.05185v1#S5) 的流程是：图片和问题 → 模型自主调用工具 → 最终答案 → GPT-4o 按 VDR rubric 判断语义正确性。本例复用其工具 dispatcher、system prompt 和裁判 prompt，通过 API 接入待评模型。每题一次，准确率为答对题数 / 总题数。

两组固定温度 0、32K 总上下文、最多 50 回合，共用工具缓存与同一个裁判模型。保留完整历史；超限、无答案和异常题计入分母。另报与 SFT train 去重后的分数。参考答案只给裁判。

## 准备

先完成[数据准备](data-preparation.md)，生成完整七域训练集及去重索引，再配置工具环境：

```bash
git clone https://github.com/shawn0728/OpenSearch-VL.git /path/to/OpenSearch-VL
git -C /path/to/OpenSearch-VL checkout 1ac26e2c35a3e813e022813c88888db457d88f7d
cp examples/opensearch_vl/configs/benchmark.env.example /path/to/benchmark.env
# 填入实际服务配置后：
source /path/to/benchmark.env
```

| 配置                                                              | 用途                                                            |
| ----------------------------------------------------------------- | --------------------------------------------------------------- |
| `OPENSEARCH_ROOT`                                                 | 上述工具代码目录                                                |
| `AGENT_BASE_URL / AGENT_MODEL`                                    | 待评模型的 OpenAI 兼容接口                                      |
| `SERPER_API_KEY / JINA_API_KEY / QWEN_API_BASE / QWEN_MODEL_NAME` | 文本搜索、网页读取与摘要                                        |
| `SERPAPI_URL / SERPAPI_API_KEY`                                   | Google Lens 图搜接口                                            |
| `IMAGE_UPLOAD_URL`                                                | multipart `file` 上传，返回外部可访问的 `{"url":"https://..."}` |
| `LAYOUT_PARSING_API_URL / SR_MODEL_PATH`                          | PP-StructureV3 接口、超分权重                                   |
| `JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY`                    | MaaS 裁判模型，使用 `JUDGE_MODEL` 明确指定                      |

MaaS 裁判配置：`JUDGE_BASE_URL` 填接口地址去掉 `/chat/completions` 的部分，`JUDGE_MODEL` 填实际可调用的模型名称，`JUDGE_API_KEY_HEADER=api-key`，密钥填入本地环境的 `JUDGE_API_KEY`。保留原评分 prompt，结果记录实际裁判型号；这是内部裁判分数，与论文分数分开报告。

工具环境需具备 Pillow、requests、HF processor、带 `dnn_superres` 的 OpenCV。模型服务保留 thinking 和原始 `<tool_call>` 文本，不启用会消费该文本的工具解析器。

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

原 Qwen 模板会丢弃历史 thinking，不能仅传入 `preserve_thinking=true`。先生成与训练一致的模板，本地计数与模型服务共用该文件：

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
export EXP_NAME=qwen3.5-4b-opensearch-vl-32k-8xgpu
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
- 逐题记录：`benchmark/<base|relax>/<simplevqa|vdr>/trajectories.jsonl`，包含工具轨迹、最终答案和裁判判词。
- 汇总：同目录 `summary.json`，包含题数、失败数、工具错误和截断次数。命令异常退出时先排错，不能只统计成功题。

结果另存于实验记录，注明实际裁判型号、模型版本和训练来源；不要将联调结果写成完整 benchmark 分数。
