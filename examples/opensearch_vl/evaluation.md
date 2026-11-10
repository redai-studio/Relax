# 训练后模型评测

> 接入状态：本文定义目标评测协议。当前分支尚未实现下述 `benchmark` 子命令和
> `configs/benchmark.env.example`，现有 `compare` 只比较留出决策评测，不能用于生成本文的 benchmark 总表。
> 下列运行命令需待 benchmark 接入后使用。英文版 `evaluation.en.md`、数据准备说明
> `data-preparation.md` 和验证记录 `results/qwen3.5-4b.md` 也待补齐。

## 评什么

对比**初始 Qwen3.5-4B**与**Relax SFT 模型**：

| 数据集             | 范围              | 指标   |
| ------------------ | ----------------- | ------ |
| SimpleVQA          | test 中全部英文题 | Pass@1 |
| VDR-Bench-testmini | 全部 500 题       | Pass@1 |

[原 benchmark](https://arxiv.org/html/2605.05185v1#S5) 的流程是：图片和问题 → 模型自主调用工具 → 最终答案 → GPT-4o 按 VDR rubric 判断语义正确性。本例复用其工具 dispatcher、system prompt 和裁判 prompt，通过 API 接入待评模型。每题一次，准确率为答对题数 / 总题数。

两组固定温度 0、32K 总上下文、最多 50 回合，共用工具缓存与同一个裁判模型。保留完整历史；超限、无答案和异常题计入分母。另报与 SFT train 去重后的分数。参考答案只给裁判。

## 准备

先完成数据准备，再配置工具环境：

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
| `JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY`                    | MaaS 裁判模型（默认 `glm-5.3-flash`）                           |

MaaS 裁判配置：`JUDGE_BASE_URL` 填接口地址去掉 `/chat/completions` 的部分，`JUDGE_MODEL=glm-5.3-flash`，`JUDGE_API_KEY_HEADER=api-key`，密钥填入本地环境的 `JUDGE_API_KEY`。保留原评分 prompt，结果记录实际裁判型号；这是内部裁判分数，与论文分数分开报告。

工具环境需具备 Pillow、requests、HF processor、带 `dnn_superres` 的 OpenCV。模型服务保留 thinking 和原始 `<tool_call>` 文本，不启用会消费该文本的工具解析器。

## 运行

在 GPU 节点部署初始模型；沿用数据准备中的 `MODEL_DIR / DATA_DIR / SAVE_DIR`：

```bash
export EXP_NAME=qwen3.5-4b-opensearch-vl-32k-8xgpu
python3 -m sglang.launch_server \
  --model-path "${MODEL_DIR}/Qwen3.5-4B" \
  --host 127.0.0.1 --port 30000 --tp 2 --context-length 32768
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

将实测结果填入后续的 Qwen3.5-4B 验证记录 `results/qwen3.5-4b.md`。
