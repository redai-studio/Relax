# OpenSearch-VL Benchmark

统一记录不同模型在 OpenSearch-VL 数据上的 SFT 与评测效果。每个模型有独立启动脚本和实验文档，共用数据处理、评测协议与指标定义。当前只评测SimpleVQA，采用无工具Direct Answer；带工具 Agentic 结果单独报告。

## 评测结果汇总

每次正式基准测试保留一行，按模型分组、时间倒序；空白表示尚无结果。分数为 Pass@1（%，原始 / 去重），失败题保留在分母。去重题集由本次实际 SFT train 决定，不同去重集合不能直接排名。

| 模型                 | 阶段 / 预算     | 状态       | 启动时间 UTC        | 训练 commit    |  SimpleVQA EN | SwanLab                                                                                                                                                                           | 实验文档                                      |
| -------------------- | --------------- | ---------- | ------------------- | -------------- | ------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| Qwen3-VL-8B-Instruct | SFT / 2 epochs  | 准备中     |                     |                |               |                                                                                                                                                                                   | [详情](results/qwen3-vl-8b.md)                |
| Qwen3.5-4B           | Base            | 完成       | 2026-11-10 08:33:48 |                | 45.11 / 46.38 | [评测](https://swanlab.cn/@redai-studio/Relax-opensearch-vl/runs/5210g8ibu4eo0y0zzg6ft)                                                                                           | [详情](results/qwen3.5-4b.md#baseline)        |
| Qwen3.5-4B           | SFT / 100 steps | 完成       | 2026-11-10 07:04:49 | `28f866220dc3` | 42.74 / 43.93 | [训练](https://swanlab.cn/@redai-studio/Relax-opensearch-vl/runs/ejccsn9ib17s2swcq6z40) · [评测](https://swanlab.cn/@redai-studio/Relax-opensearch-vl/runs/emr9mnx3q7y5z6e6z3pfi) | [详情](results/qwen3.5-4b.md#sft100-20261110) |
| Qwen3.6-27B          | SFT / 2 epochs  | 训练未完成 |                     | `1c621a84ae89` |               |                                                                                                                                                                                   | [详情](results/qwen3.6-27b.md)                |

当前分数均为 Direct Answer、temperature=0、4B的thinking开启、32K总上下文。完整生成配置、裁判和评测 commit 见模型文档。SimpleVQA全量对比中，100步SFT比Base低2.37个百分点。评测代码仅裁判重试及帮助文字有差异，已人工审计并保留各自源码SHA，见模型文档；100步是短程实验。[公开参考基准](benchmark-references.md)中的工具模式及其他裁判分数另列。

已有VDR分数保留在模型历史文档中；后续实验、数据下载和默认比较均只要求SimpleVQA。

## 目录与入口

| 路径                                                                 | 用途                                                   |
| -------------------------------------------------------------------- | ------------------------------------------------------ |
| [data-preparation.md](data-preparation.md)                           | 七域数据、图片处理、固定切分及真实 Processor 审计      |
| [evaluation.md](evaluation.md)                                       | 数据版本、Direct Answer 协议、裁判配置、运行与结果比较 |
| [evaluation-setup.md](evaluation-setup.md)                           | 可选工具环境和其他评测实现的区别                       |
| [benchmark-references.md](benchmark-references.md)                   | 公开 Direct Reasoning、Agentic、纯 SFT 参考分数        |
| [scripts/run-qwen3-vl-8B-8xgpu.sh](scripts/run-qwen3-vl-8B-8xgpu.sh) | Qwen3-VL-8B-Instruct的训练与SimpleVQA入口              |
| [scripts/run-qwen3.5-4B-8xgpu.sh](scripts/run-qwen3.5-4B-8xgpu.sh)   | Qwen3.5-4B 的训练、基准评测与比较入口                  |
| [scripts/run-qwen3.6-27B-8xgpu.sh](scripts/run-qwen3.6-27B-8xgpu.sh) | Qwen3.6-27B 模型配置及共用入口                         |
| [configs/benchmark.env.example](configs/benchmark.env.example)       | 模型、裁判与可选工具的配置占位符                       |
| [results/](results/)                                                 | 每模型一个 Markdown 文档，新实验插在旧实验之前         |
| [templates/model-results.md](templates/model-results.md)             | 新模型文档与每次实验的记录模板                         |
| [tools/](tools/)                                                     | 数据准备/审计、评测、比较、服务验证和 SwanLab 同步实现 |

## 训练

先按数据准备文档生成七域完整训练集。预处理不按长度过滤，训练在 32K 上限下跳过超长样本；默认训练 2 epochs，只在最后保存 native/HF 权重，不保存 optimizer/RNG。

在已经配置好的 8 GPU Ray 环境，从仓库根目录运行：

```bash
export MODEL_DIR=/path/to/models
export HF_CHECKPOINT="${MODEL_DIR}/Qwen3.5-4B"
export DATA_DIR=/path/to/data
export SAVE_DIR=/path/to/experiment-artifacts
export PROJECT_NAME=Relax/sft/opensearch-vl
export EXP_NAME=qwen3.5-4b-opensearch-vl-my-run
export NUM_EPOCHS=2
export GLOBAL_BATCH_SIZE=32
export MAX_TOKENS_PER_GPU=32768
bash scripts/entrypoint/ray-job.sh examples/opensearch_vl/scripts/run-qwen3.5-4B-8xgpu.sh relax
```

短程实验可另设 `TRAIN_STEPS=100`；步数和 epoch 预算取先到者，必须记录实际完成量。27B 换用对应脚本与 HF 目录；TP/PP、重算、CPU optimizer offload 按实际显存和吞吐选择。

脚本模式为 `relax`、`benchmark simplevqa <base|relax>`、`evaluate <HF目录> <base|relax>` 和 `compare`。`benchmark` 是公开数据集问答评测，`evaluate` 是留出轨迹决策评测；训练期间每 100 步的验证 loss/PPL 另行记录。`BENCHMARK_ONLY=1` 仅测试训练性能，默认 8 步、不保存权重。

## 评测与实验记录

最终 HF 导出默认在 `${SAVE_DIR}/${EXP_NAME}/final-hf-<step>`。直接用该目录启动模型服务，再按评测文档运行SimpleVQA；无需先发布权重。每题一次生成、固定提示和裁判，保留失败题，并记录实际数据与模型来源。

每个模型只维护一个 `results/<model>.md`。实验开始时在前方插入新记录，填写 run ID、启动时间、分支、训练/评测 commit 和 SwanLab 链接；结束后在同一节补齐题数、分数、失败情况与产物引用，同时更新上方总表。未确认的字段留空，未完成或限题测试不填正式分数。

JSON、JSONL、逐题轨迹、日志、数据、缓存和权重保存在仓库外的 `SAVE_DIR`，不提交到 Git。配置示例只用占位符；文档中的产物引用使用可公开分享的位置，无法分享时留空。

## SwanLab

可在运行环境安装 `swanlab==0.7.16` 并完成认证，然后设置 `ENABLE_SWANLAB=1`、`SWANLAB_MODE=cloud`、`SWANLAB_PROJ_NAME`、`SWANLAB_WORKSPACE` 和 `SWANLAB_LOG_DIR`。训练保留 ClearML，并通过 TensorBoard 同步标量。认证文件留在运行环境。

同步 hook 关闭自动环境、依赖、硬件及终端日志采集，只同步标量和指定来源字段。实验文档分别记录训练与评测链接；实际裁判型号与协议也需写入文档。
