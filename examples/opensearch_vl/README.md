# OpenSearch-VL Benchmark

统一记录不同模型在OpenSearch-VL数据上的SFT与 **V\*Bench本地裁剪工具评测**。每个模型有独立脚本和实验文档，共用数据处理、工具协议与评分规则。训练数据保持七域完整轨迹；评测只使用V\*Bench。

## 评测结果汇总

分数为Pass@1（%，原始 / 训练集去重）。已完成组统一使用v2明确最终选项归一化；仅重评分保存答案，不重新推理。每组完整191题，失败题保留在分母；空白表示尚未评测，不能用其他模型或其他协议的分数代填。

| 模型                 | 阶段 / 预算   | 状态           | 实验启动 UTC        | 训练 commit    |      V\*Bench | SwanLab训练曲线                                                            | 实验文档                       |
| -------------------- | ------------- | -------------- | ------------------- | -------------- | ------------: | -------------------------------------------------------------------------- | ------------------------------ |
| Qwen3-VL-8B-Instruct | Base          | 评测完成       | 2026-11-11 03:48:08 |                | 29.32 / 29.32 |                                                                            | [详情](results/qwen3-vl-8b.md) |
| Qwen3-VL-8B-Instruct | SFT / 1 epoch | 训练与评测完成 | 2026-11-10 16:08:22 | `f6afec0cb046` | 65.45 / 65.45 | [训练](https://swanlab.cn/@redai-studio/Relax-opensearch-vl/runs/ykm29sm5) | [详情](results/qwen3-vl-8b.md) |
| Qwen3.5-4B           | Base          | 评测完成       | 2026-11-11 05:23:15 |                | 60.21 / 60.21 |                                                                            | [详情](results/qwen3.5-4b.md)  |
| Qwen3.5-4B           | SFT / 1 epoch | 训练与评测完成 | 2026-11-10 18:15:32 | `6d31774a5b94` | 66.49 / 66.49 | [训练](https://swanlab.cn/@redai-studio/Relax-opensearch-vl/runs/6k29pgn6) | [详情](results/qwen3.5-4b.md)  |
| Qwen3.6-27B          | Base          | 评测完成       | 2026-11-11 05:43:22 |                | 56.02 / 56.02 |                                                                            | [详情](results/qwen3.6-27b.md) |
| Qwen3.6-27B          | SFT / 1 epoch | 训练与评测完成 | 2026-11-10 20:26:42 | `6d31774a5b94` | 71.73 / 71.73 | [训练](https://swanlab.cn/@redai-studio/Relax-opensearch-vl/runs/0iblvpvo) | [详情](results/qwen3.6-27b.md) |

4B修复最终选项漏判后由60.21%升至66.49%（+6.28个百分点）；旧解析曾将25条正确答案误计为失败，详见模型文档。

27B在同一模型内对照提高15.71个百分点，两组各191题且无基础设施错误。

本次8B在相同协议下提高36.13个百分点，基座耗尽轮数的题数为109，SFT为2。差异体现本地工具交互与完成作答的能力，不能把全部增益解释为纯视觉识别能力提高，也不与不同工具或图像预算的公开成绩直接排名。

## 目录与入口

| 路径                                                                 | 用途                                             |
| -------------------------------------------------------------------- | ------------------------------------------------ |
| [data-preparation.md](data-preparation.md)                           | 七域训练数据、图片处理、切分与Processor审计      |
| [evaluation.md](evaluation.md)                                       | V\*Bench下载、模型服务、裁剪协议、运行与比较     |
| [benchmark-references.md](benchmark-references.md)                   | 官方题集与社区实现的协议差异                     |
| [scripts/run-qwen3-vl-8B-8xgpu.sh](scripts/run-qwen3-vl-8B-8xgpu.sh) | Qwen3-VL-8B-Instruct入口                         |
| [scripts/run-qwen3.5-4B-8xgpu.sh](scripts/run-qwen3.5-4B-8xgpu.sh)   | Qwen3.5-4B入口与共用流程                         |
| [scripts/run-qwen3.6-27B-8xgpu.sh](scripts/run-qwen3.6-27B-8xgpu.sh) | Qwen3.6-27B入口                                  |
| [configs/benchmark.env.example](configs/benchmark.env.example)       | 模型服务、数据与输出的配置占位符                 |
| [results/](results/)                                                 | 每模型一个中文文档，新实验插在前方               |
| [templates/model-results.md](templates/model-results.md)             | 新模型及新实验记录模板                           |
| [tools/](tools/)                                                     | 数据准备、审计、本地工具评测、比较和训练看板同步 |

## 训练

先按数据准备文档生成七域训练集。预处理不按长度过滤；训练在32K上限下跳过超长样本。当前脚本默认1 epoch，每200步及最后一步保存含optimizer/RNG的native checkpoint，最多保留最近1份；HF导出也只保留最近1份。

在已配置的8 GPU Ray环境中，从仓库根目录执行：

```bash
export MODEL_DIR=/path/to/models
export HF_CHECKPOINT="${MODEL_DIR}/Qwen3-VL-8B-Instruct"
export DATA_DIR=/path/to/data
export SAVE_DIR=/path/to/experiment-artifacts
export PROJECT_NAME=Relax/sft/opensearch-vl
export EXP_NAME=qwen3-vl-8b-opensearch-vl-my-run
export NUM_EPOCHS=1
bash scripts/entrypoint/ray-job.sh examples/opensearch_vl/scripts/run-qwen3-vl-8B-8xgpu.sh relax
```

其他模型换用对应脚本与HF目录。并行、重算和offload按实际显存、吞吐选择。短训练可显式设`TRAIN_STEPS`，记录实际完成量；`BENCHMARK_ONLY=1`是训练性能短测，不是V\*Bench评测。训练中的验证loss/PPL继续保留，单独记录。

## 独立评测

训练和评测分开执行，通过持久存储交接最终HF目录。先验证导出与模型来源，再按[评测文档](evaluation.md)启动服务。模型脚本只提供以下模式：

```bash
bash examples/opensearch_vl/scripts/run-qwen3-vl-8B-8xgpu.sh benchmark vstar base
bash examples/opensearch_vl/scripts/run-qwen3-vl-8B-8xgpu.sh benchmark vstar relax
bash examples/opensearch_vl/scripts/run-qwen3-vl-8B-8xgpu.sh compare
```

基座与SFT共用数据、提示、图片预算、工具和评分规则。每题生成一条真实多轮轨迹，裁剪从原始分辨率区域执行，最终按明确的选项字母评分。首次小样本运行应检查实际回图和后续模型请求，不能只验证工具调用JSON可解析。

## 实验记录与看板

每模型维护`results/<model>.md`。新实验插在前方，填写启动时间、完整branch/commit、模型与数据来源、预算、工具和SwanLab训练曲线；结束后补齐分数及失败原因，并同步总表。生成版本与评分版本不同时分别记录，保留原始答案与修正依据。

结构化记录、逐题轨迹、日志、模型来源清单、数据、缓存和权重保存在仓库外，不提交Git。公开配置只使用占位符。最终模型和native checkpoint保留至下游验收，再按本次实验所有权清理。

训练沿用ClearML，并可通过TensorBoard同步SwanLab标量。认证留在运行环境；同步hook关闭自动Git、环境、依赖、硬件和终端采集。SwanLab用于训练曲线，评测表格记录在模型文档，不另建评测看板。
