# OpenSearch-VL：视觉工具轨迹 SFT 与评测

本例使用 OpenSearch-VL 的多模态工具调用轨迹进行监督微调（SFT），并对比基座与训练后模型在 V\*Bench 上的表现。

训练样本保留完整的“图片与问题 → 模型推理和工具调用 → 工具观察 → 最终答案”多轮历史。评测时，模型可以自主调用本地 `crop` 工具，查看返回的局部图片后继续作答；不需要配置外部搜索或模型裁判服务。

下面以 **Qwen3-VL-8B-Instruct** 为例，按数据准备、训练、评测的顺序运行。其他模型换用对应的脚本和 HF 权重目录。

| 模型                       | 启动脚本                                                                                   |
| -------------------------- | ------------------------------------------------------------------------------------------ |
| Qwen3-VL-8B-Instruct       | [run-qwen3-vl-8B-8xgpu.sh](scripts/run-qwen3-vl-8B-8xgpu.sh)                               |
| Qwen3.5-4B                 | [run-qwen3.5-4B-8xgpu.sh](scripts/run-qwen3.5-4B-8xgpu.sh)                                 |
| Qwen3.5-35B-A3B            | [run-qwen3.5-35B-A3B-8xgpu.sh](scripts/run-qwen3.5-35B-A3B-8xgpu.sh)                       |
| Qwen3.5-35B-A3B（冻结ViT） | [run-qwen3.5-35B-A3B-freeze-vit-8xgpu.sh](scripts/run-qwen3.5-35B-A3B-freeze-vit-8xgpu.sh) |
| Qwen3.6-27B                | [run-qwen3.6-27B-8xgpu.sh](scripts/run-qwen3.6-27B-8xgpu.sh)                               |

Qwen3.5-35B-A3B当前验证通过的配置冻结视觉主干，继续训练merger和语言部分；原全参配置出现视觉退化，详见[实验结果与修复对照](results/qwen3.5-35b-a3b.md)。

## 环境与目录

先按 [Relax 安装说明](../../README.md) 准备训练环境，并将初始模型的 HF 权重放到 `HF_CHECKPOINT`。以下命令均在仓库根目录执行；训练需要可用的 8 GPU Ray 环境，评测可单独部署模型服务。

```bash
export MODEL_DIR=/path/to/models
export HF_CHECKPOINT="${MODEL_DIR}/Qwen3-VL-8B-Instruct"
export SOURCE_DIR=/path/to/opensearch-source
export DATA_DIR=/path/to/data
export SAVE_DIR=/path/to/outputs
export EXP_NAME=qwen3-vl-8b-opensearch-vl
export SCRIPT=examples/opensearch_vl/scripts/run-qwen3-vl-8B-8xgpu.sh
```

`MODEL_DIR` 保存初始模型，`SOURCE_DIR` 保存原始训练数据，`DATA_DIR` 保存处理后的数据与评测题集，`SAVE_DIR` 保存本次实验产物。每次新实验使用独立的 `EXP_NAME`。

## 1. 准备训练数据

训练数据来自 [OpenSearch-VL/Search-VL-SFT-36K](https://huggingface.co/datasets/OpenSearch-VL/Search-VL-SFT-36K)，包括七个域的图文工具轨迹。脚本固定数据版本，首次运行时下载约 11.38 GB 原始文件，再直接读取图片 ZIP 进行处理，无需先解压全部图片：

```bash
python3 examples/opensearch_vl/tools/prepare_data.py \
  --download \
  --source "${SOURCE_DIR}" \
  --output "${DATA_DIR}/opensearch_vl/32k" \
  --workers 16
```

已有完整原始数据时省略 `--download`。输出目录必须是新的。默认保留全部合格轨迹，并按关联图片和问题分组切分：训练 34,607 条，验证与测试各 32 条。预处理不按长度过滤，超长样本由训练阶段按 32K 上限处理。

输出包含 `train.jsonl`、`val.jsonl`、`test.jsonl` 和 `images/`，可直接交给训练脚本。图片处理、切分规则和训练前审计见 [数据准备说明](data-preparation.md)。

## 2. 训练

在已启动的 8 GPU Ray 环境中执行：

```bash
export PROJECT_NAME=Relax/sft/opensearch-vl
export NUM_EPOCHS=1
bash scripts/entrypoint/ray-job.sh "$SCRIPT" relax
```

模型脚本会先审计数据，再进行完整轨迹 SFT。并行度、重算和 batch 参数可通过脚本支持的环境变量设置。默认每 200 步及最后一步保存 checkpoint，保留最近一份 native 和 HF 导出；最终 HF 权重位于 `${SAVE_DIR}/${EXP_NAME}/final-hf-<step>`。

## 3. 评测基座与 SFT 模型

评测使用 [V\*Bench](https://huggingface.co/datasets/craigwu/vstar_bench) 全部 191 题，覆盖属性识别和相对位置判断。两组使用相同的题目、图片预算、提示和工具协议，按最终选项计算 Pass@1。

先下载题目及图片：

```bash
export BENCHMARK_DATA_ROOT="${DATA_DIR}/benchmark"
python3 examples/opensearch_vl/tools/vstar_benchmark.py \
  --data "${BENCHMARK_DATA_ROOT}/vstar" --prepare-only
```

然后按 [评测说明](evaluation.md) 准备模型来源文件，并按 [数据准备说明](data-preparation.md#评测去重索引) 从实际训练切分生成去重索引。

先部署基座模型。以下适用于 Qwen3-VL-8B-Instruct；Qwen3.5/3.6 需要按评测说明配置保留 thinking 历史的共用模板。

```bash
python3 examples/opensearch_vl/tools/serve_sglang.py \
  --model-path "${HF_CHECKPOINT}" --served-model-name benchmark-model \
  --host 127.0.0.1 --port 30000 --tp-size 1 --context-length 32768
```

在另一个终端加载相同的目录变量后，请求该服务评测基座：

```bash
export AGENT_BASE_URL=http://127.0.0.1:30000/v1
export AGENT_MODEL=benchmark-model
export AGENT_PROVENANCE_FILE=/path/to/verified-base-provenance.json
export BENCHMARK_OVERLAP_INDEX=/path/to/train-overlap-index.json
export BENCHMARK_WORKERS=16
bash "$SCRIPT" benchmark vstar base
```

停止基座服务，在服务终端换成训练后的最终 HF 目录并重新启动：

```bash
export HF_CHECKPOINT="${SAVE_DIR}/${EXP_NAME}/final-hf-<step>"
python3 examples/opensearch_vl/tools/serve_sglang.py \
  --model-path "${HF_CHECKPOINT}" --served-model-name benchmark-model \
  --host 127.0.0.1 --port 30000 --tp-size 1 --context-length 32768
```

在请求终端同步模型路径，更新模型来源文件，然后评测 SFT 并生成对照表：

```bash
export HF_CHECKPOINT="${SAVE_DIR}/${EXP_NAME}/final-hf-<step>"
export AGENT_PROVENANCE_FILE=/path/to/verified-sft-provenance.json
bash "$SCRIPT" benchmark vstar relax
bash "$SCRIPT" compare
```

逐题轨迹与裁剪图保存在 `${SAVE_DIR}/${EXP_NAME}/benchmark/`，对照结果为 `${SAVE_DIR}/${EXP_NAME}/benchmark-comparison.json`。完整的服务配置、工具坐标、生成预算与评分规则见 [评测说明](evaluation.md)。

## 实验列表

分数为 V\*Bench Pass@1（%，原始 / 训练集去重），每组完整 191 题。保留各组实验的阶段、状态、启动时间、训练 commit、训练曲线和详情入口；具体分析见各模型文档。

| 模型                 | 阶段 / 预算           | 状态           | 实验启动 UTC        | 训练 commit    |      V\*Bench | SwanLab训练曲线                                                                                                                                                | 实验文档                           |
| -------------------- | --------------------- | -------------- | ------------------- | -------------- | ------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| Qwen3-VL-8B-Instruct | Base                  | 评测完成       | 2026-11-11 03:48:08 |                | 29.32 / 29.32 |                                                                                                                                                                | [详情](results/qwen3-vl-8b.md)     |
| Qwen3-VL-8B-Instruct | SFT / 1 epoch         | 训练与评测完成 | 2026-11-10 16:08:22 | `f6afec0cb046` | 65.45 / 65.45 | [训练](https://swanlab.cn/@redai-studio/Relax-opensearch-vl/runs/ykm29sm5)                                                                                     | [详情](results/qwen3-vl-8b.md)     |
| Qwen3.5-4B           | Base                  | 评测完成       | 2026-11-11 05:23:15 |                | 60.21 / 60.21 |                                                                                                                                                                | [详情](results/qwen3.5-4b.md)      |
| Qwen3.5-4B           | SFT / 1 epoch         | 训练与评测完成 | 2026-11-10 18:15:32 | `6d31774a5b94` | 65.97 / 65.97 | [训练](https://swanlab.cn/@redai-studio/Relax-opensearch-vl/runs/6k29pgn6)                                                                                     | [详情](results/qwen3.5-4b.md)      |
| Qwen3.5-35B-A3B      | Base                  | 评测完成       | 2026-11-11 10:49:07 |                | 68.59 / 68.59 |                                                                                                                                                                | [详情](results/qwen3.5-35b-a3b.md) |
| Qwen3.5-35B-A3B      | 全参SFT / 1 epoch     | 验收失败       | 2026-11-11 09:18:55 | `5650ce254c35` | 37.17 / 37.17 | [训练](https://swanlab.cn/@redai-studio/Relax-opensearch-vl/runs/3n2rdfwh)                                                                                     | [详情](results/qwen3.5-35b-a3b.md) |
| Qwen3.5-35B-A3B      | 冻结ViT SFT / 1 epoch | 训练与评测完成 | 2026-11-11 16:40:05 | `5650ce254c35` | 76.44 / 76.44 | [前200步](https://swanlab.cn/@redai-studio/Relax-opensearch-vl/runs/idse0ojb) / [接续训练](https://swanlab.cn/@redai-studio/Relax-opensearch-vl/runs/efq7iybh) | [详情](results/qwen3.5-35b-a3b.md) |
| Qwen3.6-27B          | Base                  | 评测完成       | 2026-11-11 05:43:22 |                | 56.02 / 56.02 |                                                                                                                                                                | [详情](results/qwen3.6-27b.md)     |
| Qwen3.6-27B          | SFT / 1 epoch         | 训练与评测完成 | 2026-11-10 20:26:42 | `6d31774a5b94` | 71.73 / 71.73 | [训练](https://swanlab.cn/@redai-studio/Relax-opensearch-vl/runs/0iblvpvo)                                                                                     | [详情](results/qwen3.6-27b.md)     |

公开实现与协议差异见 [基准参考](benchmark-references.md)。新增模型时使用 [实验记录模板](templates/model-results.md)。
