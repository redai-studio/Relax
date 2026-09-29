# Kimi K3 训练与导出

[English](./README.md)

## 概述

Relax 支持 Kimi K3 的全参和 LoRA SFT，包括图片输入、packing、固定 CP、PP 和 EP。模型配置位于 `scripts/models/kimi-k3.sh`，公开示例位于 `examples/models/kimi-k3/scripts/`。

请使用按照 `docker/Dockerfile` 或 `docker/Dockerfile_Blackwell` 中固定的 Megatron/Bridge、FLA 和 NVRx 依赖构建的训练镜像。所有训练节点都必须能读取模型和数据目录。这些脚本是参考配置；长跑前需要在目标集群验证内存占用和数值行为。

## 训练

在已有 Ray 集群上，从仓库根目录执行命令。`MODEL_DIR` 直接指向原始 HF 模型目录，`DATA_DIR` 指向准备好的数据目录。仍支持原有的 `HF_CHECKPOINT`、`HELLASWAG_DATA_DIR`、`OPENR1MM_DATA_DIR`、`POKEMON_DATA_DIR` 和 `LLAVA_DATA_DIR` 覆盖变量。

```bash
export MODEL_DIR=/shared/models/Kimi-K3
export DATA_DIR=/shared/data/openr1mm
export SAVE_DIR=/shared/checkpoints
export EXP_NAME=kimi-k3-openr1mm-full
DRY_RUN=1 bash examples/models/kimi-k3/scripts/run-kimi-k3-openr1mm-128xb300.sh
bash scripts/entrypoint/ray-job.sh \
  examples/models/kimi-k3/scripts/run-kimi-k3-openr1mm-128xb300.sh
```

| 脚本文件名                                  |      GPU | 输入                                    |
| ------------------------------------------- | -------: | --------------------------------------- |
| `run-kimi-k3-hellaswag-128xb300.sh`         | 128 B300 | `train.jsonl`、`validation.jsonl`       |
| `run-kimi-k3-hellaswag-128xb300-rawtext.sh` | 128 B300 | 同上；原始 context/target 基线对比      |
| `run-kimi-k3-openr1mm-128xb300.sh`          | 128 B300 | `train.parquet`                         |
| `run-kimi-k3-llava-onevision-128xb300.sh`   | 128 B300 | `train/` 下的 JSONL 分片及 `READY.json` |
| `run-kimi-k3-pokemon-64xb300.sh`            |  64 B300 | `pokemon_gpt4o_zh.parquet`              |
| `run-kimi-k3-pokemon-128xb300.sh`           | 128 B300 | 同上                                    |
| `run-kimi-k3-pokemon-192xgpu-b300.sh`       | 192 B300 | 同上                                    |
| `run-kimi-k3-pokemon-lora-64xb300.sh`       |  64 B300 | 同上；语言主干 LoRA                     |

设置 `SAVE_DIR` 后，脚本在固定的 `${SAVE_DIR}/${EXP_NAME}` 目录保存和加载 checkpoint；不设置则不保存。OpenR1-MM 和 OneVision 默认**仅保存模型权重**，不保留 optimizer 和 scheduler 状态。设置 `SAVE_OPTIMIZER=1` 可保存完整状态，但会明显增加主机内存和存储需求。仅模型 checkpoint 不能精确恢复训练状态。

OpenR1-MM 和 OneVision 默认冻结视觉模型；设置 `FREEZE_VISION_TOWER=0` 可联合训练视觉模型，并需重新验证内存。脚本默认向训练 actor 传递 `FLA_TILELANG=0`，支持通过环境变量显式覆盖。ClearML 使用已有运行环境配置。Ray 默认等待任务结束并跟随日志；设置 `RAY_NO_WAIT=1` 可后台提交，此时提交成功不等于训练成功。

## OneVision 数据准备

处理工具仍位于 `scripts/tools/prepare_llavaonevision.py`。

```bash
python -m scripts.tools.prepare_llavaonevision prepare \
  --output-dir /shared/data/onevision --workers 8
export DATA_DIR=/shared/data/onevision/sft
```

准备过程在 manifest 中固定数据集版本，将内嵌图片提取为共享文件，并在所有分片处理完成后发布 `READY.json`。重复执行会复用已完成的分片。`--subsets` 用于小规模验证，生成的是 `SUBSET_READY.json`，不代表完整数据集已就绪。

也可以用 `sample` 子命令从本地缓存的 Parquet 分片中选取固定数量的有效样本，其图片采用内联 data URI。完整数据集建议使用 `prepare`，并确保提取图片的路径在所有训练节点可见。

## 导出

在兼容的训练镜像中连接 Ray 集群后执行转换。`CKPT_PATH` 必须指向单个 `iter_*` 目录，原始 HF 目录提供模型配置、tokenizer 和原生量化布局。

```bash
export CKPT_PATH=/shared/checkpoints/experiment/iter_0000100
python examples/models/kimi-k3/tools/convert_kimi_k3_torch_dist_to_hf_parallel.py \
  --input-dir "${CKPT_PATH}" --origin-hf-dir "${MODEL_DIR}" \
  --output-dir "${CKPT_PATH}_hf" \
  --world-size 16 --tp 4 --pp 1 --ep 16 --expert-tp 1 \
  --cpus-per-worker 16
```

并行导出要求 PP=1、expert-TP=1、EP=world-size。工具验证输出后才发布；输出目录已存在时默认拒绝，除非显式传入 `--replace-output`。

合并语言 LoRA checkpoint 时，必须使用**训练时完全相同的原始模型**：

```bash
python examples/models/kimi-k3/tools/merge_kimi_k3_lora_to_hf.py \
  --input-dir "${CKPT_PATH}" --origin-hf-dir "${MODEL_DIR}" \
  --output-dir "${CKPT_PATH}_hf" --workers 8 --cpus-per-worker 4
```

该工具先合并 LoRA，再进行原生 MXFP4 量化，输出完整 HF 权重。此合并路径不支持视觉 adapter。同目录下的 `compare_kimi_k3_hf_exports.py` 和 `validate_kimi_k3_parallel_export.py` 提供分片比较与量化检查。

## 验证与部署

临时 SGLang 验证工具需要指定日志目录：

```bash
python examples/models/kimi-k3/tools/validate_kimi_k3_sglang.py \
  --model-path "${CKPT_PATH}_hf" --log-dir "${CKPT_PATH}_serve_logs" \
  --tp-size 8 -- \
  --weight-loader-prefetch-checkpoints --context-length 16384 \
  --max-running-requests 16 --cuda-graph-max-bs 16 --disable-decode-cuda-graph
```

内置检查仅覆盖文本。图片评估使用 `scripts/tools/eval_openr1mm.py`；通过 `--help` 查看服务地址、数据和输出参数。

## 边界

- Kimi K3 不支持 MTP、动态 CP、all-gather CP 和 VPP。固定 CP 使用 zigzag 切分，TP 大于一时需要 sequence parallelism。
- 长上下文、视觉联合训练和完整状态 checkpoint 保存需要分别验证容量。这些示例不能作为 256k 上下文已就绪的证明。
- CPU 单测和 Gloo 测试不覆盖 NCCL、多节点 Ray 导出、CUDA 量化或实际图片推理。面向生产合入前，需要在目标镜像和集群完成这些验证。

## 下一步

- [SFT 训练](../../../docs/zh/guide/sft-training.md)
- [模型 Checkpoint 转换](../../../docs/zh/guide/model-conversion.md)
- [LoRA 训练](../../../docs/zh/guide/low-rank-adaptation-training.md)
