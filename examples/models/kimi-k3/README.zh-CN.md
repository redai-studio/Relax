# Kimi K3 训练与导出

[English](./README.md)

## 概述

Relax 支持 Kimi K3 的全参和 LoRA SFT，包括图片输入、packing、固定 CP、PP 和 EP。模型配置位于 `scripts/models/kimi-k3.sh`，公开示例位于 `examples/models/kimi-k3/scripts/`。

请使用按照 `docker/Dockerfile` 或 `docker/Dockerfile.cu13` 中固定的 Megatron/Bridge、FLA 和 NVRx 依赖构建的训练镜像。所有训练节点都必须能读取模型和数据目录。这些脚本是参考配置；长跑前需要在目标集群验证内存占用和数值行为。

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

## 强化学习

Kimi K3 支持共卡 GRPO，rollout 使用原生 MXFP4 权重。训练保留 BF16 专家参数；每次更新后，Bridge 将路由专家转换为成对的 `weight_packed` / E8M0 `weight_scale` 张量。Dense、共享专家和视觉权重保留配置要求的未量化格式。量化配置同时支持顶层和 `text_config` 嵌套结构。

**所有 rollout 节点**都必须安装仓库提供的 SGLang v0.5.17 patch。重载钩子在更新 MXFP4 重打包、融合 decode buffer、MLA 投影和 AttnRes 缓存时保持已捕获的运行时存储地址。Router 支持注销和重新注册 worker，旧请求和健康检查不会影响同 URL 的新 worker。PP 权重转换广播清理后的配置副本，避免权重已收集后再次执行 TP/EP 集合通信。

| 脚本                                                                    | 布局                                 | 用途                                                    |
| ----------------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------- |
| `scripts/training/text/run-kimi-k3-5l-8xgpu-grpo.sh`                    | 训练 TP2/EP4/ETP1，rollout TP8；8 卡 | 5 层/128 专家减配 checkpoint，纯文本 DAPO math 冒烟验证 |
| `examples/models/kimi-k3/scripts/run-kimi-k3-openr1mm-128xb300-grpo.sh` | 训练 TP4/PP4/CP2/EP32/ETP1；128 B300 | 包含视觉联合训练的全参多模 OpenR1-MM GRPO               |

全参脚本使用 PP 层数 21/24/24/24、8 个各占 16 卡的 rollout engine（attention TP8/DP2、MoE EP16）、CPU optimizer offload（默认比例 0.9），通过 dummy 初始化 rollout 后完整推送权重，不加载参考模型。它清空继承的 allocator 设置、限制编译线程，并在权重同步时保持 Ray Serve 探针可响应。默认训练 200 个 rollout，每 200 个 rollout 保存一次 model-only checkpoint，只保留一份。未设置 `LOAD_DIR` 时从 HF 初始化；仅模型 checkpoint **不能恢复 optimizer 状态**。

```bash
export MODEL_DIR=/shared/models/Kimi-K3
export DATA_DIR=/shared/data/openr1mm
export SAVE_DIR=/shared/checkpoints
export EXP_NAME=kimi-k3-openr1mm-grpo
export HF_CHECKPOINT="${MODEL_DIR}"
export PROMPT_SET="${DATA_DIR}/train.parquet"
DRY_RUN=1 bash examples/models/kimi-k3/scripts/run-kimi-k3-openr1mm-128xb300-grpo.sh
bash scripts/entrypoint/ray-job.sh \
  examples/models/kimi-k3/scripts/run-kimi-k3-openr1mm-128xb300-grpo.sh
```

全参脚本默认从 `${MODEL_DIR}/Kimi-K3` 读取模型，从 `${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet` 读取数据，字段为 `prompt`、`label` 和 `image`。上例显式设置 `HF_CHECKPOINT` 和 `PROMPT_SET`，以复用 SFT 的目录布局。减配脚本读取 `DATA_DIR` 下的 `dapo-math-17k.jsonl`，`MODEL_DIR` 直接指向减配 checkpoint。其 entropy 系数 0.01 用于在未经重训的冒烟模型 reward 为零时保持梯度，并不构成效果基线。减配脚本可选设置 `SAVE_DIR` 启用完整状态保存和恢复。实验名与 checkpoint 路径固定，时间戳仅用于日志和任务名。ClearML 沿用现有运行环境配置。

全参脚本默认开启按专家路由的权重更新，设置 `COLOCATE_EXPERT_WEIGHT_ROUTING=0` 可使用广播。动态采样过滤默认关闭，可通过显式 CLI 参数开启。

两个脚本都通过 `--train-env-vars` 启用 `OPEN_TRAINING_MXFP4_FAKE_QAT_FLAG=1` 和 `FLA_TILELANG=0`。Fake QAT 使用直通梯度估计：路由专家前向使用与在线 MXFP4 导出相同的量化/反量化网格，梯度回到 BF16 master。其他启动方式默认不启用此钩子。Rollout 每张图片只发送一个原始媒体 token；训练 processor 保留原始 prompt，并独立展开图像特征占位。

### 有界同步 Checkpoint 保存

两个 Dockerfile 都在主 Megatron patch 后应用 `docker/patch/megatron/sync-save-bounded-staging.patch`。全参 GRPO 脚本显式启用；设置 `MEGATRON_SYNC_SAVE_BOUNDED_STAGING=0` 可让该脚本回到原同步 writer。其他启动方式默认关闭此优化。异步保存行为不变。

- `MEGATRON_SYNC_SAVE_BOUNDED_STAGING=1`：分批暂存并顺序写入，避免将整个 checkpoint 预载到 CPU。
- `MEGATRON_SYNC_SAVE_STAGE_BYTES`：每个 rank 的活跃暂存预算，默认 1 GiB。窗口为空时允许一个超过预算的大张量，因此边界为 `max(budget, largest_tensor)`。
- 预算 `0` 表示逐张量暂存和写入，**不表示**切回原 writer。
- 预算只约束活跃暂存张量，不约束总 RSS、optimizer/offload 内存、序列化临时空间或文件系统页缓存。降低暂存内存可能牺牲写入吞吐，需在目标存储上测量。

脚本会把两个环境变量传递到训练 actor。旧镜像没有此 patch 时，即使设置变量，也仍使用原保存实现；要获得优化，需要统一重建镜像或给所有节点安装 patch。已运行的任务不会自动加载新 patch。Checkpoint 格式和加载方式不变。异常处理与验证边界见 `docker/patch/megatron/sync-save-bounded-staging.md`。

### 训推差异诊断

`examples/models/kimi-k3/tools/probe_mxfp4_mismatch_floor.py` 在相同 token IDs 和相同 SGLang 引擎下，对比原生 MXFP4 与 BF16 反量化副本的 prompt logprob。提供 `--mxfp4-dir`、`--bf16-dir`、`--data`，可选 `--n-prompts`。该工具使用 TP1，因此被测 checkpoint 必须能放入单卡。这是诊断参照，不能独立证明在线推权正确：还需在目标镜像中检查重载存储地址、loss/梯度有限值及多次更新。CPU 测试不构成 CUDA Graph、量化 kernel 或多机保存恢复的正确性证明。

## 边界

- Kimi K3 不支持 MTP、动态 CP、all-gather CP 和 VPP。固定 CP 使用 zigzag 切分，TP 大于一时需要 sequence parallelism。
- 长上下文、视觉联合训练和完整状态 checkpoint 保存需要分别验证容量。这些示例不能作为 256k 上下文已就绪的证明。
- CPU 单测和 Gloo 测试不覆盖 NCCL、多节点 Ray 导出、CUDA 量化或实际图片推理。面向生产合入前，需要在目标镜像和集群完成这些验证。

## 下一步

- [SFT 训练](../../../docs/zh/guide/sft-training.md)
- [模型 Checkpoint 转换](../../../docs/zh/guide/model-conversion.md)
- [LoRA 训练](../../../docs/zh/guide/low-rank-adaptation-training.md)
