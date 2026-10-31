# Keye-VL 2.0 RL

[English](./README.en.md)

这里提供 `Kwai-Keye/Keye-VL-2.0-30B-A3B` 的纯文本和图像 RL 示例。模型采用约 30B 总参数、约 3B 激活参数的 MoE 语言主干，包含 48 层、128 个专家、每 token 选择 8 个专家，并使用 top-2048 稀疏注意力 indexer。图像路径包含原生 Keye 视觉塔、投影层和三轴 mRoPE。

训练使用 Megatron/Bridge，rollout 使用 SGLang **0.5.21**，8 张 GPU 共卡运行。模型适配代码位于 [relax/models/keye_vl2](../../../relax/models/keye_vl2)。请使用与仓库匹配、包含 Megatron/Bridge、Transformer Engine 和 `fast_hadamard_transform` 的训练环境；仅安装 SGLang 版本号相同不足以替代完整训练环境。

## 脚本

| 文件                                                                                       | 用途                                         | 训练并行                     | 默认 prompt × 采样数 |
| ------------------------------------------------------------------------------------------ | -------------------------------------------- | ---------------------------- | -------------------- |
| [run-keye-vl2-30B-A3B-8xgpu-math.sh](./scripts/run-keye-vl2-30B-A3B-8xgpu-math.sh)         | DAPO Math，AIME-2024 评估；每 20 步保存      | TP4 / SP / PP1 / CP1 / EP8   | 16 × 8               |
| [run-keye-vl2-30B-A3B-8xgpu-openr1mm.sh](./scripts/run-keye-vl2-30B-A3B-8xgpu-openr1mm.sh) | OpenR1-MM 图像 RL；视觉塔和投影层参与训练    | TP1 / PP1 / CP2 zigzag / EP8 | 4 × 2                |
| [keye-vl2-30B-A3B.sh](./scripts/keye-vl2-30B-A3B.sh)                                       | 复用 Qwen3-MoE 尺寸配置，设置 Keye RoPE base | —                            | —                    |

纯文本训练使用 TP4 和 sequence parallel；图像训练仍要求 TP1、PP1。不支持动态 CP、contiguous CP、视频和音频。图像训练在注入完整图像特征后按逐样本 zigzag 布局切分 CP，跨 rank 汇聚 K/V 和 indexer K。文本 rollout 默认使用 4 个 TP2 SGLang engine；`SGLANG_TP` 可调整每个 engine 的卡数。图像脚本保持 1 个 TP8 engine，参数直接写在启动脚本中。

## 模型与数据准备

以下命令均在仓库根目录执行。`MODEL_DIR`、`DATA_DIR` 是根目录，而非具体模型目录或数据文件；`SAVE_DIR` 是运行输出根目录。

```bash
export MODEL_DIR=/path/to/models
export DATA_DIR=/path/to/data
export SAVE_DIR=/path/to/runs

hf download Kwai-Keye/Keye-VL-2.0-30B-A3B \
  --local-dir "${MODEL_DIR}/Keye-VL-2.0-30B-A3B"
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir "${DATA_DIR}/dapo-math-17k"
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir "${DATA_DIR}/aime-2024"
python scripts/tools/process_aime.py \
  --input "${DATA_DIR}/aime-2024/aime-2024.jsonl"
```

OpenR1-MM 另需下载并转换图片数据，转换结果包含 `prompt`、`image` 和 `label`：

```bash
hf download --repo-type dataset lmms-lab/multimodal-open-r1-8k-verified \
  --local-dir "${DATA_DIR}/multimodal-open-r1-8k-verified"
python scripts/tools/process_openr1.py \
  --input-dir "${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001.parquet" \
  --output-dir "${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet"
```

模型目录需保留完整权重、tokenizer、processor 和自定义 Python 文件。已有上述文件时可跳过下载和转换。

## 快速运行

本机直接运行脚本会调用 `scripts/entrypoint/local.sh` 初始化训练环境。该入口会清理本机已有的 Ray/Python 训练进程，请使用独占的 8 卡节点。

先做两轮短链路检查，再启动长跑：

```bash
NUM_ROLLOUT=2 ROLLOUT_BATCH_SIZE=8 N_SAMPLES=2 GLOBAL_BATCH_SIZE=16 RESPONSE_LEN=512 \
  EXP_NAME=keye-dapo-smoke \
  bash examples/models/keye-vl2/scripts/run-keye-vl2-30B-A3B-8xgpu-math.sh

NUM_ROLLOUT=200 EXP_NAME=keye-dapo-200 \
  bash examples/models/keye-vl2/scripts/run-keye-vl2-30B-A3B-8xgpu-math.sh
```

文本脚本每轮采样 `16 × 8 = 128` 条回复，全局 batch 默认为 128（可用 `GLOBAL_BATCH_SIZE` 覆盖），即每轮一次优化更新。其余训练效果参数对齐 `scripts/training/text/run-qwen3-4B-8xgpu.sh`：响应上限 8192，不启用 dynamic filter，也不额外限制 prompt/context 长度。每 20 轮进行 AIME 评估，每题采样 8 次，评估响应上限 16384、top-p 为 0.7。使用 GRPO 组内优势、非对称裁剪、TIS、`low_var_kl` KL loss（系数 0）和恒定学习率 `1e-6`。每卡训练 token 上限 32768 和 CPU optimizer offload 保持 Keye 的性能配置。文本 decode 使用融合 RoPE、保留 BF16 舍入的分页 indexer 和 SGLang 原生分页 GQA kernel，仅访问 top-2048 KV，不展开 KV heads。decode 开启 CUDA graph，prefill 保持 eager；overlap scheduling 和 radix cache 仍关闭。图像脚本目前保持 eager 调度。

训练主 attention 关闭 TE RoPE 融合，以保留与 rollout 一致的 BF16 逐次舍入；两侧 MoE router 用 FP64 点积生成 FP32 logits，并固定 top-k 顺序；RMSNorm 使用逐行固定归约，残差相加先舍入到 BF16。MoE 激活、路由权重和专家输出合并保留训练侧的舍入顺序。训练和 rollout 共用稀疏 GQA forward，训练 backward 重算标准 masked SDPA 梯度；无需启用 router replay。 TP>1 推理保留完整的 attention 输出投影和专家 down 权重，汇聚输入后按完整 K 维计算，避免局部 BF16 舍入；QKV 和 gate/up 权重仍按 TP 分片。

训练 attention 按每个 query 分块的最后位置截断未来 KV，默认每块 1024 个 query；FP32 teacher 保留 GQA 分组，避免反复展开和转换 KV。Indexer 辅助 KL、完整层重计算和 CPU optimizer offload 仍保留。CP 分块在 zigzag 两半边界处拆开，因果关系使用原始 token 顺序，不使用压缩后的图像坐标。

文本脚本默认 `SAVE_CHECKPOINT=1`：每 `SAVE_INTERVAL` 轮保存一次（默认 20），每轮一次优化更新，即每 20 step 保存完整 checkpoint，保留最近两份。加载和保存均使用稳定的 `${SAVE_DIR}/${EXP_NAME}`；恢复时保持这两个变量不变，新实验更换 `EXP_NAME`。设置 `SAVE_CHECKPOINT=0` 可关闭保存，此时重启只能从 HF 初始权重开始。

图像 RL 的短链路检查：

```bash
NUM_ROLLOUT=2 RESPONSE_LEN=512 SAVE_INTERVAL=1 EXP_NAME=keye-openr1mm-smoke \
  bash examples/models/keye-vl2/scripts/run-keye-vl2-30B-A3B-8xgpu-openr1mm.sh
```

图像脚本默认 `SAVE_CHECKPOINT=1`，响应上限 4096，200 轮，每 20 轮保存一次，保留最近两份完整 checkpoint，保存和加载均使用稳定的 `${SAVE_DIR}/${EXP_NAME}`。保持这两个变量不变可恢复训练；新实验请换 `EXP_NAME`。视觉特征缓存已关闭，以便每轮使用更新后的视觉权重。

512-token 检查用于确认生成、奖励计算、反向和权重同步链路。截断可能使整个采样组奖励为零，此时不能据此判断 RL 效果；正式训练应保留更长的响应预算，并观察非零奖励、组内奖励差异和截断比例。

## 已有 Ray 集群与日志

复用集群时，设置实际的 Ray Jobs HTTP 地址，通过统一入口提交：

```bash
export RAY_ADDRESS=http://127.0.0.1:8265
RELAX_SKIP_CLUSTER_CLEANUP=1 WORKING_DIR=./ \
  bash scripts/entrypoint/ray-job.sh \
  examples/models/keye-vl2/scripts/run-keye-vl2-30B-A3B-8xgpu-math.sh
```

默认提交会等待任务完成并跟随日志。需要后台提交时显式设置 `RAY_NO_WAIT=1`，随后用 `ray job logs --address="${RAY_ADDRESS}" <job-id>` 查看训练输出。启动脚本的提交日志在本示例的 `scripts/log/`；后台提交时该文件只包含提交阶段输出。

两个脚本默认启用 ClearML 和 metrics service，使用已有运行环境的凭证。`PROJECT_NAME` 控制项目，`EXP_NAME` 控制实验名称前缀；每次启动建立新的 ClearML task，控制台会打印链接。
