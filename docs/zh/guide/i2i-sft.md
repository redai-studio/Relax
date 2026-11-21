# 图像编辑 SFT

图像编辑监督训练使用 Relax 的标准 `train` 入口、Controller、Actor、RayTrainGroup
和 FSDP2。CPU SFT producer 向 TransferQueue 写入样本索引，训练 worker 读取图片；
head 无需挂载模型和原始图片。模型目录、metadata 和图片必须在所有训练 worker 上可见，
checkpoint 输出目录必须跨 worker 共享。

当前模型适配为 SenseNova U1.5 MoT Edit，支持全参或 LoRA、单张 source 条件图，
以及按顺序提供 source 和 glyph 两张条件图。target 图片是监督目标，不计入条件图数量。
1024² 是像素预算，图片保持长宽比并对齐到模型 patch 大小。

## 数据

`--prompt-data` 指向 JSON 数组。路径相对于 `data_root`，也可以使用绝对路径：

```json
[
  {
    "image": "target/001.png",
    "edit_image": ["source/001.png", "glyph/001.png"],
    "prompt": "替换图片中的文字"
  }
]
```

无 glyph 时仅保留 `edit_image` 的第一项，设置 `CONDITIONING_IMAGES=1`。
`DATASET_SIZE` 必须等于 metadata 实际行数；worker 会检查数据哈希在所有 rank 一致。

## 启动

在已经准备好的 Ray 集群中，通过项目统一脚本提交。将示例路径替换为训练 worker 可访问的实际路径：

```bash
MODEL_PATH=/shared/models/SenseNova \
DATA_ROOT=/shared/data \
METADATA_PATH=/shared/data/train.json \
DATASET_SIZE=64 \
SAVE_DIR=/shared/runs/i2i-smoke \
MODE=lora CONDITIONING_IMAGES=1 STEPS=5 RAY_NO_WAIT=1 \
bash scripts/entrypoint/ray-job.sh scripts/training/diffusion/run-sensenova-i2i-sft-8xgpu.sh
```

`MODE=full CONDITIONING_IMAGES=2` 选择全参加 glyph。示例使用 BF16 计算、FP32
梯度归约与 optimizer master，线性 warmup 200 个 optimizer steps，之后保持固定学习率。
默认 LoRA rank/alpha 为 32、学习率 1e-4；全参文字替换学习率为 1e-6。

## 冻结条件缓存

LoRA 训练时，编码条件图和条件前缀的模型部分保持冻结，同一条 prompt 和条件图在每个 epoch
都会得到相同的前缀特征。条件缓存把这些特征（各层前缀 K/V 与 `indexes_image`）保存在每张训练 GPU 上；
样本再次出现时，该 micro-batch 直接复用，跳过条件图嵌入和前缀编码。缓存默认关闭，
fixed-tiers 示例脚本通过环境变量开启：

```bash
MODEL_PATH=/shared/models/SenseNova \
DATA_ROOT=/shared/data \
METADATA_PATH=/shared/data/train.json \
DATASET_SIZE=64 \
SAVE_DIR=/shared/runs/i2i-cache \
MODE=lora CONDITIONING_IMAGES=1 STEPS=24 \
CONDITION_CACHE_ENABLED=1 CONDITION_CACHE_MAX_BYTES=8589934592 \
bash scripts/entrypoint/ray-job.sh scripts/training/diffusion/run-sensenova-i2i-sft-fixed-tiers-8xgpu.sh
```

默认 8 卡、全局 batch 为 8 时，64 个样本每个 epoch 训练 8 步，24 步共训练 3 个 epoch，
样本从第 2 个 epoch 开始重复出现。未设置 `FIXED_PIXELS_TIERS` 时，该脚本与上面的脚本一样按
`MAX_PIXELS` 缩放图片。脚本把这些设置写入 `--i2i-sft-config`，其他启动方式可以直接设置相同的键：

| 环境变量 | `i2i_sft_config` 键 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `CONDITION_CACHE_ENABLED` | `condition_cache_enabled` | `0`（`false`） | 设为 `1` 开启缓存 |
| `CONDITION_CACHE_MAX_ENTRIES` | `condition_cache_max_entries` | `256` | 每个 rank 最多缓存的条件数量 |
| `CONDITION_CACHE_MAX_BYTES` | `condition_cache_max_bytes` | `1073741824`（1 GiB） | 每个 rank 缓存 Tensor 存储的字节上限 |
| `BASELINE_STATS_DIR` | `baseline_stats_dir` | 未设置 | 保存各 rank 编码统计的目录 |
| `BASELINE_WARMUP_STEPS` | `baseline_warmup_steps` | `5` | 计时汇总时排除的开头步数 |

两个上限分别作用于每个 rank，缓存的 Tensor 与训练共用 GPU 显存，每个条目按其持有的 Tensor
存储字节计费。新条件会超出任一上限时，最久未使用的条目先被淘汰；单个条件就超过
`CONDITION_CACHE_MAX_BYTES` 时不写入缓存，该 micro-batch 照常编码。

只有所有 rank 都在缓存中找到各自的条件时，该 micro-batch 才复用缓存；任一 rank 未命中时，
所有 rank 都在线编码，使各 rank 的 FSDP 通信顺序保持相同。因此第 1 个 epoch 不会复用；
每个 rank 遇到的不同条件多于预算能保存的数量，或需要同时命中的 rank 增多时，复用比例都会下降。
缓存键包含编码器版本、解码后的 RGB 像素及其顺序、prompt、预处理与目标尺寸、tokenizer、dtype
和 attention 实现，因此只有输入和编码器权重都相同时才会复用。

全参训练（`MODE=full`）会更新编码器，即使开启缓存，条件也始终在线编码。加载 checkpoint
会清空缓存。缓存内容不写入 checkpoint，恢复训练后缓存从空开始；恢复时可以修改缓存与统计设置。

训练结束时，每个 rank 输出一行以 `i2i condition cache rank=<rank> stats=` 开头的日志。
`effective_reuses` 是复用的 micro-batch 数，`local_hits` 是本 rank 命中的查询数，
`group_forced_online` 是本 rank 命中、但因其他 rank 未命中而在线编码的次数。
`evictions` 较多且 `peak_charged_bytes` 接近上限时，可以增大 `CONDITION_CACHE_MAX_BYTES`。

估算预算时，关闭缓存运行一个短任务，并将 `BASELINE_STATS_DIR` 设为所有 worker 共享的目录。
每个 rank 在其中写入 `rank_<rank>_steps.jsonl` 和 `rank_<rank>_summary.json`，任务结束后合并：

```bash
python -m relax.engine.sft.condition_tracker /shared/runs/i2i-baseline
```

命令生成 `baseline_summary.json`。其中 `feature_tensor_bytes.past_key_values_bytes` 与
`feature_tensor_bytes.indexes_image_bytes` 给出一个缓存条件的大小，
`stage_timings_ms.ref_vision_encode_ms` 和 `stage_timings_ms.prefix_encode_ms` 的 `share_pct`
表示缓存可以跳过单步训练中多少比例的时间。统计使用 CUDA event 计时，并在每步结束时等待事件完成，
常规训练不要设置 `BASELINE_STATS_DIR`。

## 恢复与约束

设置 `LOAD_DIR` 指向之前的输出目录，`STEPS` 是恢复后要求达到的总步数。
恢复最新已提交的 `i2i_sft/iter_*`，同时加载 optimizer、scheduler、各 rank RNG 和已完成步数。
Controller 用 Actor 恢复的步数重设 producer，因此预取不会改变恢复后的样本顺序。
数据、卡数、batch、LoRA 或 optimizer 配置变化会拒绝精确恢复。

每个 optimizer step 消费一个全局 batch；micro batch 为 1，全局 batch 必须是卡数的整数倍。
每个 epoch 的末尾重复开头样本补齐全局 batch。只支持同步纯 FSDP 数据并行；
文本 packing、文本 PPL/predict、RL rollout/reward 和 TP/PP/CP 不参与此路径。

SenseNova 适配包含原生模型及预处理，不依赖 DiffSynth 运行时。
改编源码的文件头保留上游版权和来源说明，Apache-2.0 许可证见仓库根目录 `LICENSE`。
模型权重与 tokenizer 由用户另行提供，仍遵循各自的许可证。

## 导出

在独立 CPU 进程中导出已提交 checkpoint：

```bash
python examples/diffusion/export_i2i_sft.py \
  --checkpoint /shared/runs/i2i-smoke/i2i_sft/iter_0000005 \
  --output /shared/exports/adapter.safetensors
```

LoRA 输出使用 DiT 相对参数名及 BF16 权重，目前要求 alpha 等于 rank。
全参模式的 `--output` 指向一个新目录，导出 BF16 模型分片并复制原模型 tokenizer 配置。
导出不包含 optimizer 状态；继续训练使用原始 DCP checkpoint。
