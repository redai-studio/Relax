# 图像编辑 SFT

图像编辑监督训练使用 Relax 的标准 `train` 入口、Controller、Actor、RayTrainGroup
和 FSDP2。CPU SFT producer 向 TransferQueue 写入样本索引，训练 worker 读取图片；
head 无需挂载模型和原始图片。模型目录、metadata 和图片必须在所有训练 worker 上可见，
checkpoint 输出目录必须跨 worker 共享。

图像编辑 SFT 路径支持两种模型适配，均提供全参数或 LoRA 训练：

- **SenseNova U1.5 MoT Edit**（默认，`relax.models.sensenova.adapter.SenseNovaSFTAdapter`）
- **Qwen-Image-Edit**（通过 `--model-adapter-path relax.models.qwen_image.sft_adapter.QwenImageEditSFTAdapter` 指定）

输入支持单张条件图或两张条件图（SenseNova 按顺序提供 source 和 glyph）。
target 图片是监督目标，不计入条件图数量。1024² 是像素预算，图片保持长宽比并按模型 patch 大小取整。

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

`MODE=full CONDITIONING_IMAGES=2` 选择全参数加 glyph。示例使用 BF16 计算、FP32
梯度归约与 optimizer master，线性 warmup 200 个 optimizer steps，之后保持固定学习率。
默认 LoRA rank/alpha 为 32、学习率 1e-4；全参数文字替换学习率为 1e-6。

训练 Qwen-Image-Edit（例如 diffusers 格式的 `Qwen/Qwen-Image-Edit-2509`，目录包含
`transformer`、`text_encoder`、`vae` 与 `processor`）时，在 `relax.entrypoints.train`
中指定 `--model-adapter-path relax.models.qwen_image.sft_adapter.QwenImageEditSFTAdapter`。
该适配类在 LoRA 与全参数模式下均保持 `text_encoder` 和 `vae` 冻结并处于 `eval()` 状态，
在 `--i2i-sft-config` 中通过 `transformer_gradient_checkpointing`（布尔值，默认 `true`）
控制 DiT 激活检查点；SenseNova 对应配置项为 `prefix_gradient_checkpointing`（布尔值，默认 `true`）。

## 冻结条件缓存

当条件编码器在训练期间保持冻结时，输入和预处理完全相同的重复样本在后续 epoch 会产生一致的条件特征。
两种 SFT 适配共用公共有界缓存 `ConditionCache` 与 FSDP 协调层 `ConditionCacheBinding`：

- **SenseNova U1.5 MoT Edit**（LoRA 模式）：缓存各层前缀 K/V 与 `indexes_image`，跳过参考图视觉编码与前缀编码。全参数训练（`MODE=full`）会更新编码器，因此即使开启缓存，SenseNova 也始终在线编码。
- **Qwen-Image-Edit**（LoRA 与全参数模式）：缓存 `prompt_embeds`、`edit_latents` 与 `target_latents`（VAE 使用确定性的 `latent_dist.mode()` 编码），跳过 `text_encoder` 和 `vae` 编码。两种训练模式下 `text_encoder` 和 `vae` 均保持冻结与 `eval()`，均可启用缓存。

缓存默认关闭。SenseNova 的 fixed-tiers 示例脚本通过环境变量开启：

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
`MAX_PIXELS` 缩放图片。脚本把这些设置写入 `--i2i-sft-config`，Qwen-Image-Edit 及其他启动方式可以直接在 `--i2i-sft-config` 中设置相同的键：

| 环境变量 | `i2i_sft_config` 键 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `CONDITION_CACHE_ENABLED` | `condition_cache_enabled` | `0`（`false`） | 设为 `1` 开启缓存 |
| `CONDITION_CACHE_MAX_ENTRIES` | `condition_cache_max_entries` | `256` | 每个 rank 最多缓存的条件数量 |
| `CONDITION_CACHE_MAX_BYTES` | `condition_cache_max_bytes` | `1073741824`（1 GiB） | 每个 rank 缓存 Tensor 存储的字节上限 |
| `BASELINE_STATS_DIR` | `baseline_stats_dir` | 未设置 | 保存各 rank 编码统计的目录 |
| `BASELINE_WARMUP_STEPS` | `baseline_warmup_steps` | `5` | 计时汇总时排除的开头步数 |

两个上限分别作用于每个 rank，缓存的 Tensor 与训练共用 GPU 显存，每个条目按其持有的独立 Tensor
存储字节计费。新条件会超出任一上限时，最久未使用的条目先被淘汰；单个条件就超过
`CONDITION_CACHE_MAX_BYTES` 时不写入缓存，该 micro-batch 照常编码。

只有所有 rank 都在相同编码器代数下命中缓存时，该 micro-batch 才复用缓存；任一 rank 未命中时，
所有 rank 都在线编码，使各 rank 的 FSDP 通信顺序保持相同。因此第 1 个 epoch 不会复用；
每个 rank 遇到的不同条件多于预算能保存的数量，或需要同时命中的 rank 增多时，复用比例都会下降。
缓存键包含编码器版本、解码后的 RGB 像素及其顺序、prompt、预处理与目标尺寸、tokenizer 或
processor 配置及实际输入摘要、dtype 和 attention 实现，因此只有输入和编码器权重都相同时才会复用。

加载 checkpoint 会清空缓存并推进编码器代数。缓存内容不写入 checkpoint，恢复训练后缓存从空开始；
恢复时可以修改缓存与统计设置。

训练结束时，每个 rank 输出一行以 `i2i condition cache rank=<rank> stats=` 开头的日志。
`effective_reuses` 是复用的 micro-batch 数，`local_hits` 是本 rank 命中的查询数，
`group_forced_online` 是本 rank 命中、但因其他 rank 未命中而在线编码的次数。
`evictions` 较多且 `peak_charged_bytes` 接近上限时，可以增大 `CONDITION_CACHE_MAX_BYTES`。

估算预算时，关闭缓存运行一个短任务，并将 `BASELINE_STATS_DIR` 设为所有 worker 共享的目录。
每个 rank 在其中写入 `rank_<rank>_steps.jsonl` 和 `rank_<rank>_summary.json`，任务结束后合并：

```bash
python -m relax.engine.sft.condition_tracker /shared/runs/i2i-baseline
```

命令按文件记录的 `tracker_schema` 生成 `baseline_summary.json`。其中 `feature_tensor_bytes`
给出单个样本缓存特征的大小（SenseNova 为 `past_key_values_bytes` 与 `indexes_image_bytes`；
Qwen-Image-Edit 为 `prompt_embeds_bytes`、`edit_latents_bytes` 与 `target_latents_bytes`），
`stage_timings_ms.<stage>.share_pct` 表示缓存可以跳过单步训练中多少比例的时间（SenseNova 为
`ref_vision_encode_ms` 与 `prefix_encode_ms`；Qwen-Image-Edit 为 `prompt_encode_ms`、
`ref_vae_encode_ms` 与 `target_vae_encode_ms`）。统计使用 CUDA event 计时并在每步结束时等待事件完成，
常规训练不要设置 `BASELINE_STATS_DIR`。

### RL 未来接入方式

公共缓存核心（`relax/engine/sft/condition_cache.py` 中的 `ConditionCache`、`ConditionFieldSpec`、
`ConditionRequest` 与 `condition_key`）不依赖具体模型，也不持有 FSDP process group。
后续扩展到多模生成 RL（例如围绕同一 prompt 和参考图的多候选 GRPO 训练）时，可按以下方式接入：

1. 在 RL 模型适配层实现 `ConditionCacheAdapter`，仅声明冻结且确定性的条件特征（如冻结文本编码与参考图特征），各候选独立的噪声、去噪轨迹、log-prob 与奖励不进入缓存。
2. 在训练或 rollout worker 进程内绑定本地 `ConditionCache`，并在同一参数分片通信组执行前向回放之前协调各 rank 的复用决定。
3. 将权重同步版本或 adapter 切换标识写入 `version_tags`，在权重同步边界调用 `bump_encoder_version` 触发失效，保证依赖变化后不再读取旧特征。

## 恢复与约束

设置 `LOAD_DIR` 指向之前的输出目录，`STEPS` 是恢复后要求达到的总步数。
恢复最新已提交的 `i2i_sft/iter_*`，同时加载 optimizer、scheduler、各 rank RNG 和已完成步数。
Controller 用 Actor 恢复的步数重设 producer，因此预取不会改变恢复后的样本顺序。
数据、卡数、batch、LoRA 或 optimizer 配置变化会拒绝精确恢复。

每个 optimizer step 消费一个全局 batch；micro batch 为 1，全局 batch 必须是卡数的整数倍。
每个 epoch 的末尾重复开头样本补齐全局 batch。只支持同步纯 FSDP 数据并行；
文本 packing、文本 PPL/predict、RL rollout/reward 和 TP/PP/CP 不参与此路径。

SenseNova 与 Qwen-Image-Edit 适配均不依赖 DiffSynth 运行时。
改编源码的文件头保留上游版权和来源说明，Apache-2.0 许可证见仓库根目录 `LICENSE`。
模型权重与 tokenizer 由用户另行提供，仍遵循各自的许可证。

## 导出

在独立 CPU 进程中导出已提交的 SenseNova checkpoint：

```bash
python examples/diffusion/export_i2i_sft.py \
  --checkpoint /shared/runs/i2i-smoke/i2i_sft/iter_0000005 \
  --output /shared/exports/adapter.safetensors
```

LoRA 输出使用 DiT 相对参数名及 BF16 权重，目前要求 alpha 等于 rank。
全参模式的 `--output` 指向一个新目录，导出 BF16 模型分片并复制原模型 tokenizer 配置。
导出不包含 optimizer 状态；继续训练使用原始 DCP checkpoint。
