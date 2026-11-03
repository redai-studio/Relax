# Qwen-Image 文生图 SFT

Qwen-Image 通过 `QwenImageSFTAdapter` 接入 Relax 的同步 FSDP2 SFT 训练循环，
复用图像 SFT 的数据游标、优化器更新和 checkpoint。该入口训练 DiT LoRA，
文本编码器和 VAE 冻结并保持 eval 模式；每个样本在线执行文本编码、VAE 编码及后验采样，
随后采样扩散噪声和时间步，计算 flow-matching loss。

默认使用 BF16 计算、FP32 梯度归约和优化器主权重。当前不支持该入口的全参训练或文本编码器 LoRA。

## 数据与启动

准备目标图像及描述的 JSON 数组，图像路径相对于 `DATA_ROOT`：

```json
[
  {"image": "images/001.jpg", "prompt": "A cat sitting beside a window."}
]
```

描述应对应目标图片；文生图数据不接受图像编辑的 `edit_image` 字段。
在已有 Ray 集群上提交，模型、数据和输出路径需要所有 worker 可访问：

```bash
MODEL_PATH=/shared/models/Qwen-Image \
DATA_ROOT=/shared/data/captions METADATA_PATH=/shared/data/captions/train.json \
DATASET_SIZE=128 SAVE_DIR=/shared/runs/qwen-sft EXP_NAME=qwen-r01 SAVE_INTERVAL=32 \
GPUS=8 GLOBAL_BATCH_SIZE=16 MICRO_BATCH_SIZE=1 STEPS=32 \
HEIGHT=512 WIDTH=512 RAY_NO_WAIT=1 RELAX_SKIP_CLUSTER_CLEANUP=1 \
TEXT_ENCODER_PLACEMENT=sharded \
bash scripts/entrypoint/ray-job.sh scripts/training/diffusion/run-qwen-image-sft.sh
```

`DATASET_SIZE` 必须与元数据条数一致。每个 rank 顺序累积自己的样本，`MICRO_BATCH_SIZE`
必须为 1，有效 batch 必须是 GPU 总数的整数倍。默认 LoRA rank/alpha 为 32、学习率 `1e-4`，
warmup 为 200 次优化器更新；短跑可显式设置 `LR_WARMUP_ITERS`。

`HEIGHT` 和 `WIDTH` 必须是 16 的整数倍。预处理依次应用 EXIF 方向、RGB 转换、
保持比例的双线性缩放和指定尺寸的中心裁剪。`RANDOM_CROP=1` 与 `CAPTION_DROPOUT`
可启用逐次随机裁剪和描述丢弃，默认关闭。此入口要求 `IMAGE_PREFETCH_DEPTH=0`，
在训练线程中完成随机预处理，保持随机数调用顺序。

文本输出长度上限为 512；编码时保留模型的完整模板上下文，再裁掉模板前缀并截取输出。
`GRADIENT_CHECKPOINTING` 默认开启；`DETERMINISTIC_ATTENTION=1` 可启用确定性设置。

脚本以 JSON 传递 `--t2i-sft-config`，数据、模型和含空格的路径均按独立命令行参数传递。
`DRY_RUN=1` 可直接运行训练脚本预览提交命令，不访问 Ray 或创建输出目录。
提交日志写入 `RUN_DIR`，默认 `${SAVE_DIR}/${EXP_NAME}/logs`；每次运行使用新的日志目录。

## 冻结文本编码器放置

`TEXT_ENCODER_PLACEMENT` 对应 `t2i_sft_config.text_encoder_placement`：

| 配置值 | 在线训练行为 |
| --- | --- |
| `sharded`（默认） | 文本编码器作为独立 FSDP 组分片，在线编码时聚合所需权重。 |
| `replicated` | 每张 GPU 保留一份完整冻结文本编码器，编码时不执行该编码器的 FSDP all-gather。 |

复制驻留通过原生 FSDP `ignored_params` 实现。适配器负责提前放置这些权重并设置 dtype；
参数仍注册在模型中，不能在训练过程中解冻或改变放置策略。复制驻留增加每卡常驻权重，
应根据实际启动和训练显存峰值选择；它不改变 DiT LoRA 的训练目标。

## 保存、恢复与导出

设置 `SAVE_INTERVAL` 控制保存频率。设置 `LOAD_DIR` 恢复最新已提交的
`t2i_sft/iter_*` checkpoint；`STEPS` 是要达到的总更新次数，而非额外追加步数。
恢复包含优化器、scheduler、各 rank 随机状态和数据游标，并检查数据、模型、batch、LoRA
及训练配置。恢复时使用相同的文本编码器放置策略。`SAVE_CHECKPOINTS=0` 可关闭保存。

现有导出工具按 checkpoint 中记录的适配器导出 Qwen-Image：

```bash
python examples/diffusion/export_i2i_sft.py \
  --checkpoint /shared/runs/qwen-sft/qwen-r01/t2i_sft/iter_0000032 \
  --output /shared/exports/qwen-image-lora
```

输出为 Diffusers 可加载的 transformer LoRA 目录，保留 FP32 主权重，要求 alpha 等于 rank。
继续训练应使用原始 DCP checkpoint；导出目录用于模型加载。

## 可选冻结条件缓存

使用 `scripts/training/diffusion/run-qwen-image-sft-condition-cache.sh`，沿用上面的
数据与训练配置。`RELAX_CONDITION_CACHE=1` 启用冻结文本特征复用，
`RELAX_CONDITION_CACHE_MAX_BYTES` 设置 payload 预算（默认每节点 1 GiB），
每次运行使用新的 `RELAX_CONDITION_CACHE_RUN_DIR`。设置 `RELAX_CONDITION_CACHE=0`
保留在线基线。随机裁剪和 caption dropout 仍逐次执行，缓存键使用其实际输出。

可选 `CACHE_VAE_POSTERIOR=1`，按有效图像与描述联合缓存确定性的 VAE 后验参数和文本特征。
后验采样、扩散噪声及 timestep 仍在线执行；要求关闭 VAE tiling/slicing。缓存记录并在
采样前恢复 posterior 的 contiguous 或 `channels_last_3d` 布局，保持随机数的元素映射。
该选项使冻结 VAE 在每张 GPU 常驻。

在线与缓存对照须保持 `TEXT_ENCODER_PLACEMENT`、`CACHE_VAE_POSTERIOR`、checkpointing
和 FSDP reshard 配置相同，不将放置策略变化计为缓存收益。分片编码器需要全组一致命中，
后端确认完整冻结副本后才可按 rank 独立决定。恢复 checkpoint 时保持相同的放置和
posterior 策略；基础在线入口缺省的 posterior 标志与 `False` 等价，恢复不要求已有热缓存。

所有权、容量、失效、数值验证及完整训练循环计时见[冻结条件缓存](./frozen-conditioning-cache.md)。
