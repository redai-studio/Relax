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
