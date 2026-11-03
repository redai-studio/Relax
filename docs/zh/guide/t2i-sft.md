# 文生图 SFT 与冻结条件缓存

Qwen-Image 通过 `QwenImageSFTAdapter` 接入 Relax 的同步 FSDP2 SFT 训练循环。
它与 SenseNova 共用数据游标、优化器更新、checkpoint、条件缓存和数值验证器。
现有 `QwenImageAdapter` 是 RL 适配器，不能替代这个 SFT 入口。

当前支持 Qwen-Image DiT LoRA。文本编码器和 VAE 冻结，BF16 计算，FP32
梯度归约和优化器主权重；不支持此入口的全参训练或文本编码器 LoRA。

## 数据与启动

准备图像及描述的 JSON 数组，路径相对于 `DATA_ROOT`。描述应对应目标图片，
不应把图像编辑指令直接当成文生图描述。该路径不接受 `edit_image` 字段。

```json
[
  {"image": "images/001.jpg", "prompt": "A cat sitting beside a window."}
]
```

在现有 Ray 集群上提交，路径替换为 worker 可访问的位置：

```bash
MODEL_PATH=/shared/models/Qwen-Image \
DATA_ROOT=/shared/data/captions METADATA_PATH=/shared/data/captions/train.json \
DATASET_SIZE=128 SAVE_DIR=/shared/runs/qwen-sft EXP_NAME=qwen-r01 SAVE_INTERVAL=32 \
GPUS=8 GLOBAL_BATCH_SIZE=16 MICRO_BATCH_SIZE=1 STEPS=32 \
HEIGHT=512 WIDTH=512 RAY_NO_WAIT=1 RELAX_SKIP_CLUSTER_CLEANUP=1 \
TEXT_ENCODER_PLACEMENT=sharded \
RELAX_CONDITION_CACHE=1 RELAX_CONDITION_CACHE_MAX_BYTES=1073741824 \
RELAX_CONDITION_CACHE_RUN_DIR=/shared/runs/qwen-sft/cached-r01 \
bash scripts/entrypoint/ray-job.sh scripts/training/diffusion/run-qwen-image-sft-condition-cache.sh
```

每个 rank 顺序累积自己的样本；micro batch 必须为 1，有效 batch 必须是卡数的整数倍。
默认 LoRA rank/alpha 为 32，学习率 `1e-4`，warmup 为 200 次优化器更新。
`HEIGHT`、`WIDTH` 必须为 16 的整数倍。预处理应用 EXIF 方向、RGB 转换、保持比例的
双线性缩放与指定尺寸的中心裁剪；这是 Relax 的固定尺寸预处理约定。
`RANDOM_CROP=1` 和 `CAPTION_DROPOUT` 可启用逐次随机裁剪与描述丢弃，默认关闭。
当前文本长度上限为 512；编码时仍保留官方完整上下文，再裁掉模板前缀和截取输出。
CPU 图像预取与条件预取暂时关闭，避免线程改变随机增强的执行顺序。

## 文本编码器放置策略

启动脚本的 `TEXT_ENCODER_PLACEMENT` 对应 `--t2i-sft-config` 中的
`text_encoder_placement`，均支持 `sharded`（默认）和 `replicated`，与是否启用条件缓存独立。

| 配置值 | 冻结文本编码器的行为 |
| --- | --- |
| `sharded` | 编码器作为独立 FSDP 组跨 rank 分片；在线编码时聚合权重，全组缓存命中时跳过编码器调用及其通信。 |
| `replicated` | 每张 GPU 常驻一份完整冻结编码器，使用本地权重编码，不发生编码器的 FSDP all-gather。 |

复制驻留使用原生 FSDP `ignored_params`。编码器仍正常注册在模型中，保留数据所有权和参数
变更检查；FSDP 不对这些参数执行分片、设备搬移、dtype 转换或梯度归约。适配器将它们以
BF16 加载到模型设备，并要求参数冻结且处于 eval 模式；训练中解冻复制驻留的编码器会报错。
该选项不改变 DiT LoRA 训练和条件缓存容量。

复制驻留会在每张 GPU 上保留完整编码器权重，直到模型释放，缓存命中时也不例外。
分片可以减少步间的权重驻留，但在线编码仍需要完整权重和通信缓冲。
`--no-fsdp-reshard-after-forward` 只将聚合后的权重保留到反向阶段，不会让编码器跨更新持续
复制驻留。选择策略前应分别测量启动及训练峰值：复制驻留不一定提高在线路径的峰值，
但相较于缓存命中时的分片编码器，会占用更多常驻显存。编码器权重不计入有界特征缓存的容量。

## 缓存范围与验证

默认仅复用确定性的冻结文本 `prompt_embeds`。缓存键由公共层的实际 checkpoint 内容指纹，
以及适配器声明的有效 token、attention mask、模板、截取规则、编码器配置、attention 设置
和 dtype 组成。它与图片文件名无关；同一有效描述可对应不同目标图片。
目标图的 VAE 后验采样、扩散噪声和时间步每次执行，不进入缓存。
编码器被修改或解冻后，公共运行时禁止继续复用旧条目。

可选的 `CACHE_VAE_POSTERIOR=1`（配置项 `cache_vae_posterior: true`）将文本特征和
VAE 的确定性后验分布参数放入同一条有界缓存记录。键还包含增强后 RGB 像素的内容、
尺寸、归一化规则和 VAE 配置。每次训练仍在原位置调用后验分布的 `sample()`，随后
生成扩散噪声和时间步；不会缓存采样结果。该模式要求 VAE 关闭 tiling 和 slicing。
后验参数仅支持 contiguous 或 `channels_last_3d` 布局；缓存会记录布局并在采样前恢复，
保持后续 `randn_like` 随机数与张量元素的对应关系。其他布局会被拒绝。
它复用的是相同的有效图像与文本组合，不再独立复用不同目标图片之间的相同描述。
随机裁剪或描述丢弃仍正常执行，并以本次产生的实际输入构键。

开启后验复用会让冻结 VAE 通过原生 `ignored_params` 常驻每张卡，在线与缓存对照必须
使用相同的 `CACHE_VAE_POSTERIOR`，不能把改变权重放置的收益计作特征复用收益。
该选项默认关闭，并增加每卡常驻权重和缓存 payload；需分别测量低复用开销与多轮收益。

`RELAX_CONDITION_CACHE=0` 保留在线路径。缓存由公共运行时按训练尝试持有，文件是不可变的
张量快照，读取方拥有独立的消费张量。容量按节点计算：上例最多 1 GiB 有效载荷，另有
独立的 1 GiB 写入 staging 上限；元数据及正在使用的张量还需额外内存。
LRU 淘汰跳过仍被借用的条目；缺失、损坏和容量不足会回退在线计算。分片编码器的 FSDP rank
共同决定是否命中，避免部分 rank 跳过编码器通信。对于适配器声明且后端确认所有编码模块
均为完整冻结副本、分支中无 collective 的情况，各 rank 独立处理命中与在线回退。
执行策略在初始化时跨 rank 校验并固定，不会随命中或权重失效动态切换。
正常结束和可捕获异常关闭当前运行的资源；
进程被强制终止后，公共缓存按 owner 存活状态回收该运行的遗留条目。

为每次运行使用新的日志目录。设置 `RELAX_CONDITION_VERIFY=1`、
`RELAX_CONDITION_VERIFY_RTOL=0`、`DETERMINISTIC_ATTENTION=1` 和 `LR_WARMUP_ITERS=0`
运行一次有效 batch，可比较在线/缓存 loss、完整累积并归约的逐参数梯度和实际 FP32 更新量。
这类验证包含额外参考计算，应与性能运行分开。性能对照应固定数据顺序、有效 batch、尺寸、
优化器及初始状态，分别报告首次构建、稳态完整训练循环和任务总耗时；低复用数据也需实测。
每组在线/缓存对照必须保持文本编码器放置策略、gradient checkpointing 和 FSDP reshard 设置
一致。显存允许复制驻留时，应比较两种策略，将更快且可运行的在线配置纳入基线，再使用相同
配置测量缓存收益。报告策略及其显存成本，不应把改变权重放置带来的收益归因于条件特征复用。

## 保存、恢复与导出

设置 `SAVE_INTERVAL` 控制保存频率，`LOAD_DIR` 恢复最新已提交的 `t2i_sft/iter_*`。
`STEPS` 为要达到的总更新次数。恢复包含优化器、scheduler、各 rank 随机状态和数据游标，
并严格核对数据、模型、batch、LoRA 与训练配置；缓存本身不作为 checkpoint 的必要状态。
文本编码器放置策略和后验复用模式也属于恢复契约，恢复时必须使用相同的
`TEXT_ENCODER_PLACEMENT` 与 `CACHE_VAE_POSTERIOR`。
两种策略均保存可训练的 DiT LoRA 权重，不会将冻结编码器权重加入 adapter checkpoint。

现有导出工具按 checkpoint 中的适配器执行，也可导出 Qwen-Image：

```bash
python examples/diffusion/export_i2i_sft.py \
  --checkpoint /shared/runs/qwen-sft/qwen-r01/t2i_sft/iter_0000032 \
  --output /shared/exports/qwen-image-lora
```

输出为 Diffusers 可加载的 transformer LoRA 目录，保留训练得到的 FP32 主权重，并要求
alpha 等于 rank；继续训练仍应使用原始 DCP checkpoint。
RL 目前没有接入此缓存；后续可由 RL 模型适配器声明同类确定性条件及依赖，复用公共存储和
生命周期，不应缓存候选的噪声、轨迹或奖励。
