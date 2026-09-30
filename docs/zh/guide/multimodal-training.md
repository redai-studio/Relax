# 多模态训练优化

多模态训练的开销分布在媒体读取与解码、processor 预处理、数据传输和模型计算几个阶段。SFT 与 RL 的数据生产路径不同，应先确认耗时发生在哪一段，再选择对应的优化。

| 适用范围 | 优化 | 作用 |
|---|---|---|
| 通用：SFT / RL | 控制媒体规模、定位 CPU / 存储 / GPU 瓶颈 | 减少不必要的数据处理与计算，避免盲目增加并发 |
| SFT：Megatron | 自动预取与 rank 侧图片重建 | 将下一步数据准备与当前训练重叠，减少图片像素经 TransferQueue 传输的体积 |
| RL：内置 SGLang rollout | 多进程 processor | 并行处理 rollout 侧的多模态输入，缓解 CPU 预处理瓶颈 |

## 通用：先确定瓶颈与数据预算

这些原则同时适用于 SFT 和 RL；各模型支持的媒体类型、分辨率与预处理方式仍以对应模型实现为准。

- **控制每条样本的媒体规模。** 图片数量、分辨率、视频帧数都会影响 CPU 处理、传输和 GPU 计算。调整图片 token 上限或视频采样配置前，先确认模型支持，并评估对任务质量的影响；这不是完全无损的性能开关。
- **区分读取、预处理和计算。** 存储慢时增加 processor worker 未必有效；CPU 已饱和时扩大预取窗口也不会提高生产速度。对比端到端耗时、GPU 等待、CPU 占用和存储吞吐后再调整。
- **按实际进程数分配资源。** 多进程 processor 会增加模型处理器副本和内存开销。SFT producer、训练 rank 和 RL rollout 的 worker 池属于不同阶段，参数不能混用。

通用训练计算与显存优化见[性能调优](./performance-tuning.md)。下面分别说明 SFT 和 RL 的数据路径。

## SFT：自动预取与 rank 侧图片重建

本节适用于 **Megatron SFT**。图片引用传输由 SFT producer 和训练 rank 配合完成，不会自动应用到 RL rollout。

### 自动选择与使用方式

Megatron SFT 自动让各 rank 从 TransferQueue 取数。用户配置在途 step 预算，框架根据数据类型和后端能力选择准备方式：

| 条件 | 自动选择的路径 |
|---|---|
| 仅允许一个 step 在途 | 同步取数，保留原多模态 payload |
| 至少两个 step 在途，且 `--multimodal-keys` 包含 `image` | 预取下一步；满足图片引用条件的样本在 rank 侧重建像素 |
| 至少两个 step 在途，无 `image`，且满足 async prepack 限制 | 后台取数、划分 microbatch、打包，并提前准备首个 microbatch 的设备数据 |
| 至少两个 step 在途，但不满足上述条件 | 普通数据预取 |

Async prepack 当前要求 NCCL、THD、PP=1、CP=1、VPP 关闭，且未启用动态 CP 或 routing/indexer replay。图片数据优先走图片预取路径。无需手动组合 per-rank fetch、train-data-prefetch 或 async prepack 开关。

在已有的 SFT 启动配置中，图片列映射与预算可以写成：

```bash
--multimodal-keys '{"image":"images"}' \
--sft-max-in-flight-steps 2
```

这是附加参数片段，不是完整启动命令；完整配置见 [SFT 训练](./sft-training.md)。在途预算包含当前训练 step，默认未设置时沿用 `--max-staleness`，其默认值 0 对应一个 step。增加预算会增加缓冲数据的资源占用，不保证吞吐提升。

### 图片引用传输的工作方式

Producer 仍执行完整 processor，生成 tokens、loss masks、图片 grid，并完成长度检查。对于可重建的图片，传输时用文件引用和形状元数据替换 `pixel_values`；训练 rank 在后台读取同一文件并重建像素。

```text
SFT producer                         Vision training rank
  full processor                       background prefetch
  tokens / masks / grid ── TQ ──────>   read shared image files
  image refs / pixel shape ─────────>   rebuild pixels on CPU
                                        validate grid and shape
                                        attach pixels before training
```

因此，这项优化减少的是像素传输量，并将 rank 侧重建与训练重叠；producer 的图片处理工作仍然存在。它也会增加训练节点的 CPU 工作和图片存储读取量。

消费端不重新生成 tokens 或 loss masks。当前重建实现覆盖 Kimi K3 SFT 图片处理路径和通用 HF image processor 路径；只有产出可识别的 `pixel_values` / `image_grid_thw` 结构时才替换像素。模型 chunk 具有 `pre_process` 能力的 vision rank 才重建图片，其余 PP stage 保留元数据。

### 输入格式与自动回退

文件引用必须在所有训练节点上指向相同且可读的图片。建议在数据准备阶段使用共享存储上的绝对路径，例如：

```json
{
  "messages": [
    {"role": "user", "content": "<image>\nDescribe this image."},
    {"role": "assistant", "content": "A cat sitting on a chair."}
  ],
  "images": ["/shared/data/images/cat.jpg"]
}
```

| 输入形态 | SFT 图片传输行为 |
|---|---|
| 文件路径字符串，或仅含路径的 `{"path": "..."}` 对象 | 满足重建条件时传文件引用 |
| `data:image/...;base64,...`、`{"base64": "..."}`、原始 bytes 或含 bytes 的对象 | 自动保留该样本的像素 payload |
| 裸 base64 字符串，无 data URI 前缀 | 按文件路径解释，不会自动识别为 base64；应改用上面的内联格式 |
| processor 输出不具备可重建的图片结构 | 保留原 payload |
| 视频、音频字段 | 不使用这项图片引用优化；是否支持该媒体仍由模型和数据处理路径决定 |

回退按**样本**生效：一条样本只要有图片不能表示为文件引用，就保留该样本的图片像素；同一 batch 的其他样本仍可使用引用。无需配置 unsupported 策略或手动开启像素回退。

文件错误要区分发生阶段：producer 读取或处理失败遵循 `--sft-invalid-multimodal-strategy`（默认 `error`，可选择 `skip`）；producer 成功后，训练 rank 读取失败，或重建出的 grid / 像素形状与描述符不一致，会报错终止训练，不会静默回退。

### 性能观察与排查

| 指标 | 含义 |
|---|---|
| `sft_rank_image_read` | 图片重建任务的读取与解码耗时，按样本累加 |
| `sft_rank_image_process` | 图片重建任务的 processor 耗时，按样本累加 |
| `sft_train_prefetch_wait` | 训练线程等待预取 future 的耗时，可能包含图片重建等待 |
| `per_rank_fetch_time` | TransferQueue 取数耗时，不含之后的图片重建 |

重建任务并行执行，前两个指标的总和可能大于实际墙钟时间。应结合完整训练 step 耗时判断收益，不能只看 TQ 流量下降或某个 rank 的取数时间。

- **rank 读取失败：** 检查各训练节点的挂载路径、权限和文件内容；producer 能读不代表其他节点已经正确挂载。
- **grid / shape 不一致：** 检查 producer 与训练节点的模型、processor 和图片预处理配置是否一致，尤其是图片 token 限制。
- **producer 供数慢：** 先检查坏样本、磁盘读取和 CPU 使用率，再调整 producer 的 `--sft-prefetch-num-workers`、`--sft-prefetch-chunk-size` 或在途预算。chunk 太小会限制可并行处理的样本数。
- **rank 重建慢：** 检查训练节点的 CPU 与共享存储负载。当前每个执行图片重建的 rank 使用固定 8 个 processor 进程；producer 的 `--sft-prefetch-num-workers` 不会改变这个池的大小。

## RL：rollout 侧多模态预处理

内置 SGLang rollout 在生成请求前会处理多模态输入，并准备训练所需的多模态张量。这里的 processor 并行度与 SFT 图片引用传输是两个独立机制；自定义 rollout 是否使用该路径取决于其实现。

当耗时集中在 rollout 侧的 HF processor，且 CPU 与内存有余量时，可配置进程池，例如：

```bash
--mm-processor-pool-size 4
```

默认值 `0` 使用线程执行；正数为每个 rollout worker 进程创建相应数量的 processor 子进程。处理器对象不存在时不会创建该池，创建失败时记录警告并回退到线程执行。评估资源时要乘以实际 rollout worker 数，而不是把这个值当作全局进程总数。

该参数不控制 SFT producer 或 SFT rank 图片重建，也不直接调整 SGLang 推理引擎的 GPU 并行度。调优时分别观察 CPU 预处理和推理引擎的耗时；后者见[性能调优](./performance-tuning.md#sglang-推理引擎调优)。

## 下一步

- [SFT 训练](./sft-training.md)
- [数据集设计](./dataset-design.md)
- [性能调优](./performance-tuning.md)
