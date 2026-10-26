# 训练侧图片特征复用

框架默认关闭，通过 `RELAX_TRAIN_VISION_CACHE=1` 开启。
`RELAX_TRAIN_VISION_CACHE_MAX_BYTES` 设置每个视觉执行 rank 的 GPU 缓存上限，默认 268435456（256 MiB）。超出上限按 LRU 淘汰，单张特征超过上限时正常计算而不缓存。此功能不改变任务数、轨迹数、图片数量或 loss。

当前适配对象是 dense `Qwen35VLModelProvider` 的实际视觉接口。以 Qwen3.5-9B 的配置为基准：27 层视觉编码器、1152 维视觉隐藏状态、4096 维 merger 输出，`deepstack_visual_indexes=[]`。当前 Bridge 内部复用 `Qwen3VLModel` 类，这是实现复用，不能据此把 Qwen3-VL 的配置和能力当作 Qwen3.5 的配置。

`--freeze-vision-model` 会冻结视觉主干，但在没有 `--freeze-vision-projection` 时仍保留 merger 可训练。因此只缓存 `vision_model.decoder` 的冻结输出，patch/position/rotary 准备与 merger 仍走原模型。merger、语言模型的前向和梯度计算正常执行。缓存入口检查 patch、position、rotary 和 decoder 冻结状态，以及 dropout、精度状态和视觉接口；不满足条件时明确报错。

图片身份来自通用 `Sample.session_id`（或 `Sample.metadata["session_id"]`）、CPU processor 图片张量的 SHA-256、dtype 和 `(T,H,W)`。metadata 随现有 `multimodal_train_inputs` 传输，不依赖 OSWorld 数据结构。缺少 Session 身份、视频或混合视频样本沿用原计算路径，不复用特征。没有新增训练参数解析或数据传输字段。

缓存保存当前 rank 所处理图片的完整 pre-merger 特征 `[patches, 1, vision_hidden]`，不是语言模型 SP 分片。TP 成员各自持有其接口上的完整特征，以实际 decoder TP 成员组成持久 Gloo 控制组，只交换图片身份和命中状态；任一成员未命中时，所有成员对相同图片执行原 decoder。GPU 特征不经过控制组。当前不共享 DP 副本缓存。

CP 分配读取实时 `model.pg_collection.cp`，包含动态 CP；metadata 按 Bridge 原有连续图片数量分配对齐，视觉计算和外层 CP gather 保持原样。原有 LM sequence parallel 保留。PP/VPP 无视觉模块的 stage 仅移除内部 metadata，不创建缓存或控制组。没有固定 TP/PP/CP/DP 数值白名单。

已明确拒绝的接口包括：其他模型 provider、非空 deepstack、可训练的缓存边界、非零 dropout、视觉 FP8/FP4 状态、视觉 CUDA graph、视觉 SP 分片、分离式 encoder/decoder，以及无法对应源图片的自定义 CP padding。这些是当前适配边界，不代表对其他模型或接口的通用支持。权重切换、训练 offload、加载 state dict 时清空缓存；冻结参数和 buffer 的版本、设备、dtype 或 autocast 状态变化也会使旧特征失效。

## 验证范围

CPU 测试覆盖完整/部分命中、去重、Session 隔离、淘汰、权重失效、merger 和语言层的梯度一致性、真实 Sample 转换及已有多模态传输/打包，以及 TP=1/2/4 的多进程控制组和 rank 命中分歧。CP metadata 测试覆盖 CP=1/2/4、空图片分配和 LM SP 开关；PP 测试覆盖无视觉模块 stage 的 metadata 处理。

这些 CPU 测试使用轻量视觉接口 fixture。实际 Qwen3.5-9B 已在 TP4/PP1/CP1/DP2、LM sequence parallel 开启的配置下完成 18 次训练更新，并观察到缓存命中和批内去重。PP>1、CP>1、动态 CP 等其他组合仍只有源码接口与 CPU 契约检查，尚未完成 GPU 端到端验证。不能将 CPU 通过等同于这些配置的 GPU 验证通过。

GPU 验证应使用当前 Qwen3.5-9B MobileGym 启动配置，分别用 `RELAX_TRAIN_VISION_CACHE=0` 和 `1` 对照，检查首次 miss、后续 hits、训练 loss/梯度及至少两次更新，并确认日志中的实际并行配置。训练切换到 rollout 或切换权重时会打印 `hits/misses/bytes` 并清空缓存。完整开关、Ray 环境传递及与 VCS、提前传输的组合方式见 [CUA 训练资源优化](cua-training-optimizations.md)。设置 `RELAX_TRAIN_VISION_CACHE=0` 可关闭。

## 复用统计

每次权重切换、offload、state dict 加载或冻结编码器状态变化时，原 `Training vision cache:` 日志附带 `reason`、TP/PP/CP/DP rank 坐标及 `stats` JSON，报告并重置上一次清空以来的统计。首次建立编码器状态签名不额外报告。统计不新增 GPU 同步、计时同步或集体通信。

| 字段 | 含义 |
|---|---|
| `calls` / `cache_calls` | 缓存入口调用次数 / 有完整缓存身份信息的调用次数 |
| `image_refs` | 有完整身份信息的图片引用次数，重复图片逐次计数 |
| `hits` / `misses` | 跨调用缓存命中 / 未命中的图片引用次数；TP 任一成员 miss 时协同 ranks 均按 miss 计 |
| `batch_dedup_images` | 未命中引用中，本微批次去重省下的图片计算次数 |
| `encoded_images` | 有完整身份信息时，实际送进原 decoder 的去重后图片数 |
| `patch_refs` / `encoded_patches` | 所有调用原本需要处理 / 实际送进原 decoder 的 patch 数，均包含绕过缓存的调用 |
| `cache_hit_patches` / `batch_dedup_patches` | 缓存命中 / 批内去重省下的 patch 数 |
| `bypass_calls` / `bypass_patches` | 缺少完整身份信息或视频等原因绕过缓存的调用数 / patch 数；此时无法仅凭 CPU 身份信息准确统计图片数，不计入图片指标 |
| `evictions` / `oversized_skips` | 因容量淘汰的条目数 / 单条特征过大未存入的次数；主动清空不计作淘汰 |
| `tp_forced_misses` | 本 rank 已有特征、但其他 TP 成员 miss 而共同重算的图片引用数 |
| `entries` / `bytes` / `peak_bytes` / `budget_bytes` | 当前条目数 / 当前占用 / 本统计区间的峰值 / 配置上限 |
| `image_hit_ratio` / `image_skip_ratio` | 图片缓存命中率 / 同时计入缓存命中与批内去重的图片计算跳过比例，仅针对身份完整的图片 |
| `patch_skip_ratio` | `1 - encoded_patches / patch_refs`，包括绕过缓存的调用；分母为零时记 0 |

例如空缓存中输入 `[A,A,B]`：`image_refs=3, misses=3, batch_dedup_images=1, encoded_images=2`，不能把三次 miss 理解成实际编码三次。不同尺寸图片应优先看 patch 指标。patch 跳过比例不等于 FLOPs 节省或整步加速比例，吞吐收益仍需相同配置下的开关对照实验。TP 成员保存复制特征，不能把所有 TP rank 的统计直接相加当作全局样本数；比较时每个 DP 副本选一个 TP 成员。使用新统计需要重启训练任务，让新进程重新加载代码。
