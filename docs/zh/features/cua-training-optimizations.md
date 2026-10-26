# CUA 训练资源优化

以下三项能力可以独立开启：VCS 减少导出的重复训练 context，冻结视觉特征缓存减少重复视觉编码，完整 Group 提前传输并释放 rollout 侧图片降低 CPU 内存峰值。它们不会减少配置中的任务数或每个任务的独立尝试次数。

## 在启动脚本中开启

把以下配置合并到 Ray `runtime_env.env_vars`，并把变量名追加到 `RELAX_PROPAGATE_ENV_VARS`，让配置经过训练 driver 继续传递给 Serve、rollout 和训练 workers。仅在提交任务的本地 shell 设置变量，不保证远端 workers 能读取到。

```python
optimizations = {
    "RELAX_VCS": "1",
    "RELAX_VCS_KEEP_IMAGES": "1",
    "RELAX_VCS_PRUNE_IMAGES": "2",
    "RELAX_TRAIN_VISION_CACHE": "1",
    "RELAX_TRAIN_VISION_CACHE_MAX_BYTES": "268435456",  # 每个视觉 rank 256 MiB
    "RELAX_TRANSFER_BATCH_GROUPS": "1",
    "RELAX_MM_RELEASE_AFTER_TRANSFER": "1",
    "RELAX_MM_PIXEL_DTYPE": "bf16",  # 可选：减少预处理像素的存储与传输体积
}
env_vars.update(optimizations)
forwarded = set(env_vars.get("RELAX_PROPAGATE_ENV_VARS", "").split(",")) - {""}
env_vars["RELAX_PROPAGATE_ENV_VARS"] = ",".join(
    sorted(forwarded | set(optimizations) | {"RELAX_PROPAGATE_ENV_VARS"})
)
# 将 env_vars 作为 runtime_env 的 env_vars 字段提交任务。
```

模型需使用实际支持的 Qwen3.5 dense 视觉接口，并配置 `--freeze-vision-model`，才能开启当前视觉缓存。缓存预算按每个视觉 rank 单独计算；`2147483648` 表示每 rank 2 GiB，应在视觉特征复用收益和训练峰值显存之间权衡。增大缓存会占用更多显存，不能代替合理的动态微批次 token 预算。

默认 VCS 和视觉缓存关闭，提前传输阈值为 `0`（沿用原模式的批次阈值），传后释放关闭，像素存储 dtype 为 `fp32`。这些变量在 worker 导入相关模块时生效，修改后需重启训练任务。

## VCS：减少训练 context 条目

共享接口位于 `relax.utils.visual_context.VisualContext`。CUA adapter 需要按 Session 调用 `prepare`、`commit`、`finish`，保存实际模型请求和回复，并导出 `records`；仅设置环境变量不会自动拦截尚未接入该接口的任意 agent。仓库内 OSWorld adapter 已接入，其他 CUA adapter 可复用同一个接口。导出的多轮 context 仍需沿用 [Agentic rollout](../guide/agentic-rollout.md) 的 custom advantage 配置。

`KEEP_IMAGES=1`、`PRUNE_IMAGES=2` 表示请求最多保留 3 张图，溢出时一次移除最早 2 张。同段内请求继续延伸相同前缀，段末导出一个训练 context，段内新增回答仍全部受到监督。历史文本保留；跨段旧回答作为上下文，不重复作为新目标监督。解析重试、历史修正或 tools/template 变化也可能提前结束一个段。

例如每轮新增一张截图的七轮回答：baseline 的三图滑窗导出 `[A→B→C] [D] [E] [F] [G]` 五条；VCS 导出 `[A→B→C] [D→E] [F→G]` 三条。它改变 context 组织与图片窗口，不应把条目减少比例直接当作训练耗时减少比例。

## 图片缓存：复用冻结视觉 decoder 的输出

当前适配 `Qwen35VLModelProvider` 的冻结 decoder 输出，保留可训练 merger 的正常前向和梯度。身份来自通用 Sample 的 Session ID、CPU processor 像素、dtype 和网格信息；特征存放在实际执行视觉计算的 rank 上，以 LRU 按字节预算淘汰。缺少身份信息和视频输入沿用原计算路径，其他模型或不兼容视觉接口会明确拒绝启用。

该功能保留原 TP/PP/CP/DP 和 LM sequence parallel 配置，不共享 DP 缓存。协同 TP ranks 统一命中/未命中决策，同一微批次内也会去重。具体模型边界、失效策略、并行验证范围以及 `hits`、`batch_dedup_images`、`encoded_patches`、`patch_skip_ratio` 等统计见 [训练侧图片特征复用](train-vision-cache.md)。

## 提前传输：降低 rollout 侧 CPU 图片峰值

`RELAX_TRANSFER_BATCH_GROUPS=1` 在非 fully-async 模式下允许一个完整且已评分的 Group 达到传输水位，不再必须等满整批 rollout。Group 包含同一任务条目的多次独立尝试及其导出训练样本，不会把未完成的 Group 提前交给训练。

`RELAX_MM_RELEASE_AFTER_TRANSFER=1` 等待 `async_put` 成功后，将本地 Sample 中的大块图片张量替换成零行占位，保留 dtype、宽度和图片网格供指标/日志使用。写入失败或取消时保留原数据；训练仍读取队列中的完整图片。默认 `RELAX_MM_RELEASE_TRIM=1` 会执行垃圾回收和 `malloc_trim`，尽量把空闲页归还系统；`RELAX_MM_RELEASE_LOG=1` 在执行 trim 时记录 `mm_release samples=... freed=... rss=... returned=...`。

当前 colocate 模式依然在整批 rollout 完成后再训练；提前写入 TransferQueue 不等于提前执行 GPU 更新。队列和仍运行的 Sessions 也占内存，因此 rollout 缓冲水位下降不等于整机内存按同样比例下降。

## 检查是否生效

- VCS：检查导出段数、每条 context 的图片数与多轮回答的 loss mask。
- 图片缓存：检查启动时的预算/并行配置，以及权重切换或 offload 时的 `Training vision cache: ... stats=...`。统计按 rank 输出，TP 复制的记录不能重复累加；Ray 日志去重可能折叠不同 DP 的数字，完整比较应读取各 worker 原始日志。
- 峰值内存：检查 `mm_release` 的张量字节和实际 RSS 变化；`freed` 不等于即时归还操作系统的字节。

当前 Qwen3.5-9B 的 GPU 实测覆盖 TP4/PP1/CP1/DP2、LM SP 开启，其余并行组合的验证范围见视觉缓存文档。需要衡量净加速时，使用同一份固定 rollout 数据比较开关；缓存命中或 patch 跳过比例不能直接换算为整步耗时减少比例。
