# 冻结条件特征缓存

图像编辑 SFT 会在多个 epoch 中重复遇到相同指令和源图。当条件编码器保持冻结时，Relax 可以复用它的输出，继续训练生成模型，减少重复编码。缓存不保存训练噪声或模型预测。

两个模型适配器共用同一缓存实现，默认关闭：

| 适配器 | 缓存内容 | 训练路径 |
|---|---|---|
| `relax.models.sensenova.adapter.SenseNovaSFTAdapter` | 冻结的文本/图像 prefix KV 张量及索引信息 | 仅训练生成分支的 LoRA；全参训练继续在线编码 |
| `relax.models.qwen_image.sft_adapter.QwenImageEditSFTAdapter` | 冻结的文本/视觉 embedding 与 attention mask | LoRA，支持一张条件图 |

Qwen 适配使用 [Diffusers `QwenImageEditPipeline`](https://github.com/huggingface/diffusers/blob/d791c5c024014784df4a8dac2601e19fb4d300fc/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py) 的条件编码和 latent 几何，训练采用均匀 sigma 的 flow-matching loss。

## 启用复用

加载模型前，在每个训练 worker 的环境中设置以下变量。容量是**每个进程的预算**，选择主机内存上限时需要计入所有 FSDP rank。

```bash
export RELAX_CONDITION_CACHE_BYTES=68719476736
export RELAX_CONDITION_CACHE_DEVICE=cpu
export RELAX_CONDITION_CACHE_IDENTITY="<verified-model-manifest-sha256>"
```

将 identity 占位符替换为已核验模型文件清单的 SHA256 身份。目录名不能代表文件内容。独立验收仓库会在核验固定版本资产后生成这一身份，固定对照的说明见下方链接。

| 设置 | 默认值 | 含义 |
|---|---|---|
| `RELAX_CONDITION_CACHE_BYTES` | `0` | 字节预算，设为零关闭复用。 |
| `RELAX_CONDITION_CACHE_DEVICE` | `cpu` | 张量存放在 `cpu` 或 `cuda`；后者占用显存。 |
| `RELAX_CONDITION_CACHE_IDENTITY` | 空 | 核验过的模型内容身份。Qwen 缓存运行必须提供；SenseNova 未提供时可计算本地模型文件哈希。 |

SenseNova 接受该变量中的 SHA256 身份。两个适配器都要求提供的身份对应实际加载的模型文件。替换文件后，应生成新身份并重新加载模型。

下一次运行需要恢复在线编码时，设置：

```bash
export RELAX_CONDITION_CACHE_BYTES=0
```

变量在模型创建时读取，运行中修改不会调整已有缓存。SenseNova 标准训练配方和 metadata 格式见[图像编辑 SFT](./i2i-sft.md)。Qwen 适配器加载本地 Diffusers 模型目录，在 I2I 训练路径中将 `--model-adapter-path` 设为 `relax.models.qwen_image.sft_adapter.QwenImageEditSFTAdapter`。

## 容量与缓存键

`ConditionFeatureCache` 是进程内 LRU，同时限制字节数和条目数，默认最多 4,096 条。写入时会淘汰最近最少使用的条目，直到满足两个上限。单条内容超过总预算时不存入缓存，该样本仍使用本次在线编码结果。

键包含预处理后的实际 prompt 和条件图像素，以及模型身份、特征结构、预处理/适配器版本、dtype 和相关 attention 设置。SenseNova 还绑定目标几何；Qwen 包含当前 prompt template 和 processor 配置。内容相同的不同文件名可以复用，处理后的条件发生变化则生成不同的键。

字节计数覆盖存储张量及树结构元数据，不是整个进程 RSS 或 CUDA allocator 保留量的上限。数据很少重复时，构键、复制和淘汰可能比省下的编码更贵。增加容量前，可通过 `model.condition_cache_stats()` 查看 `global_hits`、`global_misses`、`bytes`、`evictions` 和 `oversized`。

## 训练语义与生命周期

适配器负责判断特征能否复用。条件编码器参与训练时绕过缓存；加载状态或检测到冻结参数版本变化时清空已有条目。SenseNova 还会绕过启用了 dropout 的 prefix，Qwen 则让冻结的文本编码器和 VAE 保持 evaluation 模式。

读缓存不转换张量 dtype，并返回独立副本。SenseNova 每次重建新的 `DynamicCache`，生成过程不会向其他样本的 prefix 追加内容。噪声初始化、flow timestep/noise 和目标处理保留原来的在线采样位置；Qwen 的 VAE 编码也仍然在线执行。

每个 FSDP rank 各自持有缓存。绕过分片编码器前，各 rank 先就可缓存性和命中达成一致，避免某个 rank 的命中跳过其他 rank 仍需执行的 collective。样本跨 rank 的分配发生变化时，命中率可能下降。

模型提供 `clear_condition_cache()`、`close_condition_cache()` 和 `condition_cache_stats()`。`clear` 清空条目后仍可继续使用，`close` 释放条目并结束缓存使用。FSDP trainer 的清理方法会关闭模型缓存，更新失败或中断也会进入该路径。自定义训练循环应在正常结束、失败或取消时通过 `finally` 调用清理。条目不写入磁盘、共享内存或 checkpoint，恢复后的进程会重新建立缓存。

## 验收与性能证据

[独立验收仓库](https://github.com/tryu-ai/relax-condition-cache-validation)保存固定模型/数据清单、数值与性能对照工具，以及 H200 和 RTX 3090 的实测证据，并记录自己的源码版本和运行环境。这些工具不属于 Relax 运行时，使用缓存无需安装或执行它们。

日常训练使用标准[图像编辑 SFT](./i2i-sft.md)路径，并设置上方的缓存变量。独立对照覆盖高复用和低复用负载及冷缓存成本；实测收益不代表其他数据或硬件一定获得同样加速。
