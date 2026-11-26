# 冻结条件特征缓存

图像编辑 SFT 会在多个 epoch 中重复遇到相同指令和图像。条件编码器冻结时，Relax 可以复用其输出，继续训练生成模型。缓存默认关闭，不保存训练噪声、时间步或模型预测。

两个模型适配器共用同一存储接口：

| 适配器 | 缓存内容 | 训练路径 |
|---|---|---|
| `relax.models.sensenova.adapter.SenseNovaSFTAdapter` | 冻结的文本/图像 prefix KV 张量及索引信息 | 仅训练生成分支的 LoRA；全参训练继续在线编码 |
| `relax.models.qwen_image.sft_adapter.QwenImageEditSFTAdapter` | 冻结的文本/视觉 embedding、attention mask，以及目标图和条件图的确定性 VAE mode latent | LoRA，支持一张条件图 |

Qwen 适配使用 [Diffusers `QwenImageEditPipeline`](https://github.com/huggingface/diffusers/blob/d791c5c024014784df4a8dac2601e19fb4d300fc/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py) 的条件编码和 latent 几何，包括后验 mode 图像 latent；训练采用均匀 sigma 的 flow-matching loss。

## 启用复用

加载模型前，在每个训练 worker 的环境中设置以下变量。同一节点的 worker 应使用相同的节点本地缓存根目录。该目录应专用于 Relax 条件缓存，并为配置容量及临时写入保留足够空间。

```bash
export RELAX_CONDITION_CACHE_BYTES=68719476736
export RELAX_CONDITION_CACHE_DEVICE=shm
export RELAX_CONDITION_CACHE_DIR=/dev/shm/relax-condition-cache
export RELAX_CONDITION_CACHE_IDENTITY="<verified-model-manifest-sha256>"
```

`shm` 模式为每次训练创建独立目录，由该节点上的各 rank 共享。后续 epoch 即使把样本分配给另一个 rank，也能读取已有条目。容量按每次训练的每个节点计算，不在节点之间或不同训练任务之间共享缓存。

将 identity 占位符替换为已核验模型文件清单的 SHA256 身份，并确保它对应实际加载的文件。目录名不能代表文件内容。独立验证工具会在检查固定版本资产后生成身份；替换模型文件后，应重新生成身份并加载模型。

| 设置 | 默认值 | 含义 |
|---|---|---|
| `RELAX_CONDITION_CACHE_BYTES` | `0` | 字节预算，设为零关闭复用。`shm` 模式约束每次训练在每个节点上已提交的条目文件；`cpu`/`cuda` 模式为每进程预算。 |
| `RELAX_CONDITION_CACHE_DEVICE` | `shm` | 使用缓存根目录下的节点共享文件，或选择 `cpu`、`cuda` 进程内私有 LRU；CUDA 存储占用显存。 |
| `RELAX_CONDITION_CACHE_DIR` | `/dev/shm/relax-condition-cache` | `shm` 模式的节点本地根目录。同节点所有 rank 必须可见，不应使用跨节点网络文件系统。 |
| `RELAX_CONDITION_CACHE_MAX_ENTRIES` | `4096` | 每个共享训练目录或私有进程的最大已提交条目数。 |
| `RELAX_CONDITION_CACHE_STAGING_BYTES` | `2147483648` | 每个 rank 的共享缓存待写入预算。超出预算时放弃该次写入并计数，训练继续使用在线结果。 |
| `RELAX_CONDITION_CACHE_IDENTITY` | 空 | 可选的已核验模型内容身份。省略时，两种适配器均从小文件内容和大权重的相对路径、大小、修改时间派生本次运行身份。 |

需要私有缓存时，可将 `RELAX_CONDITION_CACHE_DEVICE` 设为 `cpu` 或 `cuda`，并计入每个 rank 的容量。变量在模型创建时读取，运行中修改不会调整已有缓存。下一次运行恢复在线编码时，设置 `RELAX_CONDITION_CACHE_BYTES=0`。

SenseNova 配方和 metadata 格式见[图像编辑 SFT](./i2i-sft.md)。Qwen 适配器加载本地 Diffusers 模型目录，在 I2I 训练路径中将 `--model-adapter-path` 设为 `relax.models.qwen_image.sft_adapter.QwenImageEditSFTAdapter`。

## 容量与缓存键

`SharedConditionStore` 通过有界主机 staging buffer 异步写入已 detach 的特征。后台写线程等待设备到主机的复制完成，写入临时文件，再通过原子重命名发布条目。进程锁串行保护淘汰和发布；写入时淘汰最近最少使用的已提交条目，直至满足字节数和条目数限制，超大条目不存入缓存。

共享字节上限覆盖已提交的条目文件，包括序列化头和对齐空间。临时写入、私有读缓冲、各 rank 的 staging buffer、文件系统分配开销及模型自身内存另计，应在预算外预留 SHM 和主机内存。写入失败会计数，训练仍可继续使用在线路径。

各 rank 就命中达成一致前，每个读者会先校验条目并把内容装入私有主机缓冲。后续淘汰不会移除该次更新已经暂存的内容。缺失或格式损坏的条目按未命中处理并重新计算。读取保持 dtype，并给每个使用者独立张量；SenseNova 每次重建新的 `DynamicCache`，生成过程不会向其他样本的 prefix 追加内容。

键包含实际 prompt、预处理后的图像像素、模型身份、特征结构、预处理/适配器版本、dtype 和相关 attention 设置。SenseNova 还绑定目标几何和编码器的设备语义，同构设备的共享缓存不绑定本地 CUDA 序号。Qwen 包含目标像素、当前 prompt template、processor 配置和 VAE 设置。不同文件名只要有效输入相同就能复用；裁剪、prompt 或模型设置变化会选择不同的键。

数据很少重复时，构键、传输、序列化和淘汰可能比省下的编码更贵。可通过 `model.condition_cache_stats()` 和 trainer 的 `cache/*` 指标观察命中、条目数/字节数、淘汰、待写字节、丢弃写入、写入错误和损坏。rank0 日志中的分支和操作计数属于该 rank；共享条目数和字节数描述该节点的训练目录。

## 训练语义与生命周期

适配器负责判断能否复用。条件编码器参与训练时绕过缓存；加载模型状态或冻结参数版本变化时清空条目。SenseNova 会绕过启用了 dropout 的 prefix，Qwen 则让冻结文本编码器和 VAE 保持 evaluation 模式。Qwen 复用确定性的后验 mode latent，每次更新仍采样 flow timestep 和噪声。有效图像像素进入缓存键，增强后的结果不会仅按样本文件名识别。

绕过分片编码器前，FSDP 各 rank 先就可缓存性和命中达成一致。某个 rank 的命中不能跳过其他 rank 仍需执行的 collective。适配器将共享存储绑定到训练进程组，自定义集成也应保持同样的 collective 参与方式。

模型提供 `clear_condition_cache()`、`close_condition_cache()` 和 `condition_cache_stats()`。`clear` 使条目失效后仍可继续使用；`close` 结束当前 owner 的使用并释放待处理工作。同节点最后一个 owner 关闭时删除该训练目录。持久根目录锁串行保护加入、关闭和过期目录回收；条目删除后可能保留零字节生命周期锁文件。

FSDP trainer 会在最后一个配置更新完成后及失败/清理路径中关闭缓存。自定义训练循环应在正常完成、失败或取消时通过 `finally` 关闭它。被强制终止的进程无法执行清理；下次缓存启动时，会回收 owner 已不存在的遗留目录。rank 仍在运行时不要手动删除缓存根目录。缓存特征不属于 checkpoint 状态，恢复训练会重新建立缓存。

## 验证与性能证据

[独立验证仓库](https://github.com/tryu-ai/relax-condition-cache-validation)保存固定模型/数据清单、数值和性能工具、原始日志及其源码身份，并按实现版本和硬件分别记录结果。这些工具不属于 Relax 运行时，启用缓存无需安装或执行它们。

日常训练使用标准[图像编辑 SFT](./i2i-sft.md)路径。独立对照覆盖高复用和低复用负载、冷缓存成本及完整训练循环耗时；结果描述对应负载，并不代表所有数据或机器都会获得相同收益。
