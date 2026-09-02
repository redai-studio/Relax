# 配置参考手册

参数分为三类：

1. megatron 参数：Relax 会读取 PYTHONPATH 中的 megatron 里设置的所有参数，可以通过传入如 --tensor-model-parallel-size 2 的方式配置 megatron；
2. sglang 参数：支持环境中安装的 sglang 的所有参数，这些参数需要以 --sglang 起始，例如 --mem-fraction-static 需要通过 --sglang-mem-fraction-static 传入。
3. Relax 自身的参数请见：relax/utils/arguments.py

常见配置用法和示例请参见[快速开始指南](./quick-start.md)。

---

## 集群与资源配置

### Ray 启动参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--rollout-num-gpus-per-engine` | int | 1 | 每个 SGLang 推理引擎使用的 GPU 数，等同于 SGLang 的 tp_size |
| `--num-gpus-per-node` | int | 8 | 每个节点的 GPU 数。如果在 colocate 模式下每个节点使用少于 8 张 GPU 需要设置此值 |
| `--resource` | json | - | Ray 资源配置，JSON 格式。`'{"actor":[副本数, 卡数], "rollout":[副本数, 卡数]}'` |
| `--colocate` | flag | False | 是否将推理引擎和训练 Actor 共置在相同的 GPU 上。开启后会同时将 `--offload` 设为 True |
| `--offload` | flag | False | 等同于同时设置 `--offload-train` 和 `--offload-rollout` |
| `--offload-train` | flag | None | 训练期间是否将训练 Actor 卸载到 CPU。`--colocate` 时始终为 True |
| `--offload-rollout` | flag | None | 训练期间是否将 Rollout 生成器卸载到 CPU。`--colocate` 时始终为 True |
| `--distributed-backend` | str | nccl | 分布式后端 |
| `--distributed-timeout-minutes` | int | 10 | 分布式超时时间（分钟） |

### TransferQueue 数据队列

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num-data-storage-units` | int | 1 | TransferQueue SimpleStorageUnit Actor 数量 |
| `--per-rank-fetch` | flag | False | 让每个 TP/PP rank 都并行地各自从 TransferQueue 拉一份数据，省掉 rank-0 pickle + TP/PP broadcast。跨 rank 一致性依赖 TQ sampler 的 `(partition_id, task_name, dp_rank, batch_index)` 缓存（PP/TP 不变量）。当 `rollout_routed_experts` 出现在 `data_fields` 时自动禁用（jagged NestedTensor broadcast 路径不兼容）。多卡训练推荐配合 `--num-data-storage-units >= TP world size` 使用。在 pickle 主导 `tgd_bcast_tp_time` 时有收益。 |
| `--max-staleness` | int | 0 | TransferQueue 数据系统的最大陈旧度（0=on-policy） |
| `--polling-mode` | bool | True | 获取 metadata 时是否使用轮询模式 |
| `--num-iters-per-train-update` | int | 1 | 全异步流水线中每个 global batch 的迭代次数 |

---

## 训练后端与模式

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|------|------|--------|--------|------|
| `--train-backend` | str | megatron | `megatron` | 训练后端选择 |
| `--qkv-format` | str | thd | `thd`, `bshd` | Megatron 后端的 QKV 布局。`bshd` 模式下不支持动态批处理，必须指定 `--micro-batch-size` |
| `--megatron-to-hf-mode` | str | raw | `raw`, `bridge` | Megatron 到 HF 权重转换方式。`bridge` 使用 megatron bridge 自动转换 |
| `--true-on-policy-mode` | flag | False | - | 是否启用真正的 on-policy 模式 |
| `--fully-async` | flag | False | - | 是否使用全异步训练流水线 |

---

## 检查点配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--hf-checkpoint` | str | None | HuggingFace 模型检查点路径。用于初始化 SGLang 和提供 tokenizer。不需要包含最新的参数，只需与训练模型架构一致 |
| `--ref-load` | str | None | 参考模型检查点路径。当 `--load` 未设置时，会作为训练的初始检查点 |
| `--ref-ckpt-step` | int | None | 参考模型检查点的步数 |
| `--load` | str | None | Actor 模型检查点加载路径。断点续训时指向此路径 |
| `--save` | str | None | 训练中模型的保存路径 |
| `--save-interval` | int | None | 模型保存间隔（步数） |
| `--save-hf` | str | None | Megatron 后端时保存 HuggingFace 格式模型的路径。路径可包含 `{rollout_id}` 占位符 |
| `--async-save` | flag | False | 异步保存检查点 |
| `--no-save-optim` | flag | False | 保存检查点时不保存优化器状态。减少检查点大小但无法从该检查点恢复训练 |
| `--rotate-ckpt` | flag | False | 是否轮换检查点。需要同时设置 `--save`、`--save-interval` 和 `--async-save` |
| `--max-actor-ckpt-to-keep` | int | None | 保留的最大 Actor 检查点数 |
| `--checkpoint-engine-backend` | str | nccl | 检查点引擎后端 |
| `--critic-load` | str | None | Critic 模型检查点路径。None 时等于 `--load` |
| `--critic-save` | str | None | Critic 模型保存路径 |

---

## 数据配置

### 数据集

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--prompt-data` | str | None | 训练 Prompt 数据集路径 |
| `--input-key` | str | input | 数据集中输入字段的 key |
| `--label-key` | str | None | 数据集中标签字段的 key |
| `--conversation-key-map` | json | None | 将非 OpenAI 格式的对话消息改写成 SFT 期望的 `{role, content}` 结构的 JSON 映射。一个扁平 map 同时覆盖两种重命名：**字段名重命名**（作用于每个消息 dict 的所有 key，例如 `from`→`role`、`value`→`content`）与 **role 值重命名**（仅作用于重命名后名为 `role` 字段的值，例如 `human`→`user`、`gpt`→`assistant`——因此消息正文中即使出现相同词也不会被误改）。仅在未设置 `--label-key`（即 `--input-key` 整列为完整的 messages list）时生效。sharegpt 示例：`'{"from":"role","value":"content","human":"user","gpt":"assistant"}'` |
| `--metadata-key` | str | metadata | 数据集中元数据字段的 key |
| `--tool-key` | str | tools | 应用 Chat Template 时的 tools 字段 key |
| `--apply-chat-template` | flag | False | 将输入作为 OpenAI message 格式应用 Chat Template |
| `--apply-chat-template-kwargs` | json | {} | Chat Template 的额外参数 |
| `--system-prompt` | str | None | 可选的系统提示词，会添加到用户输入之前。最终消息为 `<system_prompt> + <dataset_prompt>` |
| `--rollout-shuffle` | flag | False | Rollout 时是否打乱 Prompt 顺序 |
| `--rollout-seed` | int | 42 | Rollout 的随机种子，用于打乱 Prompt 和随机采样 |
| `--use-streaming-dataset` | flag | False | 使用流式数据集以节省内存 |
| `--streaming-buffer-size` | int | 10000 | 流式数据集的缓冲区大小 |
| `--prefetch-chunk-size` | int | 32 | 每轮预取时分派到线程池的样本数。较大的值可以提高吞吐量但也会增加内存压力。仅在设置了 `--use-streaming-dataset` 且数据集包含多模态数据时生效 |
| `--prefetch-max-cached` | int | 256 | 预取缓存中保留的最大预加载样本数。缓存满时后台预取线程会暂停，直到消费者释放空间。设为 0 可禁用预取。仅在设置了 `--use-streaming-dataset` 且数据集包含多模态数据时生效 |
| `--prefetch-num-workers` | int | 1 | 预取缓冲区中用于 I/O 密集型媒体解码（视频/图像）的并行工作线程数。设为 1 可序列化所有解码操作（对 FFmpeg 非线程安全问题最安全）。较高值可提高并行度，但在某些平台上可能触发 EAGAIN 错误。仅在启用预取时生效 |
| `--custom-prompt-path` | str | None | 自定义 Prompt 转换函数的 Python 点分导入路径。该函数在对话/多模态处理之前调用。函数签名：`def custom_fn(prompt, data: dict) -> prompt`。示例：`my_package.prompt_utils.add_prefix` |
| `--data-source-path` | str | `relax.engine.rollout.data_source.RolloutDataSourceWithBuffer` | Rollout 数据源类路径 |
| `--start-rollout-id` | int | None | 起始 Rollout 步数。未设置时会尝试从 `--load` 的检查点中读取 |

### 多模态数据

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--multimodal-keys` | json | None | 多模态数据字段映射。示例：`'{"image": "image_key"}'` |
| `--use-audio-in-video` | flag | False | 是否处理视频中的音频 |
| `--image-max-token-num` | int | None | 图像处理的最大 Token 数。未设置时使用默认值 16384 |
| `--image-min-token-num` | int | None | 图像处理的最小 Token 数。未设置时使用默认值 4 |
| `--video-min-token-num` | int | None | 视频帧处理的最小 Token 数。未设置时使用默认值 128 |
| `--video-max-token-num` | int | None | 视频帧处理的最大 Token 数。未设置时使用默认值 768 |
| `--video-fps` | float | None | 视频处理的目标 FPS。未设置时使用默认值 2.0 |
| `--video-fps-min-frames` | int | None | 视频处理的最少帧数。未设置时使用默认值 4 |
| `--video-fps-max-frames` | int | None | 视频处理的最多帧数。未设置时使用默认值 768 |
| `--image-resize-scale-factor` | int | None | 图像缩放尺寸对齐因子。默认使用 `patch_size * spatial_merge_size`（通常为 28）。设为 0 可禁用对齐 |
| `--audio-sample-rate` | int | None | 音频处理的采样率。未设置时使用默认值 16000 |
| `--frame-factor` | int | None | 帧数对齐因子。未设置时使用默认值 2 |
| `--mm-processor-pool-size` | int | 0 | 多模态处理器池大小。0（默认）禁用进程池，使用 ThreadPoolExecutor。设置为正整数时，创建指定数量 worker 的 ProcessPoolExecutor，实现无 GIL 竞争的真正并行 |

---

## Rollout 配置

### 采样参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num-rollout` | int | None | 总 Rollout 轮数。与 `--num-epoch` 二选一 |
| `--num-epoch` | int | None | 训练 Epoch 数。会根据数据集大小自动计算 `num_rollout`。若同时设置 `--num-rollout` 则此参数被忽略 |
| `--rollout-batch-size` | int | **必填** | 每轮采样的 Prompt 数量。总数据量 = `rollout-batch-size * n-samples-per-prompt` |
| `--n-samples-per-prompt` | int | 1 | 每个 Prompt 生成的响应数量 |
| `--rollout-temperature` | float | 1.0 | 推理引擎的采样温度 |
| `--rollout-top-p` | float | 1.0 | 推理引擎的 Top-p 采样参数 |
| `--rollout-top-k` | int | -1 | 推理引擎的 Top-k 采样参数。-1 表示不使用 |
| `--rollout-max-response-len` | int | None | 响应的最大长度，等同于 SGLang 的 `max_tokens` |
| `--rollout-max-prompt-len` | int | None | Prompt 的最大长度。设置后会在数据集初始化时过滤长 Prompt |
| `--rollout-max-context-len` | int | None | 推理引擎的最大上下文长度。不应超过 HuggingFace 模型 `config.json` 中的 `max_position_embeddings` |
| `--rollout-stop` | str (列表) | None | Rollout 时的停止词。可以是一个或多个字符串 |
| `--rollout-stop-token-ids` | int (列表) | None | Rollout 时的停止 Token ID |
| `--rollout-skip-special-tokens` | flag | False | 响应中是否跳过特殊 Token |

### 过采样与动态过滤

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--over-sampling-batch-size` | int | None | 采样批次粒度。None 时使用 `rollout-batch-size` 的值。必须 >= `rollout-batch-size` |
| `--dynamic-sampling-filter-path` | str | None | 动态采样过滤函数路径。用于实现类似 DAPO 的过滤（如排除全部正确或全部错误的样本）。示例：`relax.engine.filters.dynamic_sampling_filters.check_reward_nonzero_std` |
| `--buffer-filter-path` | str | None | Buffer 过滤函数路径。函数签名：`list[list[Sample]]` -> `list[list[Sample]]` |

### Partial Rollout

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--partial-rollout` | flag | False | 启用部分 Rollout。未完成的样本会被回收到数据缓冲区，适合长响应场景 |
| `--partial-rollout-max-aborted-count` | int | None | 样本被中断的最大次数。达到阈值后不再中断该样本，保证其完成生成 |
| `--mask-offpolicy-in-partial-rollout` | flag | False | 部分 Rollout 中是否 mask 之前的生成内容。设置后只有 on-policy 生成的 Token 参与训练 |

### 权重更新

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--update-weight-buffer-size` | int | 512MB | 权重更新的缓冲区大小（字节）。分块更新权重，对 MoE 模型有用 |
| `--update-weights-interval` | int | 1 | 权重更新间隔 |
| `--keep-old-actor` | flag | False | 是否在训练过程中保留 Rollout 模型 |

### 外部推理引擎

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--rollout-external` | flag | False | 使用外部 SGLang 实例而非框架内启动的实例 |
| `--rollout-external-engine-addrs` | str (列表) | None | 外部引擎的地址和端口列表 |

### SGLang 引擎参数

更多参数请参照 SGLang 官方文档。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--sglang-mem-fraction-static` | float | - | SGLang 静态内存分配比例 |
| `--sglang-profile` | flag | False | 启用 SGLang 引擎的 torch profiling。在 Rollout 推理期间触发，每步保存 profile trace |
| `--sglang-profile-steps` | int (列表) | None | 指定要进行 SGLang profiling 的绝对 rollout step ID（0-indexed）列表。优先级高于 `--sglang-profile-step-start/end`。例如 `--sglang-profile-steps 3 10 50` |
| `--sglang-profile-step-start` | int | None | SGLang profiling 的起始 rollout step（**inclusive**，0-indexed）。与 `--sglang-profile-step-end` 配合指定连续范围。设置了 `--sglang-profile-steps` 时被忽略 |
| `--sglang-profile-step-end` | int | None | SGLang profiling 的结束 rollout step（**inclusive**，0-indexed）。与 `--sglang-profile-step-start` 配合指定连续范围。设置了 `--sglang-profile-steps` 时被忽略。例如 start=2, end=4 会采集 step 2, 3, 4 |
| `--sglang-profile-output-dir` | str | None | SGLang profile trace 的输出目录。默认使用 `traces/<tb_experiment_name>/sglang_trace` |
| `--sglang-profile-num-steps` | int | 3 | 每轮 Rollout 中要 profile 的 SGLang 前向步数。-1 表示 profile 整个 Rollout 步，直到调用 `stop_profile` |
| `--sglang-profile-activities` | str (列表) | ["CPU", "GPU"] | 要 profile 的活动类型（例如 `CPU GPU`） |
| `--sglang-profile-by-stage` | flag | False | 按阶段（prefill/decode）分别进行 profile |
| `--sglang-profile-with-stack` | flag | False | 在 profile trace 中记录调用栈 |
| `--sglang-profile-record-shapes` | flag | False | 在 profile trace 中记录张量形状 |

### 自定义 Rollout 函数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--rollout-function-path` | str | `relax.engine.rollout.sglang_rollout.generate_rollout` | Rollout 生成函数路径 |
| `--custom-generate-function-path` | str | None | 自定义 generate 函数，替换默认 rollout 中的 generate 函数。适合实现多轮对话、function calling 等特殊逻辑 |
| `--rollout-data-postprocess-path` | str | None | Rollout 数据后处理函数，在获取完所有数据（包含 log_probs）后调用。可用于更新 loss mask |
| `--custom-rollout-log-function-path` | str | None | 自定义 Rollout 日志函数 |
| `--custom-eval-rollout-log-function-path` | str | None | 自定义评估 Rollout 日志函数 |

---

## 批处理配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--global-batch-size` | int | None | 全局批次大小。定义执行一次参数更新（optimizer.step）所需的样本量 |
| `--micro-batch-size` | int | 1 | 微批次大小。开启 `--use-dynamic-batch-size` 后该值被忽略 |
| `--num-steps-per-rollout` | int | None | 每次 Rollout 的训练步数。等效于设置 GBS = `rollout_batch_size * n_samples_per_prompt / num_steps_per_rollout` |
| `--use-dynamic-batch-size` | flag | False | 启用动态批处理。根据样本长度动态打包，使每个 micro-batch 的总 Token 数接近 `--max-tokens-per-gpu` 限制 |
| `--max-tokens-per-gpu` | int | None | 每个 GPU 的最大 Token 数。启用动态批处理时必须设置。使用 CP 时应设为约 `max_response_len / cp_size` |
| `--log-probs-max-tokens-per-gpu` | int | None | 计算 log probs 时每个 GPU 的最大 Token 数。None 时等于 `max-tokens-per-gpu` |
| `--balance-data` | flag | False | 在静态/序列长度均衡路径上使用 `karmarkar_karp` 在数据并行 rank 间平衡 Token 数量。全异步结合 `--use-dynamic-batch-size` 时，动态 batch 路径会通过 `StreamingTokenBudgetSampler` 自动做 DP token 均衡；该 flag 可传入但不会额外改变行为。注意同一 Prompt 的不同响应可能被分到不同训练步 |

---

## 并行配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--tensor-model-parallel-size` | int | 1 | 张量并行大小 |
| `--pipeline-model-parallel-size` | int | 1 | 流水线并行大小 |
| `--sequence-parallel` | flag | False | 启用序列并行 |
| `--context-parallel-size` | int | 1 | 上下文并行大小 |
| `--expert-model-parallel-size` | int | 1 | 专家并行大小（MoE 模型） |
| `--expert-tensor-parallel-size` | int | 1 | 专家张量并行大小 |

### 重计算

重计算部分使用的是 Megatron 原生参数，具体可参考 Megatron 文档。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--recompute-granularity` | str | - | 重计算粒度：`full`、`selective` |
| `--recompute-method` | str | - | 重计算方法：`uniform`、`block` |
| `--recompute-num-layers` | int | - | 重计算层数 |

---

## 优化器配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lr` | float | 1e-6 | 学习率 |
| `--optimizer` | str | - | 优化器类型（Megatron 原生参数） |
| `--lr-decay-style` | str | - | 学习率衰减方式（Megatron 原生参数） |
| `--weight-decay` | float | - | 权重衰减（Megatron 原生参数） |
| `--adam-beta1` | float | - | Adam beta1（Megatron 原生参数） |
| `--adam-beta2` | float | - | Adam beta2（Megatron 原生参数） |
| `--clip-grad` | float | 1.0 | 梯度裁剪 |
| `--seed` | int | 1234 | 随机种子 |
| `--optimizer-cpu-offload` | flag | - | 启用 CPU offload 优化器状态（Megatron 原生参数） |
| `--overlap-cpu-optimizer-d2h-h2d` | flag | - | 重叠 CPU 优化器 D2H/H2D 通信（Megatron 原生参数） |
| `--use-precision-aware-optimizer` | flag | - | 使用精度感知优化器（Megatron 原生参数） |
| `--use-distributed-optimizer` | flag | - | ZeRO-1 风格分片优化器状态（Megatron 原生参数） |
| `--overlap-grad-reduce` | flag | - | 反向计算与 grad reduce-scatter 重叠（Megatron 原生参数） |
| `--overlap-param-gather` | flag | - | reduce-scatter 与下一步 param all-gather 重叠，强制配合 `--overlap-grad-reduce`（Megatron 原生参数） |
| `--calculate-per-token-loss` | flag | False | 按 Token 计算损失（Megatron 原生参数） |

### FP16 优化器兼容默认值

| 参数 | FP16 兼容 fallback | 非 FP16 的 Megatron 原生默认值 | 说明 |
|------|--------------------|--------------------------------|------|
| `--initial-loss-scale` | `32768` | `2**32` | 动态 loss scaling 的初始 scale |
| `--min-loss-scale` | `1` | `1` | 动态 loss scaling 的最小 scale |
| `--use-precision-aware-optimizer` / `--no-use-precision-aware-optimizer` | 开启 | 关闭 | 开启或关闭 TransformerEngine 精度感知优化器 |
| `--store-param-remainders` / `--no-store-param-remainders` | 关闭 | 开启 | 控制分布式优化器是否保存参数余数 |

使用动态 FP16 loss scaling（未设置 `--loss-scale`）时，Relax 会为缺省选项保留历史兼容值，并用一条
warning 列出实际采用的 fallback。设置静态 `--loss-scale` 后，`--initial-loss-scale` 和
`--min-loss-scale` 不再生效，因此 Relax 不会为它们补默认值、执行校验或发出 warning；两个布尔优化器选项
缺省时仍使用 FP16 兼容值。显式传入当前生效的选项即可消除 warning。动态模式下，两个 scale 必须为大于零
的有限值，且 `--min-loss-scale` 不能大于 `--initial-loss-scale`。非 FP16 默认值直接取自 Megatron 的
`OptimizerConfig`。

Qwen3-4B FP16 脚本会显式配置全部四项。传给脚本的额外参数会追加到训练命令末尾，因此可以覆盖脚本值而
无需修改文件：

```bash
bash scripts/training/text/run-qwen3-4B-fp16-8xgpu.sh \
  --initial-loss-scale 65536 \
  --min-loss-scale 2 \
  --no-use-precision-aware-optimizer \
  --store-param-remainders
```

### 优化器 Flag 兼容性

| 场景 | `--use-distributed-optimizer` | `--overlap-grad-reduce` / `--overlap-param-gather` |
|---|---|---|
| 纯文本 Dense | ✅ | ✅ |
| Dense VL，CP = 1 | ✅ | ✅ |
| Dense VL，CP > 1 | ✅ | ❌ |
| MoE | ✅ | ❌ |

---

## 算法配置

### 优势估计

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|------|------|--------|--------|------|
| `--advantage-estimator` | str | grpo | `grpo`, `gspo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, `ppo`, `sapo`, `cispo` | 优势估计器。OPD 独立于该选项，使用 `--use-opd` 及对应 KL/loss 系数启用 |
| `--normalize-advantages` | flag | False | - | 是否归一化优势 |
| `--disable-grpo-std-normalization` | flag | - | - | 禁用 GRPO 标准差归一化（来自 [Dr.GRPO](https://arxiv.org/pdf/2503.20783)） |
| `--disable-rewards-normalization` | flag | - | - | 禁用 reward 归一化 |
| `--disable-compute-advantages-and-returns` | flag | - | - | 禁用优势和回报计算。用于 SFT 或自定义损失函数 |

### 损失函数

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|------|------|--------|--------|------|
| `--loss-type` | str | policy_loss | `policy_loss`, `sft`, `custom_loss` | 损失类型。`policy_loss` 跑 PPO/GRPO 等 RL 训练；`sft` 跑监督微调（详见 [SFT 配置](#sft-配置)）；`custom_loss` 需配 `--custom-loss-function-path`。`sft_loss` 是 `sft` 的已弃用别名。 |
| `--custom-loss-function-path` | str | None | - | 自定义损失函数路径 |
| `--eps-clip` | float | 0.2 | - | PPO 裁剪范围（下界） |
| `--eps-clip-high` | float | None | - | PPO 裁剪上界。None 时等于 `--eps-clip` |
| `--eps-clip-c` | float | None | - | Dual-clip PPO 的值下界（[论文](https://arxiv.org/pdf/1912.09729)） |
| `--value-clip` | float | 0.2 | - | 值函数裁剪范围 |
| `--entropy-coef` | float | 0.0 | - | 熵损失系数 |

### PPO 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gamma` | float | 1.0 | PPO GAE 折扣因子 |
| `--lambd` | float | 1.0 | PPO GAE lambda |
| `--value-clip` | float | 0.2 | PPO Critic value loss 裁剪范围 |
| `--use-rollout-logprobs` | flag | False | 使用 Rollout logprobs 作为 PPO 旧策略 logprobs；当前提供的 colocate 拓扑必须启用 |

PPO 当前支持同步 colocate 模式，并要求在 `--resource` 中包含 `critic` 与 `advantages`。资源拓扑、checkpoint 一致性和 KL 约束详见 [PPO 训练](./ppo-training.md)。

### KL 散度相关

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--kl-coef` | float | 0.0 | KL 惩罚系数，用于 reward shaping。同步 PPO 会将非零值重置为 `0.0`。不能与 `--kl-loss-coef` 同时非零 |
| `--use-kl-loss` | flag | False | 为 GRPO-like 算法启用 loss-level KL；同步 PPO 会自动禁用该选项 |
| `--kl-loss-coef` | float | 0.0 | KL 惩罚系数，添加到最终策略损失中。不能与 `--kl-coef` 同时非零 |
| `--kl-loss-type` | str | k1 | `k1`, `k2`, `k3`, `low_var_kl` | KL 损失类型 |
| `--use-unbiased-kl` | flag | False | 启用无偏 KL 估计 |
| `--ref-update-interval` | int | None | 参考模型更新间隔（Rollout 步数）。None 表示不更新参考模型 |

### SAPO 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--sapo-tau-pos` | float | 1.0 | SAPO 正优势温度 |
| `--sapo-tau-neg` | float | 1.05 | SAPO 负优势温度 |

### Critic 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num-critic-only-steps` | int | 0 | 仅训练 Critic 的步数 |
| `--critic-train-only` | flag | False | 仅训练 Critic 模型 |
| `--critic-load` | str | None | Critic 加载的 checkpoint。None 时等于 `--load` |
| `--critic-save` | str | None | Critic checkpoint 输出目录 |
| `--critic-lr` | float | None | Critic 学习率。None 时等于 `--lr` |
| `--critic-lr-warmup-iters` | int | 0 | Critic 模型线性预热的迭代数 |

### Off-Policy 修正

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-rollout-logprobs` | flag | False | 计算重要性采样比率时使用 Rollout 的 logprobs。未设置时使用 Actor 模型的 logprobs |
| `--use-tis` | flag | False | 启用 TIS（Truncated Importance Sampling）off-policy 重要性采样 |
| `--tis-clip` | float | 2.0 | 重要性采样比率的裁剪阈值上界 |
| `--tis-clip-low` | float | 0 | 重要性采样比率的裁剪阈值下界 |
| `--custom-tis-function-path` | str | None | 自定义 TIS/RS 函数路径 |
| `--custom-pg-loss-reducer-function-path` | str | None | 自定义 pg_loss reducer 函数路径。设置后 pg_loss 使用自定义 reducer，其他指标仍使用默认的 sum_of_sample_mean |

### Routing Replay 与 OPSM

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-routing-replay` | flag | False | 启用 Routing Replay（[论文](https://arxiv.org/abs/2507.18071)） |
| `--use-rollout-routing-replay` | flag | False | 启用 Rollout Routing Replay（[论文](https://arxiv.org/abs/2510.11370)）。启用后会自动开启 `--use-routing-replay` |
| `--use-opsm` | flag | False | 启用 Off-Policy Sequence Masking (OPSM) |
| `--opsm-delta` | float | 1e-4 | OPSM 阈值 |

### 其他训练选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--reset-optimizer-states` | flag | False | 每次 Rollout 后是否重置优化器状态 |
| `--use-rollout-entropy` | flag | False | 计算 logprobs 时是否同时计算熵。用于特殊 loss mask |
| `--get-mismatch-metrics` | flag | False | 是否计算 mismatch 指标。需要同时设置 `--custom-tis-function-path` |

---

## 参数冻结与选择性训练

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--only-train-params-name-list` | str (列表) | None | 要训练的参数名正则表达式列表。其余参数被冻结。不能与 `--freeze-params-name-list` 同时使用。示例：`--only-train-params-name-list experts` |
| `--freeze-params-name-list` | str (列表) | None | 要冻结的参数名正则表达式列表。其余参数保持可训练。示例：`--freeze-params-name-list embedding output_layer` |

---

## SFT 配置

以下参数仅在 `--loss-type sft` 下生效。SFT 流水线由 `SFTStreamingDataset` producer（把 packed 样本推入 TransferQueue 的 `sft_<rollout_id>` 分区）和 `MegatronTrainRayActor` consumer（训练、周期性 PPL eval、可选生成式 predict）组成。

### 训练控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--eval-size` | float | None | 从 `--prompt-data` 切出一份 holdout eval 集，而不是另外指定 `--eval-prompt-data`。值 <1 视为训练集的占比（例如 `0.05` → 5%）；值 ≥1 视为绝对样本数。行 ID 会用 `--seed` 随机切分一次，被预留的行会从训练池里移除，所以训练样本和 eval 样本永不重叠。与 `--eval-prompt-data` 互斥。 |
| `--sft-predict-interval` | int | None | 每 N 个 rollout step 在 eval 集上跑一次生成式 predict，把生成结果写到 `<save>/predict/predictions_step_<rollout_id>.jsonl`。设置该参数后会自动拉起 Rollout 角色（SGLang 必须在线）。它是 always-on 的 PPL eval（`--eval-interval`）的生成式补充。**必需**：`--save`（写到 `<save>/predict/` 下）以及至少一个 eval 数据源（`--eval-prompt-data` / `--eval-config` / `--eval-size`）。 |

### 流式数据集预取

SFT producer 用自己的 `PrefetchBuffer`，跟 rollout 数据源的 `--prefetch-*` 参数互相独立。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--sft-prefetch-buffer-size` | int | 256 | SFT 流式数据集 PrefetchBuffer 缓存的最大预加载样本数。设为 0 禁用预取，producer 会回退到基于 ProcessorPool 的 `asyncio.gather` 路径做 batch 级并行。 |
| `--sft-prefetch-chunk-size` | int | 32 | 每轮派发给 SFT 预取线程池的 chunk 大小。 |
| `--sft-prefetch-num-workers` | int | 4 | SFT PrefetchBuffer 内部用于 I/O 密集型媒体解码（视频/图像）的工作线程数。 |

### TransferQueue 分片 Producer

`RELAX_SFT_TQ_SHARDS` 控制每个 async-prepacked SFT 训练 step 生成多少个 TransferQueue 分区。这是一个实验性环境变量，便于在不增加公开 CLI 参数的情况下对 shard 数做 A/B 测试。

| 环境变量 | 类型 | 默认值 | 说明 |
|----------|------|--------|------|
| `RELAX_SFT_TQ_SHARDS` | int | 1 | SFT TransferQueue shard 数。小于等于 0 的值按 1 处理。 |

::: warning 生效条件
这个变量本身不会开启 prepack。它仅在使用 `--loss-type sft --sft-async-prepack` 时生效，否则 Relax 只使用一个分区。Async prepack 还要求开启 `--per-rank-fetch`、至少允许两个 in-flight step（`--max-staleness >= 1` 或 `--sft-max-in-flight-steps >= 2`）、PP=1、CP=1、VPP=1，并使用 THD QKV 格式。
:::

当 `N > 1` 时，step `K` 使用从 `sft_K_shard_0_of_N` 到 `sft_K_shard_<N-1>_of_N` 的分区；原有的 `sft_K` 名称只会在 `N == 1` 时使用。Consumer 会等待所有 shard 分区就绪，然后从每个分区读取相同数量的样本。因此，`global_batch_size` 和每个 DP rank 的本地 batch（`global_batch_size / data_parallel_size`）都必须能被 `N` 整除。

满足条件时，Relax 会为 train batch 启动 `N` 个远端 `_SFTBatchProducerActor`。配置的 `--sft-prefetch-num-workers` 会按每个 shard `ceil(workers / N)` 分配，且每个 shard 至少有一个 worker。Eval 仍由 coordinator 本地协调：`--eval-size` 会在每个远端 producer 内使用同一份确定性的 train/eval split，coordinator 负责渲染 holdout 样本（或 `--eval-prompt-data`）并在 eval interval 推送 `sft_eval_<step>_n<N>_<i>` 分区。

遇到以下任一情况时，remote train producer 会 fallback 到本地 coordinator 路径：

- Ray 未初始化。
- 使用 `--task-type seq_cls`。
- 设置了 `--custom-dataset-class` / `--custom-dataset-class-path`。
- `--sft-oversize-strategy` 为 `skip` 或 `custom`，或者 `--sft-invalid-multimodal-strategy` 为 `skip`。

本地 fallback 仍会把 batch 拆成 `N` 个 TransferQueue 分区，但不会创建 `N` 个远端 producer actor。日志出现 `SFT remote shard producer enabled: ... shards=N ...` 才表示 remote producer 并行已生效；fallback 路径会记录 `SFT remote shard producer disabled: ...` 及具体原因。

请在 Ray 运行时环境中配置这个值，确保 producer 和 consumer 推导出相同的分区名：

```yaml
# configs/env.yaml
env_vars:
  RELAX_SFT_TQ_SHARDS: "2"
```

### 超长样本处理

SFT producer 如何处理 tokenize + 多模态展开后长度超过单卡容量（`--max-tokens-per-gpu × --context-parallel-size`）的样本。所有分支都会给每个超长样本打一条 WARNING 日志。

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|------|------|--------|--------|------|
| `--sft-oversize-strategy` | str | keep | `skip`, `keep`, `truncate_left`, `truncate_right`, `custom` | `skip` 丢弃样本；`keep` 原样返回（可能下游 OOM）；`truncate_left` 保留末尾 `capacity` 个 token；`truncate_right` 保留开头 `capacity` 个 token；`custom` 委托给 `--sft-oversize-custom-function-path`。⚠ 直接截多模态样本可能导致 `multimodal_train_inputs` 错位——需要同时裁剪媒体输入时请用 `custom`。 |
| `--sft-oversize-custom-function-path` | str | None | - | 在 `--sft-oversize-strategy custom` 时必填。指向一个 Python 可导入函数，签名为 `def truncate(tokens, loss_mask, capacity, idx) -> (tokens, loss_mask) \| None`。返回 `None` 等同于 `skip`。 |

::: tip 相关数据集参数
SFT 还会用到通用的[数据配置](#数据配置)参数，特别是 `--input-key`、`--label-key`、`--conversation-key-map`（sharegpt 风格数据集）、`--multimodal-keys`、`--system-prompt`、`--tool-key`。
:::

---

## 评估配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--eval-interval` | int | None | 评估间隔（Rollout 轮数） |
| `--eval-prompt-data` | str (列表) | None | 评估数据集，格式：`dataset_name /path/to/data.jsonl`，可指定多对 |
| `--eval-config` | str | None | OmegaConf YAML/JSON 评估配置文件路径。设置时覆盖 `--eval-prompt-data` |
| `--eval-function-path` | str | None | 评估生成函数路径。None 时使用 `--rollout-function-path` |
| `--skip-eval-before-train` | flag | False | 是否跳过训练前的评估 |
| `--eval-input-key` | str | None | 评估数据中输入字段的 key。None 时使用 `--input-key` |
| `--eval-label-key` | str | None | 评估数据中标签字段的 key |
| `--eval-tool-key` | str | None | 评估数据中 tool 字段的 key |
| `--n-samples-per-eval-prompt` | int | 1 | 每个评估 Prompt 的采样数量 |
| `--eval-temperature` | float | None | 评估时采样温度 |
| `--eval-top-p` | float | None | 评估时 Top-p 参数 |
| `--eval-top-k` | int | None | 评估时 Top-k 参数 |
| `--eval-max-response-len` | int | None | 评估时最大响应长度 |
| `--eval-max-prompt-len` | int | None | 评估时最大 Prompt 长度 |
| `--eval-min-new-tokens` | int | None | 评估时最少新生成的 Token 数 |
| `--eval-max-context-len` | int | None | 评估时最大上下文长度。None 时等于 `--rollout-max-context-len` |

---

## Reward 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--rm-type` | str | None | 内置 Reward 模型类型 |
| `--rm-type-fallback` | str | None | 未知/缺失 Reward 类型的回退策略：`zero` 记 0 分并告警，注册名则路由到该 Reward；None 保持报错行为 |
| `--rm-type-infer` | flag | False | 无显式类型时按注册的 label matcher 推断 Reward 类型；与显式类型冲突时告警并以显式类型优先 |
| `--custom-rm-path` | str | None | 自定义 Reward 函数路径。单样本函数接收一个样本；batch/group 函数接收完整样本列表，并为每个样本返回一个结果。会绕过格式感知路由 |
| `--reward-max-concurrency` | int | 64 | 每个调用方进程内同时执行的 Reward 调用数上限。一次自定义 batch/group 调用计为一个调用 |
| `--reward-num-workers` | int | 16 | 用于执行同步 Reward 的 Ray Actor 数量。异步自定义 Reward 不使用这些 worker |
| `--reward-key` | str | None | Reward 函数返回 dict 时提取 reward 值的 key |
| `--eval-reward-key` | str | None | 评估时的 reward key。None 时等于 `--reward-key` |
| `--group-rm` | flag | False | 是否对整个 group 做 Reward 计算 |
| `--rm-url` | str | None | 远程 Reward 模型服务 URL（用于 `--rm-type remote_rm`） |
| `--custom-reward-post-process-path` | str | None | 自定义 reward 后处理函数路径。默认是 GRPO 的归一化处理 |
| `--custom-convert-samples-to-train-data-path` | str | None | 自定义样本到训练数据的转换函数。签名：`def convert_samples_to_train_data(args, samples) -> dict` |

---

## GenRM（生成式奖励模型）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--genrm-model-path` | str | None | GenRM 模型路径。设置后启用 GenRM |
| `--genrm-num-gpus` | int | 1 | GenRM 使用的总 GPU 数 |
| `--genrm-num-gpus-per-engine` | int | 1 | 每个 GenRM 引擎实例的 GPU 数 |
| `--genrm-engine-config` | json | None | GenRM 引擎初始化参数。示例：`'{"dp_size": 1, "pp_size": 1, "max_total_tokens": 8192}'` |
| `--genrm-sampling-config` | json | None | GenRM 采样参数。可用 key：`temperature`（默认 0.1）、`top_p`（默认 1.0）、`top_k`（默认 -1）、`max_response_len`（默认 4096） |

---

## On-Policy Distillation（OPD）

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|------|------|--------|--------|------|
| `--use-opd` | flag | False | - | 启用 On-Policy Distillation。需同时指定 `--opd-type` |
| `--opd-type` | str | None | `sglang`, `megatron` | OPD 类型。`sglang`：从外部 SGLang 服务器获取教师模型的 logprobs；`megatron`：通过 `--opd-teacher-load` 加载教师模型 |
| `--opd-kl-coef` | float | 1.0 | - | OPD KL 惩罚系数 |
| `--opd-teacher-load` | str | None | - | OPD 教师模型检查点路径。`--opd-type=megatron` 时必须设置 |
| `--opd-teacher-ckpt-step` | int | None | - | OPD 教师模型检查点步数 |

---

## 容错配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-fault-tolerance` | flag | False | 是否启用 Rollout 期间的容错功能 |
| `--use-health-check` | flag | False | 是否启用全局健康检查系统。Controller 的 HealthManager 监控所有服务，故障时触发自动重启 |
| `--max-global-restart` | int | 3 | 允许的最大全局重启次数。超过后训练终止。仅在 `--use-health-check` 启用时生效 |
| `--rollout-health-check-interval` | float | 30.0 | Rollout 引擎健康检查间隔（秒） |
| `--rollout-health-check-timeout` | float | 30.0 | Rollout 引擎健康检查超时时间（秒） |
| `--rollout-health-check-first-wait` | float | 0 | 开始健康检查前的初始等待时间（秒）。使用 deepgemm 时需要增大此值 |

---

## 弹性扩缩容配置

### Autoscaler

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--autoscaler-config` | str | None | Autoscaler YAML 配置文件路径。设置后启用自动扩缩容，未设置则禁用。示例：`--autoscaler-config relax/utils/autoscaler/autoscaler.yaml` |

Autoscaler YAML 配置详情请参见 [`relax/utils/autoscaler/autoscaler.yaml`](https://github.com/redai-studio/Relax/blob/main/relax/utils/autoscaler/autoscaler.yaml)。

### Scale-Out 操作参数

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|------|------|--------|--------|------|
| `--scale-out-timeout` | float | 300.0 | - | 所有扩容操作（引擎启动、连接、健康检查、权重同步等）的超时时间（秒） |
| `--scale-out-partial-success-policy` | str | rollback_all | `rollback_all`, `keep_partial` | 扩容部分成功时的策略。`rollback_all` 回滚所有引擎；`keep_partial` 保留成功的引擎 |
| `--scale-weight-sync-precheck` | bool | True | - | 扩容权重同步前跑独立 NCCL 预检查，不兼容 transport 直接 fail-closed。默认开；用 `--no-scale-weight-sync-precheck` 关闭（关闭即放弃 fail-closed 保护）。详见 [弹性 Rollout · 权重同步预检查](./elastic-rollout.md#权重同步预检查precheck) |

### Precheck 环境变量

下列环境变量用于调参与排障，均可不设（走默认）。运行时由 `relax/utils/env.py` 读取。

| 环境变量 | 类型 | 默认值 | 说明 |
|----------|------|--------|------|
| `RELAX_SCALE_WEIGHT_SYNC_PRECHECK_MIN_FREE_BYTES` | int | 536870912（512 MiB） | Stage 2 探针允许的最低空闲显存字节；低于该值返回 `INSUFFICIENT_GPU_MEMORY` 并 fail-closed。SGLang 预留 85–90% 显存，调低需谨慎（探针可能 OOM） |
| `RELAX_SCALE_WEIGHT_SYNC_PRECHECK_PORT_BASE` | int | 18000 | Stage 2 探针子进程 rendezvous 端口段下限 |
| `RELAX_SCALE_WEIGHT_SYNC_PRECHECK_PORT_MAX` | int | 20000 | Stage 2 探针子进程 rendezvous 端口段上限。**集群网络策略需放行 `[PORT_BASE, PORT_MAX]` 这段**，否则探针建连失败 → `PROBE_FAILED` |
| `RELAX_SCALE_WEIGHT_SYNC_PRECHECK_MAX_ATTEMPTS` | int | 2 | Stage 2 探针最大尝试次数 |
| `RELAX_SCALE_OUT_MAX_REASON_ITEMS` | int | 3 | 扩容失败原因在 TUI / 稳定日志里展示的最大条数 |
| `RELAX_SCALE_OUT_MAX_REASON_ITEM_LEN` | int | 120 | 单条失败原因截断长度 |
| `RELAX_SCALE_OUT_MAX_REASON_TOTAL_LEN` | int | 512 | 失败原因总截断长度（防止原始错误信息过长） |

### Scale-In 操作参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--scale-in-drain-timeout` | float | 30.0 | 等待进行中请求排空的超时时间（秒），超时后强制终止 |
| `--scale-in-shutdown-timeout` | float | 30.0 | SGLang 引擎优雅关闭的超时时间（秒），超时后使用 `ray.kill` 强制终止 |

---

## 日志与监控配置

### TensorBoard

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-tensorboard` | flag | False | 启用 TensorBoard 日志 |
| `--tb-project-name` | str | None | TensorBoard 日志目录。默认使用环境变量 `TENSORBOARD_DIR` |
| `--tb-experiment-name` | str | None | TensorBoard 实验名称 |

### ClearML

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-clearml` | flag | False | 启用 ClearML 日志 |

### Metrics Service

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-metrics-service` | flag | False | 启用集中化指标收集和报告服务 |
| `--timeline-dump-dir` | str | None | 时间线 trace 事件导出目录（Chrome Trace 格式）。未设置则禁用时间线追踪 |

### 日志选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--log-passrate` | flag | False | 启用 pass@n 通过率日志 |
| `--log-multi-turn` | flag | False | 启用多轮 Rollout 信息日志 |
| `--log-reward-category` | str | None | 记录 reward 分类统计。指定 reward dict 中的 key |

### 通知

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--notify-urls` | str | None | Apprise 通知 URL 列表（逗号分隔） |

---

## Router 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-slime-router` | flag | False | 使用 SlimeRouter 进行基于文本的路由而非 SGLang 的基于 Token 的路由 |
| `--slime-router-middleware-paths` | str (列表) | "" | 中间件路径列表 |
| `--slime-router-timeout` | float | None | SlimeRouter HTTP 请求超时时间（秒） |
| `--slime-router-max-connections` | int | None | SlimeRouter HTTP 客户端最大连接数 |
| `--slime-router-health-check-failure-threshold` | int | 3 | 连续失败多少次后标记 worker 为不健康 |
| `--slime-router-sticky` | flag | False | 启用 sticky 粘性会话路由：将路由 key（从 `X-SMG-Routing-Key` 请求头读取）钉定到固定 worker，使同一 key 的后续请求复用该 worker 的 prefix/KV 缓存。已钉定的存活绑定不会因新增 worker 而重分配（仅当其 worker 离开健康集合时才 remap）。需配合 `--use-slime-router` |
| `--slime-router-sticky-idle-secs` | float | 600.0 | sticky 的 key→worker 绑定空闲（未被路由）超过该秒数后淘汰，避免路由 key 基数无界导致映射表膨胀。需配合 `--slime-router-sticky` |

---

## 其他训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--train-env-vars` | json | {} | 训练进程的额外环境变量 |
| `--train-memory-margin-bytes` | int | 1GB (1024³) | 训练内存分配的预留空间 |
| `--disable-weights-backuper` | flag | - | 禁用权重备份以节省主机内存 |
| `--custom-model-provider-path` | str | None | 自定义模型 provider 函数路径 |
| `--recompute-loss-function` | flag | False | 是否重计算损失函数以节省显存 |
| `--log-probs-chunk-size` | int | -1 | 分块计算 log probs 的大小。用于节省显存 |
| `--allgather-cp` | flag | False | - |
| `--model-name` | str | None | 模型名称，用于 Megatron 到 HF 权重转换。未设置时从 `AutoConfig.from_pretrained(hf_checkpoint)` 推断 |
| `--only-load-weight` | flag | False | 参考模型和 actor fwd 仅加载权重 |
| `--rlsp-server-port` | int | 8234 | RLSP Server HTTP 端口 |
| `--custom-config-path` | str | None | 自定义参数 YAML 配置文件路径。文件中的 key-value 会覆盖已有参数 |

---

## 调试与性能分析参数

### 调试

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--debug-rollout-only` | flag | False | 仅运行 Rollout，不训练 |
| `--debug-train-only` | flag | False | 仅运行训练，不 Rollout |
| `--load-debug-rollout-data` | str | None | 加载调试用 Rollout 数据。设置后自动启用 `--debug-train-only` |
| `--load-debug-rollout-data-subsample` | float | None | 子采样部分调试 Rollout 数据以加速调试 |
| `--save-debug-rollout-data` | str | None | 保存 Rollout 数据，路径支持 `{rollout_id}` 占位符 |
| `--save-debug-train-data` | str | None | 保存训练数据，路径支持 `{rollout_id}` 占位符 |
| `--dump-details` | str | None | 导出所有训练细节用于事后分析 |
| `--check-weight-update-equal` | flag | False | 检查权重更新是否相等 |
| `--enable-cuda-memory-check` | flag | False | 在底层 NCCL 通信调用周围启用内存检查。在每次集合通信前记录可用 GPU 显存，通信失败时将内存信息附加到异常中 |

### 训练性能 Profiling

以下参数控制训练过程的 PyTorch Profiler 采集。Trace 文件默认保存到 `traces/<tb_experiment_name>/train_trace/` 目录下。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use-pytorch-profiler` | flag | False | 启用 PyTorch 内置 profiler 记录训练步骤的 CUDA kernel、CPU op 和通信操作（来自 Megatron） |
| `--profile-step-start` | int | 10 | 开始 profiling 的步数偏移（**inclusive**，来自 Megatron）。指从本次训练启动后的第 N 步开始采集，非绝对 rollout ID；断点续训时计数从 0 重新开始 |
| `--profile-step-end` | int | 12 | 停止 profiling 的步数偏移（**inclusive**，来自 Megatron）。含义同上。例如 start=10, end=12 会采集 step 10, 11, 12（共 3 步） |
| `--profile-target` | str (列表) | train_overall | 性能分析目标：`train_overall`、`train_actor`、`train_log_probs` |
| `--profile-with-stack` | flag | False | 在 profiler trace 中记录调用栈信息 |
| `--profile-with-memory` | flag | False | 在 profiler trace 中记录内存信息 |
| `--profile-with-flops` | flag | False | 在 profiler trace 中估算 FLOPs |

### GPU 内存 Profiling

以下参数控制 GPU 内存快照采集，用于诊断显存泄漏和 OOM 问题。Snapshot 文件可用 PyTorch Memory Viz 工具（`torch.cuda.memory._viz`）查看。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--record-memory-history` | flag | False | 启用 CUDA 内存分配历史记录（来自 Megatron）。开启后会记录每次分配/释放的调用栈和张量信息，并在发生 OOM 时自动 dump snapshot |
| `--memory-snapshot-path` | str | snapshot.pickle | 内存快照文件名（来自 Megatron） |
| `--memory-snapshot-dir` | str | None | 内存快照保存目录。默认使用 `traces/<tb_experiment_name>/memory_snapshot` |
| `--memory-snapshot-num-steps` | int | None | 在指定步数后主动 dump 内存快照（0-indexed，即设为 3 表示在第 2 步后 dump） |
| `--memory-recorder` | str | torch | 内存记录器后端：`torch`（PyTorch 内置）、`memray`（需要 `pip install memray`） |

### 网络

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--http-proxy` | str | None | HTTP 代理地址 |
| `--use-distributed-post` | flag | False | 使用分布式 POST 请求 |

---

## 环境配置

Relax 使用 `configs/env.yaml`配置运行时环境变量：

```yaml
env_vars:
  TOKENIZERS_PARALLELISM: 'true'
  NCCL_DEBUG: 'WARN'
  CUDA_DEVICE_MAX_CONNECTIONS: '1'
  GLOO_SOCKET_IFNAME: "eth0"
  TP_SOCKET_IFNAME: "eth0"
```
