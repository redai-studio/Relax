# Replay Manager

Replay Manager 将 SGLang rollout 中的离散模型决策交给 Megatron 训练重放。内置三类数据：`moe` 对应专家路由（R3），`indexer` 对应 DSA、QSA 等稀疏 attention 的选择结果，`mm_embedding` 对应冻结的视觉输出。重放让训练复用 rollout 选中的集合；QSA 还可以跳过已有 replay 行的 indexer 打分计算。

本文重点介绍 **reference 模式**：sample 只携带小型描述符，不可变的 replay body 存放在 TransferQueue（TQ）。它与记录离线 loss/advantage 复现数据的[轨迹重放](./trajectory-replay.md)是两套功能。

## 架构与数据所有权

```text
SGLang model / capture hooks
        |
        v
ProducerPublication -- encoded bodies --> existing TQ storage
        |                                       ^
        v                                       |
ReplayRef in rollout sample              ConsumerLayerReader
        |                                       ^
        v                                       |
Sample TQ partition --> Megatron layout --> layer / microbatch channel
                                                |
                                                v
                                     model forward / recomputation

ReplayInventory: run identity, sealed records, capacity, partition holds, leases
```

管理层与模型语义分开实现：

| 组件 | 职责 |
| --- | --- |
| `ReplayManager`、`ReplayKind` | 注册生产/消费适配器及进程内引擎 hook，管理 run 和 channel 状态 |
| `ProducerPublication` | 按有界行块和 layer 发布 body，写入完成后才返回 reference |
| `ReplayRef` | 记录 run、owner、record、kind、维度、codec 和 token 摘要，不携带完整 top-k 数组 |
| `ReplayInventory`、`ReplayRegion` | 管理容量、队列 partition 所有权和 reader lease，协调回收 |
| `ConsumerBatchLayout`、`ConsumerLayerReader` | 将 sample/token 坐标映射到真实 packed TP/SP/CP 布局，读取本 rank 所属层 |
| `RoutingReplay`、`IndexerReplay` | 在模型调用处应用决策，检查 forward/backward 消费顺序 |

Replay 复用 Relax 初始化后解析出的 TQ 连接，不启动第二套 TQ 集群。Body 使用 run 独立的 partition；inventory 由 Ray actor 承载，模型 worker 各自持有 channel、缓存和读取会话。

Producer 在 body 写完后封存 record。向 sample 队列发布时，先挂上 partition 引用，sample 写入成功后再转移 producer 所有权。Consumer 读取前获取 lease；即使 sample partition 已退休，仍有 reader 持有的 body 也不会被提前删除。释放前会等待未完成读取结束；producer、partition 和 reader 均不再持有数据时才可回收。全局故障恢复会生成新的 replay run 标识，避免旧 reference 混入重启后的数据。

Token 身份、kind、codec、layer 和 partition 检查会拒绝不兼容的 reference。Channel 游标检查 forward 和重计算是否少消费或多消费。管理层负责这些通用约束，模型适配器负责采集位置、block 坐标和平行布局语义。

## 如何开启

在支持的在线、colocate Megatron 配方中加入：

```bash
--use-rollout-indexer-replay
```

该参数会自动设置 `RELAX_REPLAY_REFERENCES=1`，按需生成 run ID，并把 reference 环境传给 worker，无须再配置一个 reference 开关。

MoE replay 由 `--use-rollout-routing-replay` 独立开启；`--use-routing-replay` 控制训练侧路由重放，`--optimize-routing-replay` 选择已有的路由重放优化。这些参数不会开启 QSA。Indexer replay 与 rollout routing replay 同时开启时，两类数据共用 reference 基础设施。

### 冻结视觉 embedding replay

```bash
--use-rollout-mm-embedding-replay
```

此参数同样自动启用 reference。Qwen3.8 适配器要求在线 colocate Megatron、CP1、BF16，并冻结整个视觉编码器，**包括 merger/projection**。Checkpoint 必须输出 2560 维视觉 embedding，且没有 DeepStack。此适配器不支持依赖 Critic 的算法、MTP、推测解码、partial rollout、hierarchical cache 和自定义生成路径。现有 indexer replay 的多模态限制保持不变，此参数不会开启 indexer replay。

采集边界位于视觉 merger 之后、与文本 embedding 混合之前。新计算、embedding cache 命中、预计算特征和 chunked prefill 都引用完整媒体 body。按引擎的图片 hash 去重：同 hash 复用第一次发布的 body，不再比较后续特征数值；grid 或 modality 不兼容时拒绝使用。

`bf16-raw-v1` 将 BF16 原始位按字节传输，不做量化。Body 大小为 `N_visual × 2560 × 2` 字节，1024 个视觉 token 对应 5 MiB。多个 response 共用 body，不为每条回答复制。Sample 队列仅携带 reference 和布局元数据，校验 grid 后移除原始 pixel tensor；存储、reader lease 和 partition 回收沿用现有 TQ 机制。

训练只在输入 PP stage 加载，每个 TP 组读取一次后广播给组内其他 rank。有界缓存复用于 log-probability 和训练前向，保留原有 placeholder scatter、MRoPE 及语言模型梯度。不开此参数时仍走原有视觉计算路径。

### 当前 indexer 接入范围

校验器识别以下模型族。这些是适配约束，不代表每种拓扑均已完成性能测试。

| 模型 | 训练布局 | Rollout 约束与数据表示 |
| --- | --- | --- |
| Qwen3.8（`qwen4_exp`、`qwen4_exp_text`） | CP1；zigzag partition；不使用 allgather-CP；mixed backend、TP selector；支持 TP/SP | 不开 DP attention；R4、512 个完整块；`qsa-r4-bitmap-v1`；采集 context 上限 262144 token |
| DeepSeek-V4（`deepseek_v4`） | CP>1、TP1、不开 SP；contiguous partition、开启 allgather-CP | 开启 DP attention；采集配置中的 C4 层；当前通用发布路径使用 `int32` |
| Keye DSA（`KeyeVL2`） | CP>1、TP1、不开 SP、PP1；zigzag、不使用 allgather-CP | 不开 DP attention；关闭 overlap schedule 和 radix cache；top-k 2048；`keye-bitmap-v1`，采集 context 上限 32768 token |

Indexer replay 当前拒绝离线训练、非 colocate、fully async/hybrid、多模态、agentic/自定义 rollout、partial rollout、dynamic CP、MTP/推测解码和 hierarchical cache。Indexer loss 必须为零；full activation recompute 必须使用 uniform 方法；也不支持 combined-1f1b MoE overlap。启动校验会列出不支持的组合。

需要使用带有 SGLang capture 和 Megatron 集成 patch 的兼容 Relax 训练镜像。运行时适配器安装进程内 hook，单独更新仓库源码不能补齐镜像中缺失的引擎 API。专用 bitmap attention 算子本身不需要额外修改 Megatron 源码。

## 压缩、读取与传输

模型适配器自动选择传输 codec，bitmap 压缩无须额外开关。

Keye 在 D2H 前将选中 token 集合编码为 bitmap，避免长期保存完整 int32 top-2048 host cache；全可见的因果前缀只存 header。QSA 将同样的方法用于**文档内完整四 token 块**：2048 token 预算最多对应 512 个 block ID。Codec 不记录当前未完整的尾块，attention 单独补回因果尾部。Bitmap 保留集合成员关系，不保留原始 top-k 排序。

QSA 发布时只写当前 chunk 可见的 word 范围。以 8192-token 文档为例，覆盖完整 block 域的 QSA bitmap 每行约 260 字节（header 加 64 个 word），512 个 int32 block ID 为 2048 字节，展开后的 2051 个 token ID 为 8204 字节；早期全选前缀 chunk 只需 header。这是表示大小对比，不是实测网络压缩率；bitmap 成本会随 context 增长。

两类 bitmap consumer 都在 32 MiB/128-body 上限内批量读取不同形状的编码 body，并共享有界 GPU body cache。CPU 路径缓存不可变的路由元数据，绕过 SimpleStorage 的 nested batch 重打包，将缓存未命中的 body 合并到 pinned host staging 后做 H2D。QSA 还会在当前 consumer batch 内提前读取，尚不是跨 batch 预取，也不保证通信完全隐藏。

TQ 配置为 `MooncakeStore` 且使用 RDMA 时，读取编码 bitmap body 的 CUDA consumer 会复制连接配置并选择 GDR 接收；通用 int32 DSA reader 不会自动走此路径。注册 staging 限制为 8 MiB，压缩数据经一次 D2D 解包写入独立 tensor。CPU producer 仍使用 host 传输。SimpleStorage 则走 CPU 接收加 H2D。开启 replay 本身不会配置 RDMA，也不能证明某次运行已走 RDMA。

`RELAX_REPLAY_CAPACITY_MB` 限制 inventory 管理的 body 存储，默认 16384 MiB。Consumer materialization/body cache 默认每个缓存 256 MiB；增大存储容量不会扩大 GPU cache。应按在途 rollout 数据量选择存储容量，不要混淆两种预算。

## QSA 计算路径与专用 attention

Replay 覆盖的行不再重新选 top-k；没有回退行需要计算时，连 indexer 的 query/key 投影和打分也会跳过。缺失行仍使用普通 indexer，TP 通信只同步缺失 selection。SP 下需要投影时，通过本地投影加紧凑 raw-key 交换，避免 all-gather 整个 hidden activation。Checkpoint invocation 内的 tape 复用 selection，同时保留消费检查；attention 的 Q/K/V 梯度仍正常计算。

Bitmap 从 consumer 一直保留到 mixed attention。专用 Triton 实现按 key tile 读取 32-bit selection word，结合文档内因果 mask 直接计算，无须展开 token routes 或构造 Flex BlockMask。Forward 将 GQA 的 query heads 合并到矩阵行，复用 K/V 加载；backward 计算 Q 梯度，并在所属 CTA 内归约 K/V 梯度，省去按 query head 展开的 FP32 K/V 临时缓冲及额外 reduction/cast。

默认 **Grouped mixed** 配置下，分流如下：

| 单篇文档长度 | 不开 rollout replay：selected IDs | QSA bitmap replay |
| --- | --- | --- |
| ≤2051 | 因果 Flash Attention | 因果 Flash Attention |
| 2052–16384 | 原有 FlexAttention | 专用 bitmap attention |
| 16385–32768 | 原有 Grouped attention | 专用 bitmap attention |
| >32768 | 原有 Grouped attention | 仅为 Grouped 解码 token routes |

阈值看每篇文档长度，不看单卡拼接后的总 token 数。因此 16 篇 8K 文档组成的 128K pack 能命中专用算子。关闭 rollout indexer replay 后仍走 selected-ID 路径，不会自动把原生 selection 转为 bitmap，也不会隐式启用专用算子。

上表描述默认 Grouped mixed 后端，模型侧显式分流设置仍然生效。专用算子要求 CUDA BF16、D256、整数 GQA 比例不超过 64，以及精确 CPU 文档边界。模型后端配置单独见 [Qwen3.8 QSA 后端](./qwen38-qsa.md)。

## 性能实验摘要

以下为 2026 年 11 月单张 Blackwell GPU、BF16、D256、local 6Q/1KV 的配对测速。Selection 已在 GPU，forward+backward 包含完整 Q/K/V 梯度，不包含 rollout 采集、TQ 传输、indexer 计算和分布式完整训练。

| 负载与对照 | 优化前 | 专用算子 | 耗时下降 |
| --- | ---: | ---: | ---: |
| 16 × 8192 token，forward+backward，对照为直接读 bitmap 的 Flex | 139.44 ms | 40.71 ms | 70.8% |
| 单篇 16K，forward+backward，对照为展开 routes 的 Grouped 路径 | 37.54 ms | 11.37 ms | 69.7% |
| 单篇 32K，同上 | 75.65 ms | 42.76 ms | 43.5% |

16 × 8K 场景中，扣除调用输入后的 PyTorch 新增 allocated 峰值由 Flex 的 1.648 GiB 降至 0.881 GiB，不代表进程总显存。单篇 64K 时专用实现比原 Grouped 路径慢约 10.4%，因此默认上限保留在 32K。

更早的独立 QSA 链路中，indexer 计算约占 5.8%。因此单独跳过 indexer 的空间有限，暴露的 replay 读取和同步开销还可能抵消收益。

### 端到端训练实验

以下为 32 卡 TP4/PP4/CP1/EP8、单卡动态 microbatch token 预算 128K、rollout TP4 配置的日志观测值。括号内为 Ray submission ID 的前八位，step 从零编号。表中排除冷启动 step 0，记录早期 step 1–2，并非长跑均值；算子 v1 的 step 2 受到 profiling 干扰，不纳入。

`perf/train_time` 是训练区域耗时，包含 log-probability 计算、actor 训练及相关工作；单独的 actor 更新由 `perf/actor_train_time` 记录。代码将 `perf/step_time` 定义为 `perf/train_wait_time + perf/train_time`，等待部分包含训练区域以外的 rollout、权重更新及 colocate 切换等时间。因此它不是 attention 算子耗时，也不等于简单的 rollout 加 actor 前反向。

| 实验 / run | Step | 训练 token 数（百万） | `train_time`（秒） | `step_time`（秒） |
| --- | ---: | ---: | ---: | ---: |
| 基线，不开 rollout replay (`f1849893`) | 1 | 2.978 | 72.13 | 480.56 |
| 基线，不开 rollout replay (`f1849893`) | 2 | 3.301 | 68.18 | 427.43 |
| 仅 QSA replay (`e9168b14`) | 1 | 3.163 | 81.41 | 506.51 |
| 仅 QSA replay (`e9168b14`) | 2 | 3.346 | 72.00 | 427.08 |
| QSA + R3，直接消费 bitmap (`59942491`) | 1 | 2.972 | 81.79 | 517.90 |
| QSA + R3，直接消费 bitmap (`59942491`) | 2 | 3.010 | 72.56 | 465.41 |
| QSA + R3，bitmap 算子 v1 (`b96d38fe`) | 1 | 3.444 | 82.53 | 512.26 |
| QSA + R3，bitmap 算子 v2 (`f6ee7d39`) | 1 | 3.164 | 80.31 | 534.96 |
| QSA + R3，bitmap 算子 v2 (`f6ee7d39`) | 2 | 3.093 | 63.18 | 442.27 |

以 step 2 为例，bitmap v2 的训练区域为 63.18 秒，基线为 68.18 秒；对应完整 step 却分别为 442.27 和 427.43 秒。各实验的 token 数和 rollout 长度不同，这些观测不能构成严格的 replay 或算子加速对照，也不能将上述单算子收益直接换算为 step time 收益。单开 QSA replay 没有明显降低实测 mismatch。

### 冻结视觉 embedding：OpenR1MM GRPO

2026-11-18 的 Qwen3.8-Flash-Next OpenR1MM 对照使用 32 张 Blackwell GPU（每卡 280 GB），训练 TP4/PP4/CP1/EP8、rollout 四个 TP8/EP8 引擎，单卡动态 microbatch token 预算 16384。Prompt batch 为 128，每题生成 8 个回答，global batch 为 1024；视觉编码器及 merger/projection 冻结。两组各跑 10 步，不保存 checkpoint，只切换 `--use-rollout-mm-embedding-replay`，均未开启 QSA indexer replay 或 R3。

| 耗时中位数（排除 step 0，统计 step 1–9） | 原生视觉路径 | MM embedding replay | 下降 |
| --- | ---: | ---: | ---: |
| `perf/train_time` | 81.01 s | 74.53 s | 8.0% |
| `perf/step_time` | 313.68 s | 302.81 s | 3.5% |

相同统计区间内，以训练 token 总数除以 `train_time` 总和得到的吞吐由 20.05K 提高至 24.40K token/s（约 21.7%）。生成结果和响应长度并非逐 token 相同，基线后续也触发过形状编译，因此这些是本配置下的观测收益，不能将吞吐变化全部归因于视觉计算的省略。

消费端输入 PP stage 的一个 TP0 rank 在代表步中实际读取约 126 MB，CPU 读取耗时约 0.9–1.1 秒，TP broadcast 的 CPU 调用耗时约 0.02 秒；缓存命中约 900 次、未命中约 120 次，无淘汰。这里的 CPU 耗时包含调用和等待，不是通过 CUDA 同步测得的 GPU 完成时间，也不是全局通信量。

训推一致性略有改善：两组前 10 步的 `train/mismatch_kl` 均值由 0.002208 降至 0.002012（约 8.9%），logprob 绝对差由 0.02814 降至 0.02703（约 4.0%）。后续 200 步长跑在观察时已完成 39 步，其前 10 步 mismatch KL 相对基线低约 6.5%；这不是长跑完成结果，也不足以单独证明 replay 对 mismatch 的因果影响。

## 运行时观察

Worker 日志通过 `replay_consumer` 输出 TP/PP/CP/DP 坐标。可关注 `replay/encoded_read_bytes`、`replay/encoded_h2d_bytes`、`replay/encoded_rdma_bytes`、`replay/gpu_cache_peak_bytes`、`replay/encoded_prefetch_hits` 和 `replay/encoded_prefetch_wait_seconds`。Tracking 后端记录一个代表训练 rank 的指标，不是所有 rank 的最大值。

结合 actor 和 log-probability 耗时判断：编码字节数用于观察压缩路径，H2D/RDMA 计数区分接收路线，prefetch 命中与等待时间反映前台阻塞。命中预取不等于传输已完全隐藏。

实现入口位于 `relax/distributed/replay_manager/`，QSA 适配在其中的 `backend/qwen38.py`，专用算子及 mixed 分流位于 `relax/models/qwen3_8/ops/qsa_mixed_ops/`。
