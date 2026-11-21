# Keye DSA replay 无损集合压缩方案

日期：2026-11-11。Keye 文本 DSA reference replay、GPU FULL/bitmap 编解码、SimpleStorage 和 Mooncake GDR 已接入并完成真实 RL 更新；批量 GDR 读取正在进行五步验证。下方原始设计与 PoC 章节保留研究时的记录，其中 LIST16/EXCEPT16 等候选编码没有接入生产，不应视为当前实现。

## 当前实现：如何启用

训练入口为 `examples/models/keye-vl2/scripts/run-keye-vl2-30B-A3B-8xgpu-math.sh`。
单机无 Ray 集群时直接 `bash` 该脚本，它会加载 `scripts/entrypoint/local.sh`。
以下命令会实际启动训练，不要在已有实验占用这些卡时重复执行。

功能开关彼此独立：

| 功能 | 开启方式 | 关闭方式 |
|---|---|---|
| DSA 选择捕获和训练 replay | CLI `--use-rollout-indexer-replay` | 不传此 CLI 参数 |
| reference 按需读取 | `RELAX_REPLAY_REFERENCES=1` | `0`；本次验证路径使用 `1` |
| Keye GPU FULL/bitmap 压缩与消费侧 GPU 解压 | Keye DSA reference 路径自动使用，无额外开关 | 本次未提供 raw/bitmap 切换参数 |
| Mooncake RDMA 与 replay GPU 接收 | CLI `--tq-rdma-mode required`，配置外部 master | CLI `--tq-rdma-mode off` 使用 SimpleStorage |
| 当前层、当前 microbatch 的批量 GDR 读取 | 上述 GDR 路径自动使用，无额外开关 | CPU/SimpleStorage 保持单 body 读取 |

`RELAX_REPLAY_REFERENCES=1` 本身不启用 DSA；必须同时传 DSA CLI 参数。
`--tq-rdma-mode required` 启用 TQ 的 RDMA 后端；replay 私有 reader 使用 GDR，普通样本客户端仍走 CPU 接收。
生产路径为 GPU 编码 → 压缩数据 D2H/CPU 存储 → RDMA 到 GPU 注册缓冲区 → 一次压缩数据 D2D → GPU 解压，并非全链路零拷贝。

### 三组相同配置、五步对照

在仓库根目录设置公共配置；模型、数据和输出根目录使用实际部署路径：

```bash
export MODEL_DIR=/your/model/root
export DATA_DIR=/your/data/root
export SAVE_DIR=/your/output/root
unset RAY_ADDRESS RELAX_ENTRYPOINT_MODE
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export SAVE_CHECKPOINT=0 NUM_ROLLOUT=5 DYNAMIC_SAMPLING=1
export ROLLOUT_BATCH_SIZE=16 N_SAMPLES=8 GLOBAL_BATCH_SIZE=128 RESPONSE_LEN=8192
export MAX_TOKENS_PER_GPU=8192 LOGPROBS_MAX_TOKENS=8192
export SGLANG_TP=2 SGLANG_MAX_TOTAL_TOKENS=262144 SGLANG_CONCURRENCY=32
export RAY_NO_WAIT=
export RELAX_REPLAY_CAPACITY_MB=131072
unset RELAX_REPLAY_RUN_ID  # 脚本为每次 reference 实验自动生成 UUID
```

依次运行，上一组结束后再启动下一组：

```bash
# 1. 不 replay DSA
RELAX_REPLAY_REFERENCES=0 EXP_NAME=keye-dsa-off \
  bash examples/models/keye-vl2/scripts/run-keye-vl2-30B-A3B-8xgpu-math.sh \
  --seed 1234 --rollout-seed 42 --tq-rdma-mode off

# 2. DSA reference replay + GPU bitmap，SimpleStorage，无 RDMA
RELAX_REPLAY_REFERENCES=1 EXP_NAME=keye-dsa-simple \
  bash examples/models/keye-vl2/scripts/run-keye-vl2-30B-A3B-8xgpu-math.sh \
  --seed 1234 --rollout-seed 42 --use-rollout-indexer-replay --tq-rdma-mode off

# 3. 相同 replay + GPU bitmap，Mooncake RDMA/GDR
# 必须先有可访问的外部 Mooncake master，以及可用的 RDMA/GDR 环境。
export MC_MASTER_ADDRESS=your-master-host:port
export MC_STORE_MEMCPY=0
export MC_GID_INDEX=3  # 本次机器实测值；其他环境须检查实际 RoCE GID 表
export RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB=4
export GLOG_minloglevel=2  # 可选，仅减少原生库日志，不保证所有 INFO 均被抑制
RELAX_REPLAY_REFERENCES=1 EXP_NAME=keye-dsa-rdma \
  bash examples/models/keye-vl2/scripts/run-keye-vl2-30B-A3B-8xgpu-math.sh \
  --seed 1234 --rollout-seed 42 --use-rollout-indexer-replay --tq-rdma-mode required
```

`required` 在 RDMA 不可用时失败，不允许降级后继续当作 RDMA 实验。
当前脚本固定 TP1/PP1/CP2 zigzag/EP8 训练、TP2 rollout、无 overlap、关闭 radix cache，冻结 indexer；这里只覆盖 Keye 文本路径。
五步实验没有触发独立评测，reward/accuracy 来自训练 rollout。

### 环境变量、默认值与容量区别

| 变量 | 默认值 / 是否必填 | 本次值与意义 |
|---|---|---|
| `RELAX_REPLAY_REFERENCES` | 默认 `0` | `1`，用 reference 传递 replay 数据位置 |
| `RELAX_REPLAY_CAPACITY_MB` | 默认 `16384` MiB | `131072` MiB = 128 GiB，replay inventory 的逻辑容量上限，不是预分配内存 |
| `RELAX_REPLAY_RUN_ID` | reference 脚本未设置时生成 UUID | 数据命名空间；独立实验不要手工复用同一个 ID |
| `MC_MASTER_ADDRESS` | RDMA 时必须提供 | 实际外部 master 的 `host:port`；本次临时服务地址记录于 `/tmp/keye-dsa-replay/mooncake-master.json`，不是通用默认值 |
| `MC_STORE_MEMCPY` | 框架强制 `0`，显式非零会报错 | 禁用当前 Mooncake 版本存在正确性问题的同机 memcpy 快捷路径 |
| `MC_GID_INDEX` | 由 Mooncake/部署环境决定 | 本机 `3`；是 RoCE GID 索引，不是 GPU 或网卡编号 |
| `RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB` | 默认 `4` GiB | 每个 Mooncake 客户端贡献的存储内存段，区别于 replay 逻辑容量 |
| `GLOG_minloglevel` | 可选 | 本次 `2`，期望只输出 ERROR/FATAL；不是功能开关 |

GPU 端的 64 MiB 压缩 cache、8 MiB 注册接收缓冲区，以及批量读取的 32 MiB 保守预算 / 32 key 上限是当前内部策略，无新增环境变量。批量预算不等于整个训练的物理显存上限。

### 验证是否走到预期路径

- replay 指标：`replay/encoded_read_bytes`、`replay/encoded_fetch_seconds`、cache 命中与淘汰数。
- SimpleStorage 组：`replay/encoded_h2d_bytes > 0`、`replay/encoded_rdma_bytes = 0`。
- RDMA 组：`replay/encoded_rdma_bytes > 0`、bitmap 的 `replay/encoded_h2d_bytes = 0`；reader 会检查 GDR 已启用且返回 CUDA tensor，失败即报错。
- 批量读取：`replay/encoded_read_batches < replay/encoded_read_bodies`，原生 `batch_get_into` 日志能看到多个 key；这些 reader 指标不等同于网卡硬件计数。
- 压缩传输仍有一次 D2D：`replay/gdr_staging_d2d_bytes`；H2D=0 仅描述 replay bitmap payload，不能推广到所有训练输入。

原始实验指标及源码 manifest 位于 `docs/draft/experiments/keye-dsa-compression-poc/formal/` 和 `optimized/`。
当前批量读取实验的本机启动记录为 `/tmp/keye-dsa-replay/run-optimized.sh`；仓库中的本节说明不依赖该临时脚本。

## 结论

推荐第一版为 **FULL / BITMAP / LIST16 / EXCEPT16 自适应编码**，越界范围保留 LIST32。SGLang 捕获后尽早压缩，TQ 和 CPU cache 保存压缩 bytes，训练消费当前层/本地 query 行时才展开为 attention kernel 所需的 int32 slots/counts。

不能只把 TQ 写入前的 tensor cast 成 uint16：原生 SGLang 的稠密 host cache、D2H，以及消费者反复展开的内存峰值都要一起处理。第一版不新增压缩依赖，不用有损选择，不改变 top-k 集合。

## 为什么当前代码是 int32

当前是 ABI 和已有实现约定，并非信息量要求：

- [ProducerPublication](../../relax/distributed/replay_manager/producer.py) 按 `topk * 4` 计算预算，检查 owned contiguous `np.int32`。
- [SGLang capture adapter](../../relax/distributed/replay_manager/backend/sglang.py) 检查 int32 CPU host buffer；原生 `BaseHostCache` 也使用固定稠密 int32。
- [Keye compact_indices](../../relax/models/keye_vl2/sparse_attention.py) 在消费边界转成 int32，供 native sparse-attention 的 slots/counts 使用。

存储编码与计算 ABI 可以分离。当前 rollout context-length=32768，逻辑 key ID 在标准文本路径中小于32768，uint16 足够。保存 count/row-valid，不需要在无符号数组里保留 -1 sentinel。超过范围必须改用32位或其他编码，禁止截断。

PyTorch 的 uint16 等无符号 dtype 在 eager 下支持有限，因此 wire/storage 推荐使用 `uint8` byte buffer；解码边界生成原有 int32，而非要求全链路 tensor 运算改成 uint16。[PyTorch dtype 文档](https://docs.pytorch.org/docs/main/tensor_attributes.html)

## 可无损压成集合的依据

Keye training、SGLang extend 和 paged decode 最终都会经过 `compact_indices`，先把有效逻辑 token ID 升序排序，再使用 slots/counts；teacher KL 使用 support 内的 indexer scores，而非 top-k 原始排名。该结论只适用于当前 Keye DSA adapter，不能直接推广到 MoE 的 ID/weight 配对或其他模型。

必须保留以下契约：

1. 编码输入是按原语义过滤后的、唯一的有效逻辑 ID。`compact_indices` 只排序，不去重；遇到重复必须报错，不能静默去重再声称无损。
2. ID 是文档内 token 坐标，不是 paged KV 物理地址，也不是 packed 全局行号。CP 只划分 query 行。
3. 解码输出与原 `compact_indices` 相同的升序有效 ID，保持浮点归约使用的顺序。
4. `row_valid=false`、有效 query 的空集合和缺失数据是不同状态。缺失/损坏必须失败，padding/末 token 由原 adapter 的规则处理。
5. FULL 只表示完整有效因果前缀。若存在 mask 内洞、特殊多模态规则、滑窗或其他有效性规则，不能仅凭长度推断 FULL。

## 第一版编码

对一行，`n` 为当前文档有效候选 universe `[0,n)` 的长度，`m` 为有效选择数量。第一版针对当前标准因果文本路径；一般 mask 只有在其有效性可以完整重建时才能用 FULL/EXCEPT。

| 模式 | 内容 | body 字节数，不含统一行头 |
| --- | --- | ---: |
| FULL | 全部 `[0,n)`，不保存 ID | 0 |
| BITMAP | universe 内每个位置一 bit | `ceil(n/8)` |
| LIST16 | 升序 uint16 ID | `2*m` |
| EXCEPT16 | 升序保存未选中的 ID | `2*(n-m)` |
| LIST32 | 大 ID 范围的兼容分支 | `4*m` |

在满足范围和 mask 契约的模式中，按实际 body+header 字节数选择最短。little-endian，bitmap 的 bit order 固定；尾部未使用 bit 必须为0。合法 Keye top-k 通常 `m=min(n,2048)`，因此模式大多由 CPU 已知的长度决定，不必每行 GPU→CPU `.item()`。

示例：`n=10240,m=2048`，LIST16=4096 B，BITMAP=1280 B，原定长 int32=8192 B；`n=2100,m=2048` 时 EXCEPT16 只需104 B，而 bitmap 为263 B。标准全因果前缀在 `n<=2048` 时全部选中，可采用 FULL。

这借鉴 Roaring 的“数组/位图/区间按形状选择”原则，但不直接为每行套标准 Roaring 容器：标准16位容器按65536 universe设计，2048元素通常会用4096 B数组，而本场景10K universe的位图只需1280 B。[Roaring 格式规范](https://github.com/RoaringBitmap/RoaringFormatSpec)

## 容量推导

固定48层、128条等长序列；`L` 是 prompt+response 总长度，capture 行数为 `L-1`。只计算有效 body，不计元数据、allocator、临时缓冲、TQ内部开销、冗余读取和同时保留的其他批次。

| 表示 | L=8192 | L=10240 |
| --- | ---: | ---: |
| 原定长2048×int32 | 383.95 GiB | 479.95 GiB |
| 仅改成定长uint16 | 191.98 GiB | 239.98 GiB |
| FULL前缀 + LIST16 | 143.98 GiB | 191.98 GiB |
| FULL前缀 + BITMAP | 22.51 GiB | 36.02 GiB |
| FULL/BITMAP/LIST16/EXCEPT16 自适应 | 22.41 GiB | 35.92 GiB |

位图推导：

```text
bytes = 128 * 48 * sum(ceil(n/8), n=2049..L-1)
```

每行使用1 B模式、2 B cardinality、4 B offset的保守估算，会再增加约0.33/0.41 GiB；chunk header、对齐等另算。32 prompt oversampling 候选批按条数翻倍，不能套用128条已选择样本的容量。多轮候选同时存活时还须计入其生命周期。

这些推导不要求选择聚集，因此可作为本配置第一版的可靠容量目标；不是生产实测压缩率。默认16 GiB逻辑容量仍不足最坏情况，不能在接线时忽略容量规划。

## 端到端接入位置

```text
SGLang top-k
  -> 原因果/有效性过滤 + 已有升序 slots/counts
  -> GPU encode 为压缩 bytes（仅发布 owner）
  -> 有界 D2H / stable chunk / 请求 tail
  -> TQ 按 record/layer/query-chunk 保存压缩 body
  -> 原 manifest、TokenHandle、CP row plan、lease
  -> 有界 compressed prefetch/cache
  -> GPU decode 当前层/当前本地行
  -> 原 int32 slots/counts + sparse attention
```

1. **先压缩再长期持有。** 当前 SGLang `BaseHostCache` 为整个 KV token pool 分配 `[tokens,48,2048] int32`。只在最终 publication 压缩，仍会保留巨大 host cache；应替换 Keye 的 capture storage，为已接受行发布压缩 chunk，为未发布尾部保留有界缓冲。不得在压缩前先长期积累整批 dense int32。
2. **GPU 早编码是最终目标。** 先做 CPU codec 可验证语义，但若原始 int32 已 D2H，只减少 TQ body，不能宣称减少了 D2H 或原生捕获缓存。Keye 原路径已做 token-ID 排序，encoder尽量复用，避免额外排序。
3. **每层/每 query chunk 一对象。** 初始比较128/256行 chunk，不承诺它优于1024；测 RPC开销、过读和背压。块内包含 mode、cardinality、offsets、query起点等；变长行可随机访问，不必解压整层或整批。批量TQ写入若要求等长tensor，可使用受预算约束的byte padding加真实长度，不能悄悄退化为unbounded stack。
4. **兼容的版本化 schema。** 现有 reference body/预算假定int32定长，需要增加明确codec/version/index-space描述，旧v2 raw路径仍可读，未知codec失败。TQ仍传byte字段，不新建独立存储或改Controller调度。
5. **预算按真实表示计算。** inventory按encoded bytes管理存储，producer队列计算编码输出与在途staging，prefetch同时计入压缩输入、展开输出、scratch。CPU cache优先保留压缩块；native int32只在当前层消费点展开，不能跨所有层长期缓存。
6. **复用生命周期。** seal成功后发布manifest；过滤候选、abort/retract、KV slot复用、reader lease、retire与错误清理沿用现有账本。ring/tail的覆盖必须等待写入及异步拷贝完成；不能仅凭forward结束就覆盖backward仍引用的索引。

第一版限定现有 Keye 文本脚本的无 speculative、关闭 radix cache、无 overlap rollout；以后支持 prefix reuse 时必须确保引用到的历史选择仍可读取，不能让ring覆盖它们。训练CP2 zigzag按原row plan取行，不改key ID。indexer辅助loss若启用仍需其可微scores；压缩离散选择不能代替梯度计算。

## 为什么暂不先上其他编码

- **差分+bit packing / Stream VByte**：适合升序整数，对真实gap分布有依赖。均匀、聚集、连续段需要分别测；不能把搜索引擎CPU吞吐数字套到GPU训练。[SIMD integer compression](https://arxiv.org/abs/1401.6399)、[Stream VByte](https://arxiv.org/abs/1709.08990)
- **Elias–Fano**：长上下文时值得第二阶段加入。例如n=32768、m=2048时，其不含索引辅助结构的估算约1536 B，优于4096 B bitmap；n=10240时估算约1088 B，与1280 B bitmap接近，第一版先选简单GPU解码。[Quasi-Succinct Indices](https://arxiv.org/abs/1206.4300)
- **跨相邻query的集合差分**：先测真实Jaccard。若只换入/换出d项，uint16增加/删除列表约4d B；n=8192时，要优于1024 B bitmap需d<256，即2048项中重合超过约87.5%。需要checkpoint限制随机访问依赖链，不能先假定高重合。
- **LZ4/Zstd/nvCOMP**：可在较大块上追加，但有压缩/解压、临时内存和依赖成本。nvCOMP支持GPU无损压缩及delta/RLE/bitpacking组合，不代表对本场景已有收益。[NVIDIA nvCOMP](https://docs.nvidia.com/cuda/nvcomp/)、[Cascaded encodings](https://docs.nvidia.com/cuda/archive/13.4.3/nvcomp/cascaded.html)

对独立、任意的2048元素集合，n=10240时单行信息量为`log2(C(10240,2048))`，约923 B；1280 B bitmap已接近该量级。真实选择若有跨行相关性可更小，但不能保证将任意集合无损压成几十字节。

## 验证计划与已有证据

已有CPU参考原型位于 `/tmp/keye-topk-compression-research/set_codec_prototype.py`，223组exact-set roundtrip通过，涵盖FULL、BITMAP、EXCEPT16、LIST16、LIST32、空集合、65535边界和大于65536的ID。容量推导在同目录 `capacity.json`。这不是生产codec，也没有测GPU吞吐。

落地验收：

- 比较解码后的升序有效ID/count与原`compact_indices`逐元素一致，测试sentinel、重复、未来/越界ID、mask孔洞、尾bit、截断块和损坏offset。
- 测短于/等于/长于2048、8K/10K/32K、两个文档、CP2 zigzag、dummy/末token与full recompute；比较主干输出/梯度，辅助loss开启时也比较indexer梯度。
- 独立测GPU encode/decode、D2H/H2D字节、压缩body、pinned/host/device峰值、writer背压和reader等待；观察完整step，不只压缩率。
- 先以原Keye配置做短跑raw/codec对照，再决定Elias–Fano或相邻行delta是否值得增加复杂度。不得用压缩损失或TopK集合变化换取性能。

## GPU / GPUDirect RDMA PoC 实测（2026-11-11）

代码位于 `scripts/benchmarks/keye_replay_poc/`，原始测量位于
`docs/draft/experiments/keye-dsa-compression-poc/`。单张 H20 96GB，其他七卡运行占卡任务。
输入为 GPU 上生成、排序的唯一 top-2048 集合；大批量复用 128 个随机集合。
这是 codec/搬运 PoC，不是 Keye 训练端到端收益。未修改生产 TQ 路径。

计时使用预热后的 CUDA events，30 次中位数；不含编译、分配、输入生成和校验。
长度 8192，bitmap 每行 1024B，对比原始 int32 每行 8192B：

| 行数 | 原始 / bitmap | 压缩 ms | 解压 ms | 压缩+D2H+H2D+解压 ms | 原始 D2H+H2D ms |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.25 / 0.03125 MiB | 0.0164 | 0.0162 | 0.0366 | 0.0224 |
| 256 | 2 / 0.25 MiB | 0.0165 | 0.0165 | 0.0416 | 0.0868 |
| 4096 | 32 / 4 MiB | 0.0460 | 0.0523 | 0.2746 | 1.2180 |
| 196608（4096×48） | 1536 / 192 MiB | 1.9731 | 2.2583 | 11.9309 | 57.9166 |

小批次 eager Python launch 间隙明显，event 数字不是纯 kernel 指令耗时；
32 行不值得单独编码后走 CPU staging，应聚合或融合 capture。
长度 10240、4096 行时压缩 0.0463ms、解压 0.1676ms，payload 5MiB；
解码使用 next-power-of-two word tile，非 2 次幂 universe 存在额外工作，后续可优化。
长度 32768 时 bitmap 与 uint16 都为 16MiB，但 uint16 解压约 0.0194ms，
bitmap 约 0.2038ms，选择 codec 必须考虑延迟，不应只比较字节数。
FULL 前缀 payload 为零；当前测试仍发射空编码 kernel，集成时应跳过。

### 真正的 RDMA 测试边界

使用现有 libibverbs 编译独立 RC WRITE 探针，`mlx5_bond_0` → `mlx5_bond_1`，
RoCEv2 GID index 3，单个 outstanding write，20 次平均；设备名/GID 是本机探测值，
探针仅用于当前环境。无需新增依赖、代理或修改驱动。
GPU buffer 经 `ibv_reg_mr` 成功注册，启用 `CU_POINTER_ATTRIBUTE_SYNC_MEMOPS`；
编码后同步，等待 CQ completion 后从 CPU 提交解码，避免 GPU kernel 与 RDMA 写并发读取。
内存排序依据 [NVIDIA GPUDirect RDMA 文档](https://docs.nvidia.com/cuda/gpudirect-rdma/)。
传输没有 payload D2H/H2D 或显式 D2D staging。

实际 bitmap → RDMA → bitmap → GPU 解码均与原始 int32 完全一致：

| 行数 / 长度 | bitmap payload | RDMA ms（不含建连、MR 注册） |
|---:|---:|---:|
| 4096 / 8192 | 4 MiB | 0.1988 |
| 4096 / 10240 | 5 MiB | 0.2442 |
| 196608 / 8192 | 192 MiB | 8.7460 |

原始 32MiB GPU buffer RDMA 为 1.4729ms（约22.8GB/s）；
4096行/8192 的压缩+RDMA+解压各阶段相加约0.297ms，对比原始RDMA约1.473ms。
**阶段和不是实测端到端延迟**，未包含控制面/同步开销，也未证明跨节点、TQ、网络线速或RL收益。
本轮每次 probe 重建连接与注册 MR，`including_setup_s` 保存这一整体开销；
生产必须复用连接、注册池和压缩 buffer，不能逐次重建。

下一步应在 capture 时直接写压缩 GPU buffer，TQ 仅传压缩 byte payload，
训练侧按 microbatch 解压到最终 int32 消费 buffer；避免先生成完整 host int32 cache、
再压缩，也避免额外的完整 GPU int32 副本。当前 PoC 仍以既有 int32 tensor 为输入，
尚未验证该 capture 融合路径，也未测生产 replay cache 峰值显存。

复现：先设置 `KEYE_POC_OUTPUT` 为结果目录，以及经实际探测的
`KEYE_RDMA_SRC_HCA`、`KEYE_RDMA_DST_HCA`、`KEYE_RDMA_GID_INDEX`，然后
`gcc -O2 -shared -fPIC scripts/benchmarks/keye_replay_poc/rdma.c -libverbs -o "$KEYE_POC_OUTPUT/rdma.so"`，
在空闲 GPU 上依次运行 `benchmark.py`、`check_rdma.py`、`compressed_rdma.py`。
