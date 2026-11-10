# Qwen3.8 QSA 后端与并行配置

本指南梳理 Megatron 训练镜像中的 QSA，源码基线为 PR1 提交 `97210aaf5b40db54eb63e80748b5f34b3433fb78`。底层 kernel 的本地形状要求、Relax 适配器允许范围、完整训练拓扑的验证状态是三种不同约束。后文的 NV 配方是配置示例，不能推广成所有后端和并行布局的保证。

## 执行层次

Selector 选择历史的**完整 block**；attention kernel 消费选择结果并补齐当前不完整的因果尾块。TP/SP/CP 在这些操作周围处理投影、token 布局和通信，HybridEP 则独立处理 MoE 通信。

```text
Model projections / TP-SP-CP layout
  -> selector -> QSASelection -> QSACoreAttention
       -> all-selected causal shortcut (unless forced sparse / mixed)
       -> dense_masked -> PyTorch SDPA with an explicit mask
       -> flex -> optional fused candidate -> Triton / triton_v2 / triton_v3
               -> otherwise native Flex / optimized BlockMask Flex
       -> id_sparse -> native selected-ID Triton
                    -> community=mixed -> Flash / Flex / Grouped
                                       -> Flash / cuDNN PR1765
```

Fused attention 替换 Flex 的计算，不替换 selector。Mixed 包装 selector，并接入 selected-ID attention 入口。Fused 候选不满足 `supports()` 时返回 Flex；mixed NV **不会**自动回退 Grouped。原生 CP1 输入若全部选中，可进入配置的 dense core；CP>1 有显式重建后的 causal 路径。Mixed 将 `all_selected` 绑定为 False，保留自己的短文档分流。依据：[原生 core][native]、[分发入口][dispatch]、[构建时接入][integration]、[mixed 入口][mixed]。

### 如何选择执行路线

下表为框架默认值，不含模型或训练脚本的环境变量覆盖。Driver 与训练 Actor 必须使用一致配置。实际 Megatron 实现由 [Docker 构建][docker] 应用补丁提供，不能用任意安装的 Megatron 副本替代审计基线。

| 路线与层次 | 配置 | 入口与默认值 |
| --- | --- | --- |
| 原生 dense 参考路径 | 清空 `RELAX_QWEN38_QSA_COMMUNITY_BACKEND`；`MCORE_QSA_SPARSE_BACKEND=dense_masked` | `QSACoreAttention._dense_masked_forward`；显式选择 |
| 原生 Flex | 清空 community；`MCORE_QSA_SPARSE_BACKEND=flex`；关闭 fused attention | `_flex_forward`；原生 backend 默认 `flex` |
| 优化 BlockMask 构建 | 原生 Flex 加 `RELAX_QWEN38_QSA_SPARSE_BLOCKMASK=1` | `qsa_block_mask._sparse_flex_forward`；框架默认关闭；改变 mask 构建方式 |
| Fused attention 实现 | 原生 Flex 加 `RELAX_QWEN38_QSA_FUSED_ATTENTION=triton`、`triton_v2` 或 `triton_v3` | `qsa_fused.backend`；框架默认关闭；`1`/`true` 也选择 Triton v1 |
| 原生 selected-ID kernel | 清空 community；`MCORE_QSA_SPARSE_BACKEND=id_sparse` | `_relax_native_id_sparse_forward`；显式选择 |
| Mixed 组合分发 | `RELAX_QWEN38_QSA_COMMUNITY_BACKEND=mixed`，关闭 fused attention，原生 backend 为 `id_sparse` | `qsa_community` 与 `qsa_mixed_ops`；community 非空时默认选 `id_sparse`，显式指定冲突 backend 会报错 |

[公共模型脚本][model-script] 默认**开启** BlockMask、local-query、replay、fused selector、fused attention，与[环境变量注册表][env]的框架默认不同。Community 非空时，该脚本关闭 fused selector，但没有同时关闭 fused attention。因此，继承 fused attention 为 `1` 的环境时，仅设置 community=mixed 会被 mixed producer 明确拒绝。应使用完整 mixed 配方，或在构建模型前显式设置以下组合：

```bash
export RELAX_QWEN38_QSA_COMMUNITY_BACKEND=mixed
export MCORE_QSA_SPARSE_BACKEND=id_sparse
export RELAX_QWEN38_QSA_FUSED_ATTENTION=0
export RELAX_QWEN38_QSA_FUSED_SELECTOR=0
export RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=grouped
```

这段配置只负责选路，仍须满足模型、冻结 indexer、布局和并行前提。

### Attention 输入契约

下面的 `Hq/Hkv/D` 指 **attention** 头数和维度，不是 selector 的独立投影。原生适配层接收 THD `[T,H,D]` 或 SBHD `[S,B,H,D]`，再转换成 kernel 布局。更改形状前应检查[原生方法][native]、[fused 适配][fused]、[fused kernel][fused-kernel]和 [mixed 入口][mixed]。

| Attention 实现 | 本地几何与 selection | 执行条件 |
| --- | --- | --- |
| 原生 `dense_masked` | 复制 K/V 实现 GQA；bitset selection；没有 Relax 的 12/1/D256 特化检查 | 显式 `[B,S,S]` 布尔 mask 加 SDPA，mask 内存为平方级；设备/dtype 取决于所选 PyTorch SDPA 实现，Qwen 配方使用 CUDA BF16 |
| 原生 Flex / BlockMask Flex | `enable_gqa` 支持分组；bitset 携带文档 ID 与位置；无 Relax 12/1 特化 | 需要 PyTorch FlexAttention 和编译支持。优化 BlockMask 要求 CUDA、非空 R4 bitset，否则使用原生 Flex；小输入 `S<=4096` 的 Dynamo 编译失败可走 dense-masked |
| 原生 `id_sparse` Triton | 整数 GQA、D 为不超过 256 的 2 的幂；int32 IDs/positions；top-k 为正，ratio 不超过 32 且整除 32 | 连续 CUDA Q/K/V 且形状匹配。IDs 必须是不重复、可见的完整块，填满 `min(visible,K)` 槽，空槽用 `-1`。原型没有显式 dtype 白名单，不代表任意 dtype 已有数值验证 |
| Fused Triton v1 / v2 / v3 | 共享 `supports`：CUDA、原生 `flex`、selection 单行、非空 R4 bitset、`D∈{64,128,256}`、整数 GQA | Kernel 接收 THD；非 THD 适配只接收 batch 1，否则回 Flex。Plan 含正反向 128-token tile 和可选全局 `query_ids`；入口没有 dtype 或仅限 Blackwell 的白名单，编译与数值可移植性需另验 |
| Mixed Flash / Flex / Grouped | R4 原生 IDs，按文档偏移展开 token 并补因果尾块；几何见后表 | CUDA BF16、CP1、冻结 Stage-1 indexer、indexer loss 为零。Grouped 使用 BF16 乘积与 FP32 K/V atomic 累加，不能套用 fused 的确定性 backward 结论 |
| Mixed Flash / cuDNN | R4 int32 IDs，每 token 严格 512 列，即 2048-token selection budget；几何见后表 | CUDA BF16，沿用 mixed producer 条件；长分支导入固定的可选包，短分支仍受 NV 适配器几何限制，环境要求见后文 |

[Fused v2][v2]/[v3][v3] 复用基础实现的 `supports()` 和 plan。SM103/BF16/D256 的 tile 选择是特化，不是所有 fused 路径都只支持 Blackwell。V3 的 **shared-dS backward 优化**另有窄范围保护：SM103、BF16、D256、R4、无局部 `query_ids`、Q/K 长度均为正且不超过 131072、int32 地址范围受限。共享 scratch 策略上限为 24 GiB；形状不满足或 scratch 不足时保留常规 backward。这不能改写成 v3 全部不支持 CP。见 [v3][v3] 的 `_can_share_ds` 和 `shared_backward.py`。

### Selector、packing 与训练阶段

[QSASelection][native] 保存文档内 `positions`、`doc_ids`，以及 byte bitset `selected_bits` 或完整块 ID `selected_ids`。R4 下预算 2048 token 对应最多 512 个完整 block，不是 2048 个 block ID。因果 mask 额外补最多三个当前尾块 token，不能跨 packed 文档边界。

| Producer / 适配层 | 配置与要求 | 边界与阶段条件 |
| --- | --- | --- |
| 原生 selector | `QSAIndexer` 参数来自 provider 的 indexer 配置，按 attention backend 输出 bits 或 IDs | 重建 CP 序列，并处理 SP gather；不能从 attention 的 global 24Q/2KV 推断 selector 头数 |
| Fused selector 后处理 | `RELAX_QWEN38_QSA_FUSED_SELECTOR=1`，框架默认关闭；`qsa_select.select_blocks` | CUDA 四头 selector、`bits` 输出、可用的 CPU offsets 等形状条件，不满足时走普通 selector；与 fused attention 分开控制 |
| Mixed TP selector | `RELAX_QWEN38_QSA_TP_SELECTOR=1`，默认 `0`，要求 mixed；可选 `RELAX_QWEN38_QSA_SELECTOR_COMPILE=1`，默认关闭 | CP1、冻结、IDs；selector 为 **4Q/1KV/D128**、R4、预算为正且可被 4 整除并不超过 2048；zero-centered RMSNorm、BF16 CUDA `[S,1,hidden]`、RoPE、显式 TP/单 rank CP group；gather 后序列非空且长度可被 TP 整除。按 query 分工，再 all-gather IDs |
| Mixed packed 元数据 | `qsa_mixed_ops.forward`、`bind_document_bounds` | 要求 producer 精确 CPU 边界：`qsa_physical_layout_cpu` 绑定或 `qsa_select_offsets_cpu`；selection 副本在 checkpoint replay 中保留边界。非 packed 的每个 batch 行是独立文档 |
| Stage-2 indexer loss | 原生 `id_sparse`、正 `qsa_indexer_loss_coeff` | 独立的受限 KL 路径：gather 后 `S*B<=4096`、attention dropout 为零、TP>1 时显式 TP group；THD 要求真实长度 `qsa_stage2_layout_cpu`。Mixed 拒绝该阶段，冻结 indexer replay 不能截断其投影梯度 |

Selector 入口优先级为 mixed 包装 → 已开启的 fused-selector 包装 → replay → TP-selector → CPU-layout/原生阶段。“CPU layout”指元数据处理，不是 CPU attention kernel。这些是带条件的入口，不是每个 batch 都执行所有分支。依据：[dispatch][dispatch]、[selector fusion][select]、[TP selector][tp-selector]、[mixed 入口][mixed]、[原生 loss attachment][native]。

## 并行布局与独立保护条件

| 维度 | QSA / 模型行为 | 限制所属层次 |
| --- | --- | --- |
| TP | 分片 global attention heads；TP 超过两个 KV 头时复制 KV 组，在组内缩小 Q/gate 分片 | 模型投影布局，不是 attention kernel 的完整拓扑验证；见[投影补丁][projection]与 `qsa_routed_qkv.route_table` |
| CP | 原生稀疏 core 重建完整 Q 和可求导 K/V，计算后切回本地输出 | 基础通信路径，每个 CP rank 仍持有完整激活；mixed producer 明确要求 **CP1** |
| CP local-query | `RELAX_QWEN38_QSA_LOCAL_QUERY=1` 保留本地 Q，gather K/V 并在反向汇总梯度 | 仅 CP>1、THD、CUDA、Flex、非空 R4 bitset、Stage-1/无可求导 indexer 张量、适用的稀疏 selection；不满足时回原生 core，不会启用 mixed CP>1 |
| SP | 开启 sequence parallel 时，原生/TP selector 从显式 TP group gather hidden states | 与 attention GQA 正交；TP selector 仍要求 gather 后长度可被 TP 整除，脚本的 `--sequence-parallel` 不是 kernel 单独提供的能力 |
| PP | QSA 在所属流水线层内执行，MTP 可在最后阶段增加 QSA 层 | Attention kernel 没有通用 PP4 检查，需另验层分配、MTP 和重算接入 |
| EP / ETP | MoE 通信独立于 attention 头数分组 | 即使 QSA 本地形状通过，所选 dispatcher 仍可能施加物理拓扑和包版本约束 |

对于 global 24Q/2KV/D256 模型，投影代码推导出的布局为：

| TP | 本地 attention Q/KV | KV 分配 |
| --- | --- | --- |
| 1 | 24/2 | 两个 KV 组均在本地 |
| 2 | 12/1 | 每 rank 一个 KV 组 |
| 4 | 6/1 | 每个 KV 组由两个 TP rank 共享 |
| 8 | 3/1 | 每个 KV 组由四个 TP rank 共享 |

[投影补丁][projection]在 query groups 少于 TP rank 数时 gather 投影分片，按 rank 选择 KV 组，再在组内切分 Q/gate。可选 [routed QKV][routed] 将 gather 换成显式 TP all-to-all，并在伴随反向中累加共享 KV 梯度。保护条件是 gated QSA、global 24/2/D256、TP4/8、CP1、BF16 或 FP32，排除 FP8/FP4/test mode；其他情况走原投影。这张表是**布局推导**，不是 TP1/2/4/8 完整训练通过记录。NV 适配器仍受下文的 mixed 几何表限制。

### MoE 拓扑与重算是独立契约

[HybridEP 预检][hybrid]仅在选择 HybridEP dispatcher 时执行。其 Qwen3.8 布局允许 TP2 **或 TP8**，固定 PP4/CP1/EP8/ETP1/world32；另一个 TP1/PP2/CP4/EP4/world16 来自 Qwen3.6 配方，不构成 mixed QSA CP4 支持。预检要求 EP×ETP 组内 rank 数正确且唯一、均在同一 hostname、每个成员探测到的可访问 peer 数正确、安装包匹配 `RELAX_HYBRID_EP_COMMIT` 且模块来源正确，禁止强制覆盖可访问 rank 数。因此 TP1/TP4 的合法本地几何不代表现有 Qwen3.8 HybridEP 配置会接受该拓扑。

`RELAX_QWEN38_QSA_RECOMPUTE_REPLAY=1` 将冻结的 selection/mask/plan 元数据缓存在 **checkpoint 调用独有**的 tape 中，可训练 indexer 走普通路径，K/V 保持可求导。它不同于 `RELAX_QWEN38_RETAIN_QSA_ATTENTION=1` 的“保留 attention、只重算 MLP”。后者的[保护条件][retain]刻意限定为 BF16、full/uniform/每层重算、TP8/PP4/CP1/EP8/ETP1；排除 FP8/FP4、activation offload、distributed saved activations、VPP 和 CUDA graphs，还检查层归属与 indexer 冻结状态。这是优化的保护范围，不是所有 QSA 的限制。

## Mixed 几何与文档长度分流

[geometry.py][geometry] 的共享检查 `geometry.validate_geometry` 同时用于 mixed 公共入口、Grouped 长分支适配和直接 NV 适配调用。

Mixed QSA 要求 CUDA BF16、D256、CP1，以及冻结的 Stage-1 indexer。本地头数的支持范围取决于所选长文档后端：

| 长分支后端 | 允许的本地头数布局 |
| --- | --- |
| `grouped` | Q、KV 头数为正，且 Q 头数能被 KV 头数整除，包括 12Q/1KV、24Q/2KV、6Q/1KV 和 3Q/1KV；这些示例不是白名单 |
| `cudnn_pr1765` | 当前 Relax 适配器仅接受 12Q/1KV/D256，全为短文档的 batch 也保留这一限制 |

框架中 `RELAX_QWEN38_QSA_MIXED_LONG_BACKEND` 的默认值为 `grouped`；下文的 NV 示例脚本选择 `cudnn_pr1765`。使用 24Q/2KV、6Q/1KV、3Q/1KV 等布局时，应显式设置 `RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=grouped`。NV 路线遇到不支持的布局会在执行前报错，不会自动回退到 Grouped。固定版本 NVIDIA 包的底层能力不等于当前 Relax 适配器已开放相同的头数布局。

通过本地几何检查不代表 GPU 输出与梯度数值正确性或完整 TP/CP/EP 训练拓扑已经验证，仍须满足 Megatron 和 HybridEP 的拓扑约束。Packed 输入必须携带 producer 提供的 CPU 文档边界；每篇文档独立处理 token 位置和不完整的因果尾块。


框架默认值为 `MIXED_LONG_BACKEND=grouped`、`MIXED_FLASH_MAX_LEN=-1`、`MIXED_FLEX_MAX_LEN=16384`，完整变量名均带 `RELAX_QWEN38_QSA_` 前缀。设 IDs 列数为 `K`，展开后的完整 token 预算为 `B=4*K`：

| Mixed 路线 | 短 / 中 / 长文档 | 可配置边界 |
| --- | --- | --- |
| Flash/Flex/Grouped | 生产 `K=512` 时，典型为 Flash 至 2051、Flex 从 2052 至 16384、之后 Grouped | 自动 Flash 上限为 **B+3**；Flex 上限范围 0..16384；显式 Flash 上限须在 0..B+3，且 Flex 上限必须为 0 |
| Flash/cuDNN | 强制 `K=512`，默认 Flash 至 2051、NV 从 2052 开始，无 Flex 中间分支 | Flash 上限 `-1` 解析为 2051，显式范围 0..2051；`NV_MAX_PACK_TOKENS=0` 时逐篇调用 NV，0..16384 可使用有界 mini-pack |

长度均指单篇文档，不是拼接 pack 总长。Grouped 组合会调整 selection 偏移并还原输出顺序，Flex 子组内部可 padding 到 16384；NV 从原生 IDs 构建文档内 byte mask 并补尾块。依据：[Grouped 分发][mixed-dispatch]、`flex.attention`、[NV 适配][nv]与 `routes.token_routes`。公共 QSA 训练脚本将 Flex 上限设为 **0**，因此其 Grouped 配方也不同于框架三路默认；NV 包装再将 Flash 上限设为 2051，后文配方表列出具体脚本值。

## 证据与验证边界

下表区分上述源码基线中的实现、测试覆盖和实际执行记录。“有用例”不等于已经运行。本次仓库审计未找到同时绑定源码版本、硬件和配置的 GPU 数值报告或完整分布式训练验收记录，相关列保留为**待确认**。PR1 会话中的 CPU 执行结果记录在变更说明中，不在这里转写成 GPU 验证。

| 路线 / 能力（均有实现） | CPU 契约覆盖 | GPU 数值 / 通信用例 | GPU 实跑记录 / 分布式训练记录 |
| --- | --- | --- | --- |
| 原生 dense / Flex | 布局与分发覆盖，不是 CUDA 执行的 CPU 数值证明 | `test_qsa_block_mask_patch.py`、`test_qsa_cp_patch.py` 覆盖 mask/Flex 行为与参考比较，不是完整原生 dense/Flex 分发或训练覆盖 | 待确认 / 待确认 |
| 原生 `id_sparse` | Community 分发契约 | 本次未定位到独立的原生 ID kernel 数值套件；mixed 测试执行其替换实现，不能算作该 kernel 的验证 | 待确认 / 待确认 |
| Fused v1/v2/v3 | 开关、tile 配置、v3 workspace 保护 | `test_qsa_fused_attention.py`：输出/梯度、v1/v2 几何、v3 SM103 shared-dS 与回退用例 | 待确认 / 待确认 |
| Mixed Grouped | `test_qsa_mixed_{routes,threshold,checkpoint,cudnn}.py`：几何、分流、边界/replay | `test_qsa_mixed.py`：packed、12/1、24/2、6/1、3/1 与稀疏预算 | 待确认 / 待确认 |
| Mixed NV | `test_qsa_mixed_cudnn.py`：几何拒绝、元数据、替代执行的梯度 | 同名 GPU 文件：SM100/SM103、固定包、12/1/D256/BF16，与 FP32 输出/QKV 梯度参考比较 | 待确认 / 待确认 |
| Selector | 动态 kernel 的 CPU interpreter / 离线编译与分发用例 | `test_qsa_select_patch.py`；`test_qsa_tp_selector.py` 有真实双 rank NCCL 的 TP2/CP1 用例 | 待确认 / 待确认 |
| CP local-query / replay | Replay 与布局契约 | CP query-ID 用例含**单 GPU 模拟 rank 切片**，不能证明多 rank 通信通过 | 待确认 / 待确认 |
| Routed QKV / HybridEP | `test_qsa_routed_qkv.py`、`test_qwen38_hybridep.py`：CPU 路由/伴随与模拟通信/预检 | CPU mock 不是 NCCL 或物理 peer discovery 的执行记录 | 待确认 / 待确认 |

测试位置分别见 [CPU 契约测试][cpu-tests]和 [GPU/后端测试][gpu-tests]。CUDA 或可选包缺失导致的 skip 不算通过。今后补充证据时，应保留源码 commit、镜像/包版本、GPU 架构、实际 TP/CP/SP/PP/EP/ETP、backend 与 selection budget、packed 形状、重算设置、输出/梯度比较及原始日志；分布式数值测试也须与完整训练、optimizer/checkpoint 验证分开记录。

用户提供的性能图标注 Blackwell GPU、TP2、local 12Q/1KV/D256/BF16，且明确 core timing 不含 selector/TP。本次没有其源码 commit、原始数值或训练日志，只能将其作为外部 benchmark 的描述。`mixed0`、`mixed16k` 不是当前 community backend 可选值，不能仅凭名称推断具体设置。`pr7553`、`TileLang` 是外部 benchmark 标签，不是当前 checkout 已启用的实现，community 校验会拒绝旧的 `pr7553`/`tilelang7060` 值。Flash 是组合内部的短文档 kernel，不是独立的全局 QSA selector。

### 源码索引

链接指向仓库实现，上文函数名对应具体保护条件；条件变化时应同步维护本指南。原生 QSA 由基础模型补丁和算子/投影接入共同组成，不能只看基础补丁。

- [原生 QSA、selected-ID kernel 与 Stage-2 loss][native]；[构建时算子接入][integration]；[TP/KV 投影补丁][projection]。
- [环境默认值][env]；[分发入口][dispatch]；[fused 适配][fused]；[基础 fused kernel][fused-kernel]；[v3 特化][v3]。
- [Mixed 入口与几何][mixed]；[Grouped 组合][mixed-dispatch]；[NV 适配][nv]；[TP selector][tp-selector]；[selector 后处理][select]。
- [CP/replay][cp]；[routed QKV][routed]；[attention 保留][retain]；[HybridEP 预检][hybrid]；[公共模型默认值][model-script]。

[native]: https://github.com/redai-studio/Relax/blob/main/docker/patch/megatron/20261016-qwen38-models.patch
[integration]: https://github.com/redai-studio/Relax/blob/main/docker/patch/megatron/20261020-model-operators.patch
[projection]: https://github.com/redai-studio/Relax/blob/main/docker/patch/megatron/20261020-qwen35-optimized.patch
[docker]: https://github.com/redai-studio/Relax/blob/main/docker/Dockerfile.cu13
[env]: https://github.com/redai-studio/Relax/blob/main/relax/utils/env.py
[dispatch]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/qsa_dispatch.py
[fused]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/qsa_fused.py
[fused-kernel]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/ops/qsa_fused_attention.py
[geometry]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/ops/qsa_mixed_ops/geometry.py
[v2]: https://github.com/redai-studio/Relax/tree/main/relax/models/qwen3_8/ops/qsa_triton_v2/
[v3]: https://github.com/redai-studio/Relax/tree/main/relax/models/qwen3_8/ops/qsa_triton_v3/
[mixed]: https://github.com/redai-studio/Relax/tree/main/relax/models/qwen3_8/ops/qsa_mixed_ops/
[mixed-dispatch]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/ops/qsa_mixed_ops/mixed.py
[nv]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/ops/qsa_mixed_ops/cudnn.py
[tp-selector]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/ops/qsa_mixed_ops/selector.py
[select]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/qsa_select.py
[cp]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/qsa_cp.py
[routed]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/qsa_routed_qkv.py
[retain]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/qsa_attention_recompute.py
[hybrid]: https://github.com/redai-studio/Relax/blob/main/relax/backends/megatron/hybridep_preflight.py
[model-script]: https://github.com/redai-studio/Relax/blob/main/scripts/models/qwen38-flash-next.sh
[cpu-tests]: https://github.com/redai-studio/Relax/tree/main/tests/cpu/backends/megatron/
[gpu-tests]: https://github.com/redai-studio/Relax/tree/main/tests/backends/megatron/

## NV 运行环境

使用兼容的 CUDA 13 Blackwell 训练环境，包含 Megatron/Bridge、HybridEP 和 Flash Attention。可选依赖为 [NVIDIA cuDNN Frontend PR1765](https://github.com/NVIDIA/cudnn-frontend/pull/1765)，固定到 commit **627c23f62e860fe2f76d6b38d2efcc2ee84c70cd**，CuTeDSL 使用 **4.8.1**。

仓库的 CUDA 13 Blackwell 镜像通过 PIP_CONSTRAINT 将 CuTeDSL 固定为 4.6.2，直接安装 4.8.1 会发生约束冲突。仅创建 venv 也不会清除继承的环境变量。应构建 Qwen3.8 NV 专用派生镜像，避免改动正在使用的基础环境。将下面内容保存为 Dockerfile.qsa：

~~~dockerfile
# syntax=docker/dockerfile:1.7
ARG RELAX_BASE_IMAGE
FROM ${RELAX_BASE_IMAGE}

RUN python - <<'PY'
import os
import re
from pathlib import Path

source = Path(os.environ["PIP_CONSTRAINT"])
replaced = re.compile(
    r"^\s*nvidia[-_.](?:cutlass[-_.]dsl|cudnn[-_.]frontend)\s*(?:[<=>!~@]|$)",
    re.IGNORECASE,
)
lines = [line for line in source.read_text().splitlines() if not replaced.match(line)]
lines.append("nvidia-cutlass-dsl==4.8.1")
Path("/etc/relax-qsa-runtime-constraints.txt").write_text("\n".join(lines) + "\n")
PY

ENV PIP_CONSTRAINT=/etc/relax-qsa-runtime-constraints.txt
RUN python -m pip install --no-cache-dir --upgrade \
      'nvidia-cutlass-dsl[cu13]==4.8.1' \
      'nvidia-cudnn-frontend @ git+https://github.com/NVIDIA/cudnn-frontend.git@627c23f62e860fe2f76d6b38d2efcc2ee84c70cd' && \
    python -c 'import importlib.metadata as m; assert m.version("nvidia-cutlass-dsl") == "4.8.1"; from cudnn.QSA.api import qsa_forward, qsa_backward'
~~~

这会复制基础约束，只替换 CuTeDSL 和 frontend 两个包的约束：CuTeDSL 固定版本，frontend 在安装命令中固定源码 commit。Torch、NumPy、Transformer Engine 等其余约束继续生效，不要全局取消 PIP_CONSTRAINT，也不要修改原始约束文件。

~~~bash
docker build --build-arg RELAX_BASE_IMAGE='<your-cu13-blackwell-image>' \
  -f Dockerfile.qsa -t relax-qsa:pr1765 .
~~~

所有 Ray 节点（包括 head）都要使用派生镜像。从源码构建固定版本还需要上游构建依赖和兼容的 cuDNN 安装，具体准备方式见该版本的[上游安装说明](https://github.com/NVIDIA/cudnn-frontend/tree/627c23f62e860fe2f76d6b38d2efcc2ee84c70cd)。仅有名为 cudnn 的其他 Python 包并不足够。Relax 不会自动安装或替换依赖；框架默认仍使用 Grouped，也不会导入这个可选后端。

构建和导入检查只覆盖依赖解析、CuTeDSL 版本及 QSA 入口，不代表已经验证 GPU 数值、TP/PP 重算，或其他 kernel 与新版 CuTeDSL 的兼容性。训练前须在派生镜像中验证相关路径；这份 Qwen3.8 专用环境不能直接视作所有 Blackwell 配方已验证的替代镜像。

## 选择训练入口

两套脚本均使用 4 节点 × 8 GPU、48 层、global 24Q/2KV、TP2/PP4/CP1/EP8/ETP1 和 sequence parallelism，采用 BF16、GBS512/MB1、每两层一次的 full uniform recompute，以及一层 MTP。

| 脚本 | 各 PP 阶段层数 | LR decay iterations | 输入 |
| --- | --- | --- | --- |
| 纯文本 | 14 / 12 / 12 / 10 | 996 | 文本消息和 tool calls |
| 多模态 | 12 / 14 / 14 / 8 | 2023 | 带图像引用的消息 |

纯文本分配让每个 PP 阶段各有三个主干 QSA 层。最后一段还包含 MTP QSA 和输出头，计入后 QSA 数量为 3/3/3/4。多模态脚本保留自身的 PP 分配；调整该布局属于单独的调优工作。

在仓库根目录运行。MODEL_DIR 是完整 HF checkpoint 目录，DATA_DIR 是 JSONL 文件或目录，SAVE_DIR 是所有节点都可访问的持久 checkpoint 根目录：

~~~bash
export MODEL_DIR=/models/qwen38
export DATA_DIR=/datasets/text
export SAVE_DIR=/checkpoints/qwen38
export MEGATRON=/workspace/Megatron-LM
export RELAX_HYBRID_EP_COMMIT='<installed-hybridep-commit>'
export RAY_JOBS_ADDRESS='http://<ray-head>:8265'

EXP_NAME=qsa-text-nv NUM_STEPS=50 \
  bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-nv-text-sft-32xgpu-blackwell.sh
~~~

进行图文 SFT 时，改为提供多模态 HF checkpoint 和数据，并使用：

~~~bash
EXP_NAME=qsa-mm-nv NUM_STEPS=50 \
  bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-nv-sft-32xgpu-blackwell.sh
~~~

两套脚本都读取 messages 和 tools 字段。多模态脚本额外把 image 映射到 images、将每张图像限制为 128 token，并在训练 rank 上预处理图像。纯文本脚本省去这三个多模态选项，不会把图像数据转成文本；使用该入口时应提供纯文本数据集。

脚本通过 scripts/entrypoint/ray-job.sh 准备集群，再提交 Ray Job。需要独占 Ray Serve 工作环境；直接调用时发现已有运行作业会拒绝启动，不提供并发作业隔离。默认阻塞等待提交的作业结束；只有需要单独监控返回的 job 时，才设置 RAY_NO_WAIT=1。

可以先查看有效配置，不触发 Ray 初始化、网络访问、GPU 导入或输出目录创建：

~~~bash
DRY_RUN=1 \
  bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-nv-text-sft-32xgpu-blackwell.sh
~~~

## 分流规则与对照

两套脚本会把相同设置传给 driver 和训练 Actor：

| 设置 | 脚本值 | 含义 |
| --- | --- | --- |
| RELAX_QWEN38_QSA_COMMUNITY_BACKEND | mixed | 沿用现有 selector 和 packed 输入路径 |
| RELAX_QWEN38_QSA_MIXED_LONG_BACKEND | cudnn_pr1765 | 长文档使用 NV |
| RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN | 2051 | 不超过 2051 token 的文档使用 Flash |
| RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN | 0 | 关闭中间的 Flex 分支 |
| RELAX_QWEN38_QSA_NV_MAX_PACK_TOKENS | 0 | 每篇长文档分别调用 NV |
| NVTE_USE_CUTLASS_GROUPED_GEMM | 0 | 默认关闭独立的 MoE GEMM 选项 |
| CUDA_DEVICE_MAX_CONNECTIONS | 1 | 对齐 TP/sequence-parallel 运行时 |

2051 由 512 个完整四 token block 加最多三个尾部 token 构成；长度 2052 的文档进入稀疏长分支。分流依据 producer 提供的真实单篇文档长度，而不是整个拼接 pack 的长度。

要运行相同纯文本配置的 Grouped 对照，只覆盖长分支和实验身份：

~~~bash
RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=grouped EXP_NAME=qsa-text-grouped \
  bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-nv-text-sft-32xgpu-blackwell.sh
~~~

不同 arm 要使用不同 EXP_NAME 或 SAVE_DIR，避免从对方的 checkpoint 恢复。可以单独设置 NVTE_USE_CUTLASS_GROUPED_GEMM=1 做 MoE 对照，但它既不选择 QSA 实现，也不能证明某个 MoE kernel 实际执行。NV mini-pack 属于实验选项，两套脚本均保持关闭。

## Checkpoint 与吞吐口径

脚本把 load/save 配置为稳定的 SAVE_DIR/EXP_NAME 路径，每 SAVE_INTERVAL 步保存一次（默认 50），保留两个 Actor checkpoint。恢复同一实验时保持路径不变，不要追加新时间戳；每次启动的 ClearML task 和日志名称会追加时间戳。脚本通过现有 Megatron 后端配置 checkpoint，dry-run 只检查配置，不代表 optimizer/RNG 状态已成功保存和恢复。

两套脚本关闭自定义 record-function 标签和 autotune 日志，默认不启用 PyTorch profiler；producer 队列超时为 120 分钟，自动全局重启次数为零。需要显式采集诊断 trace 时，参考[性能调优](./performance-tuning.md)。

对比 Actor 吞吐时，在相同 step 窗口使用 **有效训练 token 总数 / Actor 训练总秒数 / GPU 数量**。端到端吞吐则使用总 step 秒数，不能混算。保留慢步，将冷编译和 checkpoint 保存单列，并对齐数据、拓扑、packing 和重算设置。QSA 单算子的提速不会等比例转成整个训练流水线的提速。
