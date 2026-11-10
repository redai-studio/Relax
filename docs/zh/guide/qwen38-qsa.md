# Qwen3.8 QSA 算子与 Mixed 配置

QSA 首先由 selector 选择参与计算的历史块，再由 attention 算子完成计算。本文整理当前支持的算子、并行配置条件及开启方式。相关开关仅改变 attention 的执行路径，不改变 selector 的选择结果。

## 算子、并行条件与开启方式

表中的并行条件以当前模型的全局 **24Q/2KV/D256** 为例。TP 决定每张卡的本地头数；CP 决定同一条序列是否沿 token 维度跨卡切分。**CP1 表示不做上下文切分，不表示只使用一张 GPU。**

表中符号说明：

- **Q/KV/D**：每个 TP rank 上的 attention Q 头数、KV 头数和每头维度，不是 selector 的头数。
- **R4**：每个历史块包含 4 个 token。常用的 2048-token 选择预算对应 512 个块 ID，另补当前块内的因果尾部 token。
- **GQA**：Q 头数能被 KV 头数整除。
- 表中 `FUSED_ATTENTION`、`SPARSE_BLOCKMASK`、`MIXED_LONG_BACKEND` 等短名均需加上 **`RELAX_QWEN38_QSA_`** 前缀；`MCORE_QSA_SPARSE_BACKEND` 使用原名。

非 Mixed 路线先清空 community 并关闭 fused attention，避免继承模型脚本的默认开关；然后按表选择。环境变量要在构建模型前设置，并一致传给 driver 和训练 Actor。

```bash
export RELAX_QWEN38_QSA_COMMUNITY_BACKEND=''
export RELAX_QWEN38_QSA_FUSED_ATTENTION=0
```

| 算子 | 具体做法 | 并行配置条件 | 主要限制 | 开启方式 |
| --- | --- | --- | --- | --- |
| cuDNN PR1765 | 将选中的块转换为 byte mask，调用 NVIDIA QSA 算子 | **本模型要求 TP2、CP1**：TP2 后每张卡为 12Q/1KV/D256，符合当前适配器的固定输入要求；TP1/4/8 的本地头数不符合，CP>1 也不接受 | CUDA BF16；冻结 indexer、indexer loss 为 0；R4、512 列 int32 块 ID；需要额外依赖 | 通过 Mixed 设置 `MIXED_LONG_BACKEND=cudnn_pr1765`，完整配置见下文；`MIXED_FLASH_MAX_LEN=0` 可使所有文档使用 cuDNN |
| Grouped | 对共享同一 KV 头的 Q 头进行分组计算，将选中的块展开为 token 索引，仅访问这些 token | **要求 CP1**；本地 Q 头数能被 KV 头数整除，D256。本模型 TP1/2/4/8 的本地布局均满足头数条件，不固定要求 TP2 | 通过 Mixed 接入；CUDA BF16、冻结 indexer；反向用 FP32 atomic 累加 K/V 梯度 | Mixed 公共配置 + `MIXED_LONG_BACKEND=grouped`，将 Flash 和 Flex 上限都设为 `0` |
| Flash | 对短文档直接执行完整的因果 attention | **要求 CP1**；通过 Grouped 路线接入时，本模型 TP1/2/4/8 满足头数条件；通过 cuDNN 路线接入时，整个入口要求 TP2，即使全部文档使用 Flash 也需满足 | 仅作为 Mixed 内部分支；不能用它替代长文档的稀疏选择；需要 Flash Attention | 通过 Mixed 的 `MIXED_FLASH_MAX_LEN` 控制，默认自动取选择预算 + 3 |
| Triton v1 | 按 bitset 标记的稀疏块计算 attention，复用 selector 结果 | 本地 Q 头数能被 KV 头数整除，本模型 TP1/2/4/8 满足头数条件；CP1 直接计算，CP>1 由原生框架收集 Q/K/V 后计算，再切分本地输出 | CUDA、R4、D 为 64/128/256；selection 为单行；非 THD 输入要求 batch=1；不满足条件时回退到 Flex | `MCORE_QSA_SPARSE_BACKEND=flex`、`FUSED_ATTENTION=triton` |
| Triton v2 | 沿用 v1 的选择结果，调整循环、tile 和反向计算 | TP 和 CP 条件与 v1 相同：不固定为 TP2；CP>1 通过框架处理通信 | 同 v1 | `MCORE_QSA_SPARSE_BACKEND=flex`、`FUSED_ATTENTION=triton_v2` |
| Triton v3 | 沿用 v2 前向，反向可共享 FP32 dS，减少重复计算 | TP 和 CP 条件与 v1 相同；使用 CP local-query 时不启用共享 dS 优化，仍保留常规反向路径 | 同 v1；共享 dS 另限 SM103/BF16/D256、Q/K 长度不超过 128K，并受地址范围和临时显存限制；不满足时使用常规反向路径 | `MCORE_QSA_SPARSE_BACKEND=flex`、`FUSED_ATTENTION=triton_v3` |
| flex | 根据选择结果生成 mask，调用 PyTorch FlexAttention | 本地 Q 头数能被 KV 头数整除，本模型 TP1/2/4/8 满足头数条件；CP1 直接计算，CP>1 由原生框架收集 Q/K/V、计算并切分输出 | 需要 FlexAttention 和编译支持 | `MCORE_QSA_SPARSE_BACKEND=flex`、`FUSED_ATTENTION=0`、`SPARSE_BLOCKMASK=0` |
| BlockMask Flex | 使用 FlexAttention，优化 BlockMask 的构建 | TP 和 CP 条件与 Flex 相同；BlockMask 优化仅改变 mask 构建，不改变并行布局 | 优化要求 CUDA、R4、非空 bitset，否则回退到原生 Flex | Flex 配置 + `SPARSE_BLOCKMASK=1` |
| id_sparse | 直接读取选中的块 ID，通过原生 Triton kernel 完成计算 | 本地 Q 头数能被 KV 头数整除，本模型 TP1/2/4/8 满足头数条件；CP1 直接计算，CP>1 由原生框架收集 Q/K/V、计算并切分输出 | 连续 CUDA Q/K/V；D 为不超过 256 的 2 的幂；int32 IDs/位置；正 top-k；压缩比不超过 32 且能整除 32 | 清空 community，`MCORE_QSA_SPARSE_BACKEND=id_sparse`、`FUSED_ATTENTION=0` |


并行配置说明：

- **原生与 Mixed 的区别**：原生 `id_sparse` 可接原生 CP 流程；设置 `COMMUNITY_BACKEND=mixed` 后，同一入口的计算由 Mixed 替换，必须遵守 CP1 限制。
- **CP**：原生路径在 CP>1 时先汇集完整 Q/K/V，计算完成后切分为本地输出。Flex/Triton 可配合 `LOCAL_QUERY=1` 只保留本地 Q、汇集 K/V，但要求 THD、CUDA、R4 bitset 和冻结的 Stage-1 indexer 等条件；不满足时回退到原生路径。Mixed 当前只允许 CP1。
- **SP**：selector 会按 TP group 汇集序列。SP 不改变表中的本地 attention 头数要求；可选 TP selector 只用于 Mixed，并要求汇集后的序列长度能被 TP 整除。
- **PP / EP**：分别负责层分配和 MoE 通信，不由 QSA 算子决定。表中的条件是算子和适配层条件；TP1/2/4/8 满足头数要求不代表完整训练拓扑已经验证；使用 HybridEP 时还要满足其拓扑检查。

## Mixed 几何与文档长度分流

**Mixed 是按单篇文档的 token 长度选择 attention 算子的分流机制，本身不是独立的计算算子。** 当前有两条路线：**Flash/Flex/Grouped** 和 **Flash/cuDNN**。

第一条路线中，`mixed0` 和 `mixed16k` 是两种配置名称：分别表示关闭 Flex 分支、将 Flex 分支上限设为 16K。Grouped 是该路线中实际执行长文档计算的算子。两种配置使用相同的并行条件，仅分流边界不同。

第二条路线将长文档后端替换为 `cudnn_pr1765`：将 selector 选出的块转换为 byte mask，交给 NVIDIA cuDNN Frontend PR1765 的 QSA 实现计算；短文档仍使用 Flash，没有 Flex 中间分支。它与 Grouped 使用相同的选择结果，区别在于稀疏数据的表示与计算实现。

`mixed0` / `mixed16k` 不是 backend 可选值，两者均设置 `COMMUNITY_BACKEND=mixed`、`MIXED_LONG_BACKEND=grouped`。两条路线都要求 CUDA BF16、CP1、冻结的 Stage-1 indexer，且 indexer loss 为 0。Packed 输入需要携带 producer 提供的准确文档边界，每篇文档各自计算位置和因果尾块。

### 本地头数与 TP

| Mixed 长文档后端 | 本地 attention 条件 | 对 global 24Q/2KV/D256 模型的含义 |
| --- | --- | --- |
| `grouped` | Q/KV 头数为正，Q 能被 KV 整除，D256 | TP1/2/4/8 分别得到 24Q/2KV、12Q/1KV、6Q/1KV、3Q/1KV，均能通过几何检查 |
| `cudnn_pr1765` | 仅接受 **12Q/1KV/D256** | 当前应使用 TP2；即使 batch 中全是短文档，也会先检查此限制 |

TP4/8 时，模型投影层会在 TP rank 之间共享 KV 组。上表说明代码允许的头数布局，不是完整训练验证记录。cuDNN 路线遇到不支持的布局会报错，不会自动换成 Grouped。

### 按单篇文档长度分流

以常用的 **2048-token 选择预算（512 个 R4 块）**为例：

| 路线 | ≤2051 token | 2052～16384 token | >16384 token |
| --- | --- | --- | --- |
| mixed16k：Flash/Flex/Grouped，Flex 上限 16K | Flash | Flex | Grouped |
| mixed0：Flash/Grouped，Flex 上限 0 | Flash | Grouped | Grouped |
| Flash/cuDNN | Flash | cuDNN PR1765 | cuDNN PR1765 |

2051 = 2048 个选中 token + 最多 3 个因果尾部 token。在这个范围内，不需要丢弃可见历史，因而可用完整的因果 Flash attention。长度按**单篇文档**计算：例如 32K 的 pack 里装了多篇 1K 文档，这些文档仍各自走 Flash。

Grouped 路线的自动 Flash 边界随选择预算变化，为 `4 × 块 ID 列数 + 3`；cuDNN 路线固定要求 512 列块 ID，自动边界为 2051。

### 开启方式

先设置两条 Mixed 路线共用的配置。模型自身仍需设置为冻结 indexer、indexer loss 为 0；这些环境变量只负责选路。

```bash
export RELAX_QWEN38_QSA_COMMUNITY_BACKEND=mixed
export MCORE_QSA_SPARSE_BACKEND=id_sparse
export RELAX_QWEN38_QSA_FUSED_ATTENTION=0
export RELAX_QWEN38_QSA_FUSED_SELECTOR=0
```

**路线一：Flash/Flex/Grouped。** 以下为 mixed16k 配置，在 16K 以内保留 Flex 中间分支；将 `MIXED_FLEX_MAX_LEN` 改为 `0` 即为 mixed0，采用 Flash/Grouped 两段分流。

```bash
export RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=grouped
export RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN=-1
export RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN=16384
```

**路线二：Flash/cuDNN。** 没有 Flex 中间分支；超过 Flash 边界后使用 cuDNN。

```bash
export RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=cudnn_pr1765
export RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN=-1
export RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN=0
export RELAX_QWEN38_QSA_NV_MAX_PACK_TOKENS=0
```

cuDNN 路线需要兼容的 CUDA 13 Blackwell 环境，以及额外的 NVIDIA cuDNN Frontend PR1765 包，当前接入固定到 commit `627c23f62e860fe2f76d6b38d2efcc2ee84c70cd`，CuTeDSL 为 `4.8.1`。仓库基础镜像将 CuTeDSL 固定为 `4.6.2`，需使用准备好这些依赖的专用环境。`NV_MAX_PACK_TOKENS=0` 表示逐篇调用 cuDNN。

默认值与配置说明：

- 框架默认长后端是 `grouped`，Flash 上限为 `-1`（自动），Flex 上限为 `16384`；公共 QSA 训练脚本会把 Flex 上限设为 `0`，实际以启动脚本为准。
- Grouped 路线的 Flex 上限范围是 `0..16384`。若显式设置 Flash 上限，必须同时将 Flex 上限设为 `0`；保留 Flex 中间分支时，Flash 上限用 `-1`。
- 公共模型脚本默认开启 fused attention，Mixed 不能与它同时使用，所以公共配置中的 `FUSED_ATTENTION=0` 必须显式设置。

实现入口：[Mixed 分发](https://github.com/fxsc03/Relax/blob/fit/fengxiaoshi/QSA/relax/models/qwen3_8/ops/qsa_mixed_ops/mixed.py)、[几何检查](https://github.com/fxsc03/Relax/blob/fit/fengxiaoshi/QSA/relax/models/qwen3_8/ops/qsa_mixed_ops/geometry.py)、[cuDNN 适配](https://github.com/fxsc03/Relax/blob/fit/fengxiaoshi/QSA/relax/models/qwen3_8/ops/qsa_mixed_ops/cudnn.py)、[Triton 选择](https://github.com/fxsc03/Relax/blob/fit/fengxiaoshi/QSA/relax/models/qwen3_8/qsa_fused.py)。
