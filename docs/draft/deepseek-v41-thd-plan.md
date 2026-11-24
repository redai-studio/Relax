# DeepSeek V4.1 THD 训练支持计划

> 状态：四层 THD 适配、同环境 BSHD/THD 各 100 步 SFT、两轮在线 THD RL 及同 rollout 的 BSHD 重放已完成；完整模型与多节点验证仍待推进。调研更新于 2026-11-15。实测数值、启动重启记录与验证边界见 [实施记录](../../deepseek_v41_thd/progress.md)，不以单项测试通过或很小的平均 log-prob 差代替逐 token 精度结论。
>
> 仓库基线：`735e9b9a02277d2e7b6b64e4466ebf54859ae311`，即原 [Relax #460](https://github.com/redai-studio/Relax/pull/460) 的 V4.1 BSHD 实现合入 main 后的版本。运行环境只考虑 `docker/Dockerfile.cu13`，实际硬件按 H800 处理，不根据 L20Y 等伪装名称选择内核或镜像架构。

## 1. 路线结论

近期主线是：**保留 Relax 已有 V4.1 模型、Bridge 转换、TP/PP 和训练闭环，以 Miles #4488 为直接迁移参考，引入正确的 packed THD 语义；同时采用官方 Core #8018 的布局、CP 和重计算设计校验实现。**

三个 PR 都有价值，但承担不同职责：

| 来源 | 在本计划中的用途 | 不能据此假定的能力 |
| --- | --- | --- |
| [Megatron-Bridge #6199](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6199) | HF 配置、权重映射、量化 checkpoint 转换、Engram 加载的官方对照；后续 native 模型接入入口 | Bridge 本身不提供 THD attention 算子；当前 provider 明确限制 PP=1 |
| [Megatron-LM #8018](https://github.com/NVIDIA/Megatron-LM/pull/8018) | 原生 CSA2 的 THD、contiguous CP、跨层状态和 full recompute 的算法与测试参考 | 当前实现明确限制 TP=1、PP=1；多 rank CP 测试不等于 TP/PP 支持 |
| [Miles #4488](https://github.com/radixark/miles/pull/4488) | 在已有 V4.1 模型上补齐分段位置、压缩、indexer、Engram 的直接实现参考 | 不包含 Relax 自有 PP carry 适配；作者训练报告不能替代 H800、PP、多轮权重同步验证 |

选择现有底座的理由是保留已经做过精度对齐的行为和必需的 TP/PP 能力。它不是要求继续采用 BSHD 训练：**BSHD 留作精度和性能对照，THD 才是本次交付目标。**

不把上述三个 PR 整体叠加为安装步骤。官方 native 路线使用 HybridModel/HybridStack；当前 Relax 使用自定义 `DeepSeekV41GPTModel`。迁移算法和切换整个模型实现是两个不同工作包。待 native 路线满足所需 TP/PP，并完成训练、同步和恢复的一致性验证后，再评估整体切换。

## 2. 首期交付范围

必须交付：

- [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) 的完整语言主干训练（仅文本输入），沿用已有 BF16 checkpoint 契约和冻结 Engram 表，不借 THD 改变可训练参数集合。
- 一次 microbatch 中包含多条不同长度样本的 THD packing，支持动态 batching；不仅验证单样本加 padding。
- TP、SP、PP、contiguous all-gather CP，以及它们与 full recompute 的组合。至少覆盖历史四层基线的 TP2/PP2/CP2/EP4/ETP1。
- 保留 optimizer CPU offload、DeepEP（`flex/deepep`）、MoE routing replay（R3）、训练/推理 offload、Megatron → SGLang 权重同步、checkpoint 保存与恢复，并验证它们与上述并行和重计算配置同时开启。
- 用同一份 rollout 对比 BSHD/THD 精度，并在同一 H800 配置上测量有效 token 吞吐、显存和训练步耗时。

首期不扩展 vision/audio、MTP/DSpark、FP8/FP4 权重训练、VPP、自定义非均匀 PP layout、动态 CP 或训练 CUDA graph。attention indexer replay 与 packed THD 的组合暂不承诺支持，应明确拒绝未实现的配置；它与 MoE routing replay 是两套机制，不能用关闭 R3 来绕过验收。

首个联合验收配置固定为：**BF16 + THD dynamic batching + TP2/SP + PP2 + CP2 contiguous all-gather + EP4/ETP1 + full/uniform/1 recompute + optimizer CPU offload + flex/deepep**。现有四层 BSHD 脚本已经同时配置这些并行、重计算、offload 和 dispatcher；THD 要保留整套组合，而不是分别跑通就算完成。

边界需要明确：

- PP 首期均匀分层，不使用 VPP 或 `overlap_moe_expert_parallel_comm`；该调度开关与已有 `--overlap-cpu-optimizer-d2h-h2d` 不同，后者保留并验证。
- full recompute 首期锁定 `uniform`、`recompute_num_layers=1`，不承诺所有 selective attention/mHC 模式。
- DeepEP 首期指 `flex/deepep`，不扩大到所有 `hybridep` / DeepEP v2 实现。安装包名称不等于训练时选择的 dispatcher。
- 完整模型现有 128 卡 RL 脚本使用 `alltoall`，尚不能作为 EP64 + DeepEP 的验证证据；完整模型阶段必须补 `alltoall` 与 `flex/deepep` 的数值、通信及稳定性对照。
- CPU offload 需要验证 optimizer step、状态恢复、主机内存与传输峰值，不能以单次 forward/backward 通过代替。Miles 的 NVMe optimizer streaming 也不等于当前 Relax 的 optimizer CPU offload，本期不自动引入磁盘 offload。

## 3. 当前代码与实验基线

### 3.1 已有能力与缺口

下表描述实施起点的代码基线；新增能力的实测状态以实施记录为准。

| 层次 | 当前实现 | 本次工作的边界 |
| --- | --- | --- |
| 模型/provider/转换 | [V4.1 模型目录](../../relax/models/deepseekv41/)、[Bridge](../../relax/models/deepseekv41/bridge.py) | 保留现有注册、HF 映射、冻结策略和 checkpoint 契约 |
| 执行限制 | [execution_config.py](../../relax/models/deepseekv41/execution_config.py) 只允许 BSHD；model/attention 也主动拒绝 `packed_seq_params` | 实现完整语义后再放开限制，不能只删除检查 |
| THD 数据层 | [data.py](../../relax/backends/megatron/data.py) 已有通用 THD、真实长度、物理 padding 和 CP metadata | 接入 V4.1 专用布局解释，避免破坏其他模型 |
| attention/indexer/Engram | 当前按 BSHD 整条序列处理 | 加入样本边界，不能让前一个样本或尾 padding 影响后一个样本 |
| PP/full recompute | [runtime.py](../../relax/models/deepseekv41/runtime.py)、[model.py](../../relax/models/deepseekv41/model.py) 用显式 carry 保存跨层状态 | carry 形状与生命周期必须随 THD 一起改，这是主体工作，不是最后补一项测试 |
| RL 数据与 loss | [actor.py](../../relax/backends/megatron/actor.py)、[loss.py](../../relax/backends/megatron/loss.py)、[CP 工具](../../relax/backends/megatron/cp_utils.py) | 验证 response 顺序、有效 mask、CP 输出布局、routing replay 和归约口径 |
| Rollout | [SGLang V4.1 适配](../../relax/backends/sglang/deepseekv41/)、版本化 [0.5.21 补丁](../../docker/patch/sglang/v0.5.21.patch) | 保留现有权重更新和 Engram 生命周期修复，训练 THD 不直接要求更换推理引擎 |

当前 PP 实现要求均匀分层、`variable_seq_lengths=True`，不支持 VPP；这些边界在 THD 首期保持不变。router、retrieval indexer、Engram 表的冻结策略也保持一致。官方可训练 Engram 或新 auxiliary loss 不与本次布局改造混在一起。

### 3.2 已有 BSHD 精度材料

已读取此前提供的两份内部文档：

- [四层模型对齐实验](https://docs.xiaohongshu.com/doc/f620d56cee6752178885dd9fceb3f9da)：记录 TP2、PP2、CP2、EP4、SP、full recompute、optimizer CPU offload、训练/推理 offload、DeepEP、R3 和 per-token loss。前两步各统计 128 个回答、32768 个有效 token。
- [完整 LLM + BSHD](https://docs.xiaohongshu.com/doc/e59084f3fadb669b90938ab8d0717fa2)：记录完整模型 DAPO-Math 实验、80-step 日志入口及 PP+CP 对照组。温度从 0.8 到 1 的变化需要在新实验清单中固定，不能混用两组结果。

四层文档中的历史结果如下；这是历史报告，不是当前合入 main 后重新测得的值：

| 步数 | Miles 均值差 | Relax 均值差 | Miles 逐 token MAE | Relax 逐 token MAE |
| --- | --- | --- | --- | --- |
| 1 | 0.003381 | 0.003162 | 0.057449 | 0.056011 |
| 2 | 0.002468 | 0.002606 | 0.056414 | 0.056486 |

这里“均值差”是两边平均 log-prob 差的绝对值，不是逐 token MAE，也不能直接称为 KL。文档提到的 0.03 是该均值差检查的历史标准，不能拿来替代 THD 的逐 token 和梯度验收。四层文档还记录了 100 步 SFT loss/grad_norm 曲线对齐；下一阶段需要定位原始日志、数据顺序和配置，重新跑当前 BSHD 基线。

可复用的仓库入口是 [四层 SFT 脚本](../../examples/models/deepseek-v4.1/scripts/run-deepseek-v4.1-4layer-sft-8xgpu.sh) 与 [完整模型 RL 脚本](../../examples/models/deepseek-v4.1/scripts/run-deepseek-v4.1-flash-dapo-math-128xgpu.sh)。旧文档中的四层 RL 对齐脚本位于旧工作树，当前分支没有同名脚本；需迁移必要配置，不能照抄绝对路径。128 卡脚本的存在也不代表当前资源或 H800 显存容量已经验收。

### 3.3 Checkpoint 格式与已存在的模型

BF16 准备是当前加载、训练和在线权重同步路径的要求，不是 THD 本身要求。BSHD 切换到 THD 不改变 checkpoint 格式。所谓“转 BF16”不能理解为将每个张量全部转换成 BF16：

| 对象 | 当前格式契约 |
| --- | --- |
| Dense、MoE expert 等普通语言模型权重 | 原始 FP8 dense / FP4 experts 反量化到 BF16；保留必要 FP32 辅助参数 |
| 冻结的 Engram embedding 表 | 保持 `F8_E4M3` 数据和 `F8_E8M0` scale，或对应的 `U8` bits；不整体转 BF16，也不能删除其 scale |
| 配置与普通权重 scale | 删除模型级 `quantization_config` 和已失效的非 Engram scale；不能仅修改 config 中的 `dtype` |
| 训练前向中的 indexer/KV 量化点 | 继续保留与推理匹配的 fake quant/cache rounding；BF16 权重不意味着取消这些数值行为 |

这与 [Miles 的 checkpoint 准备说明](https://github.com/radixark/miles/blob/faecb81af3bf69f3684dcad69ef9e3f830036496/docs/models/deepseek/deepseek-v4-1-flash.md#32-prepare-the-checkpoint) 一致。Relax 的 [checkpoint 校验](../../relax/models/deepseekv41/checkpoint.py) 在训练 provider 和 SGLang 启动入口都会执行；当前不支持把更新后的 BF16 权重直接同步进原始 FP8/FP4 rollout checkpoint。训练、reference 和 rollout 应使用同一份已准备 checkpoint 与 tokenizer。

2026-11-15 已通过远端读取用户指定模型根目录下的 config、index 和 safetensors 头，执行当前 `validate_bf16_checkpoint`。未载入权重 payload、未执行转换：

| 模型子目录 | 层数 | 非 Engram 表的语言模型张量 | Engram 表 | 当前校验与用途 |
| --- | --- | --- | --- | --- |
| `DeepSeek-V4.1-4layer` | 4 | 4681 个 BF16、32 个 F32 | 第 0 层，FP8 + E8M0 | 通过；直接作为四层基线，无需再次转换 |
| `DeepSeek-V4.1-bf16` | 40 | 46644 个 BF16、320 个 F32 | 第 1、14 层，FP8 + E8M0 | 通过；直接作为完整模型基线，无需再次转换 |
| `DeepSeek-V4.1-Flash` | 40 | 原始 FP8 dense、packed FP4 experts 等混合格式，仍有 quantization config | 第 1、14 层，FP8 + E8M0 | 被当前校验拒绝；保留作原始权重来源 |

通过头部校验只说明格式符合当前代码要求，不等于所有权重内容、反量化精度或训练闭环已经重新验证。P0 仍需记录文件来源、tokenizer 和数据版本，并重跑 BSHD 基线。

[Miles 测试用四层模型](https://huggingface.co/CharyZeng/DeepSeek-V4.1-4layer) 不是完整模型的前四层截断版。当前本地配置为 `compress_ratios=[0,2,1,1]`、KV sources `[1,2]`、index sources `[1,2,3]`、candidate source `2`、Engram layer `0`。构建时继续从其 config 读取拓扑；四层通过不能替代完整模型跨层共享和 PP 边界验证。

当前无需重新准备权重。若以后需要重做转换，应先定位历史 V4.1 专用转换器或实现并验证缺失工具；不能直接运行仓库旧式 `convert_fp8_to_bf16.py`，它针对 `weight_scale_inv` / 128×128 FP8，不完整覆盖 V4.1 的 32×32 scale、FP4 experts 和 Engram 特例。

## 4. 上游快照与版本策略

### 4.1 固定参考提交

以下均是 2026-11-15 的核查结果；开始迁移时再次检查状态，并保留实际采用的 SHA。

| PR | 状态 | 本次参考 head |
| --- | --- | --- |
| Bridge #6199 | Open，未合并 | `ea83c044528bd4435f443fd34196b465d1c75b2a` |
| Core #8018 | Open，未合并 | `28aa0d871a141c4f972d3a9edabcbba1743273b6` |
| Miles #4488 | Open，未合并 | `faecb81af3bf69f3684dcad69ef9e3f830036496` |

**Miles 路线。** #4488 是已有 V4.1 模型上的 8 文件增量，没有依赖版本升级。通用 THD helpers 起源于已合并的 [#2038](https://github.com/radixark/miles/pull/2038)，但应取 #4488 head 实际引用的 [thd_utils.py](https://github.com/radixark/miles/blob/faecb81af3bf69f3684dcad69ef9e3f830036496/miles_plugins/models/deepseek_v4/ops/thd_utils.py)，不能只复制 #2038 的旧 diff。当前 head 的 GPU CI 为 skipped。新增 GPU 测试主要是 TP1、CP1/2/4 的 attention 前反向与 Engram hash 隔离；没有完整 PP、optimizer、checkpoint/resume 测试。[固定测试源码](https://github.com/radixark/miles/blob/faecb81af3bf69f3684dcad69ef9e3f830036496/tests/fast-gpu/test_dsv41_thd_packing.py)

作者报告的四层实验在 1×8 H200 上复用同一份 BSHD rollout，THD 的训推差恢复到 BSHD 水平；另报告了 TP8/CP4/EP64、256K 内部实验。这些材料有参考价值，但没有公开 PP>1 的完整证据；train-only 复现还关闭了 weight-update 检查，不能代替多轮 RL 闭环。

**官方 native 路线。** #8018 的 base 是 `pull-request/7502`，模型基础链为 [#7500](https://github.com/NVIDIA/Megatron-LM/pull/7500) → [#7501](https://github.com/NVIDIA/Megatron-LM/pull/7501) → [#7502](https://github.com/NVIDIA/Megatron-LM/pull/7502)，还依赖通用跨层状态 [#7678](https://github.com/NVIDIA/Megatron-LM/pull/7678)。其列出的冻结前缀梯度修复 [#8664](https://github.com/NVIDIA/Megatron-LM/pull/8664) 已关闭且未合并，需追踪替代提交。作者明确表示 #8018 分支还需要在 #7678 上 rebase；组合测试同时使用了 Bridge #6199。[作者说明](https://github.com/NVIDIA/Megatron-LM/pull/8018#issuecomment-6088861414)

在该固定版本中，[CSA2 配置检查](https://github.com/HollowMan6/Megatron-LM/blob/28aa0d871a141c4f972d3a9edabcbba1743273b6/megatron/core/transformer/experimental_attention_variant/csa2_config.py) 限制 TP=1、PP=1、无 VPP/CUDA graph，CP 使用 contiguous THD；CP/indexer loss 和 full-recompute/indexer loss 的组合受限，full recompute 只接受 uniform、每 region 一层。#7678 的通用 PP 测试不能证明 V4.1 已支持 PP。

Bridge #6199 的 [provider](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/ea83c044528bd4435f443fd34196b465d1c75b2a/src/megatron/bridge/models/deepseek/deepseek_v41_provider.py) 依赖原生 Core 模型；其 [转换实现](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/ea83c044528bd4435f443fd34196b465d1c75b2a/src/megatron/bridge/models/deepseek/deepseek_v41_bridge.py) 使用 attention/expert 物理层交替组织，Core 层数为 HF decoder 深度的两倍。当前 Relax 的层号、PP 分层、权重名和 replay 层号不能直接套用。#6199 列出的完整模型栈还包含 #7503、#7762、#7763；文本 THD 迁移不据此自动扩展 DSpark/vision 范围。

### 4.2 不预先整体升级依赖

| 依赖 | 当前 cu13 构建基线 | 首期处理 |
| --- | --- | --- |
| SGLang | `0.5.21` | 保持版本及合并后的版本化补丁；仅在明确遇到上游已修复的阻塞时评估升级 |
| Megatron Bridge | `6862ee170a2569607e8ebc5dde91f119779093ca` | 保留当前自定义 provider/bridge；按需迁移映射或 API 修复 |
| Megatron Core | `4a4de865795f123eda422ebf0a4c13d2418cf3cb` 加仓库补丁链 | 先迁移 THD 算法与最少必要 hook；不直接把 native PR 套到旧树 |
| Torch / TE / Transformers | 远端当前为 `2.13.0+cu130` / `2.18.0` / `5.16.0` | 精度比较阶段固定，不同时引入另一组内核和依赖变量 |

构建入口见 [Dockerfile.cu13](../../docker/Dockerfile.cu13)，H800 构建必须显式选择 `GPU_ARCH=hopper`，该文件默认值仍为 blackwell。Bridge 通过 `switch_mcore.sh dev` 选择 Core：[当前 Bridge 的 `.dev.commit`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/6862ee170a2569607e8ebc5dde91f119779093ca/.dev.commit) 固定为上述 Core SHA。**#6199 head 的 `.dev.commit` 也仍是同一旧 SHA**，所以仅更新 Bridge 到该 PR 并不会自动安装原生 V4.1 THD 所需 Core。[#6199 的 Core pin](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/ea83c044528bd4435f443fd34196b465d1c75b2a/.dev.commit)

若后续选择 native 路线，需要成对固定 Bridge/Core 组合提交，核对全套补丁，并重新完成 TP/PP、权重映射、冻结策略和 RL 验收。作者组合分支 `dsv41-main-dspark` 的 `a383da0e232263c9a58771346cb73b27ff403e10` 可作研究快照，不作为默认升级目标。

### 4.3 先消除镜像与仓库不一致

2026-11-15 远端只读核查发现：安装包版本与上述环境基本一致，但 `/root/Megatron-LM` 的源文件缺少仓库补丁要求的三个接口：

- `GPTModel._get_transformer_block_class`
- `TransformerBlock._build_hyper_connection_output_head`
- `copy_to_tensor_model_parallel_region(..., all_reduce_grad_fp32=...)`

这是源码接口检查，不是模型运行验证。**GPU 基线重跑前必须先对齐镜像**，记录基础镜像 digest、Core/Bridge 源码来源和实际补丁 hash；不能用“版本号相同”推断已包含仓库补丁。通过 Docker 构建或明确的依赖更新流程处理，不临时修改远端安装包。

另外保留当前 `requirements.txt` 的 `librosa==0.11.0` 和 cu13 兼容范围，不再为旧 cu12 回退 NumPy/HF Hub/PyArrow。远端实际 librosa 仍为 1.0.0，说明依赖声明的修正尚未体现在运行环境中；本期是文本训练，但环境清单应记录这项差异。cu13 的 SGLang 构建只读取版本化 `.21.patch`，不叠加 `latest/sglang_per_pos_topk.patch`。

## 5. 实现工作包

### A. 统一 V4.1 THD 布局契约

在数据入口形成每个 microbatch 独立的布局：真实样本边界、物理 padding 边界、有效 token、样本内 position、CP local/global 区间，以及每个压缩率对应的有效压缩组数和 offset。

Relax 已有 `cu_seqlens_q_cpu`、`valid_seqlens_q_cpu` 和 GPU `cu_seqlens_*_padded`；Miles helper 使用 `cu_seqlens_host`。**不能只重命名字段**：Miles 测试把 padding 作为独立 segment，而 Relax all-gather 路径可以延长最后一个样本的物理端点。adapter 必须排除尾 padding 的语义影响，并同时处理 CP1 与 CP>1。

布局应在数据准备阶段生成；避免 attention 热路径反复 `.item()`、`.tolist()` 或 GPU→CPU 同步。优先做 V4.1 局部适配，只有其他模型确实共享的行为才修改通用 data/cp_utils。

### B. 迁移分段 attention、压缩与 Engram

| 模块 | 需要实现的行为 | 主要落点 |
| --- | --- | --- |
| RoPE / sliding window | 每个样本重置 position；窗口不能跨样本，padding 不成为可见 key | `attention.py`、`ops/position.py` |
| Compressor | 每段独立组成压缩组；不足一个 ratio 的尾部按参考实现处理，不能跨样本拼组 | `ops/compressor.py` |
| Indexer / candidates | query 只选择本样本 keys；区分 segment-relative index 与打包后的 offset；复用层使用一致布局 | `ops/indexer.py`、`attention.py` |
| 共享 latent / index keys | CP gather 后恢复 sequence-major 顺序，所有复用层使用同一组映射 | `attention.py`、`runtime.py` |
| Engram | n-gram 历史在样本起点截断；CP 前缀不能借用前一条样本 token | `engram.py`、`model.py` |

当前 Relax indexer 是 query chunk 循环，Miles #4488 使用逐 segment 的 `indexer_select`；保留现有投影、量化和冻结行为，适配分段选择，不整文件替换。首版可用逐 segment 调用保证正确性，但必须测量其 Python 调度与小 kernel 开销，不能把 THD 标签当成性能结论。

迁移时保留 [NOTICE](../../relax/models/deepseekv41/NOTICE) 与来源说明，记录所取 helper 的固定 SHA。官方实现可交叉核查 [THD 工具](https://github.com/NVIDIA/Megatron-LM/blob/28aa0d871a141c4f972d3a9edabcbba1743273b6/megatron/core/transformer/experimental_attention_variant/csa_utils/thd_utils.py)、[CP 边界处理](https://github.com/NVIDIA/Megatron-LM/blob/28aa0d871a141c4f972d3a9edabcbba1743273b6/megatron/core/transformer/experimental_attention_variant/csa_utils/csa2_context_parallel.py) 和 [candidate 逻辑](https://github.com/NVIDIA/Megatron-LM/blob/28aa0d871a141c4f972d3a9edabcbba1743273b6/megatron/core/transformer/experimental_attention_variant/csa_utils/csa2_candidates.py)。

### C. TP/SP、CP 与 PP/full recompute 一起设计

TP/SP 保留现有 head/group 切分和复制 KV 的梯度归约精度。packed 后 token 的 gather/scatter 与 head 分片必须分开验证，不能以 TP1 正确推断 TP2 正确。

CP 首期只走 contiguous all-gather。处理跨 CP 切分点的压缩组、Engram 前缀和梯度回传，明确 canonical key ownership，避免重复压缩或重复累计梯度。通信始终使用显式 process group；官方 #8018 提到的 halo gather 和 MoE dispatcher 调度问题，应纳入多节点验证。

还要显式处理某个 CP rank 全是物理 padding、没有有效 query 或压缩 key 的情况：该 rank 仍参与相同 collective，生成合法的掩码输出和可重计算 carry。不能因本地无有效 token 跳过通信，也不能直接对空 segment 集合计算 `max(...)`；Miles 将 padding 当 segment 的测试不能覆盖 Relax 的这项差异。

PP/full recompute 的关键是改变现有 carry schema：当前 `runtime.py` 按 BSHD 的 `global_length // ratio` 分配共享 latent/index keys。THD 应按真实段计算，例如压缩组数是 `sum(floor(valid_segment_length / ratio))`，不能用总长整除替代；candidate 宽度也不能直接沿用全局 token 宽度。

每个 microbatch 的 carry 必须能恢复它自己的 layout、mHC pre、input IDs、latent、index keys、top-k 和 candidates。先明确哪些 metadata 可由所有 stage 确定性重建，哪些必须随状态传输；不能让 backward 读取下一次 forward 覆盖的可变状态。保留现有 BF16 carry 中 mHC 精度处理和卸载清理契约。

必须覆盖 KV/index/candidate 的生产层与复用层跨 PP stage、连续多个 microbatch、full recompute 与普通 forward/backward 的一致性，以及冻结模块经过时仍保留必要输入梯度。四层模型不一定覆盖完整模型全部共享距离，需要补充针对性模型配置。官方 [stateful recompute adapter](https://github.com/NVIDIA/Megatron-LM/blob/28aa0d871a141c4f972d3a9edabcbba1743273b6/megatron/core/transformer/experimental_attention_variant/csa_utils/csa2_hybrid_adapter.py) 用作语义参考，不直接替换当前 carry。

### D. 接回 RL 与权重同步

- 检查 packed response logits、advantages、returns、old log-probs、loss mask 与原样本顺序一致，CP 上有效 token 只计一次；空 response 分片保持合法反向图。
- 验证 R3 按相同逻辑 token、层号和 CP/SP 分片重放；先固定 replay 开关做同条件比较，再验证正式开启的配置。
- 不把 attention indexer replay 限制混同 R3。Miles #4488 拒绝 indexer replay 与 THD dynamic batching/多样本 microbatch 的组合，Relax 首期也应明确保留未支持边界。
- 权重同步至少验证初始加载、optimizer 更新后、多轮更新后以及 offload/resume 后；检查 SGLang log-prob 和输出，不能只使用 train-only 的 `--no-check-weight-update` 路径。
- 保持 frozen Engram 表、BF16 权重转换、原位更新和同步完成后再返回的契约，不用修改量化语义来掩盖布局误差。

## 6. 实施顺序与完成条件

| 阶段 | 工作与产物 | 完成条件 |
| --- | --- | --- |
| P0：锁定基线 | 核对 H800/cu13 镜像与三处 Core hook；保存模型/tokenizer/data/hash、依赖和补丁清单；定位历史日志；重跑当前 BSHD | BSHD 在当前环境可复现，得到同一份可重放 rollout 与数值噪声基线 |
| P1：布局与 CP1 THD | 实现 A/B，复用固定 SHA helpers；暂用 TP1/PP1/CP1 排错 | 多样本 packing 与逐样本 BSHD 的输出、loss、梯度一致，样本隔离与 padding 用例通过 |
| P2：TP/SP 与 CP | TP2/SP、CP2/4 分开验证，再组合；实现边界通信与 ownership | packed 跨 CP 边界和全 padding rank 的前反向通过，无重复梯度、通信 hang 或跨样本读取 |
| P3：PP 与重计算 | 完成 carry schema、layout 生命周期及跨 stage 共享；联合接入 full recompute、optimizer CPU offload 和 flex/deepep | TP2/PP2/CP2/EP4/ETP1 整套组合通过，多 microbatch、空有效分片 carry、重计算梯度和 optimizer 更新通过 |
| P4：训练闭环 | 保持 P3 全部开关，同时进行同 rollout 的 BSHD/THD 比较、R3、权重同步、100 步四层 SFT、checkpoint/resume | 精度不劣于重新建立的 BSHD 基线，更新后训推一致性、恢复一致性及主机/GPU 内存稳定性通过 |
| P5：完整模型与性能 | 根据实际资源运行完整模型，补齐 EP64 的 alltoall/DeepEP 对照；比较训练侧和端到端吞吐、显存，分析主要瓶颈 | 完整模型联合配置的数值、通信与稳定性通过，THD 在目标长度分布上有超出测量噪声的实际收益 |

P1/P2 的 TP1 或 PP1 只是定位问题的中间步骤，不能作为项目完成条件。P3 必须完成；不因 Miles 或官方当前缺少 PP 证据而把 PP 从目标中删除。

阶段按依赖推进，不承诺尚未测量的工期。P0 会决定可用卡数、首批上下文长度、数值阈值及性能目标。若 kernel/环境接口构成阻塞，先形成具体最小升级差异再评估，不先升级所有依赖再排查。

## 7. 验收方法

### 7.1 精度与状态一致性

采用两层比较：先比较同权重、同 tokens 的 **Relax THD vs Relax BSHD**，隔离布局问题；再用相同 rollout 和相同量化/replay 设置比较训练与 SGLang，并与历史 Miles 指标核对口径。

| 用例 | 要观察的结果 |
| --- | --- |
| 不等长多样本、奇数长度、短于压缩率、尾 padding、空 response | token 数量、位置、有效 mask、分段 hash/窗口/候选索引正确；改变前一条样本不影响后一条样本 |
| Attention / Engram / 完整模型前反向 | logits/log-prob 的 MAE、p99、最大误差，loss，输入和主要参数梯度的相对 L2/余弦相似度 |
| TP/SP、CP 边界、PP 跨层复用 | 相同逻辑 token 的结果可还原；归约不漏算/重算，无跨 microbatch 污染 |
| full recompute、冻结模块、offload | 与无 recompute 对照的 loss/梯度一致，必要梯度存在，状态可恢复且无持续内存增长 |
| optimizer step、R3、权重同步、恢复 | 更新后的有效权重、训练/rollout log-prob、loss/grad_norm 轨迹一致 |

离散布局与边界可用精确断言；浮点数值阈值在 P0 用重复 BSHD、实际 sparse backward 噪声及已有基线校准并固定，不把上游单个测试的 `1e-2` 直接设成所有层的标准。可列出超阈值参数/层，不能只看平均 loss。

使用已有 `--save-debug-rollout-data` / `--load-debug-rollout-data` / `--debug-train-only` 能力重放同一份 rollout，记录 sample IDs、有效 response mask、packing 顺序、随机种子、温度、replay 设置及 loss normalization。独立新采样的两次训练不能作为逐 token 精度对照。

### 7.2 性能与实用性

比较固定的有效 token 工作量，而不是 padded tokens/s。固定模型、GPU 数、TP/PP/CP/EP、recompute、offload、数据长度分布和优化器；记录预热后的多次测量以及样本数、segment 数和 padding 比例。

至少分别报告训练 forward/backward/optimizer 时间、峰值 GPU 显存/主机内存、权重同步、offload 和 rollout 时间，避免 SGLang 波动掩盖训练收益。分短序列、混合长度和长序列场景评估，不直接把作者 256K 内部实验当作我们的首轮规格。

验收要求 THD 在目标混合长度场景的有效 tokens/s 提升超出重复测量噪声，同时保持精度与内存稳定。具体加速目标在 P0 固定；若主要开销仍来自逐 segment indexer、candidate 或 CP gather，应依据 profiler 决定下一项优化。只跑通 THD、但速度仍无实用改善，不能宣布完成性能目标。

## 8. 代码与验证约束

- 先扩展现有 [V4.1 GPU/集成测试](../../tests/models/deepseekv41/)、[V4.1 CPU 测试](../../tests/cpu/models/deepseekv41/)、[BSHD CP 回归](../../tests/backends/megatron/test_bshd_contiguous_cp.py)、[BF16 同步测试](../../tests/backends/megatron/weight_update/test_v41_bf16_sync.py)，保留 BSHD 对照；只为真实缺口新增永久测试。
- CPU 测试验证纯布局和边界；GPU 测试执行实际 attention/Engram、梯度和分布式通信。不能用源码字符串或 AST 检查替代运行验收；依赖缺失导致的 skip 不算通过。
- 所有依赖 Megatron、SGLang、Transformers 或 GPU 的验证在远端 H800 环境执行。本地只做文档、代码和无依赖静态工作；集成测试未运行时记录硬件/环境缺口。
- 不修改 Controller/Service/Launcher 来绕过模型问题；参数解析、额外依赖等修改按仓库约束单独说明必要性。避免与目标无关的格式化或重构。
- 用户已于 2026-11-15 授权使用远端 8 卡实施和测试；环境修复优先采用运行时适配，确需源码修改时先在仓库形成 patch，再应用到远端并记录。不得执行 `ray stop` 或 Git 暂存、提交、分支修改；临时实验不进入持久测试目录，作业结束清理自身进程和临时资源。

开始 P1 前需要补齐的材料集中在 P0：可重放的历史数据与配置、当前环境的补丁来源、目标训练长度分布，以及数值/性能阈值。三个参考 PR 的后续更新继续跟踪，但不把等上游合并作为开始迁移的前提。
