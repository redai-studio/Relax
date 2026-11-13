# Kimi-K3 优化

## Kimi K3 LoRA：Expert Split 优化

### 优化思路

独立专家 LoRA 原先通过多次 `narrow` 切分各专家的 token。反向传播时，每个切片分别生成完整输入大小的梯度缓冲区，再累加到输入梯度，产生大量 BF16 `fill` 和 `add` kernel。

优化将这些独立切片替换为一次 `torch.split`，由同一个反向节点拼接各专家的输入梯度，减少重复清零、累加和显存读写。各专家仍使用独立 LoRA 权重，GEMM 路径、参数布局及 checkpoint 格式不变。

### 如何开启

在训练 worker 的环境变量中设置 `RELAX_KIMI_K3_EXPERT_SPLIT=1`。使用 Relax 训练入口时，可通过参数传入：

```bash
--train-env-vars '{"RELAX_KIMI_K3_EXPERT_SPLIT":"1"}'
```

若已有 `--train-env-vars`，将该字段合入原 JSON。worker 日志出现 `Enabled Kimi K3 grouped-expert LoRA split backward.` 表示补丁已安装。设 `RELAX_KIMI_K3_EXPERT_SPLIT=0` 可关闭，用于同配置对照。

### 代码位置

- [expert_split.py](../../../relax/models/kimi_k3/expert_split.py)：`forward`、`_forward_per_expert` 实现切片替换，`install_expert_split` 安装到 Bridge 的 `GroupedExpertLinearAdapter`。
- [training_ops.py](../../../relax/models/kimi_k3/training_ops.py)：`configure_training_ops` 在模型构建前读取开关并安装补丁。
- [env.py](../../../relax/utils/env.py)：定义 `RELAX_KIMI_K3_EXPERT_SPLIT` 开关，默认关闭。

### 收益

- 内部训练场景中，开启后单个 step 耗时从 **380 s 降至 309 s**，减少 **71 s（约 18.7%）**。
- Blackwell 局部测试（129,984 tokens、56 个本地专家、LoRA rank 16）中，grouped adapter 的前向加反向耗时：FC1 从 **39.51 ms 降至 3.57 ms**，FC2 从 **33.55 ms 降至 2.58 ms**。这是局部算子收益，不代表整步训练的加速倍数。
- 主要收益来自消除切片反向中重复的全尺寸 `fill` / `add`，不依赖共享专家权重或降低计算精度。

## Kimi K3：SiTU / AttnRes 融合优化

### 优化思路

SiTU 原先通过多个 PyTorch 算子完成 FP32 转换、tanh、sigmoid 和门控乘法，产生中间张量及多次 kernel launch。融合实现将这些计算合并到 Triton 前向、反向 kernel，保留 FP32 中间计算和输出 dtype 转换；训练路径使用精确 tanh，未启用近似数学模式。

AttnRes 原先先拼接 residual bank 和当前 prefix，再依次执行 RMS 打分、softmax 和加权求和。融合实现直接读取 bank 与 prefix，减少拼接、FP32 中间张量和显存读写；输出先转换回输入 dtype，再调用原来的 output norm。LoRA 场景下，评分权重冻结时不分配、归约其权重梯度缓冲区。

两个实现参考 NVIDIA NeMo Automodel 的融合内核，并适配 Bridge 的训练调用和梯度语义。AttnRes 使用离线选定的固定配置：chunk width 1024、4 warps、反向梯度每个 program 处理 8 行。训练中不做在线 autotune，token 数 `T` 不参与值特化，避免动态 packing 的新长度反复触发选型；首次遇到新内核变体仍有正常 JIT 编译开销。

### 如何开启

两个开关可独立控制，在训练 worker 的环境变量中设置：

```bash
--train-env-vars '{"RELAX_KIMI_K3_FUSED_SITU":"1","RELAX_KIMI_K3_FUSED_ATTN_RES":"1"}'
```

若已有 `--train-env-vars`，将两个字段合入原 JSON。worker 日志出现 `Enabled Kimi K3 fused SiTU (precise tanh).` 和 `Enabled Kimi K3 fused AttnRes aggregation.` 表示对应实现已安装。将任一字段设为 `0`，可单独关闭该优化做对照。

框架中两个开关均默认关闭；[64 卡 Blackwell 启动脚本](./scripts/run-kimi-k3-pokemon-lora-64xblackwell.sh) 默认将它们设为 `1`，启动前显式设置同名环境变量为 `0` 可覆盖脚本默认值。

### 代码位置

- [fused_situ.py](../../../relax/models/kimi_k3/fused_situ.py)：`fused_situ_and_mul` 实现 SiTU 融合前向和反向。
- [fused_attn_res.py](../../../relax/models/kimi_k3/fused_attn_res.py)：`fused_attn_res_aggregate` 适配 Bridge，`attn_res_mix` 实现融合聚合；包含固定 launch 配置和冻结评分权重的反向优化。
- [training_ops.py](../../../relax/models/kimi_k3/training_ops.py)：`configure_training_ops` 在模型构建前安装实现，同时更新 Bridge ops 和 layers 中的 AttnRes 函数引用。
- [provider.py](../../../relax/models/kimi_k3/provider.py)：`provide_language_model` 在构建语言模型前调用安装入口。
- [env.py](../../../relax/utils/env.py)：定义两个独立的融合开关。

### 收益

内部训练场景中，在 Expert Split 优化后的 **309 s/step** 基础上叠加 SiTU / AttnRes 融合，单个 step 耗时进一步降至 **253 s**，减少 **56 s（约 18.1%）**；按相同工作量计算，对应吞吐提升约 **22.1%（1.221 倍）**。

Blackwell 单卡上，使用真实训练的 4 个 PP 阶段、共 8 份样本验证：BF16、hidden size 7168、token 数 7630 / 7598，AttnRes bank 大小为 1 / 2 / 4 / 8。下表为预热后的局部前向加反向设备计时，使用 CUDA Graph 隔离 CPU 发射开销，不表示训练已启用 CUDA Graph。

| 验证范围                                       | 相对原 PyTorch 实现的加速 |
| ---------------------------------------------- | ------------------------- |
| AttnRes 聚合，包含原 TransformerEngine RMSNorm | **8.1–15.1 倍**           |
| SiTU 激活                                      | **10.2–11.1 倍**          |
| shared-expert 本地分片，包含前后两个线性投影   | **1.50–1.62 倍**          |

- AttnRes 的固定 chunk width 从 2048 调整为 1024 后，在这 8 个样本上额外提速 **2.6%–9.1%**。该配置在离线验证后固定，不在训练热路径搜索。
- 对于 7630 tokens、bank=8 的 AttnRes，单次前反向的额外峰值显存从约 **11.11 GiB 降至 1.12 GiB**；这是局部计算测量，不是整模型显存。
- 补充非零诊断梯度后，融合输出与梯度相对原实现的最大相对 L2 误差约 **0.013%**，并非逐位一致。

上述单卡局部测试不包含 TP / CP / EP 通信，不能直接换算为整步训练收益。AttnRes 反向使用保存的层输出梯度作为代理，并额外用非零诊断梯度检查正确性；shared-expert 回放仅覆盖本地 TP 权重分片及本地 token 切片，确认保存的 LoRA 输出权重为零后重建基础投影，不包含 LoRA 反向。整步收益需通过相同数据、相同并行配置下的稳定训练 step 对比确认。

## Kimi K3：编译缓存复用优化

### 优化思路

将 TorchInductor / Triton 编译产物持久化到共享目录，训练启动前恢复到各 GPU 节点的本地可写缓存。训练 worker 使用本地缓存目录，新增编译产物由各节点的缓存 agent 周期性增量归档到共享目录，并在所属训练 driver 退出时发布最后一份增量，供后续任务复用。

缓存按运行配置的 cache key 和构建指纹隔离，构建指纹包含编译依赖、GPU 架构及相关源码版本。这样可以复用兼容的已有产物，减少重复编译和训练过程中的编译等待；未命中的新内核变体仍需正常编译。

### 如何开启

启动前设置共享缓存目录，再使用原有训练入口启动：

```bash
export RELAX_KERNEL_CACHE_DIR=/shared/compiler-cache
```

目录需为所有 GPU 节点可访问的绝对路径。本地缓存由入口准备，训练 worker 的 `TORCHINDUCTOR_CACHE_DIR` 和 `TRITON_CACHE_DIR` 指向对应的节点本地子目录。

### 代码位置

- [kernel_cache.py](../../../relax/distributed/ray/kernel_cache.py)：构建指纹、本地缓存恢复、增量归档和节点缓存 agent。
- [kernel-cache.sh](../../../scripts/entrypoint/kernel-cache.sh)：生成缓存配置、准备节点缓存、管理 agent 生命周期及注入运行环境。
- [ray-job.sh](../../../scripts/entrypoint/ray-job.sh)：在提交训练任务前配置缓存并启动各 GPU 节点的缓存 agent。
- [actor_group.py](../../../relax/distributed/ray/actor_group.py)：将训练 worker 的 TorchInductor / Triton 缓存路径设置为节点本地目录。

### 收益

- 内部训练场景中，复用已有编译缓存后，**step 1 耗时从 2150 s 降至 630 s**，减少 **1520 s（约 70.7%）**，对应首个 step 约 **3.41 倍加速**。
- 同次训练观察中，后续 step 耗时几乎没有突刺，运行耗时更稳定。
- 该收益主要来自减少重复编译等待，不代表稳态 GPU 算子吞吐提高 3.41 倍；首次建立缓存、环境或内核变化导致缓存未命中时，仍可能出现编译开销。

## Kimi K3 LoRA：Grouped GEMM 冗余复制优化

### 优化思路

Bridge 已有 BF16 grouped GEMM 路径，但原实现仍将按专家排列的输入 split 后重新 cat，并通过索引复制专家权重，反向再将权重梯度 scatter 回原参数。

在 ETP1 且满足 grouped GEMM 条件时，直接使用原输入和完整的独立专家权重，通过累计 token offsets 描述专家边界；零 token 专家保留重复 offset。这样省去输入拼接、权重索引复制及对应反向操作。保留权重模块调用以触发 DDP prehooks，同时保留 dropout、激活函数及 offload 标记。FP8、ETP 大于 1、非连续输入等继续使用原有路径，各专家的 LoRA 权重及 checkpoint 布局不变。

### 如何开启

沿用 `RELAX_KIMI_K3_EXPERT_SPLIT=1`，无需新增开关。[64 卡 Blackwell 启动脚本](./scripts/run-kimi-k3-pokemon-lora-64xblackwell.sh) 默认开启；满足 ETP1 BF16 grouped GEMM 条件时自动使用快路径。

### 代码位置

- [expert_split.py](../../../relax/models/kimi_k3/expert_split.py)：`forward` 中的 ETP1 grouped GEMM 快路径。
- [training_ops.py](../../../relax/models/kimi_k3/training_ops.py)：沿用 `configure_training_ops` 安装入口。

### 收益

Blackwell 单卡局部测试：BF16、56 个独立专家、LoRA rank 16、129,984 个合成 tokens。预热后交替进行 A/B 测量，前向加反向中位耗时如下：

- FC1 adapter：**3.626 → 2.846 ms**，耗时降低约 **21.5%**。
- FC2 adapter：**2.654 → 1.928 ms**，耗时降低约 **27.4%**。
- 单次 adapter 前后向的额外峰值显存分别从 **3972 → 3067 MiB**、**2565 → 1792 MiB**；两者独立测量，不能相加作为整层峰值收益。
- 局部验证中，输出、输入梯度及两组 LoRA 权重梯度逐位一致，覆盖空专家、倾斜 token 分布、dropout、重计算及 optimizer 更新。

与 MLA 后端优化叠加后的整步收益见下一节；尚未单独测量这项改动的整步贡献。

## Kimi K3：MLA cuDNN Attention 后端优化

### 优化思路

K3 MLA 通过 Transformer Engine 的 `TEDotProductAttention` 执行 attention。原脚本使用 `--attention-backend flash`，排除了 cuDNN fused attention；改为 `fused` 后，使用现有 TE/cuDNN 路径，保留 BF16、causal mask、变长 packing 及 CP 通信语义，无需修改外部依赖。

### 如何开启

```bash
--attention-backend fused
```

[64 卡 Blackwell 启动脚本](./scripts/run-kimi-k3-pokemon-lora-64xblackwell.sh) 已默认配置。当前训练配置为 TP4 / PP4 / CP4 / EP16、ETP1、alltoall，PP 层数为 **23/24/24/22**，重计算为 `full + uniform + 1`。

### 代码位置

- [run-kimi-k3-pokemon-lora-64xblackwell.sh](./scripts/run-kimi-k3-pokemon-lora-64xblackwell.sh)：`OPTIMIZATION_ARGS` 中指定 `--attention-backend fused`。
- [model_provider.py](../../../relax/backends/megatron/model_provider.py)：现有参数传递路径将 `attention_backend` 配置传给 Bridge provider，无需新增适配代码。

### 收益

4 卡 Blackwell、CP4 p2p 局部测试：每卡 24 heads，Q/K head dimension 为 192，V 为 128。使用真实训练 dump 中的序列长度 **\[56352, 65728\]** 生成合成 Q/K/V，每个 CP rank 处理 30,520 tokens。测试环境为 TE 2.18.0、FA2 2.7.4.post1，实际运行时 cuDNN 版本为 9.25.1。

- 同进程交替 A/B 测量中，attention 核心前向、反向及 CP 通信总耗时约 **195.7 → 54.5 ms**，约 **3.59 倍加速**；另一组输入分布复测约 **3.65 倍**。
- 独立进程测量的额外峰值显存约 **6360 → 4971 MiB**，减少约 **1.36 GiB**。
- 完整输出和 dQ/dK/dV 均为有限值，相对原 FA2 路径的相对 L2 误差最大约 **0.44%**；短序列另以 FP32 reference 核对前后向。
- 收益依赖序列长度：CP4 下全长 4096 tokens 的小规模测试中，cuDNN 略慢；以上加速倍数针对长序列 attention 核心，不是完整层或整步加速倍数。

2026-10-29 的 64 卡训练实测由使用者确认：在此前 **253 s/step** 基础上，叠加本节 MLA 后端优化与上一节 LoRA 冗余复制优化后，单步耗时降至 **194 s**，且 **loss 与基线对齐**。两项合计减少 **59 s（约 23.3%）**，按相同工作量计算，吞吐提升约 **30.4%（1.304 倍）**；该整步结果不拆分归因到单项优化。

累计优化记录为 **380 → 309 → 253 → 194 s/step**。相对最初 380 s 基线，耗时降低约 **48.9%**，吞吐提升约 **95.9%（1.959 倍）**。
