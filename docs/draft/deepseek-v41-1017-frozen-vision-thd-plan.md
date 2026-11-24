# DeepSeek-V4.1-Flash-1017 冻结视觉塔的多模态 THD 训练与验收计划

**状态（2026-11-19）**　当前先交付 **DeepSeek-V4.1-Flash-1017 的图文 SFT**。已接入保留视觉资产的 BF16 转换、原生图文输入、冻结视觉的 THD 模型和离线 HF 导出。此前文本 SFT 的数值对齐使用首发版。1017 的训练效果须以固定快照和真实 OpenSearch-VL / V\*Bench 数据单独验收。[训练入口](../../examples/models/deepseek-v4.1/scripts/run-deepseek-v4.1-flash-1017-vision-sft.sh) · [权重准备](../../examples/models/deepseek-v4.1/tools/README.md) · [历史纯文本 THD 计划](deepseek-v41-thd-plan.md)

**实施优先级。** 先在四层、8×H800 上跑通“1017 图文基座 → 冻结视觉的 THD SFT → 已落盘的 Megatron checkpoint → 独立 HF → 新起 SGLang 服务读图”。这一步通过后，再投入整模 OpenSearch-VL 一轮 SFT 与 V\*Bench 正式验收。在线图文 GRPO 复用同一个输入、模型和同步实现，单独验收。按现有证据，**SGLang 和 Megatron-LM/Bridge 暂无必须升级的前置条件**；先在已验证的镜像上补 Relax 模型适配。需要重新验证的版本边界见第 2 节。[SFT 验收标准](https://docs.xiaohongshu.com/doc/1a5e9d419546f8377db1e6fdd6bab8e8)

**本轮整模实验。** 四层闭环已完成。40 层使用 128 张 H800、GBS 128，在训练节点掉卡前完成了 120 步，保留第 50、100 步完整 DCP。按负责人要求随后暂停。独立 32 卡集群上，第 50 步导出的 BF16 HF 与 BF16 基座使用同一组 80 道 V\*Bench 题，Pass@1 从 15/80 提高至 64/80（+61.25 个百分点）。原计划的第 200 步未执行。验收文档的完整口径为一个 epoch 及全部 191 题。[整模运行记录](../../deepseek_v41_vit_1017/sft-40layer-128xh800/README.md) · [逐题对照](../../deepseek_v41_vit_1017/sft-40layer-128xh800/benchmark/paired-step50.json)

## 1. 目标与冻结范围

四层冒烟使用同一 1017 快照的连续前四层，保留对应层号、视觉路由和 Engram。转换时指定 `--keep-vision --decoder-layers 4`，完整 40 层只需 `--keep-vision`。设定共享的 `MODEL_DIR`、`DATA_DIR`、`SAVE_DIR` 和转换结果 `HF_CKPT`，已有 Ray 集群从仓库根目录启动：

```bash
RELAX_SKIP_CLUSTER_CLEANUP=1 NUM_STEPS=2 GLOBAL_BATCH_SIZE=4 SAVE_INTERVAL=2 \
  bash scripts/entrypoint/ray-job.sh \
  examples/models/deepseek-v4.1/scripts/run-deepseek-v4.1-flash-1017-vision-sft.sh
```

指定 `NUM_STEPS` 的短测默认跳过模型 checkpoint。只验证 loss 时沿用默认值；需要验证恢复或导出 HF 时显式设置 `SAVE_INTERVAL`。上面的两步命令在最后一步保存一个 DCP。正式按 epoch 训练默认每 200 步及末步保存，保留最近一个 DCP。两种模式都可用 `SAVE_INTERVAL=0` 关闭模型保存。

脚本按 checkpoint 的层数选择默认拓扑。四层使用单机 8 卡和 DeepEP，40 层使用 16×8 卡和 alltoall。默认数据为 `$DATA_DIR/opensearch_vl/32k/train.jsonl`，可用 `PROMPT_DATA` 指定其他准备好的图文轨迹。正式运行去掉 `NUM_STEPS`，默认一轮；同一 `EXP_NAME` 从最近一次成功保存的 DCP 恢复。将短跑延长时，续训沿用 DCP 中的恒定 LR/WD 及 scheduler 的累计样本数。配方默认记录 TensorBoard / ClearML。正式验收使用的 SwanLab hook 需要含 SwanLab SDK 的 metrics worker 环境和已有认证，通过 `ENABLE_SWANLAB=1` 启用。11 月 19 日的八卡验证镜像未安装该 SDK。HF 导出读取训练 DCP 和同次训练的 BF16 基座。

从同一份 **1017 来源的多模态 BF16 HF 基座** 初始化 Megatron SFT/actor、需要时的 reference 及 SGLang。模型接收文本、单图和多图，输出文本。THD 混合打包不同长度的多条样本。SFT 监督完整轨迹中指定的 assistant 回合；GRPO 只在有效 response token 上计算策略损失，沿用 colocate、R3、在线更新与恢复流程。正式 SFT 交付包括可续训的 DCP、可独立评测的最终 HF 目录和同协议 Base/SFT 比较。

| 对象 | 首期行为 | 验收依据 |
| --- | --- | --- |
| 32 层 ViT、两层 aligner/projector、三个图像起止和换行向量 | 从 1017 加载并固定，首个 PP stage 执行视觉前向。这里将“冻结视觉塔”按**冻结整条视觉输入支路**处理 | 与源张量比对。优化器中无相应参数或 moments；更新、sleep/wake、恢复前后冻结资产一致 |
| 语言主干与 MoE | 延续纯文本 BF16 配方的可训练集合，包括现有可训练的 Engram 投影。沿用现有 router/indexer 冻结策略；为各主干层恢复固定的视觉选路 bias | 代表性的语言参数有非零梯度和实际更新；同一图文 prompt 在同步后使用新策略版本 |
| Engram 查找表 | 延续原生 FP8 bits + scale、宿主内存和冻结策略。图像不参与 n-gram，不接收 Engram residual | 独立样本与 packed 样本的 hash/贡献一致，覆盖图后文本和跨 CP 边界 |
| 并行与内存 | Actor 保留 TP/SP、均匀 PP、contiguous all-gather CP、EP、真正的多样本 THD、full/uniform/1 重计算、optimizer CPU offload 和 `flex/deepep`，并与 R3、train/rollout offload 联合运行 | 在同一作业记录生效参数、两轮非零更新、路由和权重版本。全模型多节点另行验收 |

首期输入为图片与文本，输出为文本。MTP/DSpark 沿用文本配方的裁剪策略。#632 的 Qwen 冻结 ViT 实验继续训练 merger；本计划沿用此前对 1017 冻结范围的选择，连同 aligner 和图像特殊向量一起固定。训练清单和效果报告均注明该范围。[Qwen 冻结范围与实测](../../examples/opensearch_vl/results/qwen3.5-35b-a3b.md)

## 2. 可复用的基线和远端实况

**近期 PR 与模型身份。** 较早的 #460 涉及 V4.1、BSHD 和镜像，仍是 draft。开放中的 #728 包含 `dc52199`、`20be67b` 的文本 THD 工作。`939c8cb` 调整训练脚本，`8ea1b11` 将四个 SFT/RL 入口迁入模型示例目录，`fbc2d0f` 收回了部分通用环境和后端改动，`57b69b9` 新增纯文本 BF16 转换器及测试。验收使用的示例和评分协议来自已合入的 #632，现有启动脚本针对 Qwen。历史转换日志的源 revision 为 `8b3cec70c307624fb098c3a04bdbfc118e7abda9`；本次 1017 的基座、裁剪代理、DCP 和最终 HF 都须追溯到独立的 1017 快照。[PR #460](https://github.com/redai-studio/Relax/pull/460) · [PR #728](https://github.com/redai-studio/Relax/pull/728) · [PR #632](https://github.com/redai-studio/Relax/pull/632) · [转换来源](../../examples/models/deepseek-v4.1/tools/README.md) · [首发版](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) · [1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)

**四层文本精度基线。** 四层 8×H800 已在 TP2/SP、PP2、CP2、EP4/ETP1、full recompute、CPU optimizer offload、DeepEP 和 R3 同开时完成两轮在线更新，SGLang 权重版本为 1→2→3。同两份 rollout 的 THD 对 SGLang 的 response logprob MAE 为 0.05676、0.05684，BSHD 为 0.05644、0.05711。该实验使用人为构造的非零 reward，可用作对齐方法。另有四层 100 步 SFT 参数更新对照，以及恢复 SFT 数据游标并提交下一次保存的记录。[四层联合运行与恢复记录](../../deepseek_v41_thd/progress.md) · [同轨迹逐 token 比较](../../deepseek_v41_thd/rl-logprob-comparison-20261115.md) · [100 步参数比较](../../deepseek_v41_thd/sft-100step-parameter-comparison.md) · [四层性能记录](../../deepseek_v41_performance_4layer/thd/performance-report.md)

**40 层文本基线。** 128×H800、TP2/PP2/CP2/EP64 下，文本 RL 的 THD 与历史 BSHD 在共同 70 步内各自的 train/rollout mismatch 步均值为 0.02147 和 0.02132，两组独立采样。11 月 17 日整模 SFT 另有 Relax THD 对 Miles BSHD 的完整 100 步比较，loss MAE 为 0.0000889343，grad norm MAE 为 0.00510159。有序样本 ID 和监督长度匹配，该次 Relax 未保存逐步 token/mask 字节指纹。当前分支的 128 卡 SFT 入口为 **TP4/PP2/CP1/EP32**，DAPO 入口为 **TP2/PP2/CP2/EP64**，均使用 alltoall。四层 SFT 脚本默认 BSHD，验收时须显式选择 THD。后续以展开后的命令和运行日志确认 CP、THD、DeepEP 同时生效。[RL 共同 70 步](../../deepseek_v41_thd_full/matched-70step-summary.md) · [SFT 共同 100 步](../../deepseek_v41_thd_full/miles-sft/comparison-1117/README.md) · [现有训练脚本](../../examples/models/deepseek-v4.1/scripts/)

**11 月 18 日远端核查。** 通过 SSH 读取共享工作树和正在运行的 Ray 环境，使用短时前台进程。下列图像检查只验证预处理。

| 对象 | 观测及对实施的影响 |
| --- | --- |
| 硬件与解释器 | 实际为 **8×H800**。管理工具显示伪装的 L20Y/8.9；沿用 Ray 进程的 CUDA compat 库路径后，`/opt/sglang/bin/python` 的 Torch 实测 8 卡、capability **(9, 0)**。裸 SSH 的 `/usr/bin/python` 缺少训练包，裸调虚拟环境也可能加载旧驱动库。调试复用镜像 PATH、PYTHONPATH、CUDA_HOME、LD_LIBRARY_PATH |
| 软件栈 | 实际镜像 manifest 为 **cu13**，Torch 2.13.0+cu130、SGLang 0.5.21、Transformers 5.16.0、TE 2.18.0、Ray 2.56.1.2；有 `deep_ep`。Core/Bridge 位于 `/root/Megatron-LM`，manifest 的 Bridge commit 为 `6862ee17`；SGLang 位于 `/sgl-workspace/sglang/python`。已核对旧计划涉及的两个 Core 构造 hook 及 FP32 TP 梯度归约参数。安装位置见 [Dockerfile](../../docker/Dockerfile)，版本以实机及 [cu13 配方](../../docker/Dockerfile.cu13) 为准 |
| 模型资产 | 核查的共享模型目录中，四层和整模 `DeepSeek-V4.1-bf16` 都是 flat text config，索引中视觉参数与 `bias_vl` 均为 0。带原生视觉的 `DeepSeek-V4.1-Flash` 本地下载元数据指向 `8b3cec70`；其索引含 259 个 `vision.*`、4 个 `aligner.*`、3 个 `image_*`，以及 40 个主干和 3 个 draft 的视觉 bias。尚未在已查模型目录确认可用于训练的 1017 快照；第一项交付是固定来源的新产物 |
| 图像路径 | 该 Transformers 原生 CONFIG_MAPPING 无 `deepseek_v41`。已安装 SGLang 自带 V4.1 配置归一化、image processor、ViT/aligner 及视觉权重 loader。远端用两张内存 PNG（高×宽 301×539、703×267）比较共享首发版官方预处理与 SGLang PIL 路径，分别展开 **192、191** 个 token，patch BF16 值和角色序列均逐元素相同。Rust/GPU 预处理、1017 权重前向和在线采样另验 |
| Rollout 与数据 | 实机 SGLang `deepseek_v4.py` 的视觉分支要求 rollout 的 attention CP=1、PP=1、MoE A2A 关闭，支持 TP/EP/DP；这是生成侧约束。共享 OpenR1 图文数据的已转换 parquet 有 **7,689** 行，列为 `prompt/image/label`，原始表同样行数。复用 [OpenR1 图文 RL 脚本](../../scripts/training/multimodal/run-qwen35-27B-8xgpu-openr1mm-colocate.sh) 的数据与 [reward](../../relax/engine/rewards/openr1mm.py)，再核对 renderer 和可判分样本 |

**版本决策。** 11 月 19 日在 128 卡 H800 集群中用单机八卡实测，现有 Megatron-LM/Bridge 已能从保留视觉的四层 BF16 基座做图文 THD SFT 并保存 DCP。SGLang 0.5.21 可独立冷加载从 DCP 导出的 HF，处理单图及带工具返回图的多轮输入。这条 SFT 链路可沿用现有镜像。40 层图文多节点运行和在线 RL 分别验收。

## 3. 实施顺序

### A. 建立可追溯的 1017 多模态 BF16 checkpoint

1. 模型和数据集由需求方下载。已下载的 1017 快照固定为官方 revision `08c8657f02291e11b6f6b63e0e16014f8e748518`。与下文调研时引用的 `1bae0e10` 相比，这次快照增加了 `chat_template.jinja`，权重分片和原生图像、消息编码实现的对象均相同。登记原始 index、shard 摘要、配置、tokenizer、官方 encoding / 图像预处理及转换程序版本。在仓内 `convert_to_bf16.py` 上增加显式多模态模式，保留已有纯文本行为。该工具源自 Miles `6f52e0b`，已实现逐张量解量化、分片预检及恢复；默认文本模式会删除视觉键及视觉 bias、展平配置。图文模式使用 `--keep-vision`，四层裁剪保留原始层号。让 `--check-only` 按多模态目标列出键数、各类 dtype 和容量，`--resume` 校验模式、源 revision、配置与资产清单，避免沿用文本分片。在开全模转换前核算源模型、训练基座、DCP 与最终 HF 同时存在时的磁盘配额。旧版文本产物约 1.31 TB，只用作容量参考。[官方 1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) · [仓内转换器](../../examples/models/deepseek-v4.1/tools/convert_to_bf16.py) · [Miles 配方](https://github.com/radixark/miles/blob/main/docs/models/deepseek/deepseek-v4-1-flash.md#32-prepare-the-checkpoint)
2. 语言部分沿用现有 FP8 dense、MXFP4 expert 解量化及 FP32 例外，两个冻结 Engram 表及 scale 保留原始表示。逐项拷贝 1017 的原生视觉参数，保留源 dtype（本次快照 266 个视觉输入张量均为 BF16，语言侧视觉路由 bias 为 FP32）、三个特殊向量和每个保留主干层的 `ffn.gate.bias_vl`。完整 BF16 产物保留 composite `text_config/vision_config`、image token、量化后实际 dtype 描述；去掉会使已反量化主干被再次量化加载的配置。actor 和 SGLang 从同一份产物各自派生内部配置。沿用官方 prompt 编码，记录任何兼容 Chat Template 的等价性。[checkpoint 校验](../../relax/models/deepseekv41/checkpoint.py) · [训练配置](../../relax/models/deepseekv41/configuration.py)
3. 在现有 header 校验上按目标模型核对 key、shape、dtype、重复或缺失映射、主干与 draft 边界；对冻结资产做逐字节校验，对解量化的代表性 dense、expert、FP32 参数和 Engram 行做数值抽验。真实导入后核对 actor 与 rollout 的视觉层数、已加载参数、特殊向量和视觉 bias。实测 BF16 wrapper/loader 之后的视觉 norm dtype，并用同一组真实 patches 分别与官方 ViT、aligner 输出比较，覆盖奇数网格的补边和 3×3 channel-major 重排，避免训练与推理两侧犯同一个错误。文本脚本里的 `SGLANG_SKIP_CHECKPOINT_LOAD_CHECK` 不能代替此验收。旧 flat checkpoint 继续用于纯文本回归，收到图像时须在入口报明缺失资产。
4. 从相同 1017 快照准备 **8 卡可运行**的四层验证产物。保留完整宽度的 ViT、aligner 和匹配的语言 embedding。记录语言层、专家、Engram、KV/index/candidate source 与 `bias_vl` 的裁剪和重映射；沿用已验证四层规模以控制容量。Engram 的哈希乘数随原始 layer ID 生成，重编号时须保留相应的种子与表行语义；如采用新的四层拓扑，则把它明确标作独立验证代理并为两端建立同配置参考。已有 Miles 四层模型是专门拓扑，不能将旧权重拼上视觉塔作为 1017 对齐证据。模型前向以相同裁剪配置和权重的未打包参考作对照；40 层的跨 PP 共享和容量另验。

完整 40 层源索引的预期为 259 个 `vision.*`、4 个 `aligner.*`、3 个 `image_*`、40 个主干和 3 个 draft 的视觉选路 bias，以及两对 Engram table/scale。A 阶段保留前 266 个视觉张量和 40 个主干 bias，并在目标 schema 中明确排除 draft。第三方已有相同 1017 revision 的多模态 BF16 转换，保留了 MTP/DSpark，可以直接用公开清单交叉核对 dtype 和覆盖范围；作者报告了离线反量化抽验，尚未在该产物上验证运行。正式训练资产以本次实际产物的清单和冷加载为准。[原版索引](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/model.safetensors.index.json) · [第三方 BF16 转换记录](https://huggingface.co/EnvLoop/DeepSeek-V4.1-Flash-1017-bf16/blob/c37b570ad54344140c2ac089b9a3d57fc1514461/README.md)

### B. 建立训练与 rollout 共用的图文输入语义

在 V4.1 的 processor/renderer 适配处，从原图与消息顺序生成一个 compact prompt，以及 actor 使用的 expanded token IDs、原生 patches、每图网格、样本内 span 和角色。SGLang 接收 **每图一个占位符的 compact IDs + 对应 image_data**，由它展开。传输保留相同图像内容，防止通用 Qwen resize、二次裁剪或有损重编码引入差异。检查 rollout 端的实际展开结果；若服务端不回传 expanded IDs，用固定输入在服务端 processor 边界采集同一请求对照。[rollout 入口](../../relax/engine/rollout/sglang_rollout.py) · [图像读取](../../relax/utils/multimodal/image_utils.py) · [官方图像处理](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/image_processor.py)

对齐单位是**每条样本的真实 expanded 序列**。一个图像 span 的所有位置使用同一个 image token ID；由 start / patch / newline / end 区分内容。patch=14，aligner 下采样 3×3，LLM 网格 `h×w` 消耗 `h*(w+1)+2` 个 token。纯文本样本保留零图槽位，回答和 padding 具有独立有效性标记。长度预算按展开后的序列计；图数不符或截断穿过 span 时按样本报错或走已有超长过滤并计数。response 起点、old/ref/train logprob、loss mask 和 rollout R3 的 `seq_len-1` 对齐在同一 ID 序列上；采到保留 image token 时核对采样/训练 token 身份，避免静默替换后仍使用旧 logprob。

复用 `Sample.multimodal_train_inputs`、TransferQueue 和流式 loader，图文 payload 随样本一起 shuffle、重排、切 microbatch。定义每样本图数、每图 patch 数与 span；保留混批里的零，不对长度不一的 `[1,L_i]` role tensor 直接按 batch 维 cat。普通 GRPO 也要验证图像与占位符一一对应；当 tokenizer fallback 把 processor 解析为 None 时，带图样本在进入 rollout 前失败。[处理器加载](../../relax/utils/data/processing_utils.py) · [样本准备](../../relax/utils/data/data_utils.py) · [流式传输](../../relax/utils/data/stream_dataloader.py)

**SFT 的输入从真实训练入口单独验收。** 使用 `OpenSearch-VL/Search-VL-SFT-36K` 的固定 revision `2c1c460af4fa15bd63210cbf426a96664b959944`，复用七域转换、问题和原图像素分组切分、去重索引。既有配方切分后得到 train/val/test 各 34,607/32/32 条，长度过滤发生在训练阶段。保留数据集准备后实际交给模型的 PNG、原图摘要和完整“提问 → 推理与工具调用 → 带图观察 → 最终答案”历史；按 **1017 展开后**的 32K 上限执行既有超长跳过策略，记录实际消费、跳过及监督 token 数。当前 V4.1 整模文本脚本读取 `problem/generated_solution`，图文 SFT 改走真实 messages/images 路径。[数据准备](../../examples/opensearch_vl/data-preparation.md) · [SFT 多模态入口](../../relax/engine/sft/dataset/multimodal.py)

沿用 V4.1 模板对 `thinking_mode` 和历史 reasoning 的处理。用含两次 assistant 发言及带图 tool 观察的真实轨迹，对照官方整段编码逐 token 检查：assistant 的思考、文本工具调用及最终答案得到监督，system/user、工具返回、图像 span 和 padding 全部 mask。配方以文本标签表达工具行为，保留其原文，验证跨轮后没有丢历史或重复注入工具标记。现有 `audit_data.py` 使用 Qwen 的 `<|image_pad|>`；接入 1017 processor 后从模型元数据取 image token，检查真实 expanded IDs、监督 mask 及图序。将该样本同独立语言样本一起 pack，比较单样本和 THD 的监督 token 总数、loss、语言梯度及一步更新。[DeepSeek 模板](../../relax/engine/sft/dataset/deepseek_chat_template_patch.py) · [真实 SFT 数据审计](../../examples/opensearch_vl/tools/audit_data.py)

### C. 复用 V4.1 THD，在首个 PP stage 接入冻结视觉前向

- **先拆开两个现有判断。** “模型接收多模态字段”与“模型要求通用 VL unsplit/zigzag 重打包”独立选择。当前 `get_batch` 遇图可能构造另一套 padded offsets，训练 forward、old/ref forward 的 mm kwargs 门控也不同。统一 V4.1 在微批调度、训练、重算 logprob、R3、response gather 与 loss 中的布局选择，其他 VLM 保留各自的分片行为。只在需要消除两条调用路径分歧时提取共享逻辑。[batch 与调度](../../relax/backends/megatron/data.py) · [两条 forward](../../relax/backends/megatron/model.py) · [R3 与 response gather](../../relax/backends/megatron/actor.py) · [loss](../../relax/backends/megatron/loss.py)
- **图像归属使用现有 `ThdLayout`。** 从真实/物理 cu_seqlens、sample-local span 和 contiguous CP 区间求交；图可跨 CP，不能跨样本或落入 padding。保留已有 CSA2、RoPE、滑窗和 indexer 样本隔离，图像位置与文本位置共享该样本的位置序列。用未打包逐样本对照验证 split 前后的顺序、奇数网格、padding-only CP rank。沿用已有动态 token 预算，另记图像 patch 数对首 stage 峰值的影响。[THD 布局](../../relax/models/deepseekv41/thd/layout.py)
- **首 stage 合并，语言链保留梯度。** 建议第一版在首 PP stage 的 TP/CP 副本上复制冻结 ViT/aligner，在有限图像预算下各自处理本地需要的整图，按全图顺序取该 rank 负责的 span 行；以相同的 TP/SP embedding 分布作替换，确保只切分一次。若选带 collective 的视觉实现，无图或纯 padding rank 也必须按同序参与。仅视觉计算设 eval/no_grad，合并后的语言 embedding 保持梯度，再进入首层 mHC。后续 PP stage 使用已有隐藏态/carry 和本 microbatch 模态元数据，不传递原始像素。跨 CP 重复计算成本在八卡测量后再决定是否引入视觉特征分发。[provider](../../relax/models/deepseekv41/provider.py) · [模型入口](../../relax/models/deepseekv41/model.py)
- **视觉路由与 Engram 必须一起接。** 官方按图像位置选择 `bias_vl`，按文本位置选择原 bias，仅改变选 expert 的分数；混合权重的计算保留原语义。R3 记录与消费都使用同一 expanded 顺序，并另做关闭 replay 的对照，防止 replay 遮蔽错误选路。Engram 对图像位置按官方 mask 处理 hash 历史并屏蔽 residual；重点验证最后一张图后的前几个文本 token、跨 CP 前缀和相邻文档。对所有 MoE 层提供模态信息，包括最后一个 Engram 层之后的 stage。[1017 官方模型](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py) · [Relax Engram](../../relax/models/deepseekv41/engram.py)
- **PP 和 full recompute。** 优先由各 stage 的同一微批元数据重建 CP/SP-local mask；无法重建时才扩展现有 carry。当前 input IDs 的 carry 生命周期仅到最后一个需要它的 Engram 层。新增模态信息须绑定每次 forward，在交错的多个 microbatch、1F1B 及 backward 重算时可恢复。验冻结视觉前向调用次数和释放时机，防止把 pixels/整图 features 驻留到下一个微批或重放错误状态。参考上游的 packed image 梯度测试，但保留 Relax 的 TP/SP/PP 实现。[Relax carry](../../relax/models/deepseekv41/runtime.py) · [MCore #8018](https://github.com/NVIDIA/Megatron-LM/pull/8018)

### D. 接好在线更新、卸载与恢复

训练、reference 和 rollout 从相同 immutable base 初始化；冻结项在 optimizer 构造之前登记，沿用 BF16 训练权重到 SGLang 的 Bridge 同步。当前迭代器会枚举 parameter 和 persistent buffer，单设 `requires_grad=False` 不会使其自动退出同步。优先沿用已有同步方式跑通，逐项核对首载覆盖、offload 备份与恢复；仅在能从同一 base 保留或恢复冻结视觉资产时，按显式资产清单跳过每轮传输。保留所有可变语言张量的正常同步。启动、仅 sleep/wake、一次非零语言更新后、从 DCP resume 后分别检查视觉指纹和同图输出。对缓存固定图像的 rollout，权重版本切换必须使依赖语言权重的 KV 失效；重复同图与换图的结果都要比较。[Bridge](../../relax/models/deepseekv41/bridge.py) · [权重迭代器](../../relax/backends/megatron/weight_update/hf_weight_iterator_bridge.py) · [SGLang 加载与 offload](../../relax/backends/sglang/sglang_engine.py)

恢复沿用已有 Megatron `torch_dist`（DCP），记录每种冻结资产由 DCP 携带还是由固定摘要的 BF16 基座重载。冷恢复时核对已提交的 optimizer 更新数、optimizer/RNG、下一批样本和图片顺序、语言参数及视觉指纹；SFT 另核对 `sft_data_state`，RL 另核对策略版本。导出器从明确的一次已完成保存读模型权重；在清单中同时记录 DCP iteration 和实际完成的更新数，防止零基步号或 tracker 后移选错最终权重。

### E. 把 SFT 最终 DCP 导出为可独立评测的 HF 模型

**Miles 可以复用到哪里。** 主线 `fa13e2ec` 的离线工具已能从 DCP 组装张量、调用 V4.1 文本反向映射、写 safetensors 和新 index。但映射显式跳过冻结的 Engram table/scale，缺少 ViT、aligner、image 特殊向量和视觉 bias；`--origin-hf-dir` 只复制顶层非权重文件。Relax 当前文本 Bridge 声明 `SUPPORTS_HF_PRETRAINED_EXPORT=False`，仓内 CPU 导出器的模型映射仅覆盖指定 Qwen 架构。已有在线 SGLang 同步能提供语言张量的映射对照。另有 Miles fork 的开放 PR 记录了原生量化 1017 整模导出、96,085 个键和独立推理重载，公开了视觉键映射。该 PR 的图像训练实测为 CP1、BSHD、MBS1；公开变更中未找到生成该完整目录的专用离线命令，复用其映射和校验方法仍需接入 Relax DCP。[Miles 离线工具](https://github.com/radixark/miles/blob/fa13e2ec42622f6e77aabd8e0c295ab8b4347eac/tools/convert_torch_dist_to_hf.py) · [Miles V4.1 映射](https://github.com/radixark/miles/blob/fa13e2ec42622f6e77aabd8e0c295ab8b4347eac/miles/backends/megatron_utils/megatron_to_hf/deepseekv4_1.py) · [fork 导出实测](https://github.com/apexin-ai/miles/blob/38c5092628b779bf6cda6966f4797d85655d898a/docs/models/deepseek/deepseek-v4-1-1017-vision-sft.md#complete-native-hf-export-and-inference) · [fork 公开改动](https://github.com/apexin-ai/miles/pull/5/files) · [Relax Bridge](../../relax/models/deepseekv41/bridge.py)

| 交付物 | 来源与验收 |
| --- | --- |
| **训练基座**，1017 多模态 BF16 HF | A 阶段生成，固定 revision、转换配置和全部 shard 摘要。作为训练初始态、正式 Base 评测输入及导出时冻结资产的来源。基座和最终模型保持相同的 BF16 语言格式、原生 Engram、视觉配置及无 MTP/DSpark 的推理设置。可用官方量化原版对固定图文 prompt 做初始转换校准，单独记录格式差异。 |
| **续训 checkpoint**，DCP + SFT 数据状态 | 绑定训练 commit、基座清单、并行拓扑、数据版本和最终保存步。先核对 DCP 可冷恢复，记录可训练参数及冻结资产清单。Engram 的大表在当前实现中是从 HF mmap 加载的普通属性，DCP 恢复依赖该基座。[Engram 实现](../../relax/models/deepseekv41/engram.py) |
| **最终评测目录**，`final-hf-<step>` | 按模型实际保留的架构输出完整、自包含的 safetensors 分片、index、复合配置、tokenizer、模板、1017 编码/图像处理所需文件及来源清单。分片和逐键校验全部通过再原子发布，可由新推理进程仅凭此目录加载；优化器、数据游标留在续训产物。 |

实施优先扩展现有流式 CPU DCP 导出器，复用其有界内存读取、原始字节拷贝和临时目录发布。在真实四层 DCP 上列出模型 state keys，再补 V4.1 选择和映射，核对 PP 全局层号、TP 分片、EP 专家、fused gate/up、mHC 与 Engram **可训练投影**等。每个应有的可训练 HF 键必须由指定 DCP 提供；对训练模型明确拥有、但没有对应目标键的项逐个说明。只允许从同一次训练基座按白名单补冻结资产，含保留层的 Engram 表/scale、固定视觉参数和选路 bias；若 DCP 也携带冻结项则两处验同。禁止用基座静默填回漏导的可训练项。[CPU 导出器](../../scripts/tools/convert_checkpoint_to_hf_cpu.py) · [CPU 映射入口](../../scripts/tools/convert_checkpoint_mapping.py) · [训练侧映射](../../relax/models/deepseekv41/bridge.py)

**先验证零更新，再验证非零更新。** 在四层上先用不改变权重的保存对照逐键验证 BF16/FP32 和原始量化字节，再实际做至少一步 SFT、落 DCP 并离线导出。给语言更新留下可重建的代表性张量值与变化量；核对每个冻结资产的摘要、index 实际覆盖和所有 shard 的摘要。另起无训练进程、无基座权重目录可见的 SGLang 服务，只提供导出目录；固定文本、单图、换图及带工具返回图的多轮输入，将导出后 teacher-forced logits/logprob 与同一步保存前模型配对，也检查零步与训练后的变化。完整 40 层最终 DCP 重复做覆盖检查和冷加载。首个交付为训练精度的 BF16 HF，保留原有 FP32 例外和 Engram FP8 bits/scale。若部署确需原生 FP8/MXFP4，再单列重新量化及独立精度验收。

### F. 按 OpenSearch-VL / V\*Bench 协议验收整模 SFT

1. 用同一 1017 BF16 基座的 **Base** 与 **冻结视觉、OpenSearch-VL 1 epoch 后的 trained-final** 做配对。记录训练 commit、开始时间、拓扑、全局 batch、有效训练量、恢复边界及最终 DCP/HF 摘要；按验收标准在 redai-studio 下留存 SwanLab 训练曲线链接。8×H800 四层覆盖链路与并行正确性，40 层训练及两组评测需按权重和上下文容量安排多节点资源。同一套评测协议先在 Base 上用两题冒烟，验证从真实 crop、返回图片到下一轮识图的链路；正式评测中模型自主决定是否调用工具。[验收标准](https://docs.xiaohongshu.com/doc/1a5e9d419546f8377db1e6fdd6bab8e8) · [已有配方与实验模板](../../examples/opensearch_vl/README.md)
2. 复用已有 evaluator 的提示词、crop 坐标、返回原图区域、完整历史和 v2 选项评分。1017 的 `HTTPRunner` 使用原生消息编码器、图像 processor 和展开后的 token 计数（含每行换行与首尾标记），通过 SGLang `/generate` 提交紧凑 token ID 和原图。以服务返回的 prompt token 用量核对含裁剪图的第二轮计数。保留原始 `<tool_call>` 与 `<response>` 文本，验证服务对 reasoning 和工具标记的呈现。Base 和 trained-final 使用相同模板哈希、thinking 设置、每轮历史和评分器；遇到计数或服务异常先定位并重跑受影响的评测。[HTTP 客户端](../../examples/opensearch_vl/tools/benchmark.py) · [V\*Bench 入口](../../examples/opensearch_vl/tools/vstar_benchmark.py)
3. 固定 `craigwu/vstar_bench@d9ae62c903da0c98336e85c5ee89cd863b04b4da`，每组完整 **191 题**，温度 0，输入输出合计 32K，最多 10 轮，每张显示图最多 1,024 个视觉 token。每题生成一次，模型超限、无效选项或无效工具调用等计入分母。按实际 SFT train 切分的问题/原图像素 hash 另报去重分数及分母。保留逐题输出、真实工具轨迹、失败分类、协议/模型/评测代码指纹、`summary.json` 和 `benchmark-comparison.json`。[固定评测协议](../../examples/opensearch_vl/evaluation.md) · [同协议比较器](../../examples/opensearch_vl/tools/compare_benchmark.py)
4. 分开记录工程链路完成与模型效果结论，报告 Base / SFT 原始和去重 Pass@1、命中数、百分点差及对错翻转样本。标准目前给出了其他模型的结果，**未规定 1017 数值门槛**。正式训练前用 1017 自身 Base 登记效果判据；出现退化时对照输入/导出检查、工具调用与答案格式，把效果验收留待归因。跨模型分数、训练 loss 和四层冒烟各有独立用途。[结果模板](../../examples/opensearch_vl/templates/model-results.md)

## 4. 验证阶梯与通过条件

进入 128 卡整模前，先在原生四层、8×H800 的真实图文轨迹上验证 loss 下降。记录连续训练步的 loss、前后窗口均值，以及固定的未参与训练的图文轨迹在多个更新点的 loss；两个不同 batch 的单步数值不足以判定趋势。冷恢复的第一步还需与该 DCP 导出 HF 在同样本、同监督位置上的 NLL 配对。

**1017 四层基座实测（2026-11-19）。** 从上述固定快照保留原层 0–3，输出 8 个分片、4,977 个张量，权重载荷 215,375,883,552 字节。冻结的视觉、四层视觉路由 bias 与一对 Engram 数据合计 272 个张量，源与 BF16 产物逐张量 SHA-256 全部一致。完整 40 层按同样格式只读预检为 47,274 个张量，约 1.308 TB。

**OpenSearch-VL 数据实测。** 核对 recipe 固定 revision 的七域 JSON 和图片 ZIP 后，36,592 条原始轨迹经原配方校验，得到 train/val/test 各 34,607/32/32 条，1,921 条记录在拒绝清单中。按 1017 原生 tokenizer 和图像网格规划逐条计算，所有训练样本均在 32K 内，展开长度中位数 8,857，最大 32,542。训练集共 340,372,389 个输入 token，其中 31,753,987 个 assistant token 受监督。全局 batch 32 按现有训练步数规则完成 1,081 个整批，尾部 15 条。另对每个切分各 8 条走实际图像处理和 loss mask 审计；35 条八卡诊断样本全部属于正式训练切分。

在 H800 上，两张相同尺寸的图片分别经过 1017 自带的预处理和 32 层视觉参考实现。Relax 的 ViT 输出、aligner 输出及包含图像起止/换行向量的 212 个 embedding 均逐元素相同。固定问题并换图，四层 BF16 基座冷启动的 SGLang 单图请求中，候选 `red` 的下一 token logprob 从 −15.9265 变为 −15.4819；双图工具历史中的展开长度为 502 个 token。这项检查覆盖图像对语言输出的影响，正式任务得分在整模上另测。

八卡 SFT 使用 TP2/SP、PP2、CP2、EP4/ETP1、DeepEP、full/uniform/1 重算与 CPU optimizer offload。四条实际轨迹打成两个 THD 微批。首个更新前，Megatron 对 2,469 个 assistant token 的平均 NLL 为 14.03320；同一 BF16 基座的 SGLang 逐条 teacher forcing 为 14.03534，绝对差 0.00214。两次更新后的 DCP 保存模型、optimizer 和随机状态，同一实验目录的 `sft_data_state/` 保存下一批的游标。CPU 导出从 DCP 重建 4,705 个语言张量，从同一冻结基座补齐 272 个张量。Engram 可训练投影、mHC、注意力投影和输出头均检出更新。

从这份 DCP 冷恢复时，训练数据状态为 `step=2, epoch=0, position=8`。恢复后第一批 3,026 个监督 token 在 Megatron 上的 NLL 为 14.00899，在该步导出的 HF 上逐条 teacher forcing 得到 14.01331，绝对差 0.00432。延长至 32 次更新的实验取训练集 35 条轨迹（七域各五条），固定验证集另取四条。全局 batch 为 4，最大序列长度为 16,384，恒定学习率为 `1e-6`。训练 NLL 前八步均值为 13.89853，末八步为 11.78591。首、末 PP stage 记录到的 PyTorch 峰值分配分别为 50.75 和 50.07 GiB，峰值预留分别为 52.78 和 55.68 GiB。

固定四条验证轨迹包含 2,586 个 assistant 监督 token。与正式训练集 34,607 行逐行核验，验证轨迹与训练集在轨迹 ID、归一化问题、原图像素摘要三项均无重叠。下表是相同图文内容与相同监督位置的逐 token 平均 NLL。训练作业成功结束，最终保存的零基迭代为 `iter_0000031`，SFT 数据状态为 `next_step=32, epoch=3, position=23`。[切分核验](../../deepseek_v41_vit_1017/split-overlap.json) · [逐步指标](../../deepseek_v41_vit_1017/sft-4layer-8xh800/loss.csv) · [训练与验证曲线](../../deepseek_v41_vit_1017/sft-4layer-8xh800/loss.png)

| 完成更新数 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 验证 NLL | 14.09041 | 13.82746 | 13.64142 | 12.94998 | 12.44208 | 12.18970 | 11.98022 | 11.10910 |

第 32 步独立导出的 HF 在 SGLang 冷加载后，对固定验证集的 NLL 为 11.11106，与同一步 Megatron 相差 0.00196。初始训练批的 2,469 个监督 token 在 SGLang 上由基座 14.03534 降至 11.06068。[实验记录与逐 token 证据](../../deepseek_v41_vit_1017/README.md)

| 阶段 | 执行内容 | 推进条件 |
| --- | --- | --- |
| P0 · 资产和输入 | 固定 1017，产出四层图文 BF16，完成 header/权重抽验。用无图、单图、不同网格双图、图文交错和无效占位样本对照官方 renderer 与实际 SGLang processor。原生 BF16 patches、角色、图序和 token IDs 留指纹 | 输入身份、expanded 长度、视觉键和 dtype 完整。冷启动训练端、rollout 端都实际读到 1017 视觉资产；同一问题换图能改变预定答案 token 的分数 |
| P1 · 单样本基准与混合 packing | 固定权重与 responses，同进程逐样本 BSHD/未打包对照多样本 THD，含图、文、双图；交换前一条样本图像/长度、交换样本次序。核对 span embedding、各层 route、Engram hash、response logits、语言梯度和一步参数增量。另跑纯文本历史样本 | 不相干文档不受扰动；本样本的正确图/错图有可分辨响应。整数身份/布局严格一致。连续量以同 dtype、kernel、seed 的重复 reference 测得噪声，预先登记容差及尾部指标 |
| P2 · 八卡联合配置 | 四层 8×H800，训练侧 **TP2/SP + PP2 + CP2 contiguous/all-gather + EP4/ETP1 + THD dynamic + full/uniform/1 + CPU Adam offload + flex/deepep**。一次 pack 至少两条样本，覆盖跨 CP 的图片、后 stage 路由、纯 padding shard 和多微批反向。以 alltoall 和关重算作定位对照 | 记录实参、训练侧 dispatcher、有效监督数量、PP 首 stage 显存和 CPU 内存峰值。所有 rank 完成 forward/backward/optimizer，视觉无更新，语言有更新 |
| S1 · 八卡 SFT 与 HF 往返 | 在 P2 配置上消费真实 OpenSearch-VL 图文轨迹，做零步和非零更新后的 DCP 导出，含一次保存、冷恢复并继续更新。复用 E 的白名单及 F 的两题工具链路冒烟 | 每条样本监督位置正确，续训消费预期的下一批，冻结资产一致，导出 HF 仅靠自身即可在新服务读图，并与同一步 actor 的配对数值相符。此项通过后再启动 40 层正式 SFT |
| S2 · 40 层 SFT 与正式评测 | 在多节点 H800 上先验证 32K 长轨迹的显存、首 PP stage 图像负载及全层映射，再以 TP/PP/CP、THD、重算、CPU offload 和 alltoall 完成 1 epoch。DeepEP 在单机覆盖，多机按集群可用性另测。导出最终 HF，用 F 的同协议完整评测两组 | 提供训练曲线、DCP/数据游标、可独立部署的最终 HF、Base 与 trained-final 各 191 条轨迹、原始及去重分数和效果结论。诊断结果同时列出数据过滤和失败类别 |
| R1 · 八卡在线图文 GRPO | 基于 P2 开启 **R3**。Rollout 复用 TP8/EP8、attention DP1/CP1、PP1、MoE A2A none。用可判分的图文数据，至少两次非零策略更新后再采样。同一份 rollout 重放未打包/THD old/ref/train，覆盖 sleep/wake 和保存后冷恢复 | 组内 reward 有差异、实际消费非零 advantage。报告权重版本、逐 token logprob MAE/p95/p99、按图后距离的误差与语言参数增量。冻结资产摘要一致 |
| R2 · 40 层多节点 RL | 从同 revision 的完整图文基座启动。可从历史及当前 DAPO 的 TP2/PP2/CP2/EP64 拓扑做同批 alltoall 对照，再切换 flex/deepep，运行在线更新和冷恢复。覆盖后半模型的来源层、全部视觉 bias 和跨节点通信 | 单独报告数值与稳定性，按图密度和长度报告有效 response tokens/s、总有效 tokens/s、视觉前向、重算、optimizer、权重同步及 offload 时间，以及各 PP stage 显存和节点内存峰值 |

统计时锁定模型/数据版本、图像字节、chat/thinking 模式、response token、路由、精度、随机种子与权重版本。至少区分**同策略同轨迹配对误差**和**不同策略独立在线轨迹的趋势**。没有图文噪声底之前，先用 P1 的重复对照确定阈值；若出现跨样本影响、图片丢失、错位路由或冻结权重变化，直接定位，无须等待平均指标恶化。

## 5. 最小改动与实施边界

第一批变更限定在 V4.1 的资产、processor、模型和必须共享的布局选择点。迭代中的文本代码，以实际入口的行为为准；可先提取 old/ref 与 train 的共同布局/kwargs 选择，配对验证纯文本 THD 后再接图像。复用现有 RL 数据结构、奖励、Bridge、调度器和 launcher。若实施确需改公共参数解析、Controller/Service/Launcher、新依赖或远端安装包，单列变更原因和影响，再按项目约定确认。

持久化测试优先扩展已有用例。给[正向转换](../../tests/cpu/tools/test_deepseek_v41_conversion.py)和[CPU 导出](../../tests/cpu/tools/test_convert_checkpoint_to_hf_cpu.py)补一个小型含视觉与 Engram 的真实往返，检查非零语言更新写入最终 HF，并在缺失应训练键时失败。用真实 SFT 预处理复现工具返回图与历史监督错位，复用[多模态数据用例](../../tests/cpu/engine/sft/dataset/test_multimodal.py)。跨 CP、PP 重算、同步或唤醒后冻结资产丢失的风险复用[THD 布局](../../tests/cpu/models/deepseekv41/test_thd_layout.py)与[BF16 同步](../../tests/backends/megatron/weight_update/test_v41_bf16_sync.py)覆盖合适层级。需要反向或通信的行为留在远端 GPU 验证；完整 benchmark、临时探针及 dump 留在仓库外。

依赖 Megatron、SGLang、Transformers 或 GPU 的调试均在远端既有镜像进行。使用共享代码目录与有界前台探针。RayJob 上保留 Ray 服务，结束后检查本任务的作业、进程及显存。跨节点盘、镜像和环境指纹检查可从 head 使用 [run_on_each_ray_node.py](../../scripts/tools/run_on_each_ray_node.py) 的 `--include-cpu`。

可用的独立对照包括 [AutoModel packed vision SFT 的跨 CP 实跑](https://github.com/NVIDIA-NeMo/Automodel/pull/4565)、[Miles fork 的 1017 原生视觉加载/路由与图像 SFT](https://github.com/apexin-ai/miles/pull/5)、[Spindle 1017 文本 packed GRPO](https://github.com/modal-labs/spindle/pull/193) 和 [verl Geo3K rollout/actor 对齐记录](https://github.com/verl-project/verl/pull/7483#issuecomment-6618780337)。分别用于校验视觉 packing、精确版本资产、文本 RL 和图文 logprob；本计划的联合验收以 Relax 实跑为准。
