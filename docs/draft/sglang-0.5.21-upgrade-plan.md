# CUDA 13 镜像升级到 SGLang 0.5.21

## 版本与构建

`docker/Dockerfile.cu13` 使用官方 `lmsysorg/sglang:v0.5.21`（CUDA 13），并将 TileLang 对齐到 0.1.12。SGLang 补丁使用独立的 `SGLANG_PATCH_VERSION=v0.5.21`；Megatron 的 `PATCH_VERSION` 不受影响。普通 `docker/Dockerfile` 的 CUDA 12 镜像及对应补丁固定在 SGLang 0.5.17，不再升级；后续 SGLang 升级仅维护 CUDA 13 配方。`latest` 下两份 SGLang 补丁均通过符号链接指向 0.5.21；0.5.17 的 per-position 补丁保存在 `docker/patch/sglang/v0.5.17_per_pos_topk.patch`。

目标源码为官方 tag `v0.5.21`，commit `e00930c5489053f26d86b179cee0d087f846acbb`。迁移需求以 Relax 原提交 `8e667d5c5815fcf223bb17fc2ce90099b6602f7c` 的 `docker/patch/sglang/v0.5.17.patch` 为准（SHA256 `783bd672f4be44312bce14b5bb0df2dbe243dd8bf0b70bdb1d9c5457e8969d4a`），逐项移植到官方源码。THUDM/slime commit `a4202e0595467d9374d9a68d45a34f45c3f8d230` 仅用于对照新版接口，不作为补丁骨架或需求来源。两份补丁位于 `docker/patch/sglang/v0.5.21.patch` 和 `docker/patch/sglang/v0.5.21_per_pos_topk.patch`，分别复制为镜像内的 `sglang.patch` 和 `sglang_per_pos_topk.patch`，按此顺序应用。

新基线包含 Torch 2.13、FlashInfer 0.6.18、sglang-kernel 0.4.7。旧 0.5.17 镜像上的二进制扩展不可直接复用：TE、FlashAttention 和 INT4 扩展必须针对新 Torch 编译，FlashInfer cubin/cache 版本必须与 Python 包一致。构建配方在新基线内编译扩展；B300 sampling 路径按安装包定位，兼容官方镜像的 `/opt/sglang` 虚拟环境。

0.5.21 默认使用 Rust radix tree。迁移 HiCache 行为时也要更新 Rust 实现、正常 PyO3 导出及 Python adapter，单独修改 Python tree core 不足以生效。镜像复用 router builder 的 Rust 工具链，通过 `docker/scripts/build_sglang_tree_core.py` 执行锁定依赖的 release 构建，沿用官方 Torch C++ ABI、兼容头和相对 RPATH 设置。安装时检查 Torch/Python ABI 和产物 SHA256，覆盖准确的 `mem_cache${EXT_SUFFIX}`，再用新进程验证实际 loader 加载的文件与新增生产接口。镜像设置 `SGLANG_RUST_BUILD_MODE=never` 使用已构建的原生扩展，避免 editable 源码布局在运行时另选 JIT 产物；仍使用 Rust 后端。

## 补丁迁移决定

保留并适配以下 Relax 行为：

- 权重更新后处理接口、NCCL LoRA 更新及按 bucket 接收的内存上界。
- Kimi K3 与 MXFP4 权重重载保留 CUDA Graph 捕获的存储；失败的 post-load 保持更新会话，成功后才结束。
- BF16 MoE reduce-scatter 使用 FP32 累积，保留目标缓冲区、分片大小和 all-gather dtype。
- BF16 MoE 热更新恢复加载形状时，同时覆盖 TRT-LLM 普通和 routed 变体，保留原存储地址。旧客户端直接恢复生成时，仅对本次恢复过加载形状的 MoE 层重排块布局；显式后处理会清除标记，避免重复重排。后处理失败时保持暂停。
- 新版 MXINT4 MoE 在加载时恢复带 `weight_loader` 属性的原参数，推理时复用固定参数与存储；后处理前后清理置换索引缓存，避免 colocate 卸载显存后复用失效索引。重复后处理不再次重排，转换失败保留加载状态。
- BF16 非 routed 和 MXINT4 的 FlashInfer 融合 MoE 通过内核的 `routing_replay_out` 捕获实际专家 ID，使用连续 int16 暂存后写入原生 int32 cache，避免 R3 返回全零路由。保留 draft opt-out；捕获关闭时不分配 GPU 暂存，内核异常时不发布无效路由。
- DSA query head 重复与 head-gate 对齐，兼容新版 graph helper 路径；移除迁移残留的旧工具函数导入。
- Qwen3.5 多维位置编码走可处理 MRoPE 的准备路径；一维位置仍可使用融合 CUDA 路径。
- DeepSeek V4 更新权重后原地刷新已经创建的 attention sink 缓存，保留官方新增的按 kernel head 数返回 view 和延迟分配；分桶加载时保留原 Relax 的有界未加载参数日志。
- 多模态处理器异常传播、扩散 rollout 的确定性噪声/时间步及内存内 LoRA 更新；LoRA pipeline 迁到新版包路径，按官方 `_weight_update_context` 和 `snapshot_base` 处理 layerwise offload，并在更新前失效 conditioning 缓存。
- per-position OPD 查询校验、嵌套查询和 base64 logprob 传输；适配新版 pinned D2H 和 log-normalizer 路径。
- Qwen 原生图像 OPD 的 uncached 路径保留官方 `load_mm_data()`，不采用 Slime 参考补丁中直接调用 `legacy_load_mm_data()` 的改动。由官方入口保存原始 `input_ids` 并传递给快速或回退分支，让 `MM_AVOID_RETOKENIZE` 保留采样 token；回归覆盖两条分支及非恒等重新分词。保留 expanded prompt、响应 ID、R+1 logprob 长度及有限值的严格校验。
- 保留原 Relax 的 decode 等待超时，并接入官方 rank 一致的失败归约和清理；不引入原补丁没有的 prefill bootstrap/transfer 超时、retry count 或 decode 全量回撤策略。
- HiCache 的 chunked Mamba checkpoint 延迟 write-through 在新版 `insert_req` 中、请求解锁前补记一次；通过 `root_node_handle(extra_key)`、`is_root()` 及节点和受保护长度验证身份，abort/retract/跳过插入时清除标记。Python 和默认 Rust tree core 均复用各自原生的命中阈值与 BackupKV 路径，拒绝已回收、未锁定、无 Mamba、已备份和 pending 节点，也不额外增加祖先命中计数。暂停期间在官方共享的 scheduler hook 中轮询 HiCache DMA，覆盖原四处循环。
- 保留旧分桶更新的路由：无显式会话时 distributed 默认 target，tensor 按旧 draft flag 选择；调用方显式指定的 selector 优先，显式新会话沿用其 selector。

使用新版原生实现替换旧补丁：

- routing capture 的异步结果复制、快照和 host cache 生命周期；启用 Relax 的旧优化开关时不再导入已删除的旧 capturer 类。
- 单节点多卡多模态 CUDA IPC 判断，已在 `multimodal/transport` 按 `nnodes` 实现。
- DeepSeek RoPE 结构、state capturer 销毁、DP rank 别名与严格反序列化允许列表。
- KV cache/请求状态已迁到新的组件、receipt 和 Rust tree core 结构，沿用目标版本的插入与生命周期实现，不重放旧结构上的 hunk。

Kimi 的 THD/CP 训练包装层移除 pinned Bridge 工具不接受的旧参数，并将 packed embedding 转成 sequence-first 连续布局后再做 TP scatter。无 supervision mask 时直接返回语言模型 Tensor，保留有 mask 且 CP>1 时的 tuple 契约，让 Relax 的 logprob 和训练前向均可使用。该兼容修复作为独立 Megatron 补丁在 CUDA 13 镜像中应用，TP=1/2、CP=1/2、短样本和布尔/整型 mask 的 12 项布局回归，以及 8 项返回值、mask 和梯度契约回归通过。

Relax 的可重载进程组转发 Torch 2.13 新增的 `all_gather_single`、`reduce_scatter_single`、`all_to_all_single` 及 coalesced 方法。Megatron 提前导入的通信别名也必须落到当前内部进程组，避免 colocate 恢复后使用没有 CUDA backend 的包装组。回归覆盖真实双进程 Gloo 和双卡 NCCL 的三轮销毁、重建与通信。

Relax 侧通过 SGLang 的 `record_fields` 读取新版 msgspec `ServerArgs`，同时兼容旧版 dataclass。构建检查和 GPU 导入检查会校验 `RELAX_SGLANG_VERSION`，避免在旧源码上误报验证成功。

## 跳过启动 warmup 时的 GC 兼容

Relax 原有的普通 colocate policy rollout 启动路径设置 `skip_server_warmup=True`，在服务启动后同步训练权重、恢复并预热引擎；GenRM 等未跳过 warmup 的路径继续沿用正常预热。0.5.17 的 HTTP 服务没有自动冻结 GC 的启动调用；0.5.21 官方新增的 `_wait_and_warmup()` 在跳过 warmup 后仍立即请求自身 `/freeze_gc`，此时可能尚未监听端口。实际训练日志出现了连接拒绝，因此不能仅因训练自然结束就忽略这段异常。

0.5.21 补丁只在未显式跳过 server warmup 时执行官方自动 GC 冻结，保留旧 Relax 跳过路径的启动行为。正常 warmup、失败返回、权重就绪等待和 launch callback 的顺序保持不变；未显式跳过 warmup 的 elastic joiner 沿用官方行为。该修改不添加重试，也不将日志过滤当作修复。回归直接执行安装源码中的两个函数，覆盖跳过 warmup、正常与失败 warmup、elastic joiner、权重等待及可选 callback。当前源码与运行时已完成默认 Qwen 五轮，原 GC 连接拒绝未再出现，实际 R3 和权重更新正常。验收采用下述训练指标及性能口径。

## 多模态 CUDA IPC 默认值

Relax 在注册 SGLang 参数后，将 `sglang_mm_feature_transport` 的默认值设为 `cuda_ipc`。普通 rollout、PD 的 prefill/decode 引擎及 GenRM 均沿用现有参数传递路径；显式 `--sglang-mm-feature-transport cpu` 可覆盖默认值，各引擎 YAML/config 的 `mm_feature_transport` 保持现有最高优先级。原始 Qwen R3 脚本无需追加传输参数。

CUDA IPC 负责同一节点内多模态预处理进程与推理进程之间的 GPU 张量传递，TP/NCCL 和 PD/Mooncake 的通信方式仍由各自参数控制。SGLang 对 `cuda_ipc` 要求 `nnodes=1`；多节点引擎应显式选择其支持的传输方式。`mm_processor_worker_num=0` 保留 0.5.21 原生按模型选择的策略，CLI 和 `server_info` 中的 0 不代表没有 worker，内部有效并发数由处理器另行决定。

新增回归使用实际安装的 0.5.17/0.5.21 CLI，分别检查普通、prefill、decode、GenRM 的默认值与显式配置优先级；与 CPU HiCache 参数回归合计每栈 27 项通过。此检查只证明参数路径，实际 CUDA IPC 传输、任务精度、更新后生成与性能仍以使用框架默认值的完整 GPU 实验为准。

## 验证口径

本次升级采用任务级验证：重点比较同配置多轮训练中的 `train/train_rollout_logprob_abs_diff`、平均生成长度和 `rollout/raw_reward`，观察指标量级与持续趋势，并注明样本量、截断率及奖励定义。训推 diff 是每个版本内部训练前向与 rollout 的差异，不要求两个版本的逐 token logprob、隐藏状态或完整梯度严格一致。原有数值诊断结果保留，但不再作为升级验收的阻断门槛。

仍须检查请求与 token 对齐、数值有限性、实际权重更新、下一轮生成、自然退出和性能。对齐错误、NaN/Inf、持续训推 diff 恶化或任务指标持续退化仍需定位。实验不保存 checkpoint。

必须在全新的官方目标源码上分别执行 `git apply --check`，并检查所有修改的 Python 文件。运行现有 K3、MXFP4、BF16 通信、后端参数和 OPD 回归；测试夹具同步识别新版 runtime context，数值、缓冲区地址及失败状态断言保持不变。

GPU 验证使用 `scripts/training/multimodal/run-qwen35-35B-A3B-8xgpu-openr1mm-r3.sh`，保持原有 8 卡、batch、R3 和响应长度配置，设置 `NUM_ROLLOUT=5` 做自然结束的短程回归。验收需覆盖多模态 rollout、奖励、actor 更新、更新后的权重同步及下一轮 rollout，最终 Ray job 为 `SUCCEEDED`。模型和数据通过 `MODEL_DIR`、`DATA_DIR` 注入，使用 `RAY_NO_WAIT=1` 经 `scripts/entrypoint/ray-job.sh` 提交。

另验证 `scripts/training/text/run-kimi-k2.6-2layers-8xgpu-int4.sh`：保留 8 卡、DP attention、EP=8、CP=2、R3、INT4 推理及 BF16 QAT 训练配置，同样设置 `NUM_ROLLOUT=5`。模型配置保留调用方的 `NLAYERS=2` 覆盖，避免原先无条件重置成 61 层。INT4 QAT 的 STE 权重不支持 TE fused wgrad accumulation，因此脚本显式设置 `--no-gradient-accumulation-fusion`；保留原有 INT4 QAT、优化器和 batch 配置。INT4 和 BF16 检查点的 text config 均须为 2 层；验收包含量化权重回传后继续 rollout，最终 Ray job 为 `SUCCEEDED`。

两层 Kimi 检查点的索引也应与层数一致。如果 config 已改为两层但索引仍引用完整模型分片，推理加载器仍会遍历整个索引。本次通过 `HF_INT4` 指向隔离的两层索引视图：保留全部非语言层权重和语言层 0/1，只引用原始文件中的 5 个分片，配置、tokenizer 和权重内容与原目录一致。

旧镜像的 TileLang 0.1.11 在真实 Qwen3.5 GDN 反向中出现了已独立复现的梯度错误，不能直接作为正确性参照。原始失败和原包保持不变，另在旧 Torch/SGLang 栈中隔离使用与候选相同的 TileLang 0.1.12，完成独立 FP32 参照和固定真实 batch 的核验后再做训练对照。固定 batch 的旧版修正参照与候选梯度范数相对差为 1.62%，满足事先冻结的 2% 上限；这不等于证明完整梯度方向一致，也不表示原始旧镜像通过。对照结果须明确区分原始旧环境、修正参照及候选默认/共同推理后端。

Vision OPD 使用 `examples/on_policy_distillation/vision_opd/run-vision-opd-qwen3.5_9b-35ba3b-8xgpu-colocate.sh`，学生为完整 Qwen3.5-9B，教师为 Qwen3.5-35B-A3B，8 卡 colocate，保留每轮 4×8 样本及 1024 响应上限。该脚本固定 warmup 为 10 轮，首轮成对验证使用 `NUM_ROLLOUT=14`，补充对照采用同配置 seed=43、`NUM_ROLLOUT=26`，每版含 512 条 warmup 后样本；`NUM_ROLLOUT=3` 不满足调度器约束。验收逐轮检查 32 条非空图像样本、原生教师打分、有限且非零的训练梯度、权重同步及下一轮生成，最终作业须自然 `SUCCEEDED`。该配置使用 random reward 且关闭 RL reward，`raw_reward` 仅作为运行诊断，不能解释为任务正确率。HTTP 200 本身不能替代 token/logprob 对齐检查。新增 CPU 回归覆盖单图、双图、非规范短响应及 514-token 长响应，并在真实 GPU 上对照同次前向的 self-top-k 验证 per-position logprob 数值。

Qwen3.5 的视觉 head_dim=72 在 SM103 上触发 FA4 的反向编译限制：先在 backward preprocess 的越界 mask 处形状不匹配；只修该 mask 后，完整反向仍会触发 dQ reduction 的 tile 对齐断言。因此 TE 2.18 的选择器在 SM10x/11x 训练时，遇到 Q/K 或 V head_dim 非 32 对齐，就移除 FA4 候选，继续沿用原生 FA2、fused 等后端筛选。不会强制重新启用被其他规则禁用的后端，也不会改动语言 head_dim=256 的 FA4 路径。独立 TE 补丁在镜像中按零 fuzz 应用；FA4 包保持原版。前一处错误与上游 [flash-attention #2852](https://github.com/Dao-AILab/flash-attention/issues/2852) 的报告一致。独立 GPU 诊断覆盖视觉 head_dim=72 的短序列、跨 128 边界和 packed varlen 输入，记录 FP32 前向及 dQ/dK/dV 对照，同时检查 aligned-head、语言 GQA、推理及 CP 的原生选择。各项原始通过或失败结果保留，不再用跨版本逐元素一致性替代完整训练指标的判断。Kimi 的 Q/K=192、V=128 均对齐，且 CP>1 原本已禁用 FA4，此限制不改变其训练路径。

FA2 2.7.4.post1 的构建另显式指定 `FLASH_ATTN_CUDA_ARCHS=100`，为 SM103 提供可执行的通用 SM100 内核；仅设置 `TORCH_CUDA_ARCH_LIST` 不会覆盖该版本 FA2 的默认构建目标。

Mini SWE 的真实首轮训练还暴露了 TE 2.18 原生选择器的架构过滤缺口：head_dim 大于 192 时使用精确 compute-capability 白名单，包含 SM100 但遗漏当前 SM103，而 CP 又会禁用 FA4。固定回放保留的 16 条真实样本，在八个训练进程捕获到 CP4、Q8/KV1、head256、BF16 THD、32752-token 全局序列和无序列间 padding，并重现无可用后端。独立 TE 补丁仅将该 FA2 架构判断扩展到 SM10x，保留 head 上限、对齐及其他全部过滤条件；`num_splits=1` 本身合法，无须改变。旧栈隔离层与候选使用同一修正，训练 CP、长度、flash 配置不变。生产回归覆盖训练及 log-prob 阶段的后端选择；独立八卡数值实验覆盖两组四卡 CP、完整全局上游梯度的 FP32 前后向对照，以及真实 32k 长度前向。

该独立数值实验原有的严格 FP32 逐元素断言在两栈均为失败，原始结果完整保留：共有 8 个 dV 坐标和 1 个输出坐标超过逐元素容差，全部张量相对 FP32 的 L2 误差均小于 0.3942%。独立 CPU FP64 公式核验参考值，再按源码固定的 BF16 概率转换、分头结果、GQA 归约、CP 累加及每个加权 partial 的转换顺序，精确重建这 9 个原生值。原生 A/B 比较使用原容差：输入和 FP32 参考逐位相同，保存的输出、dK、dV 逐位相同，dQ 最大相对 L2 差约为 8.23e-6。未放宽原容差或改变生产计算，严格 FP32 对照的失败不能写成通过；这些独立分析仅用于确认共同运行时的已知 BF16 行为，完整任务质量、更新、缓存及性能仍需端到端验收。

MOPD 示例同步将旧的 logprob chunk 环境变量别名替换为 `SGLANG_ENABLE_LOGPROB_CHUNK` 和 `SGLANG_LOGPROB_CHUNK_SIZE`，包括 Ray 环境传播列表。0.5.21 已移除旧别名；继续使用会使 8192 的分块设置失效。新名称在 0.5.17 与 0.5.21 均可读取，实际 getter 与 shell 语法检查单独记录。

本次远程机器的显示名称为 H20G，但 NVIDIA 查询和 Torch 查询的实际 compute capability 均为 10.3（SM103，每卡约 268 GiB），按 B300 对应的 Blackwell Ultra 目标验证，按本次验证范围保留现有 `GPU_ARCH=b300` 和构建入口，编译目标仍为 `10.0a 10.3a`。旧镜像上的隔离环境验证不能替代完整 Docker build，也不能据此断言所有型号、模型、PD 或 HiCache 配置均通过。实验的实际环境指纹、检查结果、job ID 和日志应随本次交付记录保存。

## 升级前后成对验证要求

Qwen R3、Kimi 两层 INT4、Vision OPD、mini SWE PD + CPU HiCache 均需与原 0.5.17 环境比较。mini SWE 沿用 8 卡 2P2D、每引擎 TP2 的现有配置，使用真实 R2E parquet 与 SIF。两边固定模型、任务集合、参数和随机种子，分别记录代码、环境及模型配置指纹；不要覆盖已有基线文件。

验收检查各版本内部的训推 diff、token/logprob 对齐与有限性、训练 loss/梯度与权重更新、平均输出长度、真实任务 reward/沙箱测试结果，以及 PD、CPU 缓存、会话导出和清理行为。CPU cache 必须有 host 命中/回读证据，并对照不走缓存的固定请求和权重更新后的失效，不能仅检查启用参数。Agentic rollout 保存真实消息、工具调用、任务标识和 finalized Samples，确保得分对应实际执行的上下文。

性能报告首次加载/JIT 开销及排除这些开销后的整轮耗时、吞吐和延迟，结合实际输入及生成工作量比较。持续的任务级退化仍需定位；分阶段尖峰和独立微基准单独记录，不能把输出更长造成的耗时变化直接归因为内核退化，也不能用端到端变快掩盖独立微基准变慢。精度/成功率报告样本量和波动范围；本轮是短程升级回归检查，短程训练或全零 reward 均不能证明长期模型质量等价。

成对实验固定样本集合与配置，旧实验记录中的严格跨版本数值阈值作为历史诊断保留：Qwen、Kimi 和 mini SWE 均为 5 轮，排除首轮后比较；Vision OPD 为 14 轮，排除 10 轮 warmup 后比较。mini SWE 固定 10 个真实任务、每任务 8 个 session，独立重放原生沙箱测试判分，并检查导出样本、工具消息、PD 传输、两个 prefill 引擎的 CPU 回读和沙箱实例退出。缺少实际 CPU 命中时只能判为覆盖不足。两层 Kimi 若数学 reward 全零，需另以非零梯度诊断核验实际量化权重内容变化，并保留其不能证明模型质量的限制。

Mini 的逻辑上下文上限仍为 32768，响应上限仍为 8192。本次验证最初额外固定的 32768-token 池不足以容纳实际 32329-token 输入加原生 512-token decode reservation，导致队头请求不能预分配并阻塞后续请求；原始脚本并未要求该池大小。因此两版本共同池大小改为 65536，在正式重跑前分别通过同一份原生 1P1D TP2 长请求预检：并发提交原失败的 10 种输入长度，最长请求完整达到 32768 上下文，全部输出有限、P/D 请求身份对应且退出后 GPU 清空。预检还发现并修复本次迁移遗漏的 Rust root 接口与延迟命中补记；真实 Python/Rust 树的 92 项迁移回归通过。失败原始材料保留，任务级训练指标、性能及正式 2P2D CPU 缓存回载仍需检查。这些容量和接口回归不能替代正式端到端验收。

早期固定输入回放曾要求严格的跨版本 logprob 和 top-1 一致性；部分用例不满足该诊断标准，原始失败记录保留。后续分层采集确认差异从 BF16 full-attention 附近出现并传播；本次验收不再继续以逐元素一致性追查作为前提，改按上述训推 diff、平均长度和 raw_reward 判断实际影响。固定工作量性能记录仍保留 TTFT、端到端与 decode 延迟和设备状态，用于解释任务耗时变化。

原 0.5.17 镜像的默认 BF16 TRT-LLM 路径存在 R3 捕获的静态缺口，默认路径的完整热更新未完成验证；视觉 head_dim=72 的 FA4 backward 则有实际失败证据。成对 Qwen 与 mini SWE 采用已经可用的共同 Triton MoE 路径，候选版本另跑默认路径。这不能表述为旧默认 TRT-LLM 已观察到相同的 GPU 失败。旧环境通过独立 PYTHONPATH 视图使用同一 TE head 对齐及 SM103 CP 兼容补丁，原全局 TE/SGLang/Torch/Triton 文件保持原样。两栈共享的 `/root/Megatron-LM` 中 Kimi Bridge 兼容文件已原地调整，保存的任务前版本也含用户已有改动；因此不能把整个旧环境或 Bridge 称为原版未改。比较结果必须明确这些共同夹具，不能称为原镜像默认配置通过。

## 2026-10-22 任务指标验收结果

按本次调整后的验收口径，当前代码通过所列短程任务回归检查，可整理提交。这里的结论限于实际运行的配置，不表示跨版本逐元素一致或所有部署场景性能相同。旧严格诊断的失败记录保持原样，不改写为通过。

下表前三项指标使用各任务全程样本；Vision 补充实验使用预热后样本。整轮耗时均排除预热：Qwen、Kimi 和 Mini 排除第 1 轮，Vision 排除前 10 轮。箭头表示旧 0.5.17 修正参照到新 0.5.21。

| 任务 / 对照配置 | 每版统计样本 | 训推 diff | 平均生成 tokens | raw_reward | 预热后平均整轮耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen R3：旧共同 Triton / 新框架默认路径 | 1280 | 0.008791 → 0.008788 | 1563.4 → 1562.4 | 0.3266 → 0.3141 | 568.6 → 543.1 秒 |
| Kimi 两层 INT4 | 1280 | 0.014218 → 0.013298 | 4034.2 → 4035.0 | 0 → 0 | 263.6 → 188.5 秒 |
| Mini SWE：共同 Triton 路径 | 80 | 0.017369 → 0.017029 | 8038.2 → 8553.1 | 29/80 → 27/80 | 436.9 → 374.6 秒 |
| Mini SWE：旧共同路径 / 新默认路径 | 80 | 0.017369 → 0.017636 | 8038.2 → 9080.0 | 29/80 → 23/80 | 436.9 → 448.9 秒 |
| Vision OPD：seed 43，26 轮中后 16 轮 | 512 | 0.012180 → 0.012295 | 751.8 → 767.1 | 0.4941 → 0.4922 | 69.70 → 64.57 秒 |

Vision 补充实验每版实际完成 26 轮、832 条样本和 27 次权重同步，输入身份及分组逐条对应，作业自然结束。预热后平均输出长度增加 2.04%，截断比例为 55.66% → 56.25%，训推 diff 增加 0.94%；后 8 轮 diff 为 0.011599 → 0.011613，未出现持续扩大。后 8 轮长度增加 6.40%，因此不能由总体均值推断没有采样波动。此前 seed 42 的 14 轮实验在预热后仅有每版 128 条样本，长度增加 13.55% 的结果仍保留；补充实验未维持这一幅度的总体增长。Vision 的 random reward 不代表任务准确率。

Kimi 主实验的两层模型 reward 和梯度全零，不能提供模型质量结论；另行完成的两轮非零梯度诊断已核验实际量化权重内容变化、原生绑定及更新后继续生成。Mini 两种配置均覆盖 PD、CPU HiCache 命中与回读、真实工具执行及样本导出。Mini 默认路径少 6 个成功 session，且输出更长；10 个任务、每任务 8 个 session 的短实验不足以证明成功率等价。一次约 300 秒的 PD WaitingForInput 超时返回 HTTP 500 后，同一请求重试成功，根因尚未确定；后续 2708 次物理派发未复现错误，不能据此称该问题已修复。

任务整轮性能总体可接受，但独立固定工作量微基准仍有退化：例如 9B、4096 输入、并发 8 的吞吐比为 0.8805，约下降 12%；其他阶段和延迟触发同样保留。新默认 Mini 的预热后整轮时间增加 2.76%，同期生成长度增加 15.81%，不能把两者直接等同于内核速度变化。

实际安装环境的迁移与 GC 回归为 99/99，通过且无跳过；旧、新环境的 CLI/默认 CUDA IPC/HiCache 参数回归各 27/27，通过且未初始化 CUDA。补丁构建、权重生命周期、通信及此前专项检查的原始证据均保留。完整 Docker 镜像尚未构建，因为验证机没有 Docker、Podman 或 buildctl；其它 GPU 架构、多节点和长期训练未覆盖。交付报告另记录最终 pre-commit、补丁重建与源码指纹，不能用隔离环境替代完整镜像验证。

## 原补丁逐文件核对

核对对象是原提交中的真实补丁，不是中间的归一化副本。主补丁共 63 个文件、161 个 hunk；逐文件去留如下（路径省略 `python/sglang/`）。保留项已对照旧新增/删除行和新版调用链，不能仅凭 `git apply` 成功判定语义保留。

| 原 Relax 文件 | 原 hunk 数 | 0.5.21 落点 | 处理依据 |
| --- | ---: | --- | --- |
| `kernels/jit/csrc/deepseek_v4/c_plan.cuh` | 1 | 官方原生 / 共享 hook | 官方已吸收原 int64 索引或 warp reduction 修复，不重复覆盖 |
| `multimodal_gen/configs/pipeline_configs/qwen_image.py` | 6 | `multimodal_gen/configs/pipeline_configs/qwen_image.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `multimodal_gen/configs/post_training/rl_rollout.py` | 1 | `multimodal_gen/configs/post_training/rl_rollout.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `multimodal_gen/configs/sample/sampling_params.py` | 1 | `multimodal_gen/configs/sample/sampling_params.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `multimodal_gen/runtime/entrypoints/post_training/io_struct.py` | 1 | `multimodal_gen/runtime/entrypoints/post_training/io_struct.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `multimodal_gen/runtime/entrypoints/post_training/weights_api.py` | 2 | `multimodal_gen/runtime/entrypoints/post_training/weights_api.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `multimodal_gen/runtime/entrypoints/utils.py` | 1 | `multimodal_gen/runtime/entrypoints/utils.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `multimodal_gen/runtime/managers/scheduler.py` | 2 | `multimodal_gen/runtime/managers/scheduler.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `multimodal_gen/runtime/models/schedulers/scheduling_flow_match_euler_discrete.py` | 1 | `multimodal_gen/runtime/models/schedulers/scheduling_flow_match_euler_discrete.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `multimodal_gen/runtime/pipelines_core/lora_pipeline.py` | 2 | `multimodal_gen/runtime/pipelines_core/lora/pipeline.py` | 迁到新目录；保留 tensor-LoRA 入口，遵循新的 offload/conditioning 契约 |
| `multimodal_gen/runtime/pipelines_core/stages/denoising.py` | 3 | `multimodal_gen/runtime/pipelines_core/stages/denoising.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `multimodal_gen/runtime/pipelines_core/stages/latent_preparation.py` | 5 | `multimodal_gen/runtime/pipelines_core/stages/latent_preparation.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `multimodal_gen/runtime/post_training/gpu_worker_post_training_mixin.py` | 4 | `multimodal_gen/runtime/post_training/gpu_worker_post_training_mixin.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `multimodal_gen/runtime/post_training/scheduler_post_training_mixin.py` | 1 | `multimodal_gen/runtime/post_training/scheduler_post_training_mixin.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `multimodal_gen/runtime/post_training/scheduler_rl_mixin.py` | 6 | `multimodal_gen/runtime/post_training/scheduler_rl_mixin.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `multimodal_gen/runtime/post_training/weights_updater.py` | 1 | `multimodal_gen/runtime/post_training/weights_updater.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `srt/disaggregation/decode.py` | 8 | `srt/disaggregation/decode.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `srt/disaggregation/mooncake/conn.py` | 2 | `srt/disaggregation/mooncake/conn.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `srt/disaggregation/prefill.py` | 2 | 官方原生 / 共享 hook | 原两处暂停轮询并入 scheduler 的共享 paused hook，不添加 Slime 超时策略 |
| `srt/entrypoints/engine.py` | 4 | `srt/entrypoints/engine.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `srt/entrypoints/http_server.py` | 6 | `srt/entrypoints/http_server.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `srt/layers/attention/dsa/dsa_indexer.py` | 8 | `srt/layers/attention/dsa/dsa_indexer.py` | 保留 Relax 环境覆盖、query repeat、head-gate 对齐；适配新版调用点 |
| `srt/layers/attention/dsa/dsa_prefill_cuda_graph.py` | 2 | `srt/layers/attention/dsa/dsa_prefill_cuda_graph.py` | 使用 Relax 公共 helper；fake shape 随 repeated query heads 调整 |
| `srt/layers/attention/dsa/utils.py` | 1 | `srt/layers/attention/dsa/utils.py` | 保留 Relax 公共 head-gate repeat helper 与原排列 |
| `srt/layers/quantization/compressed_tensors/compressed_tensors.py` | 2 | `srt/layers/quantization/compressed_tensors/compressed_tensors.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `srt/lora/layers.py` | 1 | `srt/lora/layers.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `srt/managers/io_struct.py` | 3 | `srt/managers/io_struct.py` | 恢复旧接口类型和旧 draft flag；selector 省略值区别于显式新 API |
| `srt/managers/schedule_batch.py` | 2 | `srt/managers/schedule_batch.py` | 迁移 deferred backup 标记与 retract 清理；per-position 在独立补丁 |
| `srt/managers/scheduler.py` | 3 | `srt/managers/scheduler.py` | 旧 LoRA/postprocess 与暂停 DMA 轮询；保留新版 scheduler 结构 |
| `srt/managers/scheduler_components/profiler_manager.py` | 1 | `srt/managers/scheduler_components/profiler_manager.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `srt/managers/scheduler_components/weight_updater.py` | 2 | `srt/managers/scheduler_components/weight_updater.py` | 旧无会话 bucket 更新与路由，保留新版显式会话及 BF16 后处理 |
| `srt/managers/tokenizer_control_mixin.py` | 5 | `srt/managers/tokenizer_control_mixin.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `srt/managers/tokenizer_manager.py` | 2 | `srt/managers/tokenizer_manager.py` | 旧单节点 MM 传输由官方 transport 处理；其余配合 OPD 副补丁 |
| `srt/managers/tp_worker.py` | 3 | `srt/managers/tp_worker.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `srt/mem_cache/hiradix_cache.py` | 2 | 官方原生 / 共享 hook | 旧类已删除；新 core 的 parent 叶集合更新覆盖原修复 |
| `srt/mem_cache/memory_pool.py` | 1 | `srt/mem_cache/memory_pool.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `srt/mem_cache/radix_cache.py` | 2 | `srt/mem_cache/radix_cache.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `srt/model_executor/model_runner.py` | 5 | `srt/model_executor/model_runner.py` | 原分桶 LoRA、draft offload guard、graph token 数及 postprocess；适配新模块路径 |
| `srt/models/qwen3_5.py` | 1 | `srt/models/qwen3_5.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `srt/models/deepseek_v4.py` | 9 | `srt/models/deepseek_v4.py` | 部分官方吸收；补回 sink 原地刷新和分桶日志，保留新版 kernel/view 结构 |
| `srt/models/deepseek_v4_nextn.py` | 2 | 官方原生 / 共享 hook | 官方已有 input_ids clone 与 replicated gather |
| `srt/models/qwen3_vl.py` | 1 | 官方原生 / 共享 hook | 官方 get_deepstack_embeds 已含安全切片，保留其原生策略开关 |
| `srt/multimodal/processors/glm4v.py` | 3 | `srt/multimodal/processors/glm4v.py` | 保留 image EPD/mrope；兼容 receiver 的额外 None 元数据（原基线已有签名缺陷） |
| `srt/multimodal/processors/base_processor.py` | 1 | `srt/multimodal/processors/base_processor.py` | 保留 raw-image placeholder 折叠和 precomputed 布局保护 |
| `srt/multimodal/processors/executor.py` | 1 | `srt/multimodal/processors/executor.py` | 保留 StopIteration 转 ValueError，避免 async 调用挂起 |
| `srt/observability/req_time_stats.py` | 4 | `srt/observability/req_time_stats.py` | 原 PD timing 和正时间转换；删除额外 retry 计数及重复键 |
| `srt/speculative/eagle_draft_cuda_graph_runner.py` | 1 | `srt/speculative/eagle_draft_cuda_graph_runner.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `srt/utils/common.py` | 1 | 官方原生 / 共享 hook | 退役旧宽前缀许可；真实发送方仅传官方 Tensor/metadata，沿用精确 globals 并验证 roundtrip |
| `kernels/ops/attention/dsa/dequant_k_cache.py` | 2 | 官方原生 / 共享 hook | 官方已吸收原 int64 索引或 warp reduction 修复，不重复覆盖 |
| `kernels/ops/attention/dsa/quant_k_cache.py` | 1 | 官方原生 / 共享 hook | 官方已吸收原 int64 索引或 warp reduction 修复，不重复覆盖 |
| `kernels/ops/attention/dsv4/dequant_k_cache.py` | 1 | 官方原生 / 共享 hook | 官方已吸收原 int64 索引或 warp reduction 修复，不重复覆盖 |
| `kernels/ops/attention/dsv4/quant_k_cache.py` | 1 | 官方原生 / 共享 hook | 官方已吸收原 int64 索引或 warp reduction 修复，不重复覆盖 |
| `srt/layers/quantization/mxfp4.py` | 1 | `srt/layers/quantization/mxfp4.py` | 保留 serialized/runtime 存储生命周期；不重复补官方 Hopper padding |
| `srt/state_capturer/base.py` | 5 | 官方原生 / 共享 hook | 官方已含 pinned-host 分配、销毁和 overlap 快照 |
| `srt/state_capturer/indexer_topk.py` | 1 | 官方原生 / 共享 hook | 官方已有销毁接口，保留原生实现 |
| `srt/state_capturer/routed_experts.py` | 1 | `srt/state_capturer/routed_experts.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `srt/configs/model_config.py` | 1 | `srt/configs/model_config.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |
| `srt/models/kimi_k3.py` | 11 | `srt/models/kimi_k3.py` | 保留缓存、权重原地更新和 kernel 预编译，适配新版 accessor |
| `srt/mem_cache/unified_cache/unified_tree_core.py` | 1 | `srt/mem_cache/unified_cache/unified_tree_core.py` | 保留受保护 Mamba checkpoint 的延迟 hit/backup guard |
| `srt/mem_cache/unified_cache/unified_tree_core_interface.py` | 1 | `srt/mem_cache/unified_cache/unified_tree_core_interface.py` | 保留可选 completion hook，默认无操作 |
| `srt/mem_cache/unified_radix_cache.py` | 3 | `srt/mem_cache/unified_radix_cache.py` | 旧 chunked completion 迁入 insert_req/on_release，更新 req.kv 字段和锁时机 |
| `srt/layers/dp_attention.py` | 1 | `srt/layers/dp_attention.py` | 保留 BF16 的 FP32 reduce-scatter 累积与原 dtype 恢复 |
| `srt/distributed/parallel_state.py` | 1 | `srt/distributed/parallel_state.py` | 保留原 Relax hunks，按官方 0.5.21 的接口、字段及原生控制流适配 |

原 `sglang_per_pos_topk.patch` 的三个文件（logprob processor、scheduler batch、tokenizer manager）独立迁移；保持 per-position 查询、跨 chunk 行切片及 base64 传输，适配官方 pinned D2H 与 log-normalizer。

额外撤回参考补丁中的 GetLoads inflight RPC 整组改动、未使用的 dump metadata 输入类型、GLM4V MoE encoder/language-only 分支、Qwen legacy-loader 替换、prefill 超时与 retry 指标、decode 全量回撤策略、running prefill 强制 abort、FastAPI path fallback，以及旧官方不存在的 FP8/FP4 runner aliases。其余 graph/NSA/mamba CLI aliases 在 0.5.17 官方已有，保留是兼容旧 Relax 启动参数。

新增 BF16/MXINT4 权重生命周期和 FlashInfer 融合路由捕获，是指定两项 E2E 实际失败后所需的版本适配，不来自 Slime。HiCache、DeepSeek sink、旧路由及 diffusion LoRA 的 CPU 行为回归单独保存；三项指定模型的短程 GPU 结果不能替代所有其它模型、PD、HiCache 或 diffusion 的 GPU 验证。

旧 SafeUnpickler 的 `slime.`/`Relax.` 前缀不是“被官方吸收”。核对真实 Megatron/FSDP 发送方后，传输对象仅为普通容器、Tensor 和 SGLang 原生 FlattenedTensorMetadata，均已有精确许可；因此退役未使用的宽前缀。CPU 回归覆盖混合 dtype bucket、空 bucket、LoRA CPU dict 和 FSDP named tensors，检查实际 globals 及重建数据。
