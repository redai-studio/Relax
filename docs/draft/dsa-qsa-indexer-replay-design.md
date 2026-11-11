# DSA/QSA indexer replay 调研与性能设计

本文记录 PR #486 后续调研、方案比较与性能设计，作为内部设计稿，不加入用户指南导航。调研日期为 2026-11-11；外部项目状态是该日期的快照。

代码基线为内部 `github/replay-manager` 分支的 `0336755c0ca2788a767c23a5fa5d473895eaa945`。该提交合并了 PR #486 与当时的 `github/main`，并修复 reference 生命周期和 SGLang 接口适配；原 GitHub PR 分支未修改。本文新增的性能优化仍是设计，尚未实施。

MoE 数据通路的后续实施和单机对照结果见 [Qwen3.5-35B-A3B reference R3 实验记录](./experiments/2026-11-11-qwen35-moe-r3/README.md)。该实验仅覆盖非 DSA 的预取与 TQ 热路径优化，不代表本文 DSA/QSA 方案已完成。

## 1. 目标与范围

目标是让 Relax 的 DSA/QSA rollout-to-train replay 具有很低的端到端开销：中间环节只处理元数据和布局，模型消费点直接使用录制选择，尽可能跳过原 indexer 的打分、top-k 以及仅为选路服务的计算和通信。

本轮明确的约束：

- 重点研究 reference delivery，暂不优化 full delivery。
- 保留 Relax 的 Controller、Service、训练循环、TQ、样本调度与 packing 框架；改动集中于 replay manager 的数据通路和模型 adapter。
- 先落实现有 DSV4 C4 路径，再补 QSA 捕获与消费。不能把新设计描述为已支持能力。
- 冻结 indexer 且辅助 loss 为零时，争取跳过整个 indexer 支路；需要辅助 loss 时保留必要的可微计算。
- 性能必须按完整 step 比较，并检查主干梯度正确性。文章中的低开销结果是对标材料，不是 Relax 的保证。

## 2. Replay 的语义

Router replay 记录每个 token 在 MoE 层选择的 expert；indexer replay 记录稀疏 attention 选择的位置或 block。训练与推理的 kernel、精度、batching、packing 和并行布局可能不同，微小数值误差可能改变离散 top-k，从而改变 attention 图、logprob 和梯度。

Replay 固定的是离散选择，不能自动消除其他算子的数值差异。参数更新后继续使用 rollout 的稀疏图，也不等价于当前模型自然重新选择；这一训练约束需要通过稳定性和收益评估。

需要区分以下概念：

| 概念 | 录制与复用的对象 |
| --- | --- |
| MoE R3 | Rollout 的 expert IDs |
| DSA/QSA indexer replay | Rollout 的 attention 位置或原生 block IDs |
| QSA checkpoint replay | 训练前向生成的 selection、mask 等，供 activation recompute 复用 |
| CUDA graph replay | 捕获的 GPU 执行图 |
| 跨层 index sharing | 不同 attention 层复用选择 |

QSA 现有 checkpoint tape 不能作为已经支持 rollout replay 的证据。

## 3. 外部框架调研

### 3.1 Miles：已接通的 rollout replay

源码快照：`radixark/miles@cefb2a19cb7e13d7df503e5b697caf234ef4c4c7`。

`IndexerReplayManager` 继承 `BaseReplayManager`，设置 `rollout_indexer_topk` 数据列、索引语义和检查阈值。完整链路还包括 SGLang 捕获、训练数据整理、actor 阶段切换与模型 hook，不能仅用这个类的行数比较框架复杂度。

链路为：SGLang 捕获 → rollout 数据列 → 按训练 microbatch 整理 → 每层 CPU replay 队列 → forward/recompute 消费。

值得借鉴的设计：

- 区分训练侧 record/replay 与跨引擎 rollout replay。
- 使用 module 注册和 top-k 函数包装，模型接入相对局部。
- 通过 `stream_idx` 表达 capture stream 与实际模型层的映射。
- 使用独立 forward/backward 游标，old-policy logprob forward 消费后重置 forward 游标。
- 可在 CI 重算 top-k，统计与录制集合的 overlap。Indexer 的默认行 overlap 阈值是 0.8，再限制不达标 token 的比例；这是诊断标准，不代表逐项严格一致。

当前限制与成本：

- Rollout indexer replay 的入口仍要求 CP=1。
- 普通 GLM-5 路径会将 sample-relative KV 索引转换成 packed 坐标；该规则不能直接套到 DSV4 压缩索引。
- Replay 阶段可以跳过被包装的原 top-k，但 GLM-5 路径仍计算 indexer logits；开启 replay 后将 query 分块大小改成整个 query 长度，可能增加峰值内存。
- 已有 OutputStore reference 输出，但所查主线仍会物化、拼接对应字段，再填 replay 队列。
- 按本 rank 行读取的 PR #4573，以及按本 PP stage 层读取的 PR #4643，调研时均未合入。前者的 indexer 路径仍要求 CP=1。
- 删除 indexer replay、冻结 indexer 的 PR #3879 调研时是未合入的 draft，不能将其中主张当作主线行为或已验证替代。

参考：[Replay manager](https://github.com/radixark/miles/blob/cefb2a19cb7e13d7df503e5b697caf234ef4c4c7/miles/utils/replay_base.py)、[训练数据整理](https://github.com/radixark/miles/blob/cefb2a19cb7e13d7df503e5b697caf234ef4c4c7/miles/backends/training_utils/replay/data.py)、[GLM-5 hook](https://github.com/radixark/miles/blob/cefb2a19cb7e13d7df503e5b697caf234ef4c4c7/miles_plugins/models/glm5/glm5.py)、[OutputStore](https://github.com/radixark/miles/blob/cefb2a19cb7e13d7df503e5b697caf234ef4c4c7/miles/utils/output_store/plan.py)、[PR #4573](https://github.com/radixark/miles/pull/4573)、[PR #4643](https://github.com/radixark/miles/pull/4643)、[PR #3879](https://github.com/radixark/miles/pull/3879)。

### 3.2 Slime：算子与数值路径对齐

GLM-5 技术报告 §3.2 讨论了 k=2048 的 replay 存储与通信成本，采用更确定的 top-k，并默认冻结 indexer。论文将训练稳定性改善归因于这一选择，但不能把其经验外推为任意形状与设备上的严格一致性保证。

Slime 已合入的 PR #2262 进一步对齐 FP8 GEMM、indexer logits、量化、RoPE、top-k 顺序、sparse attention 和 MoE 通信等路径。

两边并非全部使用 BF16：

| 环节 | Megatron 训练侧 | SGLang rollout 侧 |
| --- | --- | --- |
| 被对齐的 Linear/MoE GEMM | SGLang-compatible block-FP8 forward | Block-FP8 forward |
| Indexer 打分 | 使用 SGLang FP8 indexer kernel | FP8 indexer |
| 对应 GEMM backward | BF16 dgrad/wgrad | 无 backward |
| 主 attention KV | BF16，或通过 FP8 KV QAT 模拟量化 | 对应配置为 BF16 或 FP8 KV cache |

对齐环境启用 `MEGATRON_USE_SGLANG_FP8_INDEXER=1`；训练侧量化 indexer Q/K，并调用 `deep_gemm.fp8_mqa_logits()`。`kv_fp8_qat=False` 表示主 attention 使用 BF16 KV，并不表示 indexer 打分变成 BF16。

作者报告了约 `1.89e-7` 的 logprob MAE，以及部分 decoder 层 hidden-state 零差异；这是外部验证材料，本轮未复跑。仅冻结参数或替换 `torch.topk` 不足以复现整个对齐方案；PyTorch 官方也不保证相同分数元素的 top-k 索引稳定。

参考：[GLM-5 技术报告](https://arxiv.org/html/2602.15763v2#S3.SS2)、[Slime PR #2262](https://github.com/THUDM/slime/pull/2262)、[对齐配置](https://github.com/THUDM/slime/blob/79aa41a884ab0bf65222c94418f6357d3c31192b/slime/backends/megatron_utils/alignment/env.py)、[Indexer 实现](https://github.com/THUDM/slime/blob/79aa41a884ab0bf65222c94418f6357d3c31192b/slime_plugins/models/glm5/ops/indexer.py)、[Linear forward/backward](https://github.com/THUDM/slime/blob/79aa41a884ab0bf65222c94418f6357d3c31192b/slime/backends/megatron_utils/alignment/deepgemm_forward.py)、[torch.topk 契约](https://docs.pytorch.org/docs/2.15/generated/torch.topk.html)。

### 3.3 Megatron、verl 与推理引擎

| 项目 | 调研结论 | 状态和限制 |
| --- | --- | --- |
| Megatron-Core | Issue #5384 / PR #5386 提供 DSA indexer replay 原型 | PR 已关闭且未合入；当前主线不能视为已有该能力 |
| verl | 有 DSA 配方和 MoE R3 | 所查主线未找到 indexer top-k 的 rollout→batch→trainer 完整接线 |
| SGLang | 有 `IndexerTopkCapturer` 和 top-k 输出能力 | 需要逐模型确认捕获位置、逻辑坐标及层映射；不能由公共接口存在推断 QSA 已接通 |
| vLLM | PR #49555 提议通过 AuxOutput SHM 导出原始 int32 top-k | 调研时未合入，主要覆盖推理输出侧，尚非完整 RL 闭环 |
| RTP-LLM | PR #1615 提供 GLM indexer tensor capture/replay | 调研时未合入；主要是落盘后离线重现计算的开发工具 |

Megatron 原型有独立 record/forward/backward replay，并能向 fused indexer loss 传入录制索引。但其推理路径拒绝 packed sequence，且 `no_grad` 分支没有完整 replay 消费，不能直接保障 old-logprob 重算使用 rollout 同一选择。

verl 核对版本为 `63913d1d5fef4d028658c330f397c0bd2a909213`，MCore 主线为 `dd4c8e20dcc0bf299941af279d2f3b6dc5d55d07`。结论仅针对公开版本，不推断私有分支。MCore 名为 `test_indexer_replay.py` 的 CP 测试验证的是 RoPE 重复执行确定性，不能据文件名认定其支持 RL replay。

参考：[MCore issue #5384](https://github.com/NVIDIA/Megatron-LM/issues/5384)、[MCore PR #5386](https://github.com/NVIDIA/Megatron-LM/pull/5386)、[verl GLM-5.2 配方](https://github.com/verl-project/verl/blob/63913d1d5fef4d028658c330f397c0bd2a909213/examples/ascend_extras/grpo_trainer/run_glm5_2_megatron.sh)、[vLLM PR #49555](https://github.com/vllm-project/vllm/pull/49555)、[RTP-LLM PR #1615](https://github.com/alibaba/rtp-llm/pull/1615)。

### 3.4 Relax TQ host-RDMA：PR #256 已合入并包含在当前分支

[PR #256](https://github.com/redai-studio/Relax/pull/256)（`feat/enable-tq-rdma`）于 2026-10-21 合入，合并提交为 `50da7ad05`。已通过 Git 祖先检查确认当前 `github/replay-manager` 包含该提交，无须再次移植其 Controller、worker attach 或资源清理逻辑。

该 PR 接入 TransferQueue 的 MooncakeStore/host-RDMA，默认 `--tq-rdma-mode=off`；`auto` 在满足回退条件并完成清理后使用 SimpleStorage，`required` 要求 RDMA 可用。它复用外部管理的 Mooncake master，并处理有界初始化、全节点 attach、统一配置、correctness contract 与 owner/worker 清理。当前固定 `use_gdr=false`，不提供 GPU-to-GPU 直传。

Replay 的 `register_local_owner()` 和 `tq_client_factory()` 已传递并使用所选 TQ backend，因此 reference body 可沿用 MooncakeStore 配置。现有本地 replay smoke 验证的是 SimpleStorage；尚未独立验证 replay 在 RDMA 上的完整发布、读取与回收，也没有该路径的性能数据。

配套 [PR #278](https://github.com/redai-studio/Relax/pull/278) 是旧版 Mooncake loss-guard 运行时补丁，调研时已关闭且未合入。当前方案要求 pinned TransferQueue 自带 correctness contract，不应重新引入该旧补丁。

操作与验证参考 [TransferQueue host-RDMA 文档](./transfer_queue_rdma.md) 和 [跨节点 benchmark](../../scripts/benchmarks/tq_cross_node_bench.py)。其 C0/C1/C2 分别为 SimpleStorage、仅作 benchmark 对照的 Mooncake/TCP、Mooncake/host-RDMA。需要区分 attach 成功与真实线路证明：正式 RDMA 验收应同时提供 byte-exact、指定 HCA/port 的 wire proof、版本信息与资源清理结果。本轮仅核对代码和 PR 材料，未复跑真机 RDMA benchmark。

## 4. 当前 Relax 与 Miles 的差异

| 维度 | Relax 当前 reference 路径 | Miles 所查实现 |
| --- | --- | --- |
| 数据准备 | 绑定 lazy reader，模型消费时读取 | 整理 tensor，填入各层 CPU replay 队列 |
| 持有内容 | Manifest、布局、reader lease 和有预算的 CPU cache | 各层 microbatch replay tensor，直到 clear |
| 读取粒度 | 本地层、所需 query 行对应的 chunk | 当前已物化字段；更细粒度读取在推进中 |
| Indexer 模型接入 | 专用 DSV4 C4 hook | Module 注册与 top-k 包装，模型范围更广 |
| 并行边界 | 当前 DSV4 限定静态连续 allgather CP>1、THD、TP1、无 SP | Rollout indexer replay 限定 CP1 |
| 消费检查 | Forward reset 和 step 结束检查完整消费 | 基础队列主要检查越界，另有可选 overlap 诊断 |
| 生命周期 | Run、publisher、partition、reader lease 与显式回收 | 训练队列管理更简单；存储生命周期另由数据通路负责 |
| 原 top-k | 已计算后再替换 | 包装点可跳过 top-k；部分模型仍计算 logits |

Relax 当前入口还要求 colocated Megatron、零 indexer teacher-loss coefficient，拒绝 fully-async/hybrid、dynamic CP、MTP/speculative、agentic/custom rollout 等未接通配置。GLM-5 DSA 和 QSA 的 rollout replay 尚未接入，不能将当前 DSV4 支持范围泛化为全部 DSA/QSA。

主要代码：[模型布局与消费](../../relax/distributed/replay_manager/backend/megatron.py)、[CPU cache 与 lazy reader](../../relax/distributed/replay_manager/consumer.py)、[存储读取](../../relax/distributed/replay_manager/storage.py)、[配置](../../relax/distributed/replay_manager/config.py)、[入口限制](../../relax/utils/arguments.py)。

维护上，Miles 的训练侧更局部、更容易接模型；Relax 承担分布式 reference 数据服务，发布、租约、预取和 scheduler 适配的维护成本更高。当前也仍有进程级 `IndexerReplay` 状态，不能宣称完全消除了隐式生命周期。

性能只能作有条件推断：长序列、大 payload、有 PP/CP 局部消费时，按需读取有结构优势；工作集较小且内存充足时，预先准备的 CPU 队列可能更快。尚无同模型、同并行配置的 GPU A/B 结果。

## 5. 文章的数据流原则

来源为用户提供的[知乎文章](https://zhuanlan.zhihu.com/p/2086215865142400545)全文。网页访问被安全验证拦截；此处根据会话中提供的正文归纳，不复制全文。正文中的 DeepSeek-V4 报告引文是文章引用，本轮没有独立核验该引文。

文章报告的旧 POC 增加约 8 倍 step 耗时、优化后不足 7% 额外开销，以及 3.7 GB payload、每 token 96 KiB→24 B，均是作者特定实现和录制口径的结果，不能直接套用到 Relax。GB→KB 指控制链路携带的描述信息，payload 本体并未消失。

核心原则：

1. 推理侧暂存可变尾窗，仅将已接受、稳定的行按 layer/chunk 写入共享存储，后台写有字节背压。
2. 仅指定 owner 发布对象；响应、样本归并、调度、packing 携带轻量 manifest。
3. 多轮覆盖修改区间归属，避免读回 payload 拼大数组；回收必须包含被覆盖但仍已分配的对象。
4. 布局阶段将 manifest 展开为六个 int32 的 token handle，复用既有行变换，不让大索引参与无关通信。
5. 布局完成后，仅物化当前消费者需要的层和行，消费结束后释放。

当前 [TokenHandle](../../relax/distributed/replay_manager/ref_data.py) 已使用 UUID 四个 int32、行号、tag，共 24 B，并区分 RECORD/PAD/GAP；当前 GAP 会拒绝消费，不能宣称多轮插入与缺口语义已经实现。Handle 经过布局必须位保持，不能让任意数值归约或 dtype 转换破坏 UUID。

当前 reference 骨架已满足 write-once、按层分 chunk、轻量 manifest、lazy reader、唯一生命周期账本和后台写背压。独立有界 speculative tail buffer、多轮区间合并以及 QSA rollout 捕获仍需各自适配；不能因此放开现有入口限制。

“物化一次”应理解为 payload 不陪中间环节反复变换，按消费生命周期管理；它不自动保证多次 forward/recompute 只发生一次 GET 或 H2D，更不要求全 batch 常驻 GPU。

## 6. 高性能目标方案：保留框架，提前消费

```text
Payload: capture -> stable layer/chunk -> TQ body -> bounded prefetch -> model
Metadata: response -> manifest -> scheduling/packing -> token handles -> row plan
```

### 6.1 直接消费接口

在现有 manager 内增加内部入口，直接返回录制索引和有效行信息。保留现有覆盖接口兼容旧 patch；不为单个模型再建一套存储、注册表或训练状态机。

模型在 indexer 计算前作出 replay 决策：未启用时走原实现；启用时使用录制选择，有效训练行缺失即报错。Padding 由 adapter 构造合法表示，无须为 padding 计算整条 indexer。

现 `IndexerReplay.replay(computed_topk=None)` 会消费记录后返回 None，不能简单提前 hook 并传 None。生产 reference 只在写入完成后发布；对已 sealed 记录的读取失败需明确失败，不静默忽略、不回退原 top-k。无需直接照搬文章中的无限等待策略。

### 6.2 DSV4 C4：跳过专用 indexer 支路

当前 [Megatron patch](../../docker/patch/megatron/20260915-4a4de8657.patch) 在 `compute_cp_indexer_topk()` 后才替换结果。此前 indexer compressor、K all-gather、Q/W 投影、RoPE/Hadamard、打分与 top-k 已完成。

在冻结、无辅助 loss 的路径中，应在专用 indexer 支路之前取得录制的 document-relative compressed IDs，跳过上述工作。必须保留：

- 主 attention 的 Q/K/V、压缩 KV compressor 与 K/V 通信。
- Boundary/window KV、压缩布局，以及逻辑 compressed ID 到物理 KV 地址的转换。
- 主 attention backward 和远端 K/V 梯度归约。

原 CSA 的 `use_indexer_loss` 在所查路径中按 `training_with_grad and compressed_topk is not None` 判断，不以 coefficient=0 排除。提前注入 replay 后必须显式选择无辅助 loss 路径，否则可能引用未生成的 indexer 激活或 reduce-scatter edge，甚至错误 detach 主 attention KV 梯度。

跳过 indexer 专用 collective 必须在整个相关进程组中作一致决策。不能因单个 rank 的有效 query 行为空而改变 collective 顺序。

### 6.3 QSA：捕获原生 block ID，直接构造 selection

SGLang QSA 先选择 complete block，再展开为 token 位置；原生 Megatron `QSASelection.selected_ids` 使用 document-relative complete block IDs。可在 prefill/decode 的 block top-k 之后、token expansion 之前捕获，训练侧直接构造原生 selection。

这是选择的原生无损表示。例如 compress ratio=4、token budget=2048 时，512 个 int32 block ID 可表达被选中的 complete blocks；causal tail 由既有 attention 规则单独补齐，无须再次录制。不得保存 paged-KV 物理地址，也不得直接采用另一模型的 packed offset 规则。

最小训练接入口为 [qsa_dispatch.indexer_forward](../../relax/models/qwen3_8/qsa_dispatch.py)。在冻结、无辅助 loss 时，可跳过 indexer 专用 hidden SP gather、CP reconstruction、Q/K projection、norm/RoPE、pool、score/top-k 和 selector 专用通信，继续运行原 attention。

还需把 rollout replay 纳入 [build-time wrapper](../../docker/patch/megatron/20261020-model-operators.patch) 的入口条件；当前 wrapper 仅在已有 QSA 优化开关打开时进入 dispatch。保留 mixed backend 的 document bounds 等元数据，确保 checkpoint 的 dataclass 复制不会丢失字段。

QSA CP 布局需要单独适配：当前部分 backend 的 selection 按 full gathered sequence 索引；DSV4 reader 则按 CP-local query 行读取。CP1 可先验证捕获与消费闭环，CP>1 的高性能验收必须让 mask/plan 与本地 query 行对应，避免每个 CP rank 全量物化。沿用既有 token layout 与 CPU physical-boundary metadata，不再写一套独立并行布局算法。

现有 [qsa_cp.py](../../relax/models/qwen3_8/qsa_cp.py) 是 invocation-local checkpoint tape。接入时必须明确 selection 的唯一所有者，避免重复消费 rollout 游标或同时保留两份大 GPU selection。

### 6.4 辅助 loss 的边界

有 indexer 辅助 loss 时，仅保存 metadata 的 selection 不足以计算梯度。DSV4 仍可能需要 indexer Q/K/W、压缩 K 与相关通信；QSA Stage-2 需要可微 `index_query`、`compressed_key` 等张量。可复用离散选择、跳过重新 top-k；是否还能省去全量 score 取决于具体 loss，不在首轮承诺范围。

Manifest/adapter 必须明确模型类型、ratio、top-k、capture layer 映射、索引坐标、无效槽与 padding 语义，避免只根据 tensor shape 选择解释方式。

## 7. 消费链路的具体热点

### 7.1 大 payload 下预取容易失效

默认 `materialization_bytes=256 MiB`、`publication_bytes=64 MiB`。当前预取先为读取 scratch 预留 `2 × publication_bytes=128 MiB`，再为候选层预留两倍输出空间，当前层也已进入 cache。按该公式粗估，下一层大小 S 需要满足 `3S + 128 MiB <= 256 MiB`，即 S 不超过约 42.7 MiB。

例如 `20480 rows × 2048 topk × 4 B = 160 MiB/layer`；topk=1024 也有 80 MiB/layer。这些形状无法启动下一层预取，模型消费点会同步等待 authorization、TQ metadata 与 GET。以上为源码预算推导，非 GPU 实测。

优先按实际 read plan 计算 scratch，区分 indexer 与 MoE 的临时空间；采用有界的小窗口，绑定后提前启动首层读取，避免首个结果被整个多层 window 的完成时间拖住。预算仍需计入在途输出，不能通过漏算内存“修复”预取。

### 7.2 减少重复打包、CPU 临时数组与控制面 RPC

当前 TQ GET 后很快拆回 blocks；可调查复用现有低层批量读取，减少中间 batched container 拼装，再直接填入 pinned staging。改动尽量限制在 replay client，不修改 TQ 公共协议。

`output[destinations] = block[rows]` 会产生高级索引临时数组；连续 source/output spans 可走 slice copy，离散布局保留 gather。在有效 lease 的 session 内复用已解析的不可变 block metadata 与授权结果，减少重复控制面往返，同时保留并发 release、retire 和 restart 的所有权保证。

### 7.3 H2D 与重算生命周期

当前 CPU cache 命中后仍会 H2D，拷贝在消费线程当前 stream 上排队。`non_blocking=True` 不等于与模型计算重叠。

可评估有界 copy stream/event 预取窗口，提前搬下一次消费所需数据。若使用可复用 GPU buffer，必须等 attention backward 不再引用该索引后才能覆盖；不能仅等 forward 结束。QSA checkpoint tape 和 full recompute 需使用各自正确的消费顺序。

优先减少远端重读与关键路径等待，不要求跨 old-logprob、训练和 backward 永久缓存整批 GPU 数据。CPU cache 预算也不等于进程总内存上限，TQ 内部和 pinned allocator pool 需另行测量。

### 7.4 Chunk 与 rank 复制分别测量

默认 1024 行、topk=2048 的单层 chunk 为 8 MiB。当前读取整个命中 chunk，再裁出有效行；长连续 CP 切片的浪费通常集中在边界。缩小 chunk 会增加对象和 RPC 数量，不能只追求最小过读字节。

当前支持的 DSV4 路径为 TP1。未来 QSA/TP 扩展中，indexer selection 未必按 TP 切 query 行，可能出现各 rank 独立读取同一份数据；这与 chunk 边界过读不同。现各进程独立 client/cache，不存在自动的节点级共享。先完成局部优化；若实测证明重复读取是主要瓶颈，再考虑现有 backend 内的 owner 读取与组内分发，不先新增节点缓存服务。

### 7.5 沿用 TQ RDMA，并补足 replay 容量预算

Host RDMA 对跨节点大 payload 有潜在收益，但不会消除原 indexer 计算、布局处理、metadata RPC、chunk 过读或最终 H2D。同节点或小请求场景的收益需要实测，不能由 backend 名称推断。当前 host 路径仍涉及接收 buffer 分配、注册和数据组装，不是自动的零拷贝消费。

当前 `relax/utils/tq/config.py` 的 `estimate_payload_bytes()` 只估算文本与多模态样本字段，没有计入独立 replay body。启用该组合前，需要把实际 capture stream 数、top-k、query 行数、在途 batch 和保留生命周期纳入 Mooncake segment 容量预算；小 manifest 不能代表小存储占用。Replay 的逻辑 capacity、Mooncake 的实际 segment 和在途 buffer 是不同预算，需同时验证。

优先方案是保留 PR #256 的现有接线，结合提前消费和有界预取验证。先测 SimpleStorage 与 host-RDMA 的同布局 replay A/B，再决定是否需要调整注册 buffer 复用或更低层的读取接口；不另建独立 Mooncake 服务或 RDMA 传输框架。

## 8. 算子对齐与 replay 的取舍

算子对齐免去大块 top-k 传输，也能改善其他数值误差，但 kernel、量化、归约顺序、硬件和并行配置变化后需要重新验证。确定性路径可能降低某些 kernel 的速度，不能预设它始终更快。

Replay 能直接固定离散选择，对新模型/精度路径更容易隔离问题；代价是捕获、存储、读取、H2D 与生命周期管理。Reference 只降低控制链路与不必要搬运，不消除 payload 本体。

在未明确极高性能 replay 目标前，长期减少索引传输有吸引力；当前需求已明确选择高性能 replay 为主要建设路线。算子对齐作为互补手段与对照实验，不要求先完成全栈对齐才能接入 replay。

量级示例：假设 64 个 indexer 层、top-k=2048、int32，原始索引为 512 KiB/query token；32768 个被捕获 query token 约 16 GiB，未计副本和 staging。此为通用估算，不是 DSV4 C4 的固定成本；实际需用真实 capture stream 数和表示宽度计算。

## 9. 实施顺序与验收

1. DSV4 当前支持配置：增加直接消费入口，前移 hook，显式保留主 attention 梯度，省去专用 indexer 支路。
2. 消费链路：修正实际可用的预取预算与启动时机，减少无意义打包、CPU copy 和重复 metadata RPC，再验证 H2D overlap。
3. QSA：补齐 block-ID capture、层映射和 selection 构造，先核对基础闭环，再完成现有 backend 所需的 CP-local query 适配。
4. 根据真实端到端瓶颈决定 chunk 大小、跨 rank 共享等后续工作，避免先做大范围重构。

每一阶段分别验证：

| 目标 | 证据 |
| --- | --- |
| 不再计算原选择 | Profiler 中对应 indexer projection/score/top-k 及可省的专用 collective 消失；CPU 测试可用会报错的原计算回调覆盖接线，但不能代替 GPU trace |
| Attention 与梯度正确 | 与“算完再覆盖”路径对照输出、主干参数梯度、远端 K/V 梯度，检查无辅助 loss 分支 |
| 行与层对齐 | Packing、多文档、不同长度、padding、causal tail、PP/CP 与支持的 TP 配置；使用可独立预测的行/层值 |
| 重算正确 | Old-logprob、训练 forward、full/selective recompute 的游标完整性与 selection 生命周期 |
| 存储可靠 | Abort/retraction、写背压、重复读取、partition retirement、reader release、run 重建后的旧引用隔离 |
| 搬运有效 | 有效行字节、实际读取字节、PUT/GET 与 metadata RPC 数、host copy、H2D 字节、预取命中与消费者等待 |
| 内存有界 | CPU cache、在途读取、pinned memory、GPU selection 及 attention backward 的实际峰值 |
| 完整 step 收益 | 同模型/样本/权重/并行/精度下，比较关闭 replay、算完再覆盖、提前消费三组；记录 warmup 与稳态、rollout/train 时间及吞吐 |

文章的不足 7% 额外开销可作为对标，不设置未经实测的保证。跳过 indexer 后净开销甚至可能下降，也必须通过完整 step 而非局部 kernel 时间证明。加入算子对齐组时，还需比较 logprob 差异、entropy 和训练收益。

## 10. 已有验证与尚未验证的内容

此前在用户指定的 Relax 镜像中完成以下验证；镜像 tag 为 `qs-20261029-bd81bfd0-cu12`，内部 registry 地址不写入仓库：

- 选定 CPU/reference 回归集合：293 passed，无 skip；不是整个仓库测试套件。
- 真实本地 Ray + TQ smoke：manifest 交接、按层读取、重复 replay、回收通过。
- 重建 TQ/registry 后旧引用与 lease 隔离通过。
- 镜像内真实 SGLang 0.5.21 scheduler、collector、abort、output、detokenizer hook 安装通过。

此前 SGLang hook 的问题是内部接口迁移：输出收集从 `SchedulerOutputProcessorMixin` 迁至 `SchedulerBatchResultProcessor`，对应方法增加 `_` 前缀；KV slot 入口从旧的 `req.req_pool_idx` 迁到 `req.kv.req_pool_idx`；detokenizer 编码入口也需要适配。`0336755c0` 已增加新旧接口识别和现代请求 KV 字段读取。上述 hook 安装结果验证了接线兼容，不能替代真实 GPU 生成中的捕获、接受行与响应输出验证。这与训练侧“算完原 top-k 再覆盖”的性能问题分属不同路径。

当时本地 GPU 不可用，未执行真实模型训练、多节点 GPU 集成、attention 数值/梯度对照或 GPU 性能 A/B。上述结果只支撑已有 reference 基础设施，不能作为提前绕过 indexer、QSA rollout replay 或低于 7% 开销的验证。

后续实现还会涉及当前入口支持范围。若需要修改参数解析、Controller/Service/Launcher 或依赖，按仓库约定单独确认；本设计不要求先改这些公共框架层。
