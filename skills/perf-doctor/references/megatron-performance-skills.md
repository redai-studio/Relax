# Megatron 官方调优 skills：Relax 使用参考

供 `perf-doctor` 的静态审核和 `megatron-expert` 的只读咨询使用。本文提供选用方式和关键边界；详细约束按实际运行版本核对上游代码，不把官方技能当作已安装工具或自动执行授权。

核查日期：2026-09-25。原文基准为 `NVIDIA/skills` commit `d8519c57da6db5d9bea274ec1724a4a7a56a3dee`（Apache-2.0）；文末提供固定版本引用。后续刷新时同时检查各 skill 的 `card.yaml`，区分源码核对、实测结果和待验证项。

## 共同使用方式

1. 根据症状选择下列小节；必要时读取该 skill 原文及验证卡。未安装也可通过链接读取，不自动安装依赖。
2. 收集模型/任务、GPU 型号与拓扑、镜像和 MCore/TE/CUDA/NCCL 版本、序列长度、MBS/GBS、并行布局、精度、路由、recompute、dispatcher 和 graph scope。已有硬件证据优先于 `perf-doctor` 的默认 H20 假设。
3. 将上游建议映射到 Relax：读启动脚本及其 `source` 的模型配置，再追踪 `relax/utils/arguments.py`、`relax/backends/megatron/arguments.py`、`model_provider.py`、`model.py` 和训练镜像中的 MCore/TE 实现。核实参数是否存在、是否被覆盖、是否传入运行时；找不到对应能力就标注“未核实/未接入”。
4. 输出候选、限制、Relax 代码依据和验证计划。区分“保持训练语义的候选”与“仅 benchmark 的设置”；强制均衡路由、mock data、关闭检查或跳过 optimizer/checkpoint 不能证明真实训练收益。

上游的 `cfg.model`、`cfg.comm_overlap` 和带下划线的 performance harness 参数都是 Bridge 接口示例，不能直接抄进 Relax 启动脚本。规划 GPU 实验时沿用仓库既有远程提交流程；这份参考本身不执行实验，也不放宽调用方的只读边界。

## 1. perf-moe-optimization-workflow

完整名称：`nemo-mbridge-perf-moe-optimization-workflow`。

**功能与适用场景**：组织 MoE 整体吞吐优化、扩卡效率或性能回归调查；先区分显存、通信、计算和 host/launch 开销，再选择具体优化，不按 GPU 名称直接决定 dispatcher。

**使用示例**：

> 按 `nemo-mbridge-perf-moe-optimization-workflow` 审核给定 Relax MoE 启动脚本。列出测量条件、显存可行性、Attention/MoE 并行布局，以及需要补齐证据的单变量实验；仅输出诊断与验证计划。

**应用要点**：

- 顺序是固定比较条件 → 模型可靠装入显存 → 并行扩展 → profile 定位 → 单变量调整 → 验证。优先评估 selective recompute，仍无法容纳时再考虑更重的重计算/offload。
- 分开记录 `world_size = TP × CP × DP × PP = ETP × EP × EDP × PP`，避免把 EP 当成额外独立乘数重复计算卡数。
- dispatcher、EP overlap、低精度和 graphs 分别作为候选；不把同时改动的总收益归给其中一个开关。
- 验证计划应使用未开启 profiler 的固定稳态窗口衡量 step time、tokens/s 或 model TFLOPS/GPU，同时检查 loss、NaN/跳步、峰值显存与必要的恢复路径。上游建议短筛选后对入选方案运行至少 50 步；这是实验设计参考，不能替代 Relax RL 更新/变长 batch 的代表性窗口选择。

**引用**：[skill 原文][workflow]；[验证卡][workflow-card]；[Bridge MoE 优化说明](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/training/moe-optimization.md)。

## 2. perf-expert-parallel-overlap

完整名称：`nemo-mbridge-perf-expert-parallel-overlap`。

**功能与适用场景**：在 MoE、`EP > 1` 且 dispatch/combine 明显暴露时，评估通信与专家计算重叠；`delay_wgrad_compute` 是额外候选，不能与 plain EP overlap 混为一项。

**使用示例**：

> 按 `nemo-mbridge-perf-expert-parallel-overlap` 检查该配置。先判断 plain EP overlap 是否可用，再单独评估 delayed wgrad，列出 dispatcher、VPP、重计算及版本门槛，设计其余设置相同的 A/B。

**应用要点**：

- 上游入口为 `cfg.comm_overlap.overlap_moe_expert_parallel_comm`、`cfg.comm_overlap.delay_wgrad_compute`；核对 `model.moe_token_dispatcher_type` 的 `alltoall` 或 `flex` 路径。只设置 flex backend 名称不证明运行时已启用 DeepEP/HybridEP。
- 按本次上游基准，核查 BF16/FP16、PyTorch ≥ 2.6、禁止 full recompute/显式 recompute method 与 layers、不能同时打开 shared-expert overlap，以及 PP > 1 时的 VPP 要求。结合目标版本源码确认，不把这些条件视为跨版本常量。
- delayed wgrad 与梯度归约 overlap、CUDA graphs 的组合另有 TE 版本、梯度累积 fusion、attention bias 等约束，按原文逐项核对。先比较 overlap off/on 且 delayed wgrad off，再单独测 delayed wgrad；Bridge 的 `moe_a2a_overlap` 快捷项可能同时开启两项。
- 按 rank/device 对通信与计算区间分别取并集，再求交集；不能将各 stream 的 kernel duration 相加当作墙钟时间。以稳定 step time 和显存验证收益，不能仅以配置 dump 或重叠箭头作为证据。

**引用**：[skill 原文][ep]；[验证卡][ep-card]；[Bridge 通信重叠说明](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/training/communication-overlap.md)。上游具体模型、硬件、路由和 graph 设置下的收益仅是参考，不写成本任务的预期加速。

## 3. perf-cuda-graphs

完整名称：`nemo-mbridge-perf-cuda-graphs`。

**功能与适用场景**：profile 显示 host/launch 开销，或 graph 配置变化后发生性能/显存回归时，选择并验证训练侧 graph 实现与捕获范围。

**使用示例**：

> 按 `nemo-mbridge-perf-cuda-graphs` 评估该 Relax 训练配置。核对动态 batch/sequence、routing、RNG、recompute 与 EP overlap；给出 eager 对照、最小 graph scope 和 replay 验证要求，不假设 capture 成功就是加速。

**应用要点**：

- Bridge 用 `cfg.model.cuda_graph_impl` 选择 `transformer_engine` 的模块级 graph 或 `local` 的 `full_iteration`；相关项包括 `cuda_graph_scope`、warmup steps 和 TE RNG tracker。具体字段传递须按 Relax 运行版本验证。
- MoE 先考虑最小可行范围，如 `moe_router`、`moe_preprocess`；有证据再扩展到 `attn`。动态专家工作不宜直接套 full-iteration；不要为了 graph 而默默改成 drop-and-pad 路由。
- 检查捕获范围内 shape 是否稳定。Relax 动态 batch、变长序列与 packing 不能假设满足固定 shape，需确认是否有合法的分桶/重捕获支持。训练 graph 和 SGLang rollout graph 是不同配置路径。
- 核对 RNG、graph 额外显存、recompute、delayed wgrad、scope 互斥及 allocator/NCCL 注册条件。按本次上游基准，`full_iteration` 只配 `local`；`moe` 与 `moe_router` scope 不混用，`moe_preprocess` 必须同时包含 `moe_router`。原文部分示例关闭 NaN 检查，应明确记录这种验证能力变化，不能作为默认优化偷偷应用。
- 对照 eager 与实际 replay，排除 capture/warmup 时间；检查 graph launch、稳态耗时和峰值显存。修改 dispatcher/overlap/precision 后重新比较，不能沿用之前的收益结论。

**引用**：[skill 原文][graphs]；[验证卡][graphs-card]；[Bridge CUDA graphs 说明](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/training/cuda-graphs.md)。

[workflow]: https://github.com/NVIDIA/skills/blob/d8519c57da6db5d9bea274ec1724a4a7a56a3dee/skills/nemo-mbridge-perf-moe-optimization-workflow/SKILL.md
[workflow-card]: https://github.com/NVIDIA/skills/blob/d8519c57da6db5d9bea274ec1724a4a7a56a3dee/skills/nemo-mbridge-perf-moe-optimization-workflow/card.yaml
[ep]: https://github.com/NVIDIA/skills/blob/d8519c57da6db5d9bea274ec1724a4a7a56a3dee/skills/nemo-mbridge-perf-expert-parallel-overlap/SKILL.md
[ep-card]: https://github.com/NVIDIA/skills/blob/d8519c57da6db5d9bea274ec1724a4a7a56a3dee/skills/nemo-mbridge-perf-expert-parallel-overlap/card.yaml
[graphs]: https://github.com/NVIDIA/skills/blob/d8519c57da6db5d9bea274ec1724a4a7a56a3dee/skills/nemo-mbridge-perf-cuda-graphs/SKILL.md
[graphs-card]: https://github.com/NVIDIA/skills/blob/d8519c57da6db5d9bea274ec1724a4a7a56a3dee/skills/nemo-mbridge-perf-cuda-graphs/card.yaml
