# FLOPs 感知调度

在纯文本 Qwen3.5 Dense 和 MoE 训练中，FLOPs 感知调度按估算的训练计算量分配样本，而不仅按序列长度分配。Token 容量约束与成本估算分别处理。这里估算的是模型理论计算量，不是实测 GPU 耗时。

功能默认关闭。在已有的 Megatron 动态批处理配置中加入：

```bash
--use-dynamic-batch-size \
--flops-scheduling both
```

`--flops-scheduling` 支持四种模式：

| 模式 | 微批构造 | DP 分配 |
| --- | --- | --- |
| `off`（默认） | 保持原有行为 | 保持原有行为 |
| `microbatch` | 按 FLOPs 调度 | 保持原有行为 |
| `dp` | 保持原有行为 | 按 FLOPs 调度 |
| `both` | 按 FLOPs 调度 | 按 FLOPs 调度 |

默认顺序为 `--flops-scheduling-order dp-first`：先将样本行分配到数据并行 Rank，再在各 Rank 内构造微批。如需先全局打包，再分配 DP，使用：

```bash
--flops-scheduling both \
--flops-scheduling-order pack-first \
--polling-mode True \
--qkv-format thd
```

Pack-first 通过原 Sampler 选择完整 cohort，固定微批成员，再将微批分配到 DP Rank。训练端直接执行计划，不会在各 Rank 内重新打包。零训练贡献的 dummy 微批用于对齐执行次数。队列消费标记不代表优化器更新已经完成。

## 配置范围

启用模式要求同步、动态批处理、无 MTP 的普通纯文本因果语言模型训练。多模态输入、分类或偏好头、LoRA、自定义参数冻结、Wholepack、CPU prepack、完全异步训练和动态 global batch size 会被拒绝。

SFT 的 DP 分配要求 `n_samples_per_prompt=1`。RL 的 DP 分配要求 `--balance-data` 及原有 Balanced identity sampler；一个 logical identity 即使包含多条 physical row，也仍按完整 identity 准入。

Pack-first 还要求 polling 与 THD packing，不支持 hybrid、动态 CP、SFT 数据预取、绕过真实数据的 debug 路径及 routing/indexer replay。SFT 每个 partition 支持一个 optimizer mini（`--num-steps-per-rollout 1`）；Balanced RL 保留显式的多 mini 边界。Checkpoint/数据身份或执行几何变化会使旧计划无法重放，不要将保存的计划用于不同样本或模型配置。

SFT adapter 依赖仓库固定的 TransferQueue `assignment_transform` 补丁，请使用项目匹配的依赖构建。运行限制会在发布计划前检查；CPU 测试不能单独证明 GPU 或分布式拓扑可用。

比较调度模式时，应固定有效 batch、checkpoint、优化器状态及训练配置。估算 FLOPs 更均衡不保证吞吐更高；应测量完整训练循环，并先验证 loss、归约后的梯度和 FP32 参数更新量，再作性能判断。

训练准备见 [SFT 训练](./sft-training.md)，测量工具见 [性能调优](./performance-tuning.md)。
