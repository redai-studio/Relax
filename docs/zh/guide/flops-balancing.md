# FLOPs 感知 SFT 调度

`--balance-by-flops` 为普通文本 causal-LM SFT 开启 FLOPs 感知的 DP 分配和容量安全的动态微批打包，默认关闭。它不修改选定的有效 batch、优化器、loss 归一化或 `--max-tokens-per-gpu` 的单位。

## 配置

使用 Megatron、`--loss-type sft`、`--use-dynamic-batch-size` 和 `--n-samples-per-prompt 1`。需要显式选择普通 SFT 路径，例如 `--sft-max-in-flight-steps 1` 可避免自动启用前瞻预打包。新开关不会替用户修改这个设置；自动进入 async prepack 时会明确报错。

官方 Docker 已提供固定版本 TransferQueue 的 `assignment_transform` 接口；缺少该接口的安装会明确失败。适配器只改变已选定样本的 DP 分配，选择、缓存发布、重放及 checkpoint 恢复仍由原 sampler 负责。FLOPs KK 候选只有在 rank FLOPs max/mean 严格改善且预测 DP MAX 微批数不增加时才被接受，否则保持 token 分配。guard 使用实际 packing strategy，并在 worker 构建前同时检查 CP raw/padded 两种可能布局。

首版成本模型支持 Qwen2/Qwen3 文本 Dense/MoE、Qwen3.5 文本 Dense/MoE 及其 text 配置名，复用现有 FLOPs 公式。其他或不完整配置会报错，不静默回退。Qwen3.5 配置可以包含 vision 配置，但本调度器不支持实际图片输入。

MoE 要求辅助损失/z-loss 系数全部为零、专家容量无限制（`moe_expert_capacity_factor=None`）。辅助损失和专家 token 丢弃依赖微批组成，辅助 autograd 路径也可能为 dummy 产生梯度，因此首版拒绝这些配置，以保持训练语义。

同样拒绝依赖微批组成的 Sinkhorn/quantile 路由和非零 expert-bias 更新。辅助系数为零的普通 top-k 路由仍可使用；开关不会偷偷修改路由设置。

## 算法与容量

FLOPs 是均衡目标，token 长度是容量约束。先由现有容量安全的 token scheduler 在原 DP MAX/VPP 对齐数量下构造 baseline，再做固定 K repair；FLOPs 优化不会增加微批数。原有超长 singleton 例外保留且不参与 repair；它不代表该样本一定能放进 GPU 显存。

固定 K 下先尝试 single move，无改善再尝试 pair swap；只接受严格改善，目标依次为最大 FLOPs、spread、FLOPs 平方和。source 保持非空，两个受影响 bin 都必须满足 token cap。最多 32 轮，每轮只考察两重/两轻 bin、每 bin 最多 8 个样本，稳定 index 决定 tie-break。无改善时逐项返回原 baseline。

原有 DP MAX、VPP grouping、CP physical padding 和 dummy 构造不变。每个 optimizer step 真实样本不丢不重；dummy 副本和 offset 保持 token baseline 的结果，不参与 repair，保留零有效 loss/token 语义。

## 支持范围及验证边界

| 组合 | 实现边界 | 硬件验证 |
| --- | --- | --- |
| 普通文本 SFT、dynamic batch、Dense/MoE | 包含 | 待验证 |
| DP 分配与本地微批构造 | 包含 | 多卡待验证 |
| TP、静态 PP/CP、EP、VPP | 保留现有同步/拓扑约束 | 多卡待验证 |
| RL、fully async、hybrid、wholepack、async prepack | 开启时明确拒绝 | 不支持 |
| preference、多模态、动态 CP、routing replay、MTP、LoRA、自定义语言模型冻结 | 开启时明确拒绝 | 不支持 |

CPU 和模拟 collective 测试不能证明真实 NCCL、专家路由、显存峰值或分布式数值正确性，也不能证明训练提速。

## 观测与实验准备

`LOG_LEVEL=DEBUG` 时，现有日志体系输出 DP 分配前后的 token/FLOPs 总量、max/mean 不均衡率、微批容量 token/FLOPs、同步数量、dummy 数、move/swap/iteration 次数、DP 候选接受/拒绝原因和调度耗时。工作量汇总仅统计真实样本；dummy 额外开销由数量单独标识。计时只用于诊断，不参与确定性决策。

`python scripts/tools/benchmark_flops_balancing.py --help` 可查看 CPU 调度比较工具。它读取模型配置和固定 batch 的长度 JSON，不加载模型权重；输出不代表 GPU 吞吐。

正式 A/B 实验固定 checkpoint、优化器状态、随机种子、数据顺序、有效 batch、精度、拓扑、token 容量和 dropout 策略。两版均使用普通 SFT 路径，不能把预打包基线与关闭预打包后的新版本混比。先重复基线、固定容差，再比较样本 ID、有效 loss-mask token、完整 batch 的 loss、累积归约后的梯度及 FP32 参数更新量。保留原始日志、命令、版本和数据选择规则，区分预热与测量窗口，统计包括调度和数据准备的完整训练循环。真实 Dense/MoE 对照与至少 10% 的吞吐收益仍需后续硬件验证。
