# Replay Manager

Replay Manager 将 SGLang rollout 的模型决策和冻结视觉输出交给 Megatron 训练复用。本分支一期实现 `moe`（专家路由 R3）与 `mm_embedding`（视觉 embedding）；QSA/DSA indexer 只保留扩展接口，不包含其 replay 实现。

Reference 模式下，sample 携带描述符，不可变 body 存放在 TransferQueue（TQ）。它与用于离线复现的[轨迹重放](./trajectory-replay.md)不同。

## 数据流

SGLang 采集路由和视觉 embedding，将 body 写入现有 TQ；sample 只携带 reference。Megatron 按本 rank 的 layer、microbatch 和 token 布局读取并重放。

写入完成后才发布 reference。队列和 reader 分别持有数据引用，所有使用方释放后才回收；全局恢复会生成新的 run 标识，避免读取旧数据。

## 开启 R3 与 MM replay

在线 Megatron 配方中加入：

```bash
--use-rollout-routing-replay
--use-routing-replay
--optimize-routing-replay
--use-rollout-mm-embedding-replay
--freeze-vision-model
--freeze-vision-projection
```

R3 的前三个开关分别控制 rollout 采集、训练重放与路由优化。支持的在线 R3 配方默认使用 reference，MM replay 也使用 reference。关闭 replay 的基线移除四个 replay 开关，仍冻结视觉模型和 projection。

MM 支持 Qwen3.5 MoE 和 Qwen3.8 多模态 checkpoint，分别要求 2048 和 2560 维视觉输出、spatial merge size 2、无 DeepStack。要求 BF16、packed THD、CP1，同时冻结视觉编码器和 merger/projection。支持内置 SGLang 与 Agentic rollout 的 colocated、hybrid、fully async；异步 MM 要求 PP1、无 interleaving，fully async 还要求动态 token batching、inline actor log-probability，且不使用独立 reference policy。Critic、partial rollout、MTP、推测解码、hierarchical cache、自定义生成、Slime router 和 OPD 不在 MM 支持范围。

采集发生在视觉 merger 之后，训练在输入 PP stage 读取并向 TP 组广播，复用 BF16 视觉输出，不做量化。R3 reference 要求静态 CP，关闭 combined 1F1B MoE overlap。需要带匹配 SGLang 和 Megatron patch 的镜像；本轮实验使用 SGLang 0.5.21。

仅开启 R3 且保存 debug rollout 数据时，默认保存完整路由，便于跨作业重放。显式开启 reference 或 MM replay 时，引用只在当前作业有效。

## 拆分分支实验记录（2026-11-22）

模型为 **Qwen3.5-35B-A3B**，每个实验独立使用单机 **8 张 280G Blackwell GPU**。Replay 为 **R3+MM reference**，不开 QSA/DSA。每组 10 步，不保存模型；下表为剔除 step 0 后 step 1–9 的算术均值，不报告 reward。

| 任务 / 模式 | 基线 秒/步 | Replay 秒/步 | 耗时变化 | Mismatch KL：基线 → Replay | 状态 |
| --- | ---: | ---: | ---: | --- | --- |
| DeepEyes fully async，TQ 修复后 | 70.98 | 62.72 | −11.6%* | 0.003724 → 0.000825 | 两组均完成 10 步 |
| DeepEyes hybrid | 69.61 | 68.55 | −1.5% | 0.004220 → 0.001261 | 两组均完成 10 步 |
| DeepEyes colocated | 124.03 | 121.57 | −2.0%† | 0.003942 → 0.000634 | Replay 完成 10 步 |
| OpenR1MM colocated | 129.71 | 119.80 | −7.6% | 0.001488 → 0.000296 | 两组均完成 10 步 |

\* Fully async 为两台机器各自单机 8 卡的诊断对照，使用含额外 sealer 调整的代码，不能将耗时差全部归因于 replay。当前分支无额外 sealer 调整、无 profiler 的确认实验也已完成 10 步：**59.09 秒/步，mismatch KL 0.001119**，单独记录，不混入该对照。

† DeepEyes colocated 使用历史基线。Hybrid OFF/ON 在同一机器运行。短跑中的轨迹长度和 token 数会变化。

### 关键配置

| 配置 | DeepEyes fully async / hybrid | DeepEyes / OpenR1MM colocated |
| --- | --- | --- |
| GPU 分配 | Actor 4 卡 + Rollout 4 卡，独立 GPU | Actor、Rollout 分时共用 8 卡 |
| 训练 TP / PP / CP / EP / ETP | 2 / 1 / 1 / 4 / 1 | 2 / 1 / 1 / 8 / 1 |
| Sequence parallel | 开 | 开 |
| Rollout 引擎 | 2 个引擎，各 TP2 | 4 个引擎，各 TP2 |
| Staleness | 2 | 0 |
| Prompt batch × 每题采样数 | 16 × 8 | 16 × 8 |
| Global batch size | 128 | 128 |
| Micro batch / 动态 token 预算 | micro-batch-size=1；开启动态 batching，max-tokens-per-gpu=16384 | 同左 |
| Prompt / Response / Context 上限 | 4096 / 2048 / 16384 token | 同左 |
| SGLang context / max running / max total tokens | 16384 / 32 / 131072（每引擎） | 同左 |
| SGLang 静态显存比例 | 0.75 | 0.75 |
| 精度 / Attention | BF16 / Flash | BF16 / Flash |
| 视觉模型 / Projection | 均冻结 | 均冻结 |
| 优化器 | Adam，lr=1e-6，CPU offload、D2H/H2D overlap、precision-aware optimizer 均开启 | 同左 |
| 算法 | GRPO + TIS，记录 mismatch | 同左 |

DeepEyes agentic concurrency=8，timeout=1200 秒，judge 为各机独立 CPU Qwen2.5-1.5B-Instruct。OpenR1MM 使用内置 `openr1mm` reward 路径，无 agentic rollout。通信环境统一为 Socket + IPv4，关闭 IB、网络插件和 NVLS；各实验本身均为单机。

Colocated R3 的启动示例见 [OpenR1MM 训练脚本](../../../scripts/training/multimodal/run-qwen35-35B-A3B-8xgpu-openr1mm-r3.sh)。
