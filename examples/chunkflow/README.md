# ChunkFlow

[English](README.en.md) · [完整指南](../../docs/zh/guide/chunkflow.md)

将长序列拆成 chunk，复用模型原生 Attention / GDN 计算，并连接跨块状态与梯度。**优先用于长上下文、显存受限的训练；是否加速取决于模型和配置。**

## 实验收益

以下是已完成的短跑对照，不代表所有模型的收益。两组使用相同数据和训练配置，仅切换 ChunkFlow；Dense 的 K1 会额外启用 ChunkFlow 重计算。时间取第 1～29 步均值（排除第 0 步），显存取全部步骤、全部 rank 的最大值。加速比 = Baseline / ChunkFlow，低于 1 表示变慢；“—”表示未核验，不填估算值。

### Dense · Qwen3-4B

**8 卡 · TP2 / PP4 · GBS1 · LongAlign · 长度上限64K · chunk4096 · K1 · 30步**。未启用原生重计算；这是历史 K1 对照，不能当作 K0 结果。

| 方法          | 平均训练耗时 | 平均 step time | MFU | 训练加速比 | 端到端加速比 | 峰值 allocated / reserved | allocated 节省 |
| ------------- | -----------: | -------------: | --: | ---------: | -----------: | ------------------------: | -------------: |
| Baseline      |       3.89 s |              — |   — |      1.00× |            — |         52.28 / 79.03 GiB |              — |
| **ChunkFlow** |   **2.20 s** |              — |   — |  **1.76×** |            — |     **21.01 / 22.58 GiB** |     **59.81%** |

Loss 平均相对差异 **0.51%**。本配置同时体现训练加速与显存下降；不能外推到大 batch 或 K0。

### MoE · Qwen3.6-35B-A3B

**8 卡 · TP2 / PP4 / EP2 · GBS8 · LongAlign · 长度上限64K · chunk4096 · K0 · 30步**。双方均开启原生 selective `gdn moe` 重计算，关闭 ChunkFlow 自身重计算。

| 方法          | 平均训练耗时 | 平均 step time | MFU | 训练加速比 | 端到端加速比 | 峰值 allocated / reserved | allocated 节省 |
| ------------- | -----------: | -------------: | --: | ---------: | -----------: | ------------------------: | -------------: |
| Baseline      |      15.99 s |              — |   — |      1.00× |            — |         56.38 / 70.26 GiB |              — |
| **ChunkFlow** |  **18.90 s** |              — |   — |  **0.85×** |            — |     **37.79 / 40.58 GiB** |     **32.97%** |

Loss 平均相对差异 **0.39%**。本配置节省 **18.59 GiB**，训练耗时增加约 **18.2%**，适合优先解决显存压力的场景。

显存收益主要看 **allocated**；reserved 包含分配器缓存。短跑 loss 接近不代表梯度等价或长期收敛已验证。

## 什么时候开启

| 场景                                     | 建议                                               |
| ---------------------------------------- | -------------------------------------------------- |
| 长上下文 Dense、小 batch、流水线空闲较多 | 值得做对照，历史 K1 配置有加速收益                 |
| 长上下文 MoE、显存紧张                   | 优先尝试，接受可能增加的计算时间                   |
| 短序列、显存充足、追求最大吞吐           | 默认不开；先测再决定                               |
| QSA / PLE 模型                           | 实验性使用；前缀重放会重复计算，不保证省显存或加速 |

当前面向 **packed 纯文本、CP1、零 dropout、关闭 RoPE fusion、per-token loss**。PP4 需实验性 opt-in（示例已设置）。不直接支持 CP>1、VPP、多模态、full-layer / selective `core_attn` 重计算。QSA 仅允许原生 selective `moe` 重计算。

Qwen3.8-Flash-Next 已完成8K上限、K0、10步对照，但使用了长度筛选子集：allocated 仅下降2.42%，未体现加速。原始 LongAlign 的32K对照尚未完成，不能据此推荐长上下文 QSA 性能。

## 快速开启

使用 [relax.version](relax.version) 固定的 Relax 版本及其训练镜像。示例脚本自动安装集成补丁，不安装依赖，不保存 checkpoint。

```bash
export MODEL_DIR=/path/to/models
export PROMPT_DATA=/path/to/LongAlign-10k/long.jsonl

# Dense：脚本已开启 ChunkFlow，默认64K / chunk4096 / K1
bash examples/chunkflow/scripts/run-qwen3-4B-sft-8xgpu.sh

# MoE：默认64K / chunk4096 / K1；K0关闭ChunkFlow自身重计算
KEEP_LAST_K=0 bash examples/chunkflow/scripts/run-qwen36-35B-A3B-sft-8xgpu.sh
```

## 代码目录

```text
examples/chunkflow/
├── adapters/                 # 原生计算适配
│   ├── __init__.py           # 注册表与 Dense Attention
│   ├── gdn.py                # GDN 卷积、递推状态连接
│   ├── qsa.py                # 原生 QSA 前缀重放
│   ├── ple.py                # 原生 PLE 前缀适配
│   └── prefix.py             # 共享前缀布局、RoPE 与 SP 处理
├── integration.py            # 模型创建时安装适配器
├── scheduler.py / units.py   # 分块与执行计划
├── schedules.py              # 单流水线 stage 的 forward/backward
├── pipeline_schedules.py     # 多 stage 的 forward/backward
├── runtime.py / state_store.py # 跨块状态、梯度与生命周期
├── unit_batch.py / metadata.py # batch 切分与边界元数据
├── rope.py                   # 分块位置偏移
├── arguments.py              # 开关与约束检查
├── setup.py / relax.patch / relax.version # 固定版本集成
├── configs/                  # GDN、QSA 原生配置
├── scripts/                  # Dense、MoE、QSA 三个入口及公共提交逻辑
└── test/{cpu,gpu}/           # 核心回归与 CUDA 验证
```

运行链路：**脚本 → 集成补丁 → 安装适配器 → 分块计划 → 原生 forward → 跨块 backward → 清理状态**。实现细节与测试命令见[完整指南](../../docs/zh/guide/chunkflow.md)。
