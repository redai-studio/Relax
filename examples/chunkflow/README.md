<h1 align="center">ChunkFlow</h1>

<p align="center">长序列分块训练 · 复用原生计算 · 降低显存压力</p>

<p align="center">
  <a href="#实验收益">实验收益</a> ·
  <a href="#开启建议">开启建议</a> ·
  <a href="#快速开启">快速开启</a> ·
  <a href="README.en.md">English</a>
</p>

______________________________________________________________________

## 实验收益

### Dense · Qwen3-4B

> **实验配置**　8 卡 · TP2 / PP4 · GBS8 · chunk4096

数据：5,000 条合成中文 SFT 样本。长度分布：0–2K 占 70%，2–8K 占 20%，8–16K 占 8%，16–32K 占 2%。

| 方法          | 平均训练耗时 | 平均 step time |  训练 MFU | 训练加速比 | 端到端加速比 |
| ------------- | -----------: | -------------: | --------: | ---------: | -----------: |
| Baseline      |       3.08 s |         4.38 s |     23.1% |      1.00× |        1.00× |
| **ChunkFlow** |   **1.86 s** |     **3.10 s** | **38.2%** |  **1.65×** |    **1.40×** |

**显存参考**：LongAlign / 64K / chunk4096 / GBS1 / K1，峰值 allocated 从 **52.28 GiB → 21.01 GiB**，节省 **59.81%**。

### MoE · Qwen3.6-35B-A3B

> **实验配置**　8 卡 · TP2 / PP4 / EP2 · GBS8 · chunk8192 · LongAlign

| 方法          | 平均训练耗时 | 平均 step time |  训练 MFU | 训练加速比 | 端到端加速比 |
| ------------- | -----------: | -------------: | --------: | ---------: | -----------: |
| Baseline      |       16.2 s |         17.5 s |     13.8% |      1.00× |        1.00× |
| **ChunkFlow** |   **17.0 s** |     **18.4 s** | **13.1%** | **0.950×** |   **0.950×** |

**显存参考**：LongAlign / 64K / chunk4096 / GBS8 / K0 ，峰值 allocated 从 **56.38 GiB → 37.79 GiB**，节省 **32.97%**。

______________________________________________________________________

## 开启建议

| 场景                                     | 建议                             |
| ---------------------------------------- | -------------------------------- |
| 长上下文 Dense、小 batch、流水线空闲较多 | 值得做对照，已观察到训练加速收益 |
| 长上下文 MoE、显存紧张                   | 优先尝试，接受可能增加的计算时间 |
| 短序列、显存充足、追求最大吞吐           | 默认不开；先测再决定             |

______________________________________________________________________

## 快速开启

**版本依赖**：基于 Relax [d548769b](https://github.com/redai-studio/Relax/commit/d548769b77f12e445e464f99d4332290a3e5bd3d) 及其配套训练镜像，固定版本见 [relax.version](relax.version)。

```bash
export MODEL_DIR=/path/to/model
export PROMPT_DATA=/path/to/data
```

**Dense · Qwen3-4B**　64K / chunk4096 / K0

```bash
bash examples/chunkflow/scripts/run-qwen3-4B-sft-8xgpu.sh
```

**MoE · Qwen3.6-35B-A3B**　64K / chunk4096 / K0

```bash
KEEP_LAST_K=0 bash examples/chunkflow/scripts/run-qwen36-35B-A3B-sft-8xgpu.sh
```

### 接入现有训练脚本

**① 安装接入补丁**

```bash
python3 examples/chunkflow/setup.py
```

**② 添加 ChunkFlow 参数**

```bash
--use-chunkflow \
--chunkflow-chunk-size 4096 \
--chunkflow-keep-last-k 0 \
--no-rope-fusion
```

原脚本已满足下列设置时，无需重复添加：

| 设置                                           | 原因                                                                 |
| ---------------------------------------------- | -------------------------------------------------------------------- |
| `--context-parallel-size 1`                    | 当前跨块状态管理不支持 CP 切分                                       |
| `--calculate-per-token-loss`                   | 按 token 统计，避免切块改变样本权重                                  |
| `--no-rope-fusion`                             | 使用支持分块位置偏移的 RoPE 路径                                     |
| `--attention-dropout 0.0 --hidden-dropout 0.0` | 当前适配要求关闭 dropout，避免分块与重放的随机掩码问题               |
| Megatron 重计算兼容性                          | 禁用 full-layer / selective `core_attn`                              |
| 训练目标                                       | 暂不支持 GSPO / OPSM / M2PO 等依赖完整序列或 batch 统计的目标        |
| MTP 训练                                       | 暂不支持跨 chunk 的多 token 预测；请关闭 `--enable-mtp-training`     |
| 分布式优化器                                   | `--num-distributed-optimizer-instances 1`（默认）                    |
| 专家并行（EP）                                 | 同一 EP 组须有一致的分块执行计划；不一致时全组提前报错，避免通信挂起 |

### 重要参数

| 参数                      | 建议起点       | 调整方式                                                   |
| ------------------------- | -------------- | ---------------------------------------------------------- |
| `--chunkflow-chunk-size`  | `4096`         | 小于实际长样本长度才会分块；减小后块数与调用开销增加       |
| `--chunkflow-keep-last-k` | `0`            | `0` 不重计算；`1` 只保留最后一块激活，较早块在反向前重计算 |
| `--seq-length`            | 按数据长度设置 | 样本长度上限，不是 chunk 大小；超过上限的样本跳过          |
