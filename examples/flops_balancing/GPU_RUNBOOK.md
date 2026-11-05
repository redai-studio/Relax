# Task 16 上机验证操作手册

**Qwen3.5 多卡正确性和真实吞吐验证尚未完成。** 以下是固定数据、数值对照和性能复现流程；候选布局需要硬件验证。

## 1. 四组实验与候选布局

| 模式 | 微批构造                       | DP 分配     | 用途               |
| ---- | ------------------------------ | ----------- | ------------------ |
| A    | 原有 First-Fit 估数量、KK 分组 | 原有长度 KK | 保留已有优化的基线 |
| B    | 带容量约束的 FLOPs 打包        | 原有长度 KK | 单看微批收益       |
| C    | 原有微批路径                   | FLOPs KK    | 单看 DP 收益       |
| D    | FLOPs 打包                     | FLOPs KK    | 组合效果           |

两开关独立，默认关闭。候选脚本用普通同步 SFT、THD、BF16、TP2/PP2/CP1；八卡对应普通 DP2。MoE 另设 EP2/ETP1，不能再把 EP 乘进普通 DP 卡数。MTP、dropout、辅助/z loss 关闭，不改 loss mask 或有效 batch。

八卡上 FP32 Adam 状态很大，所以四组都开启 `--optimizer-cpu-offload --optimizer-offload-fraction 1.0 --use-precision-aware-optimizer`，读取优化器实际 FP32 CPU 主权重。CPU RAM、内存带宽与 PCIe 可能成为瓶颈，offload 也可能压低优化收益。改精度、offload、batch、重计算、布局或预算时，所有对照必须一起改并重新冻结。

## 2. 固定资产和真实数据

| 资产                                                                                     | 固定版本                                   |
| ---------------------------------------------------------------------------------------- | ------------------------------------------ |
| [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)                                   | `fc05daec18b0a78c049392ed2e771dde82bdf654` |
| [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)                           | `59d61f3ce65a6d9863b86d2e96597125219dc754` |
| [OpenMathReasoning-mini](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini) | `5ea611a9fde449e7df8e72aefe698996a9882390` |
| 项目 Docker 固定 Bridge                                                                  | `af17edf52c514c58c79cd291b04a9b42bc5c57f3` |
| Bridge 切换 dev 后的 Core                                                                | `0e6ac576fba8995fca541ceb6af5f35eb0e4f16e` |
| 项目固定 TransferQueue                                                                   | `58054a33834aadbcf76aacd6b1e32e25c030f2c9` |

`download_assets` 下载固定数据、配置和 tokenizer，不下载模型权重。基础 Parquet 有 19252 行。规则：种子 1605 打乱物理行号，依序保留前 2048 条满足长度区间的原始行；不截断、不重复、不按长度排序。使用生产 `SFTStreamingDataset` 生成长度和 mask。

每个模型分别保存版本、manifest 和长度分布。长度分布与适用条件应和最终性能结果一起报告。

准备过程沿用已有 Qwen 模板 patch 和 per-message mask fallback，保留相关 warning。manifest 记录源行号、逐条长度、mask sum、文件/配置摘要及渲染摘要。GPU 数值运行另记录训练端实际收到的 token/mask 摘要。

复现准备，不需要 GPU：

```bash
cd /path/to/Relax
export ASSETS=/path/to/task16-assets
python -m examples.flops_balancing.download_assets --output "$ASSETS"
python -m examples.flops_balancing.prepare_workload \
  --source "$ASSETS/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet" \
  --model "$ASSETS/Qwen3.5-27B" --output "$ASSETS/workload-dense" \
  --source-revision 5ea611a9fde449e7df8e72aefe698996a9882390 \
  --model-revision fc05daec18b0a78c049392ed2e771dde82bdf654 --samples 2048
```

MoE 替换模型/输出目录与模型版本。长序列另用新目录，增加 `--max-length 32768`。已有输出目录会拒绝覆盖。

## 3. 资源边界与上机检查

候选是单机 8×A800 80GB，需核对实际互联，建议主机 RAM 约 1TB；先验证基线可加载并走完一步。单卡小模型通过不代表完整模型放得下。

全量数值导出包含 FP32 初始值、梯度、更新量，约 `12 × 参数量` 字节。27B 单次约 324GB，35B 单次约 420GB，尚不含副本与文件开销。需预留基线和当前对照的磁盘空间。比较器逐参数读，不把全模型向量一次加载进 RAM。

完整权重下载到相同固定版本目录：

```bash
hf download Qwen/Qwen3.5-27B \
  --revision fc05daec18b0a78c049392ed2e771dde82bdf654 \
  --local-dir "$ASSETS/Qwen3.5-27B"
hf download Qwen/Qwen3.5-35B-A3B \
  --revision 59d61f3ce65a6d9863b86d2e96597125219dc754 \
  --local-dir "$ASSETS/Qwen3.5-35B-A3B"
python -m examples.flops_balancing.preflight --require-gpus 8 --output /tmp/task16-preflight.json
```

优先用项目固定 Docker/依赖及补丁。CPU venv 或单独加入 PYTHONPATH 的 Core 源码不是完整训练镜像。A800 候选用标准 MoE alltoall，没有启用参考 Hopper 路径的 DeepEP。preflight 只读设备、拓扑与版本，不装驱动。

## 4. 离线预览与原生数值对照

```bash
TASK16_DRY_RUN=1 TASK16_MODEL=dense MODE=D \
  HF_CHECKPOINT="$ASSETS/Qwen3.5-27B" WORKLOAD_DIR="$ASSETS/workload-dense" \
  RUN_DIR=/tmp/task16-dense-D-preview \
  bash scripts/training/sft/run-qwen35-flops-task16.sh
```

预览校验数据/tokenizer 摘要，记录完整参数、代码与依赖，不启动 Ray。正式运行还检查所有权重 shard 存在。每次使用新的 RUN_DIR，从同一 HF 权重和新优化器开始，不续训、不保存 checkpoint、不评估。

在首次 GPU 对照前固定候选容差并留档：

| 量                         | atol   | rtol   |
| -------------------------- | ------ | ------ |
| loss                       | `1e-5` | `1e-3` |
| 归约、反缩放后且裁剪前梯度 | `1e-5` | `1e-3` |
| 实际 FP32 Δθ               | `1e-7` | `1e-3` |

逐分量判定 `abs(y-x) <= atol + rtol*abs(x)`，另报相对 L2。非有限值、漏参数、初始值/输入/mask/归一化变化均失败。不能看过优化结果后放宽阈值。

先跑 A1、A2 检查基线自身波动，再 B/C/D；先用 `NUM_ROLLOUT=1` 完成一次有效 batch 更新。以下地址和目录需填目标机器的实际值：

```bash
export RAY_ADDRESS=http://actual-head:8265
export HF_CHECKPOINT="$ASSETS/Qwen3.5-27B"
export WORKLOAD_DIR="$ASSETS/workload-dense"
export TASK16_MODEL=dense
export RUN_DIR=/path/to/results/dense-A1
MODE=A TASK16_CAPTURE=1 NUM_ROLLOUT=1 RAY_NO_WAIT=1 \
  bash scripts/entrypoint/ray-job.sh scripts/training/sft/run-qwen35-flops-task16.sh
python -m examples.flops_balancing.collect_run \
  --address "$RAY_ADDRESS" --job-id actual-job-id --run-dir "$RUN_DIR"
```

训练结束后执行 collect_run。`submit.log` 只是提交状态，`training.log` 才是原始训练日志。capture 环境变量被显式传入 Ray actor，仅导出 rollout=0、step=0。它读取原本一次 prepare_grads 之后的梯度，在原本一次 step 后算 Δθ，不重复 prepare/unscale/step。offload 用 HybridDeviceOptimizer 实际 FP32 主权重映射，不支持的精度感知/FSDP 实现会报错。

```bash
python -m examples.flops_balancing.compare_step_captures \
  /path/to/results/dense-A1/capture /path/to/results/dense-D/capture \
  --loss-atol 1e-5 --loss-rtol 1e-3 \
  --grad-atol 1e-5 --grad-rtol 1e-3 \
  --update-atol 1e-7 --update-rtol 1e-3 \
  --output /path/to/results/dense-A1-vs-D.json
```

A2/B/C/D 分别与 A1 比较。MoE 改 `TASK16_MODEL=moe` 和相应 HF/数据目录重做。原生验证覆盖实际启用的 TP2/PP2/DP2，以及 MoE EP2/ETP1；单卡或 CPU 元数据替身测试不能算这些组合已验证。预打包另受原有 PP=CP=VPP=1 等限制，正式使用时还需单独实测。

## 5. 训练循环吞吐

关闭导出：`TASK16_CAPTURE=0`，确保 runtime env 没有 `RELAX_FLOPS_CAPTURE_DIR`。带同步/磁盘导出的运行不能参与性能比较。

保持 `NUM_ROLLOUT=30`、global batch=64；预先约定丢弃前 5 个 rollout，测接下来的 20 个。每个模型 A 与候选各至少 3 次独立运行；先短跑 B/C/D 筛查，再补有希望方案的重复，A 与候选交错跑，观察 GPU/主机负载和显存。Dense/MoE 都要保留对照，即使某个无收益。

```bash
python -m examples.flops_balancing.summarize_runs \
  /path/to/results/dense-A1-perf /path/to/results/dense-A2-perf /path/to/results/dense-A3-perf \
  /path/to/results/dense-D1-perf /path/to/results/dense-D2-perf /path/to/results/dense-D3-perf \
  --warmup 5 --steps 20 --output /path/to/results/dense-throughput.json
```

主指标 `sum(真实非 padding 输入 token) / sum(train_wait + train)` 包含供数等待、调度、通信和优化器更新，另报监督 token 吞吐。dummy 不计入真实 rollout token。口径取现有生产计时器，不包含启动/编译/checkpoint，需和原始日志一起说明。先算各次总 token/总时间，再比较中位数及 min/max/std，不能简单平均逐步 token/s。

汇总器检查固定参数、数据、代码、原生源码、依赖和设备记录；拒绝诊断/预览/失败任务、重复目录、缺步和 token 数不一致。10%/20% 标记只来自真实日志。达到阈值仍需结合基线波动、微批数、等待和显存解释收益来源。
