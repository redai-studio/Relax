---
name: perf-doctor
description: Diagnose Relax training performance and GPU memory from launch scripts or PyTorch profiler traces. Use for config audits, Train Actor profiling setup, trace processing, low GPU utilization, PP bubbles, or compute/communication/load imbalance analysis. Produces evidence-backed findings and concrete verification steps.
argument-hint: <launch-script-or-trace-path>
---

# perf-doctor

诊断 Relax 训练启动脚本（`scripts/training/**/*.sh`），找出影响 **执行性能（耗时 / MFU）** 或 **显存占用（所需卡量）** 的不合理配置；也支持开启 Train Actor profiling、处理 trace，并区分计算、通信等待、数据供给和流水线负载问题。

## 使用方式

```
/perf-doctor scripts/training/text/run-qwen36-35B-A3B-8xgpu.sh
```

参数：启动脚本、原始 `.trace.json.gz` 文件或 trace 目录路径；也可直接询问如何开启 profiling。

## 选择分析入口

- **启动前配置审核**：按下方执行步骤读取脚本、规则和 baseline。
- **开启 Train Actor profile / 分析已有 trace**：先读 [Train Actor profiling 与性能分析](references/train-actor-profiling.md)，核对当前调用点、采样范围和脚本用法。默认使用 `train_overall`；不能只因 argparse 接受 `train_actor` 就认定它已接入训练循环。
- 允许在本地执行标准库 trace 处理和统计；没有 GPU 的机器不运行训练或 CUDA 代码。仅有截图时先给可观察现象，定量结论需要原 trace。
- 配置审核仍使用 Performance / Memory 模板；trace 分析在 Performance 中给出来源、窗口、rank/stream、过滤口径、测量结果和待验证假设。没有显存证据时明确写未测量，不从时间线猜显存。

## 配置审核步骤

1. **读取脚本** — 收集 `*_ARGS=( ... )` 数组与 `ray job submit ... train` 行里的所有 `--flag value`。同时 follow `source ${MODEL_CONFIG_DIR}/...` 拿模型架构（dense / MoE、是否 multimodal）。
2. **抽取 context** —
   - 文件名解析：`run-<model>-<size>-<NxgpuY>(-async|-image|-video)?.sh` → 总 GPU 数、节点数、模式、模态
   - flag 解析：TP/PP/CP/EP/ETP、`--colocate` vs `--fully-async`、`--rollout-max-response-len`、`--max-tokens-per-gpu`、`--resource`、`--num-iters-per-train-update`、`--max-staleness`、`--num-data-storage-units`
   - **默认 GPU 假设：H20 96GB**，除非用户在 prompt 里给出别的（A100 80G / H100 80G 等）
3. **加载规则** — 读 `references/rules.md`，逐条判断 applies / borderline / not-applicable；涉及 MoE 吞吐、EP overlap 或训练侧 CUDA graphs 时，按下方入口加载官方调优参考的对应小节
4. **对照 baseline** — 读 `references/baselines.md`，若用户脚本与某条 baseline 同模型 + 同 GPU 数量级，把 baseline 的并行 / batch / mem 配置作为合理区间锚点；偏离 ≥ 2 档时把 baseline 数值写进对应 finding 的 `Cost` 一栏佐证
5. **输出报告** — 严格按下方 [输出模板](#输出模板) 渲染

## 触发判断原则

- **不要机械触发**：CPU offload 三件套在 35B-A3B 4×H20 这种边界 case 是必需的；规则 `Skip when` 节里写了什么时候它就是对的，要尊重
- **不要 false positive**：脚本里如果有 `# NOTE(...)` 注释解释为什么开 / 关某个 flag，把它当作有效理由，降级到 info 或跳过
- **借助推理而非穷举**：`references/rules.md` 是知识库不是判定表 — 模型大小 × dtype 估算显存预算、TP×CP×PP 是否合理、async 资源比是否平衡，都要 case-by-case 算

## Megatron 官方调优参考

遇到以下问题时，读取 [Megatron 调优参考](references/megatron-performance-skills.md) 的对应小节；其中包含完整 skill 名称、使用示例、官方原文及验证卡链接。

| 问题 | 官方 skill（名称省略 `nemo-mbridge-` 前缀） | 在本 skill 中的用法 |
| --- | --- | --- |
| MoE 吞吐低、扩卡收益差、优化顺序不清楚 | `perf-moe-optimization-workflow` | 整理测量条件、并行布局和瓶颈假设，给出单变量 A/B 验证计划 |
| dispatch/combine 通信开销、EP overlap 与重计算冲突 | `perf-expert-parallel-overlap` | 检查开关、dispatcher、VPP、精度和版本约束；区分 plain overlap 与 delayed wgrad |
| CPU launch 开销、训练 CUDA graph 的 scope 或 replay 问题 | `perf-cuda-graphs` | 检查 graph 范围、shape、RNG、重计算与 overlap 的兼容性；给出 eager/replay 对比方案 |

配置审核应用静态检查；已有 profile 时按 profiling 参考做实测分析。外部 skill 的实验步骤不代表执行训练或 benchmark 的授权。没有 profile 或实测证据时，写“待验证”，不把配置推断写成瓶颈结论。

官方示例属于 Megatron Bridge，不能直接视为 Relax 参数。先核对训练镜像实际的 MCore/TE 版本及 Relax 参数传递路径，再给出修改建议。新 finding 沿用 Performance / Memory 报告格式，在 `Setting` / `Fix` 中引用 Relax 依据和官方参考；上游收益不作为本任务的预计收益。

## 输出模板

```markdown
# perf-doctor: <script-name>

**Context:** model=<X> (<dense|MoE>, <text|mm|video>) · GPUs=<N> (<nodes>×<g/n>) · mode=<colocate|fully-async> · TP<x>/PP<x>/CP<x>/EP<x>/ETP<x> · max-resp-len=<X> · GPU=H20 96GB (assumed)

---

## 🚀 Performance findings

### [WARN] R-P0X — <short title>
- **Setting:** `--flag value`（脚本行号或所在 ARGS 组）
- **Cost:** <估算的 MFU / 耗时影响>
- **Fix:** <可直接照抄的 flag 修改>
- **Skip if:** <什么情况下当前设置反而是对的>

(more findings...)

## 💾 Memory findings

### [WARN] R-M0X — <short title>
- **Setting:** `--flag value`
- **Cost:** <显存影响 / 卡量影响>
- **Fix:** <修改建议>
- **Skip if:** <justified condition>

(more findings...)

---

## Summary
- Critical: N · Warn: N · Info: N
- **Top action:** <一句话最该改的>
```

## 严禁

- ❌ 不要因调用本 skill 自动修改训练配置、提交训练或运行 benchmark；默认提供配置片段和分析建议，用户明确要求实施时按其授权范围执行
- ❌ 不要把离线 trace 分析扩大为集群故障排查或擅自启停任务
- ❌ 不要把 annotation 时长、NCCL 等待时间或单 stream 占用率当成整卡有效计算率
- ❌ 不要建议不在 `references/rules.md` 里、且自己不能给出 Relax-specific 依据的"通用 ML 优化技巧"

## Rule catalog

完整规则在 `references/rules.md`。每条规则字段：`Category` / `Severity` / `Trigger` / `Why` / `Fix` / `Skip when`。新规则直接往该文件追加即可，无需改 SKILL.md。

## Baselines

经过验证的参考配置在 `references/baselines.md`，作为合理区间锚点用。新 baseline 按文件末尾模板追加即可。
