# perf-doctor known-good baselines

经过验证、能正常跑通且性能合理的参考配置。perf-doctor 在诊断时如果用户脚本与某条 baseline **同模型 + 同 GPU 数量级**，应把 baseline 的并行维 / batch / mem 类配置作为 "合理区间锚点"：

- 用户配置与 baseline 差异 ≤ 1 档（如 TP 2→4、CP 4→2）：视为正常调优，不报
- 差异 ≥ 2 档或方向相反（如 baseline EP=16 用户给 EP=1）：作为佐证写进对应 R-PXX finding 的 `Cost` 一栏

baseline 不是硬规则，是 cross-reference；脚本注释明确说在做对照实验则忽略。

---

## Qwen3.5-35B-A3B · 64×80G Hopper · sync (colocate)

| 维度 | 值 |
|---|---|
| GPUs | 64（8 节点 × 8 卡 80G Hopper）|
| Mode | `--colocate`（sync）|
| Model | Qwen3.5-35B-A3B（MoE，总参 35B / 激活 ~3B / 128 experts）|
| TP | 2 |
| PP | 2 |
| CP | 4 |
| EP | 16 |
| ETP | 1 |
| DP | 4（= 64 / (TP·PP·CP)）|
| GBS | 256 |
| max-resp-len | 40960（40K context）|
| max-tokens-per-gpu | 见下注 |
| 关键 flags | `--use-dynamic-batch-size` · `--balance-data` · `--moe-token-dispatcher-type flex` · `--moe-flex-dispatcher-backend deepep` · `--attention-backend flash` · `--sglang-load-format dummy` |

**Why this is balanced:**

- **TP=2**：MoE A3B 激活参数小，TP 大了 all-reduce 占比反而上升；2 够装单层
- **PP=2**：35B 在 TP2·EP16 下单层可装，但 40K context 的 activation 需要纵向再切一刀
- **CP=4**：40K context 必须切 sequence 维（CP=1 时 attention 显存 O(seq²) 爆），10K/CP rank 是 80G Hopper 的舒适区
- **EP=16**：128 experts / 16 = 每 EP rank 8 expert，配合 DeepEP dispatcher 通信成本最低
- **DP=4 + balance-data**：变长 + 大 DP 必开 balance，否则 straggler 拖整批
- **sync 模式**：colocate 复用 64 卡，避免 fully-async 在 80G Hopper 上 actor/rollout 拆分难匹配

**Memory budget（粗算）：**

- Weight: 35B × 2B (bf16) / (TP·EP) = 70G / 32 ≈ **2.2G/卡**（expert）+ dense 部分 / TP·PP ≈ 数 G
- Optimizer (Adam fp32): 35B × 12B / (TP·EP·DP) ≈ **3.3G/卡**（不开 cpu offload）
- Activation: bsz/DP × seq/CP × hidden × layers/PP ≈ **20~40G/卡**（dynamic batch + selective recompute）
- KV cache (SGLang colocate): mem-fraction-static 0.75 → **~50G/卡** 共享
- **结论：** sync 时 weight+optim+act ≤ 50G，剩 30G 给 SGLang 切换够用；不需要 `--optimizer-cpu-offload`

**何时偏离这条 baseline 是合理的：**

- 64×96G Hopper（显存容量高于上述配置）→ 显存更宽，PP 可降到 1；通信带宽不同，CP 可能降到 2
- 改 fully-async → 需要拆 actor/rollout 资源池，参数会重新算
- context 降到 8K 以下 → CP 可以从 4 降到 1

---

## 模板：新增 baseline 时按此填

```markdown
## <Model> · <N>×<GPU type> · <mode>

| 维度 | 值 |
|---|---|
| GPUs | N (nodes × g/n) |
| Mode | colocate / fully-async / hybrid |
| Model | 名称 + dense/MoE + 关键尺寸 |
| TP / PP / CP / EP / ETP / DP | … |
| GBS / max-resp-len | … |
| 关键 flags | 列必备 perf flag |

**Why this is balanced:** 每维度一句话说明
**Memory budget:** 粗算，避免迷信
**何时偏离合理:** 列 1–3 个常见变体场景
```
