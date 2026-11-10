# perf-doctor rule catalog

每条规则按下面格式写。`Skip when` 一定要写 — 它是防 false positive 的关键。

---

## 💾 Memory rules (M-series)

### R-M01 — Optimizer offload trio
- **Category:** memory (省显存 / 严重伤性能)
- **Severity:** warn
- **Trigger:** 同时出现 `--optimizer-cpu-offload` + `--overlap-cpu-optimizer-d2h-h2d` + `--use-precision-aware-optimizer`
- **Why:** 把 optimizer state offload 到 CPU，可让显存预算砍掉一半（如 35B-A3B 能 4×H20 96G 跑），但 step 时间通常 +30~60%、MFU 显著下降
- **Fix:** 估算 `param_count × (fp32_master + adam_m + adam_v) ≈ param_count × 12 bytes`，再加 grad + activation。如果 `(总 GPU 显存 - 估算) > 30%` 余量，三件套全删；先去掉 `--use-precision-aware-optimizer`、再去 `--overlap-cpu-optimizer-d2h-h2d`、最后才考虑保留 `--optimizer-cpu-offload`
- **Skip when:** GPU 数 × 单卡显存 < `param × 12B / EP / TP`（确实不开就 OOM）；或脚本注释明确说"边界 case 不开会 OOM"

### R-M02 — Recompute granularity vs headroom
- **Category:** memory (省显存 / 伤性能 ~10-20%)
- **Severity:** info → warn（看 headroom）
- **Trigger:** `--recompute-granularity full` + `--recompute-method uniform` + `--recompute-num-layers 1`
- **Why:** Full recompute 每层重算前向，吃约 10-20% step 时间。在显存充裕场景属于浪费
- **Fix:** 估算 activation memory ≈ `batch × seq × hidden × layers / TP / CP`，若 < 单卡显存的 40%，改成 `--recompute-granularity selective` 或干脆关掉
- **Skip when:** 长 context (>16K) 训练、单机训不下 dense 大模型、或脚本注释说"关了 OOM"

### R-M03 — `--max-tokens-per-gpu` vs parallelism
- **Category:** memory + performance
- **Severity:** warn
- **Trigger:** `--use-dynamic-batch-size` 开启且 `--max-tokens-per-gpu` 偏离合理区间
- **Why:** 设太低 → micro-batch 太小、kernel 利用率低；设太高 → OOM / 频繁 retry
- **Fix:** 经验值 `max-tokens-per-gpu ≈ rollout-max-response-len × N`，其中 N 由 TP×CP 和可用显存定。H20 96G + TP2 + CP1 + 8K resp-len 常见值 = 20480；TP2 + CP2 = 10240。Cross-check 已工作过的同型脚本
- **Skip when:** 脚本是新模型 / 新场景且作者已手动调过

### R-M04 — `--sglang-mem-fraction-static`
- **Category:** memory (rollout 侧)
- **Severity:** info
- **Trigger:** 值 < 0.65 或 > 0.92
- **Why:** 过低 → KV cache 不足、长序列 OOM；过高 → 留给 weight sync / activation 的余量不够
- **Fix:** 常用 0.7~0.85。MoE 模型 + EP 时偏低端；dense + 单 engine 偏高端
- **Skip when:** weight-sync 阶段已知 OOM，需要更多 headroom

---

## 🚀 Performance rules (P-series)

### R-P01 — `--sglang-load-format dummy` 缺失
- **Category:** performance (启动耗时)
- **Severity:** info
- **Trigger:** SGLANG_ARGS 中没有 `--sglang-load-format dummy`
- **Why:** Relax 用 NCCL broadcast 做 actor→rollout 权重同步，rollout 真正的初始权重永远来自训练侧；让 SGLang 启动时去磁盘读 HF 权重纯属浪费几十秒到几分钟
- **Fix:** 加 `--sglang-load-format dummy`
- **Skip when:** 调试场景需要验证 rollout 单独的权重正确性

### R-P02 — `--use-streaming-dataset` 缺失
- **Category:** performance (数据 stall + 启动)
- **Severity:** warn（多模态）/ info（纯文本）
- **Trigger:** ROLLOUT_ARGS 没有 `--use-streaming-dataset`
- **Why:** 不开会一次性把整个数据集加载进 driver 内存，多模态场景（image / video）经常 OOM 或启动慢到分钟级；流式则随 rollout 边读边消费
- **Fix:** 直接加 `--use-streaming-dataset`。多模态再确认上下游字段对齐
- **Skip when:** 数据集极小（< 1k 条）的快速 debug 脚本

### R-P03 — Fully-async 资源比 / staleness 失衡
- **Category:** performance (吞吐)
- **Severity:** warn
- **Trigger:** `--fully-async` 模式下出现下列之一：
  - actor:rollout GPU 比例与 `--num-iters-per-train-update` 严重不匹配（如 actor 8 / rollout 8 + iters-per-update 32，rollout 大概率追不上）
  - `--max-staleness 0` 但同时设 `--num-iters-per-train-update > 1`（语义矛盾，会回到伪同步）
  - `--num-data-storage-units` 远小于 rollout 数量（transfer queue 阻塞）
- **Why:** 全异步收益来自 actor / rollout 流水线并行；任一侧成瓶颈或语义错配则吞吐反而劣化到 sync 之下
- **Fix:** 计算 rough estimate：单个 rollout step 时间 × iters-per-update ≈ 单个 actor train step 时间。两端差距 > 30% 时调 GPU 配比，或调 `--num-iters-per-train-update`。`max-staleness` 一般 >= 1
- **Skip when:** 脚本注释里说明是在故意做 staleness 实验

### R-P04 — CP 用法
- **Category:** performance
- **Severity:** warn
- **Trigger:**
  - `--context-parallel-size > 1` 且 `--rollout-max-response-len <= 8192`（CP 开销不值）
  - `--context-parallel-size = 1` 且 `--rollout-max-response-len >= 16384` 且单节点（极可能 OOM 或重 recompute）
- **Why:** CP 在层内切 sequence 维做 all-gather，固定 overhead 在短序列上得不偿失；长序列下没 CP 又必须重 recompute / offload
- **Fix:** 短序列：CP=1，把这维并行预算给 TP 或 rollout engine；长序列：CP=2 或 4 起步
- **Skip when:** 用户脚本是有意做 CP scaling 实验

### R-P05 — MoE 未开 EP
- **Category:** performance (MoE 核心优化)
- **Severity:** warn → critical
- **Trigger:** 模型是 MoE（model 名含 A3B / A40B / 总参 ≫ 激活参）且 `--expert-model-parallel-size = 1`
- **Why:** EP 是 MoE 训练最大杠杆，不开等于把所有 expert 复制到每张卡，显存 / 通信都浪费
- **Fix:** `EP = total_experts / N`，常用值 4 或 8。同时 `--expert-tensor-parallel-size 1`（除非显存极紧）
- **Skip when:** 单节点 dense MoE 实验、或 expert 数 = 1 的退化情况

### R-P06 — TP 跨节点
- **Category:** performance (critical)
- **Severity:** critical
- **Trigger:** `--tensor-model-parallel-size > 8`（H20 / H800 单节点 8 卡）或 TP × 节点内其他并行 > 单节点 GPU 数
- **Why:** TP 每层都 all-reduce，跨 NVLink boundary 后走 IB，带宽下降一个数量级；几乎一定显著掉 MFU
- **Fix:** TP <= 8，多出来的并行放到 PP / EP / DP
- **Skip when:** 几乎不存在合理 skip 场景；若用户明确说在做跨节点 TP 调研可降级 warn

### R-P07 — 不必要的 PP
- **Category:** performance
- **Severity:** info
- **Trigger:** `--pipeline-model-parallel-size > 1` 且模型在该 GPU 数 + TP/EP 下可装下
- **Why:** PP 引入 bubble、调度复杂度、显存碎片；只有 dense 大模型装不下时才必要
- **Fix:** 估算 `param × 2 bytes / (TP × EP) < 单卡显存 × 0.6` 时 PP=1
- **Skip when:** 200B+ dense 模型、或长 context 需要纵切 layer

### R-P08 — MoE dispatcher
- **Category:** performance
- **Severity:** info
- **Trigger:** MoE 模型未设 `--moe-token-dispatcher-type flex` + `--moe-flex-dispatcher-backend deepep`
- **Why:** DeepEP dispatcher 在 EP 通信上显著快于 alltoall 默认实现
- **Fix:** 加上两个 flag；老脚本可能尚未迁移
- **Skip when:** 模型 / Megatron 版本不支持 DeepEP

### R-P09 — Attention backend
- **Category:** performance
- **Severity:** warn
- **Trigger:** MISC_ARGS 没有 `--attention-backend flash`（或 `fa3`）
- **Why:** 默认 backend 比 FA2/FA3 慢且更耗显存
- **Fix:** 加 `--attention-backend flash`。注：MLA 模型例外
- **Skip when:** 模型是 MLA / 不兼容 FA

### R-P10 — `--no-rope-fusion` 无注释
- **Category:** performance
- **Severity:** info
- **Trigger:** 纯文本脚本中出现 `--no-rope-fusion` 但附近没有解释性注释
- **Why:** RoPE fusion 默认是性能优化；纯文本场景关闭通常是为绕过某个算法精度问题。无注释意味着可能是"复制粘贴的遗留"
- **Fix:** 确认是否仍需要；不需要就删掉
- **Skip when:**
  - **多模态脚本（image / video / omni）必须关闭 RoPE fusion**，这是正确配置，不要触发
  - 旁边有 `# NOTE(...)` 说明（如 async 35B 文本脚本里"to avoid algorithm performance degradation"）

### R-P11 — 缺 `--use-dynamic-batch-size`
- **Category:** performance
- **Severity:** warn
- **Trigger:** PERF_ARGS 没有 `--use-dynamic-batch-size`
- **Why:** 静态 batch 在变长序列下要按 max 长度 pad，浪费算力 30%+；动态 batch 按 token 数打包
- **Fix:** 加 `--use-dynamic-batch-size` + 合理的 `--max-tokens-per-gpu`（见 R-M03）
- **Skip when:** 数据是定长（如纯多选题 eval）或在调试 batching bug

### R-P12 — sync 模式 SGLang CUDA graph
- **Category:** performance (rollout 吞吐)
- **Severity:** info
- **Trigger:** `--colocate`（sync）模式下 `--sglang-cuda-graph-bs` 被注释或未设
- **Why:** sync 模式 batch size 稳定，CUDA graph 能省每步 launch overhead 5-15%
- **Fix:** 参考 async 脚本：`--sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)`
- **Skip when:** rollout batch 高度动态、或 SGLang 版本 graph 有 bug

### R-P13 — `--balance-data` 使用与模式约束
- **Category:** performance (DP 负载均衡)
- **Severity:** critical（纯 fully-async 误开）/ warn（sync/hybrid 漏开）
- **Trigger:** 满足任一：
  - **(critical)** 出现 `--fully-async` 且**没有** `--hybrid`，但 ARGS 里还有 `--balance-data` —— Relax 启动校验会直接 `ValueError` 退出（见 `relax/utils/arguments.py:2369`）
  - **(warn)** `--colocate` 或 `--hybrid` 模式 + 数据是变长（开了 `--use-dynamic-batch-size` 或 `--rollout-max-response-len >= 4096`）+ DP（= world_size / TP / PP / CP）> 1，但**没**带 `--balance-data`
- **Why:**
  - `--balance-data` 用 SeqlenBalancedSampler（Karmarkar-Karp）按 token 数把 sample 均摊到各 DP rank，消掉 straggler，变长序列大 DP 场景下省 10~30% step 时间
  - **纯 fully-async 模式下 actor 通过 StreamDataLoader 消费 rollout 流，和静态 balance 语义不兼容**，框架直接 raise（错误信息：`--balance-data is not supported in pure fully-async mode`）；想用 balance 必须切到 `--hybrid`
  - 同 prompt 的不同 response 可能被分到不同 step，对纯算法精度影响通常可忽略，但对依赖"同 prompt 同 step"的算法（如某些 group-norm advantage）需要确认
- **Fix:**
  - 纯 fully-async：从 ARGS 里**删掉** `--balance-data`，或同时加 `--hybrid` 切到混合模式
  - sync / hybrid 变长场景：加 `--balance-data`
- **Skip when:** 数据定长（多选题 / 固定长度 eval）；DP=1；算法依赖 "同 prompt response 必须在同一 train step"；脚本注释里明确说明在做 baseline 对照实验
