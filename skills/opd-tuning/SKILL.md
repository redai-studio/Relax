---
name: opd-tuning
description: Tune Relax On-Policy Distillation (OPD/MOPD) runs — teacher SGLang engine knobs, logits-memory limits, student/teacher GPU split, and multimodal caches. Use when a user asks to speed up or debug an OPD run, mentions "OPD", "MOPD", "on-policy distillation", "蒸馏", a teacher engine OOM, teacher timeouts, idle student/teacher GPUs, or any `--teacher-sglang-*` / `--opd-*` flag. Produces a diagnosis report with cited flags, measured evidence, and concrete fixes; applies the edits only when asked.
argument-hint: <path-to-opd-launch-script> [已观察到的现象]
---

# opd-tuning

诊断并调优 Relax 的 On-Policy Distillation 训练（`--use-opd --opd-type sglang`），
覆盖 **teacher engine 参数**、**logits 显存**、**student/teacher 分卡**、**多模态缓存** 四块。

**边界**：actor 训练侧的通用性能/显存问题（TP/PP/CP、recompute、optimizer offload、
`--max-tokens-per-gpu`）归 `perf-doctor`；hang 归 `debug-hang`。本 skill 只管 OPD 特有的东西。

## 使用方式

```
/opd-tuning examples/on_policy_distillation/mopd/run-mopd-qwen35-9b-8xgpu-colocate.sh
/opd-tuning <script> teacher OOM 在 logits[input_logprob_indices]
/opd-tuning <script> teacher 卡满载、rollout 卡全程 0%
```

第二个参数可选：已观察到的现象。给了就**从现象倒推**，不要从头遍历规则。

## 执行步骤

1. **读脚本**，抽出 OPD 上下文：
   - 拓扑：`--resource` 的 `actor`/`rollout`/`teacher` 三项、`--rollout-num-gpus`、
     `--teacher-num-gpus-per-engine`；推导 `replicas = teacher_gpus / num_teachers / TP`
   - 教师形态：`--teacher-hf-checkpoint`（单）vs `--opd-teacher-routes`（MOPD 多教师，二选一）
   - 全部 `--teacher-sglang-*` 与 `--opd-*` flag
   - env 块：logits chunk 开关与 chunk size、VLM cache、OPD patch 开关、
     `RELAX_PROPAGATE_ENV_VARS`，以及提交方式（直接 `python3 -m` 还是 `ray job submit`）
   - 模态与规模：`--multimodal-keys`、`--image-max-token-num`、`--rollout-max-prompt-len`、
     `--rollout-max-response-len`、`--rollout-batch-size` × `--n-samples-per-prompt`
   - 学生侧对照：`--rollout-num-gpus-per-engine`、`--sglang-mem-fraction-static`
2. **拿硬件事实，别假设**：单卡显存从 `nvidia-smi --query-gpu=memory.total --format=csv`
   取；拿不到就问用户，并在报告 Context 行里显式标注 "(assumed)"
3. **拿模型事实，别猜**：从 student / teacher 的 `config.json` 取 `vocab_size`、
   `num_hidden_layers`、`num_key_value_heads`、`head_dim`（多模态模型在 `text_config` 下），
   多图场景还要 `vision_config.out_hidden_size`。这些是 R-T01/R-T02/R-T08 全部算式的输入。
   路径不可达就问，**不要用经验值顶替**
4. **读规则**：`references/teacher-knobs.md`（R-T01..R-T10）逐条判 applies / borderline / N/A。
   该文件开头有三条**自验证配方**（定位运行时 sglang、构建真 parser 查 flag、查 env 默认值）——
   凡是要断言"某 flag 存在 / 默认值是 X"，先跑配方，别凭记忆
5. **判是否失衡**：读 `references/balance.md`。有现象描述或用户给了日志/指标就实际算；
   没有就只输出「怎么量」，不要拍脑袋下结论
6. **对锚点**：`references/baselines.md` 有同量级配置就作为合理区间。
   该文件是快照，用之前先按其开头的命令刷新
7. **出报告**，按下方模板

## 决策顺序（不能乱）

teacher 是 `max_new_tokens=0` 的纯 prefill，由此推出的链条有依赖关系：

```
1. 开 logits chunking          ← 是 2 的前置（仅当 2 会让 logits 峰值超预算时才必须）
2. 拉大 chunked-prefill-size / max-prefill-tokens
3. 压低 teacher mem-fraction-static（把显存让给 logits / 激活）
4. 关 decode CUDA graph（prefill graph 别盲关）
5. 多图 VLM 才动 VLM cache 上限
6. 以上都做完还失衡，才谈重新分卡
```

**先做 2 再做 1 = OOM。** 先做 6 再做 1-5 = 在没优化的基线上重新分卡，白折腾。

## 触发判断原则

- **不要机械触发**。规则命中 ≠ 值得改。命中后必须回答"改了省多少"——
  算出来接近零就降级到 info 并明说，别为了凑 finding 数量硬推
- **脚本里的 `# NOTE` / 推导注释是有效理由**，降级或跳过；但注释也会过时，
  与变量默认值冲突时以默认值为准
- **数字要算不要抄**。三个核心算式（`teacher-knobs.md` 里有完整形式）：
  - logits 峰值 = `N_positions × vocab/TP × 4B`
  - KV 需求 = `并发数 × 单请求 token × per-token KV`，
    `per-token KV = 2 × layers × kv_heads × head_dim × dtype_bytes`（`kv_heads < TP` 时复制不切分）
  - 单图 embedding = `image_token_num × vision_out_hidden_size × dtype_bytes`
- **不要把"我记得的行为"当现状**。凡涉及 sglang flag 名/默认值、或 Relax 内部函数的行为
  （例如 teacher 副本路由是否有 group 亲和），**先跑自验证配方或读那个符号**，
  再决定要不要写进报告。仓库里的分析文档（`docs/draft/**`）是时间点快照，同样要复核
- **单步数据不能下结论**：多图 OPD 单步噪声可以很大，至少 5 步（`references/balance.md` §1）

## 输出模板

```markdown
# opd-tuning: <script-name>

**Context:** student=<X> → teacher=<Y> · <单教师|MOPD N教师> · GPUs=<G> (actor <a> / rollout <r> / teacher <t>)
· teacher TP=<tp> × <n> replicas · <colocate|fully-async> · <text|multimodal> · vocab=<V> · <单卡显存,标注是否 assumed>

---

## 🔥 Findings

### [CRITICAL|WARN|INFO] R-T0X — <短标题>
- **Setting:** `--flag value` 或 `env VAR 未设置`
- **Evidence:** <算出来的数 / 引用的实测 / 代码符号名>
- **Cost:** <显存 GiB 或 时间 %；算不出就写"未量化">
- **Fix:** <可直接照抄的改动>
- **Skip if:** <什么情况下现状反而是对的>

(按依赖顺序排，不是按严重度排)

## ⚖️ Balance

<两侧耗时比与空闲占比；数据不足时只写"怎么量"，列出 references/balance.md §1 的三种方法>

## 📋 建议改动（按顺序）

1. `<改动>` — <一句理由>

## Summary
- Critical: N · Warn: N · Info: N
- **Top action:** <最该改的一条>
- **怎么验证:** <跑几步、看哪个 metric、噪声带多少>
```

## 应用改动时

默认**只出报告不改脚本**。用户明确要求应用时：

- 最小改动，只动直接相关的行
- 沿用脚本已有的 `"${VAR:-default}"` 惯用法，不要写死
- 每处非显然的改动补一行注释说明**为什么**（对齐仓库现有 OPD 脚本的注释风格）
- **动 env 变量必须把传播链路走完**：`export` + 加进 `RELAX_PROPAGATE_ENV_VARS`；
  走 `ray job submit` 的脚本还要并进 runtime-env JSON（R-T09，漏了是静默无效）
- 不要顺手改学生侧 / 训练侧参数，那是 `perf-doctor` 的地盘

## 严禁

- ❌ 编造 flag 或 env 名。新版 sglang 的 flag 由 `ServerArgs` dataclass 自动生成，
  grep `"--flag-name"` 会**假阴性**；Relax 侧的 `--teacher-sglang-*` 还有一张跳过名单。
  一律用 `teacher-knobs.md` 开头的配方 ②/③ 实测，或直接说"没核实"
- ❌ 核对 sglang 行为时对着任意一份源码目录。镜像里可能存在多份不同版本的副本 ——
  只认 `python3 -c "import sglang, os; print(os.path.dirname(sglang.__file__))"` 的输出
- ❌ 承诺不存在的 metric。**teacher 侧没有任何耗时/失败 metric**，
  失败只在日志里；overlap ratio 之类的指标也不存在
- ❌ 按 sglang 日志行求和算指标 —— Ray 会折叠大量重复行成 `[repeated Nx]`
- ❌ 把行号写进结论。引用代码给**文件 + 符号名**，让读者自己定位

## References

- `references/teacher-knobs.md` — R-T01..R-T10 规则目录；开头是三条自验证配方，
  然后是因果链和每条规则。新规则直接追加，无需改本文件
- `references/balance.md` — student/teacher 产能平衡：怎么量、分卡公式与整除约束、
  「student offload 后原地起 teacher」为什么现在做不了
- `references/baselines.md` — OPD 示例脚本的配置快照 + 刷新命令 + 「明显没调过」的判断模式
