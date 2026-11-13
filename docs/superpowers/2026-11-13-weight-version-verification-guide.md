# 权重版本追踪与滞后审计 — 验证指南

日期：2026-11-13
分支：`feat/weight-version-tracking`（基于 `main` @ 78d704e）
适用对象：评审人或合并前自查。本文按"5 分钟冒烟 → 30 分钟完整验证"分层，命令均可直接复制；预期输出均为本机真实执行结果。

> 前置说明：本机 `/opt/miniconda3/bin/python` 下 pytest 在**通过时也可能返回退出码 255**（机器怪癖），判断结果以 stdout 的 `N passed` 为准；部分 `ls` 类命令同样如此。

## 一、5 分钟冒烟（合并前最少验证）

### 1.1 跑本特性全部测试（34 个，约 2 秒）

```bash
cd /home/nianliu/code/xhs/Relax
/opt/miniconda3/bin/python -m pytest \
  tests/cpu/test_weight_version_scenarios.py \
  tests/cpu/test_weight_version_state.py \
  tests/cpu/test_weight_version_service.py \
  tests/cpu/test_weight_version_observability.py \
  tests/cpu/test_weight_version_train_hooks.py \
  tests/cpu/utils/test_version_audit.py \
  tests/cpu/utils/test_weight_version_types.py \
  -q
```

预期末行：

```text
34 passed, 32 warnings in 2.29s
```

34 个测试即验收标准 1-6 的全部证据（对应关系见[收尾记录](./2026-11-13-weight-version-tracking-completion.md)的证据表）。

### 1.2 验证 CLI 本身能跑（不需要 GPU / 集群）

见下文 §3 的手动演练，2 分钟即可完成，输出可直接与预期逐字节比对。

## 二、完整验证（约 30 分钟）

### 2.1 全量回归对比（证明旁路性，验收标准 5）

核心问题：改动是否影响既有行为？方法是与基线（合并前的 `main`）对比。

**第一步：跑本分支全量 CPU 套件**（约 15 分钟，本机缺 megatron/transformers/transfer_queue 依赖，须加 `--continue-on-collection-errors`）：

```bash
/opt/miniconda3/bin/python -m pytest tests/cpu --continue-on-collection-errors -q 2>&1 | tail -1
```

预期末行：

```text
391 failed, 5814 passed, 965 skipped, 246 errors in ...
```

**第二步：跑 main 基线并对比**

```bash
git stash -u && git checkout main
/opt/miniconda3/bin/python -m pytest tests/cpu --continue-on-collection-errors -q 2>&1 | tail -1
git checkout feat/weight-version-tracking && git stash pop
```

基线预期：

```text
391 failed, 5811 passed, 965 skipped, 246 errors
```

**判读**：两行逐位对比，`failed`/`errors`/`skipped` 完全一致，`passed` 恰好 +3（新增的 3 个场景测试）。既有 391 个失败全部是环境性的（可选依赖缺失、huggingface-hub 版本 pin），与分支无关。delta 不为 +3 或出现新增失败 → 不通过。

> 注意：`git stash -u` 会把未提交工作全部暂存；如工作树不干净，先确认无未保存内容。另有一条更保险的路：`git worktree add ../relax-baseline main`，在独立目录跑基线，不动当前工作区。

### 2.2 分文件明细（可选，定位问题时用）

```bash
for f in test_weight_version_scenarios test_weight_version_state test_weight_version_service \
         test_weight_version_observability test_weight_version_train_hooks; do
  /opt/miniconda3/bin/python -m pytest tests/cpu/$f.py -q 2>&1 | tail -1 | sed "s/^/$f: /"
done
/opt/miniconda3/bin/python -m pytest tests/cpu/utils/test_version_audit.py tests/cpu/utils/test_weight_version_types.py -q 2>&1 | tail -1 | sed 's/^/utils: /'
```

预期（单文件实测）：scenarios 3、state 4、service 4、train_hooks 2、observability 1、version_audit 16、types 4；另有 span 相关断言分布在既有套件 `tests/cpu/test_agentic_observability.py`（合并跑 122 passed）。

### 2.3 代码规范

```bash
/opt/miniconda3/bin/ruff check relax/utils/agentic_obs/version_audit.py relax/agentic/session/state.py \
  relax/agentic/session/service.py relax/agentic/observability.py relax/utils/types.py \
  relax/utils/agentic_obs/records.py tests/cpu/test_weight_version_scenarios.py \
  tests/cpu/utils/test_version_audit.py
/opt/miniconda3/bin/ruff format --check <同样的文件列表>
```

预期：`check` 无输出（退出码 0），`format --check` 报告全部 `would keep unchanged`。完整 20 个分支改动文件的清单见收尾记录。pre-commit 本机未安装，ruff 等效覆盖（CI 会跑完整 pre-commit）。

## 三、CLI 手动演练（真实数据流走一遍）

`version_audit` 是纯离线工具：读观测目录里的 JSONL，不碰 GPU。下面手工构造一个最小 run 目录，2 分钟走完整个数据流，预期输出经过真实执行核对。

### 3.1 构造样例数据

场景：两个样本。样本 1001 跨两轮生成（第 1 轮 2 token 来自 v1，工具等待期间权重更新到 v3，第 2 轮 3 token 来自 v3），消费时 current=5；样本 1002 引擎未上报版本（null），验证未知版本显式呈现。

```bash
rm -rf /tmp/wv_verify_demo && mkdir -p /tmp/wv_verify_demo/run-1/trajectories /tmp/wv_verify_demo/run-1/train

cat > /tmp/wv_verify_demo/run-1/trajectories/shard_00.jsonl <<'EOF'
{"schema_version": "3.2.0", "run_id": "run-1", "session_id": "sess-1", "group_id": "g-1", "sample_export_id": "1001", "lineage": [{"kind": "obs", "state_hash": "h0", "token_count": 2, "weight_versions": [], "weight_version_spans": []}, {"kind": "resp", "state_hash": "h1", "token_count": 2, "weight_versions": ["1"], "weight_version_spans": [{"version": "1", "token_start": 0, "token_end": 2, "attempt_seq": 0}]}, {"kind": "obs", "state_hash": "h2", "token_count": 4, "weight_versions": [], "weight_version_spans": []}, {"kind": "resp", "state_hash": "h3", "token_count": 3, "weight_versions": ["3"], "weight_version_spans": [{"version": "3", "token_start": 0, "token_end": 3, "attempt_seq": 0}]}]}
{"schema_version": "3.2.0", "run_id": "run-1", "session_id": "sess-2", "group_id": "g-1", "sample_export_id": "1002", "lineage": [{"kind": "obs", "state_hash": "h0", "token_count": 2, "weight_versions": [], "weight_version_spans": []}, {"kind": "resp", "state_hash": "h1", "token_count": 2, "weight_versions": [null], "weight_version_spans": [{"version": null, "token_start": 0, "token_end": 2, "attempt_seq": 0}]}]}
EOF

cat > /tmp/wv_verify_demo/run-1/train/step_00000003_dp_000.jsonl <<'EOF'
{"schema_version": "3.2.0", "run_id": "run-1", "update_id": "actor:3:2", "dp_rank": 0, "samples": [{"sample_export_id": "1001", "consumed_weight_version": "5"}, {"sample_export_id": "1002", "consumed_weight_version": "5"}]}
EOF
```

### 3.2 跑审计报告

```bash
/opt/miniconda3/bin/python -m relax.utils.agentic_obs.version_audit /tmp/wv_verify_demo/run-1
```

预期输出（逐字节）：

```json
{
  "sample_count": 2,
  "lag_sample_count": 1,
  "oldest_lag_mean": 4.0,
  "oldest_lag_median": 4,
  "oldest_lag_max": 4,
  "newest_lag_mean": 2.0,
  "newest_lag_median": 2,
  "newest_lag_max": 2,
  "token_weighted_lag_mean": 2.8,
  "generation_span_mean": 2.0,
  "span_missing_rate": 0.3333333333333333,
  "token_missing_rate": 0.2857142857142857,
  "sample_missing_rate": 0.5
}
```

```text
sample=1001 session=sess-1 turn=1 attempt=0 tokens=[0,2) version=1 age=4
sample=1001 session=sess-1 turn=2 attempt=0 tokens=[0,3) version=3 age=2
sample=1002 session=sess-2 turn=1 attempt=0 tokens=[0,2) version=None age=None
```

**手工核对口径**（验证标准 4 的口径区分，数值是算出来的不是背的）：

| 指标 | 手算 | 命中行 |
| --- | --- | --- |
| generation_span | 3 − 1 = 2 | `generation_span_mean: 2.0` |
| oldest_lag（样本 1001） | 5 − 1 = 4 | `oldest_lag_max: 4`、明细 `age=4` |
| newest_lag / post_generation_lag | 5 − 3 = 2 | `newest_lag_max: 2`、明细 `age=2` |
| token_weighted_lag | (4×2 + 2×3) / 5 = 2.8 | `token_weighted_lag_mean: 2.8` |
| span_missing_rate | 1 未知 / 3 总 span | `0.3333...` |
| token_missing_rate | 2 未知 / 7 总 token | `0.2857...` |
| sample_missing_rate | 1 全未知 / 2 样本 | `0.5` |
| `lag_sample_count` | 仅样本 1001 有已知版本参与 lag | `1`（不是 2） |

**未知版本检查点（验收标准 3）**：样本 1002 的版本是 `null`，明细行显示 `version=None age=None`；它**不进**任何 lag 均值（`lag_sample_count=1` 证明），也不被当成 0 或 5 猜测——只通过三个缺失率出现。

### 3.3 过滤与 JSONL 模式

```bash
# 只看有未知版本/未知 token 的样本，一行 JSON 明细
/opt/miniconda3/bin/python -m relax.utils.agentic_obs.version_audit /tmp/wv_verify_demo/run-1 --missing-only --jsonl
```

预期：仅 1 行，`sample_export_id` 为 `1002`，且其 `oldest_lag` 等 lag 字段全为 `null`、`span_missing_rate: 1.0`。

```bash
# 只审计某个 session
/opt/miniconda3/bin/python -m relax.utils.agentic_obs.version_audit /tmp/wv_verify_demo/run-1 --session sess-1
```

预期：汇总块 `sample_count: 1`（只剩样本 1001），`span_missing_rate: 0.0`。

### 3.4 容错行为抽查（可选）

撕裂尾行（模拟进程崩溃留半行 JSON）：

```bash
printf '{"schema_version": "3.2.0", "run_i' >> /tmp/wv_verify_demo/run-1/trajectories/shard_00.jsonl
/opt/miniconda3/bin/python -m relax.utils.agentic_obs.version_audit /tmp/wv_verify_demo/run-1 2>&1 | head -3
```

预期：打一条 `version_audit: skipping malformed line ...shard_00.jsonl:3` 告警日志，报告照常输出、数值与 §3.2 完全一致。非目录入参则直接 `parser.error` 退出。

## 四、真实训练 run 的审计（有 GPU 环境时）

以上是离线验证。在线链路（生成期 span 采集 → 训练消费戳写入）需真实训练环境，两种方式：

1. **跑一次带 agentic 观测的小型训练**（配置见 [Agentic Rollout](../../zh/guide/agentic-rollout.md) 文档的 dump 开关），结束后：

   ```bash
   python -m relax.utils.agentic_obs.version_audit <dump_dir>/<run_id>
   ```

   检查点：`span_missing_rate` 接近 0（引擎正常上报版本）；多轮 session 的明细行能看到不同 turn 的 version 不同（异步重叠 update_weights 时）。

2. **GPU/完整依赖环境补测**：`tests/cpu/backends/megatron/test_weight_version_helper.py`（actor 侧版本采集，本机因缺 megatron 跳过）与 DCS 模式的 `DeviceDirectBackend.weight_version` 链路——这是本机验证的已知盲区，需在 CI 或 GPU 机器上确认。

## 五、验证结论判读

| 项 | 通过标准 |
| --- | --- |
| §1.1 特性测试 | `34 passed`，零失败 |
| §2.1 回归对比 | 与 main 基线逐位一致，delta 恰为 +3 passed，零新增失败/错误 |
| §2.3 ruff | check 无告警，format 全部 unchanged |
| §3.2 CLI 演练 | 汇总与明细输出与本文逐字节一致，8 项口径手算全命中 |
| §3.2 未知版本 | None 不进均值、不被猜测、只进缺失率 |
| §4 GPU 环境 | test_weight_version_helper 通过 + 真实 run 的 missing_rate 合理 |

全绿即满足 task.txt【No.15】验收标准 1-6 的可验证闭环。
