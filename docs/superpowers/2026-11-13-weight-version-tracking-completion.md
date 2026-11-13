# 权重版本追踪与滞后审计 — 工作收尾记录

日期：2026-11-13
仓库：Relax（`/home/nianliu/code/xhs/Relax`）
分支：`feat/weight-version-tracking`（基于 `main` @ 78d704e，15 个 commit，未推送）
任务来源：`/home/nianliu/code/xhs/task.txt`【No.15】
关联文档：
- 设计：`docs/superpowers/specs/2026-11-10-weight-version-tracking-design.md`
- 实施计划：`docs/superpowers/plans/2026-11-10-weight-version-tracking.md`
- 报告样例：`docs/superpowers/examples/weight-version-audit-report.md`

## 状态：实现完成，待合并

所有 8 个计划任务（任务 0-8）已完成并逐任务通过 spec 合规 + 代码质量两阶段评审；整体实现通过最终代码评审（verdict: APPROVED, ready to merge）。分支未推送、未合并——按约定由用户自行合入远程。

## 提交清单

| Commit | 说明 |
| --- | --- |
| d53d27e | feat(agentic): WeightVersionSpan 类型 + `Sample.weight_version_spans` |
| 33583fc | feat(agentic): spans 贯穿 IR / MsgNode / append_resp / build_sample |
| aa998c1 | test(agentic): 修正 state 测试的过时坐标注释并关闭事件循环 |
| 3d4c8d2 | feat(agentic): 在生成 attempt 边界采集 spans（service.py） |
| 553dc8e | feat(obs): spans 快照进 trajectory 流（schema 3.1.0） |
| 2750909 | test(obs): ruff format + None 版本 round-trip 覆盖 |
| e6bb24e | feat(train): 训练消费版本 `consumed_weight_version` 写入 train 流 |
| 8a9b978 | test(obs): isolation fake lineage node 补 spans 默认值 |
| ec85898 | fix(train): 按模式取消费版本，schema 升 3.2.0 |
| db92bc6 | feat(obs): version_audit 核心统计 |
| 2bfa8b5 | feat(obs): version_audit CLI（trajectories × train 流 join） |
| b2bd6fc | fix(obs): 加固流 join 与明细输出（overflow 分片、零除、CLI 校验） |
| 75841de | fix(obs): 容忍 JSONL 撕裂的多字节尾行 |
| 9a6f761 | feat(obs): 验收场景测试 + 报告样例 + zh/en 文档 |
| 069d5a6 | docs(obs): 报告样例改用节点内相对坐标 |

变更范围：23 个文件（+2124/−17）。核心实现 9 个（`relax/utils/types.py`、`relax/agentic/session/state.py`、`relax/agentic/session/service.py`、`relax/agentic/observability.py`、`relax/utils/agentic_obs/records.py`、`train_hooks.py`、`version_audit.py`（新）、`relax/backends/megatron/actor.py`、`model.py`），测试 11 个，文档 3 个。

## 验收标准证据表

| 验收标准 | 证据 |
| --- | --- |
| 1 跨版本多轮 / 暂停恢复 / 缺失版本 / 取消重试 | `test_weight_version_service.py::test_apply_generate_result_records_span_per_attempt`（多 attempt）、`::test_spans_survive_abort_requeue_and_commit`（abort 恢复）、`::test_apply_generate_result_records_unknown_version_as_none`（缺失）；`test_weight_version_scenarios.py::test_cancelled_ir_spans_never_reach_any_stream`（取消） |
| 2 token 区间只对应成功提交内容 | 采集点在 attempt 边界（service.py `_apply_generate_result`），spans 仅经 `_terminal_response_locked → append_resp` 入 node；IR 丢弃即消失，scenario 测试显式断言所有非提交 node spans 为空 |
| 3 未知版本显式呈现 + 缺失率 | `version=None` 全链路保留（JSON null）；`version_audit.py` 三口径缺失率（span/token/sample）；`test_version_audit.py::test_all_unknown_reports_null_not_zero` 等 |
| 4 版本年龄按发布序号、口径区分 | 单一序号源：`weight_updater.weight_version`（colocate/hybrid）与 DCS `DeviceDirectBackend.weight_version`；`test_metric_distinction_generation_span_vs_consumption_vs_waiting` |
| 5 旁路审计不侵入 | service.py 仅在既有异常护栏内累加数据；actor/model/train_hooks 为加法式 kwarg 透传；回归 delta = +3 passed，零新增失败 |
| 6 实现 + 测试 + 可复现示例 | `test_acceptance_6_tool_wait_two_weight_updates_reproducible_example` + `docs/superpowers/examples/weight-version-audit-report.md`（数值经真实 CLI 逐字节核对） |

## 最终验证结果（2026-11-13）

- 全量回归：`pytest tests/cpu --continue-on-collection-errors` → **391 failed / 5814 passed / 965 skipped / 246 errors**；基线（合并前 stash 对比）为 391/5811/965/246 —— **delta = +3 passed（新增场景测试），零新增失败/错误**。既有失败均为环境性（megatron/transformers/transfer_queue 缺失、huggingface-hub 版本 pin），与本次改动无关。
- 权重版本相关测试文件全绿：state 4、service 4、observability 7(+1 skip)、scenarios 3、version_audit 16、train_hooks+helper+types 7、扩展的既有 observability 套件 200。
- Ruff check + format：20 个分支改动 `.py` 文件全部通过。pre-commit 本机未安装，提交前以 ruff 等效覆盖。
- Megatron actor 侧逻辑（`test_weight_version_helper.py`）因本机无法 import megatron 按测试自身声明跳过——需在 GPU/完整依赖 CI 上复核。

## 设计要点速查（合并后维护时先读）

1. **坐标系双语义**（评审 Minor，已三处文档化）：`weight_version_spans` 在 MsgNode/LineageNode 上是 **resp 节点内相对坐标**（每轮从 0 起算），在 `Sample` 上经 `build_sample` 展平为 **整条 response 连续坐标**。两者从不在同一流中混用；明细报告打印的是前者。
2. **未知版本**：`version: None` 显式呈现，不进 lag 均值分子分母；非数字版本字符串按未知处理（`_version_or_none`），绝不按 0 或最新版本猜测。
3. **旧字段并存**：`Sample.weight_versions`（attempt 级）按用户决策保留，无消费者；新链路统一走 `weight_version_spans`。
4. **版本序号源**按模式分：colocate/hybrid 走 `weight_updater.weight_version`，DCS 走 `DeviceDirectBackend.weight_version`（`actor.py::_consumed_weight_version` 统一封装）；多 policy 各自独立序号。
5. **JSONL 撕裂尾行**：writer 契约允许进程崩溃留半行；`version_audit` 以二进制读取 + 单一 guard 同时覆盖 JSON 解析失败与多字节截断（writer `ensure_ascii=False` 使 CJK 截断常见）。
6. **overflow train 分片**：背压写入重命名为 `overflow_step_*`，是同 (rollout_id, dp_rank) 的**独立** update；audit 按数值序 (rollout_id, dp_rank, is_overflow) 排序，避免字典序导致旧戳覆盖新戳。
7. schema `3.2.0`：3.1 = lineage spans 快照，3.2 = train 样本 `consumed_weight_version`。

## 遗留事项（均非阻塞）

- 评审 Minor 未处理项：scenario 测试 fixture 与 `test_weight_version_service.py` 约 110 行同构复制（本仓库 tests/cpu 无共享 conftest 惯例，第三个消费者出现时再抽取）；`test_metric_distinction` 的链式 `!=` 仅断言相邻不等（数值已被逐项 pin 死，无实际风险）；取消测试走数据不变量而非生产取消路径（CPU 验收测试的合理取舍）。
- 用户指南的审计小节链接了 `docs/superpowers/specs/...` §6（仓库内路径，VitePress 站点不可导航）——特性文档 `docs/{zh,en}/guide/weight-version-audit.md` 已补独立口径页并互链（见下），后续可考虑把指南内链改为指向特性页。
- GPU/完整依赖环境需复核：`tests/cpu/backends/megatron/test_weight_version_helper.py`（本机 skip）与 DCS 版本链路的端到端行为。

## 合并指引

分支 `feat/weight-version-tracking` 已就绪（工作树 clean，HEAD = 069d5a6）。合并方式由用户决定（本地 merge / push + PR）；按约定不代为推送。
