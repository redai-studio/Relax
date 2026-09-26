# Task 4 — GenRM 弹性扩缩容演示与证据

GenRM 弹性扩缩容（PR #370 / RFC #351）的契约演示、GPU E2E driver 与机器判定证据。English mirror: [`README.md`](README.md).

## 契约演示（CPU，无需 GPU/Ray）

在本目录执行：

```bash
python -m unittest -v test_contract_demo.py
python contract_demo.py
python render_demo.py results/contract-demo.json --output results/contract-demo.html
```

用浏览器打开 `results/contract-demo.html`（GitHub 的 `blob` 视图只显示源码；请下载文件或使用本地 checkout 运行）。页面回放四个确定性场景：正常 1→2→1、健康检查失败、排空期间后端仍在处理、清理失败后的显式 reconcile。拖动时间轴可查看路由成员、准入请求、PG 归属与恢复边界。JSON 是页面使用的精确事件轨迹。

演示覆盖请求与生命周期契约，包括未知请求 `404`、绝对目标校验、幂等键重放（带键 `NOOP` 会被记录并在容量变化后逐字重放，绝不执行新操作）、键/请求体不匹配或未决清理时的 `409`，以及显式 retry/reconcile 路径。引擎、worker、健康检查、准入、后端空闲与 PG 均为内存 fake；`workers` 字段是契约信号而非真实 worker 进程。演示不含 Ray、SGLang、GPU、Autoscaler 或真实打分，也不证明 Task 3 提供不可取消的排空或全量 worker 清理，更不证明扩缩期间文本训练不中断——这些属于 [RFC #351](https://github.com/redai-studio/Relax/issues/351) 的集成与验收工作。

## GPU E2E drivers

所有 driver 均需单节点空闲 GPU 与 SGLang 运行时；只部署自己拥有的 Ray Serve 应用，并在外层 `finally` 中清理。

- `e2e_genrm_scale.py` —— 持续打分下的手动 `1→2→1`：为扩容 PG 探测物理 GPU，记录各阶段得分、每引擎服务计数、GPU 快照与机器判定。
- `e2e_autoscaler_load.py` —— `LOW→HIGH→STEADY→LOW'` 负载曲线下的 Autoscaler 全周期（按服务 GenRM 阈值）；以 ~1 Hz 采样容量、决策与历史。
- `e2e_autoscaler_preregistered.py` —— 预注册验收轮：冻结阈值与子断言（A1–A8、B1–B3），带流量的 Round A 与真实空闲的 Round B，外加门控 TUI 截图。
- `e2e_failure_injection.py` —— 排空/中止/强杀场景：deadline-abort 触发 fail-closed 停泊排空栅栏，SIGKILL 在飞 victim 后全部请求零丢失重试到初始引擎。
- `e2e_reward_consistency.py` —— 生产 dapo-genrm judge 协议跑固定输入集并按引擎归属：greedy 判定必须逐输入一致；官方采样稳定性一并上报。
- `e2e_sampling_divergence.py` —— 对抗一致性：初始引擎先独跑 N 个随机请求，弹性引擎再以相同 server 种子启动，固定探针集仍须产出一致的逐引擎判定（发散的 RNG 历史）。
- `e2e_train_continuity.py` —— 训练配方的 sidecar 监控：对运行中的 DAPO+GenRM 训练驱动 `scale_out`/`scale_in`，并断言每个可观测窗口（前/之间/后）内训练进度、无 >120 s 停滞、8/8 rollouts 与终容量回到初始引擎。

与监控器配套的训练配方位于 `scripts/training/genrm/`。

## 证据

已记录运行的机器判定摘要位于 `results/`（仅最终 verdict 与冻结的预注册文档）；run→commit 映射、SHA256 pin、复现命令与已知限制见 [`EVIDENCE.md`](EVIDENCE.md)。完整的逐请求日志、事件时间线、图表与失败/被取代的中间轮保留在 `evidence/task4-genrm` 分支（`EVIDENCE.md` 内为不可变 commit 链接）。
