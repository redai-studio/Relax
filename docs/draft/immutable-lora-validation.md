# 不可变 LoRA：2026-09-25 验收记录

本次交付的限定验收配置 **通过**。机器摘要：[summary.json](../../tests/engine/lora/evidence/2026-09-25/summary.json)；设计、容量和失败行为：[RFC](./immutable-lora-publication-redesign-rfc.md)。

## 环境与可复现性

- 验收时仓库基线：`6d78410f577aa98a0671221651de108af371bb4e`，加当时尚未提交的本次改动；机器摘要记录实际源码 SHA-256，交付时归入本地功能提交。
- 两张 NVIDIA RTX 6000 Ada Generation，物理 GPU 2、3；每个受管引擎 TP=1。没有停止已有 GPU 作业。
- 基座：本地 Qwen3-1.7B，BF16；Triton LoRA 与确定性 Triton attention；CUDA graph 和 radix cache 保持开启。
- PyTorch 2.11.0+cu129、SGLang 0.5.17、Ray 2.58.0、Transformers 5.12.1。
- 使用独立加载 A/B 的参考引擎；fixture 固定 seed 1707，rank=8、alpha=16、q_proj/v_proj，B 的 LoRA-B 矩阵取反。
- 运行前固定 logprob/KV `atol=0.001, rtol=0.001`；没有为通过实验放宽容差。
- SGLang patch 在干净 v0.5.17 源码应用成功；54 个补丁文件与实际安装源码逐字一致。

命令：

```bash
python -m tests.engine.lora.acceptance \
  --model /workspace/models/Qwen3-1.7B --gpus 2 3 --deterministic \
  --output /tmp/relax-no7-acceptance-20260925-r2
```

六个任务均使用当前 `/root/Relax` 代码，分别启动隔离的 Ray/Serve 与引擎。总报告为 `status=PASS, full_acceptance=PASS, missing_evidence=[]`。完整任务证据压缩保存于 `tests/engine/lora/evidence/2026-09-25/`，摘要包含 SHA-256；可用 Python `gzip.open(path, "rt")` 读取，不依赖临时目录保存的原始日志。

交付摘要和压缩证据中的内网地址统一替换为 `engine-host-N.invalid`；数值、版本/实例身份、时间和资源计数保持原样。归档 SHA-256 对应脱敏后的交付文件，`raw_report.sha256` 对应本机原始报告。

## 验收结果

| 项目 | 实测结果 |
| --- | --- |
| B 单目标就绪 | 默认仍是 A，期间新会话仍匹配 A |
| B 双目标就绪 | 默认切换 B，新会话匹配 B，旧会话保持 A |
| 单引擎失败、迟到确认、取消、重复发布与内容冲突 | PASS；失败清理与版本状态有原生回执 |
| 旧会话、工具轮次、新会话与 abort/resume | 65 个 Session 数值对照，最大 logprob 绝对误差 0；发布故障任务另有 3 个对照，误差 0 |
| A→B 热缓存、B→A 热缓存 | 每方向 64 个引擎/输入对照，最大 logprob 绝对误差 0 |
| 负对照 | 错误 KV、错误 slot 内容均能被验收检出；fixture 存在数值分离 |
| 容量 2：A 有引用且 B 已发布时发布 C | 507 容量错误，拒绝未到达原生 prepare |
| GPU 使用未完成或 slot clear 未完成 | A 保持 RETIRING，占用不提前释放 |
| A 完全释放后 | 两个引擎各实际卸载 1 次，C 复用释放的物理 slot；B 续写与 C 生成通过 |
| 缺版本 | 明确 ADAPTER_NOT_READY，无基座/最新版回退 |
| CUDA graph | 各目标有实际 replay 证据 |
| 真实导出 | optimizer 更新→现有 DCP checkpoint→PEFT export→快照→双目标发布→实际生成 PASS |

真实训练导出：56 个 tensor 发生更新，最大更新量 0.001007080078125；导出 112 个 tensor 与更新后模型逐 tensor 完全相等。它验证单进程真实更新与现有 FSDP checkpoint/export 路径，不代表完整分布式 FSDP 或 Megatron RL 训练验收。

## 持续流量

每个测量窗口 30 秒，固定 1024-token 请求，两个目标均有旧请求跨越整个发布区间。每窗口内接纳的 8 个请求全部完成，另保存窗口前后所有请求和逐 token 时间记录，避免把被截断的请求排除后声称成功。

| 窗口 | token/s | p50（秒） | p95（秒） | 全局最大进度间隔（秒） | 失败 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 普通 LoRA 参考 | 326.09 | 12.349 | 13.145 | 0.0371 | 0 |
| 受管稳态 A | 320.12 | 12.532 | 13.183 | 0.0498 | 0 |
| 发布 B，保留 A 流量 | 320.79 | 12.454 | 13.126 | 0.0490 | 0 |
| 发布后 A/B 稳态 | 323.59 | 12.213 | 13.123 | 0.0490 | 0 |

发布 API 完成耗时 **0.417 秒**，管理器记录的发布耗时 **0.355 秒**。发布窗口每台引擎最大进度间隔分别为 0.0497/0.0505 秒；全局 pause、flush、强制权重变更审计调用数为 0。版本占用从 1 到 2，原始资源采样保留在 `performance.json.gz`。

测量覆盖预登记门槛：p95 比值≤1.5，吞吐比值≥0.8，最大进度间隔≤2 秒。此短时固定配置实验证明本次负载下的行为，不外推所有模型、并行配置或长期性能。

## 回归与检查

- 从当前仓库运行 `RELAX_SGLANG_SOURCE=/workspace/sglang-no7 python -m pytest ...`：**442 passed、7 skipped**。
- 7 个 skip：6 个 GPU pytest 入口默认不自动启动设备实验（已通过上述显式完整 GPU 命令验证）；1 个已有 LoRAMerge 测试因当前 Megatron-Bridge 不提供该类跳过。
- GPU 布局和权重同步补充回归：67 passed、1 skipped，与前一选择集有重叠，不累计计数。
- `pre-commit run --all-files` 及所有新增文件的 hooks 通过；`git diff --check` 通过。
- 已核对默认切换/首绑定线性化、close-before-bind fence、迟到 ACK、原生请求与 GPU 使用分别排空、未知结果不释放容量。

使用 `python -m pytest`，避免裸 `pytest` 的安装路径优先导入环境中另一份 `/workspace/Relax` checkout。最初该命令混用了旧模块，相关失败和旧计数不作为最终结论。

## 本次发现与修复

1. 本地排队 permit 取消遗留 WAITING：仅删除未授予且不存在 RPC 重放的本地记录，新增取消/容量回归；远端不确定 acquire 仍保留补偿与 fence。
2. 原生回收测试断言落后于实现：会话 GPU fence 完成前必须继续持有请求引用；故障回归按该安全语义校正，未放宽实际数值或资源要求。
3. 本机 HTTP 代理导致引擎直连健康但启动探测阻塞：隔离验收子进程显式绕过代理，新增环境回归。
4. Ray worker 使用独立进程组，启动中断后可能存活：验收子进程携带随机 ownership token，仅清理本次 token 所有的进程，新增 detached-child 和保护无关进程回归。

第一轮实验因代理启动阻塞而中止，清理自身进程后重跑得到本报告；没有把失败轮次的部分结果与第二轮拼成通过。

## 未覆盖范围

多节点 GPU、TP/PP 多 rank、colocate/offload 与扩缩容的真实集成实验未运行：本次只有本机两张显式选定的空闲 GPU，没有多节点集群配置。相关接线和 CPU 测试不能替代该硬件验证。协调器崩溃恢复和永久失联实例强制回收不在实现承诺内；后者保持占用并明确失败。
