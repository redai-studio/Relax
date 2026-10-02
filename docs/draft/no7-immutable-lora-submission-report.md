# 〖No.7〗不可变 LoRA 版本的在线发布、会话绑定与安全回收 — RFC

**提案人：@treasevenzk · 导师：@yuanlehome · 状态：待评审**

**官方任务：No.7「不可变 LoRA 版本的在线发布、会话绑定与安全回收」**

## 摘要

本方案在 Relax 中实现不可变 LoRA 版本的在线发布、Agent 会话绑定及安全回收。

每个 adapter 版本绑定唯一内容摘要。新版只有在全部目标引擎加载、预热并确认驻留后，才原子更新默认版本。会话在首次生成时固定版本，后续工具轮次、重试和 abort/resume 均保持绑定。

回收同时受会话引用、后端在途请求、GPU 使用完成及 slot 清理状态约束。正常发布不触发全局暂停生成、清缓存或强制中断旧会话。

当前实现已完成故障注入回归及真实双 GPU 验收，提交评审。

## 背景与问题

持续训练会不断产生新的 LoRA adapter。如果直接覆盖同名 adapter，可能出现：

- 同一 Agent 会话在工具调用前后使用不同权重，导致上下文对应的策略不一致。
- 新版仅在部分引擎加载成功，却已被新会话使用。
- 客户端取消或 HTTP 返回后，GPU 仍在访问旧权重，而版本已被卸载。
- 底层 LRU 淘汰仍有会话引用的 adapter。
- 不同 adapter 错误复用 KV cache，版本日志正确但实际计算结果错误。
- 发布沿用全局暂停、清缓存的权重更新流程，影响其他会话。

因此，需要分别管理内容身份、发布状态、会话绑定及后端资源生命周期。

## 目标与设计约束

1. **内容不可变**：同版本同内容重试幂等；同版本不同内容明确拒绝。
2. **发布原子性**：两个目标引擎全部就绪后，通过单一切换点更新默认版本。
3. **会话稳定性**：首次生成绑定版本，后续工具轮次、重试及 abort/resume 不重新选择版本。
4. **明确失败**：绑定目标缺少版本时返回错误，不回退到基座或最新版。
5. **安全回收**：有会话引用、在途请求或未完成 GPU 使用的版本不可卸载。
6. **容量可控**：容量不足时拒绝发布，不淘汰仍被引用的版本。
7. **生成连续性**：正常发布不全局暂停、不清缓存、不强制中断旧会话。
8. **结果可验证**：使用实际 logprob、缓存对照和资源回执验收，不仅检查版本日志。

首个实测配置为固定 dense Qwen3 基座、两个 TP=1 引擎。多节点、多 rank、colocate/offload 等组合不作为本次实测通过范围。

## 设计

### 1. 不可变快照与身份

快照包含 adapter 配置、权重及来源信息，并生成 SHA-256 内容摘要。

通过独立 staging、文件校验、fsync 和原子目录提交完成封存。正式版本目录不可覆盖，引擎加载前再次校验制品。

主要身份包括：

| 身份 | 用途 |
| --- | --- |
| `version_id`、`digest` | 固定版本与内容的对应关系 |
| `request_id` | 发布意图的幂等重试 |
| `operation_id` | 区分具体发布尝试 |
| `cohort_id`、engine ID、boot ID | 拒绝旧协调者或旧引擎回包 |
| `native_lora_id` | 标识后端具体加载实例 |
| `owner_epoch`、`session_id` | 会话绑定与关闭的幂等身份 |
| `rid` | 标识一次后端生成 attempt |

### 2. 双目标发布

发布状态流转：

```text
PREPARING → PUBLISHED → RETIRING → RETIRED
     │
     └──失败或取消──→ RETIRING → ABORTED
```

发布流程：

1. 校验快照、重试身份和剩余容量。
2. 固定本次发布的目标集合。
3. 各目标完成 pinned 加载、实际预热及驻留确认。
4. 核对全部目标的版本、摘要、实例和引擎代际。
5. 在同一把锁内更新默认版本。

默认版本提交与会话首次绑定共用锁，形成单一线性化点。部分失败时保留旧默认版本，并清理所有目标上的未发布实例，包括结果未知的目标。

### 3. 会话绑定与实际生成

会话在首次真正获得生成资格时绑定版本。

后续请求携带对应的 `lora_path=native_lora_id` 和原生绑定身份，直接发送到绑定引擎。abort/resume 可以创建新的 attempt，但不改变 Session binding。

响应中的实例身份也会校验；缺版本或身份不匹配明确失败。

### 4. 安全回收

版本回收需要同时满足：

- 已不是默认版本；
- 会话引用归零；
- 后端请求完成或得到排空证明；
- 最后一次 GPU 使用完成；
- slot 清理完成。

HTTP 返回、超时、客户端取消及 abort ACK 均不能单独作为资源释放依据。

加载使用 `pinned=True`，防止底层 LRU 淘汰受管 adapter。退休操作针对具体实例安装 fence，重复操作与迟到回包不会导致重复卸载。

### 5. 导出与缓存隔离

实现接入现有 checkpoint/PEFT 导出能力，完成：

```text
训练更新 → adapter 导出 → 不可变快照 → 引擎加载 → 发布 → 实际生成
```

KV cache 使用原生 LoRA 实例身份隔离；重新加载的实例不复用旧身份。正常发布与全局权重更新、显式 offload 流程分离。

## 配置与使用

启用 Agentic rollout、LoRA adapter mode，并指定发布配置：

```bash
--use-agentic-rollout \
--lora-adapter-mode \
--lora-publication-config /your/shared/publication.yaml
```

配置示例：

```yaml
artifact_store: /your/shared/adapter-store
capacity: 2
engines_per_gpu: 1
bootstrap_version_id: A
auto_publish: false
prepare_timeout_seconds: 120
cleanup_timeout_seconds: 120
max_lifecycle_records: 100000
```

封存并发布完成的 PEFT 导出：

```bash
python -m relax.engine.lora.cli seal \
  --source "$EXPORT" --model "$MODEL" \
  --store "$STORE" --version-id B --source-step 10

python -m relax.engine.lora.cli publish \
  --rollout-url "$ROLLOUT_URL" \
  --version-id B --request-id publish-B --wait-seconds 120
```

查询状态及触发回收：

```bash
python -m relax.engine.lora.cli status --rollout-url "$ROLLOUT_URL"
python -m relax.engine.lora.cli collect --rollout-url "$ROLLOUT_URL"
```

`capacity` 是业务版本容量。`max_lifecycle_records` 是 cohort 的累计历史记录预算，包含重试防护记录，并非并发会话上限；耗尽后明确拒绝新工作，需要安全排空后重建 cohort。

## 故障场景与预期行为

| 场景 | 预期行为 |
| --- | --- |
| B 仅在一个引擎就绪 | 默认保持 A，新会话仍使用 A |
| B 在全部目标就绪 | 新会话使用 B，旧会话继续使用 A |
| 单引擎加载失败 | 保留旧默认版本，清理未发布资源 |
| 迟到 READY 或旧代际回包 | 校验身份与 fence，不复活已取消操作 |
| 同版本同内容重复发布 | 返回已有操作，不重复发布 |
| 同版本不同内容 | 返回内容冲突 |
| 取消与完成并发 | 通过幂等终结与原生完成证明处理，不提前释放 |
| 绑定目标缺少版本 | 明确失败，不回退 |
| 容量 2，A 有引用且 B 已发布，发布 C | 返回 507 容量错误 |
| A 引用归零但 GPU 或 slot 清理未完成 | 保持 RETIRING，继续占用容量 |
| A 完全释放 | 每个目标实例仅实际卸载一次，再允许 C |
| 引擎永久失联 | 保留未知状态和资源占用，不强制宣称清理完成 |

## 验证方法

### 故障注入与回归

覆盖发布幂等、内容冲突、部分就绪、迟到确认、取消竞争、close-before-bind、请求排空、容量限制及 slot 复用。

协议 doubles、真实类接线测试和 GPU 实验分别记录，避免将 CPU 模拟结果当作设备验收。

### 真实 GPU 数值验证

- 使用两份固定 seed 的可区分 LoRA fixture。
- 分别建立独立加载 A/B 的参考引擎。
- 运行前固定 logprob 和 KV 容差：`atol=0.001`、`rtol=0.001`。
- 比较旧会话、新会话及 abort/resume 的实际 logprob。
- 双向验证 A 预热→B 请求、B 预热→A 请求。
- 加入错误 KV 和错误 slot 内容负对照，验证测试能够检出错误。
- 保持 CUDA graph 与 radix cache 开启。

### 持续流量验证

发布期间维持生成，记录逐 token 进度、请求完成、延迟、失败数、发布耗时和版本占用。

预登记性能门槛为：p95 比值不超过 1.5、吞吐比值不低于 0.8、最大进度间隔不超过 2 秒。

## 实验结果

以下 GPU 实验结果对应 2026-09-25 验收版本。PR #377 后续评审修复及新增回归见[评审修复记录](./no7-review-fixes.md)。

测试环境：两张 NVIDIA RTX 6000 Ada Generation、Qwen3-1.7B、BF16、两个 TP=1 目标引擎。

**六项 GPU 实验全部通过，28 项验收检查全部 PASS，缺失证据为空。**

| 验证项 | 结果 |
| --- | --- |
| 会话绑定与续写 | 65 个 Session 对照，最大 logprob 绝对误差 0 |
| 发布故障任务中的会话对照 | 3 个对照，最大误差 0 |
| A→B 缓存隔离 | 64 个对照，最大 logprob 绝对误差 0 |
| B→A 缓存隔离 | 64 个对照，最大 logprob 绝对误差 0 |
| 容量与回收 | C 正确拒绝；A 在两个引擎各实际卸载一次 |
| 真实导出 | 112 个导出 tensor 与更新后模型逐 tensor 完全一致 |
| 发布 API 耗时 | 约 0.417 秒 |
| 发布期间流量失败 | 0 |
| 全局暂停、清缓存等禁止操作 | 审计调用数 0 |

持续流量每窗口测量 30 秒：

| 窗口 | token/s | p95 延迟（秒） | 最大进度间隔（秒） |
| --- | ---: | ---: | ---: |
| 普通 LoRA 参考 | 326.09 | 13.145 | 0.0371 |
| 受管稳态 A | 320.12 | 13.183 | 0.0498 |
| 发布 B，保持 A 流量 | 320.79 | 13.126 | 0.0490 |
| 发布后 A/B 驻留稳态 | 323.59 | 13.123 | 0.0490 |

本地回归 **442 passed、7 skipped**。其中六个 GPU pytest 入口已通过独立完整 GPU 命令验证；另一个已有测试因 Megatron-Bridge 未提供 `LoRAMerge` 跳过。全部 pre-commit hooks 通过。

真实导出实验验证了单进程 optimizer 更新及现有 FSDP checkpoint/export 路径；不代表完整分布式 FSDP 或 Megatron RL 训练通过。多节点、多 rank 和其他部署组合仍需独立硬件验收。

## 代码入口

| 模块 | 入口 |
| --- | --- |
| 快照与内容身份 | `relax/engine/lora/snapshot.py` |
| 制品语义校验 | `relax/engine/lora/artifact.py` |
| 版本管理、发布与回收 | `relax/engine/lora/publication.py` |
| 管理 CLI | `relax/engine/lora/cli.py` |
| 会话绑定与关闭 | `relax/agentic/session/service.py` |
| 实际生成与原生身份校验 | `relax/agentic/pipeline/runtime.py` |
| 发布协调与引擎管理 | `relax/distributed/ray/rollout.py` |
| adapter 导出 | `relax/backends/megatron/checkpoint.py` |
| SGLang 执行与回收补丁 | `docker/patch/sglang/v0.5.17.patch` |
| 故障注入测试 | `tests/engine/lora/`、`tests/backends/sglang/test_native_lora_*.py` |
| GPU 验收入口 | `tests/engine/lora/acceptance/` |
| 归档实验结果 | `tests/engine/lora/evidence/2026-09-25/` |

**本地提交：** `0d0fa7a feat(lora): publish immutable adapter versions`

归档证据已统一脱敏内网地址，保留数值、版本身份、时间和资源计数，并提供 SHA-256 校验值。

## 参考资料

- [Miles PR #3127：Publish immutable adapter versions alongside generation](https://github.com/radixark/miles/pull/3127)
- [SGLang LoRA Serving](https://docs.sglang.io/docs/advanced_features/lora)
- [完整 RFC](./immutable-lora-publication-redesign-rfc.md)
- [实验与验收记录](./immutable-lora-validation.md)
- [机器可读验收摘要](../../tests/engine/lora/evidence/2026-09-25/summary.json)
