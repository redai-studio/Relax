# RFC #410：增量权重同步实现记录

工作目录：`/workspace/github_relax/worktrees/delta-weight-sync`。分支：`feat/delta-weight-sync`。开发基线与本次 fetch 得到的 upstream/main 均为 `551c4ff4dd598def1842e26147e125ece764e15c`。

这是实现交接与验证记录，尚不是 RFC 全部阶段的完成报告。

## 磁盘是否必要

**delta 编码与网络交付不必经过磁盘。本 RFC 的离线消费、生产者退出后的可重建性和同目标恢复，需要持久化材料；选用原生 SGLang 完整 checkpoint Loader 时，还需要节点可访问的完整 checkpoint。**

| 边界 | 当前实现 | 磁盘的作用 |
| --- | --- | --- |
| 捕获与编码 | 独立 CPU tensor 快照；对内存原始字节编码 | 编码 API 不访问磁盘 |
| 内容发布 | 内容寻址对象、逻辑目录、交付 manifest、原子版本索引 | 保证退出后的持久可读 |
| 恢复材料 | owner-local canonical 块、目录和 manifest | 与 delivery 损坏隔离，不依赖受损 delta 链补 FULL |
| TCP | 按缺失对象获取，一次连接一个对象 | 网络传输本身不要求共享磁盘；接收端缓存已验证对象 |
| 重建 | 分块写入完整 safetensors shard 与索引 | 为后续完整 Loader 提供文件；准备阶段不操作运行中模型 |
| 激活 | 尚未接入 | 仍需组级准入、实际权重检查和生成探针 |

参考来源：

- [RFC #410](https://github.com/redai-studio/Relax/issues/410)：SharedFS MVP、独立加载、离线读取、后续 TCP 与源端汇聚优化。
- [slime delta 文档](https://thudm.github.io/slime/zh/advanced/delta-weight-sync.html)：disk transport、canonical HF checkpoint、本地 apply 与 `update_weights_from_disk`。
- [Miles #1235](https://github.com/radixark/miles/pull/1235)：非 colocate disk-delta 路径。
- [verl #6974](https://github.com/verl-project/verl/pull/6974)：经 NCCL 的 sharded delta，说明增量交付本身不依赖磁盘。

现有 Relax NCCL/CUDA IPC 路径继续保持原状。这份实现没有改变默认训练行为，也没有将上述项目的性能数字作为 Relax 收益。

## 已实现的软件范围

代码位于 `relax/distributed/weight_sync/`，仅使用现有依赖。

- `format.py`：格式 1、严格 JSON、逻辑 root 与 artifact ID 分离、固定块边界、COPY/SPARSE/RAW/RAW_ZLIB/XOR_ZLIB，按包含描述字段的实际编码长度选择；对原始位模式操作，保留 BF16/FP16/FP32/FP64、整数、bool、NaN payload 与正负零。
- `publisher.py`：在调用线程捕获独立 CPU 快照；单个后台 CPU worker 编码和提交；一个 pending slot；显式 submit/finalize；FULL 初始化和周期 anchor；schema/base 无效时 FULL；同 update_id 重试核对目标内容，拒绝把不同快照当成同一更新。
- `store.py`：POSIX flock 串行化状态修改，fsync 和原子 rename；内容对象先于版本索引可见；持续消费者游标、活动引用、最近两个 anchor 窗口及依赖闭包回收；owner-local canonical 恢复；同目标 FULL 追加、既有 FULL 对象/manifest 修复；封存。
- `consumer.py`：先 pin，获取目标和依赖，按对象摘要复用库存；本地获取事务及持久 pin 防止 GC 删除下载中的内容；流式重建标准 safetensors，逐块验证；损坏本地 checkpoint 可重建；返回 PreparedCheckpoint，不返回 ACTIVE。
- `tcp.py`：有界活动 worker 与 listen backlog；控制帧 64 KiB，内容块 8 MiB，manifest 16 MiB；短读、截断、重试、繁忙退避和绝对截止时间；完整缓存对象复用；接收字节包含失败尝试已收到的 payload。用于可信网络或已有认证隧道。
- `config.py`：严格配置对象。发布者的 canonical_root 与 store_root 分离，防止把恢复用的完整原始块误放进跨端交付路径。
- `__main__.py`：独立 publish、prepare、query、seal、release、gc、serve 入口。

逻辑 root 包含 schema_context、参数名、shape、dtype、固定块边界和目标字节摘要；不包含源/目标 GPU 地址，也不包含 DELTA/FULL 编码选择。同一目标的 FULL 与 DELTA 因而 root 相同、artifact_id 不同。

store 的磁盘预算计 canonical、delivery、manifest 和索引文件的逻辑字节，并为原子索引替换检查空间。引用释放和 GC 保留回收入口，不因索引临时副本超过逻辑配额而形成无法释放引用的闭环。文件系统物理可用空间与逻辑配额是两项约束，真实部署仍需验证共享文件系统的 fsync、flock 与 rename 语义。

Consumer 的 checkpoint 配额包括实际 safetensors header、数据、索引、config 和 sidecar；cache 的配额由传入的 SharedFSStore 独立配置。部署方需要把两者以及 Loader 的额外内存/磁盘工作集合并规划。目前不提供正在使用的 checkpoint 文件的自动删除策略。

## 可运行的独立入口

先准备一份不会再被写入的完整 HF safetensors checkpoint。这里的 publish 入口用于独立冻结文件的交付，不是已经接通 optimizer 更新边界的训练命令。

发布配置示例，所有路径与容量由部署方填写：

```json
{
  "mode": "delta",
  "stream": "qwen3-vl-experiment",
  "store_root": "/path/to/shared/delivery",
  "canonical_root": "/path/to/publisher-local/canonical",
  "chunk_bytes": 8388608,
  "compression": true,
  "anchor_interval": 20,
  "host_budget_bytes": 17179869184,
  "disk_budget_bytes": 137438953472,
  "sync_timeout_s": 600,
  "full_only": false
}
```

```bash
python -m relax.distributed.weight_sync --store "$STORE" publish \
  --config "$CONFIG" --checkpoint "$FROZEN_HF_CHECKPOINT" \
  --update-id "$UPDATE_ID" --result "$PUBLISHED_JSON"

python -m relax.distributed.weight_sync --store "$STORE" prepare \
  --stream "$STREAM" --version "$VERSION" \
  --cache "$NODE_CACHE" --checkpoints "$NODE_CHECKPOINTS" \
  --disk-budget-bytes "$CHECKPOINT_BUDGET" --result "$PREPARED_JSON"

python -m relax.distributed.weight_sync --store "$STORE" \
  --canonical-root "$CANONICAL" seal \
  --stream "$STREAM" --versions 0 1 --seal-id offline-example

python -m relax.distributed.weight_sync --store "$STORE" query \
  --result "$INVENTORY_JSON"
```

封存为指定目标生成 FULL 并持有引用。生产者退出后，新进程可以用 `prepare --full` 读取已封存的 FULL，无需访问生产者的 canonical_root 或训练服务。封存前后的完整目录复制与真实跨节点挂载语义尚需部署验收。

通过 TCP 获取同一版本：

```bash
python -m relax.distributed.weight_sync --store "$STORE" \
  --canonical-root "$CANONICAL" serve --host "$BIND_ADDRESS" --port "$PORT"

python -m relax.distributed.weight_sync --store "$LOCAL_CONTROL_DIRECTORY" prepare \
  --endpoint "tcp://$PUBLISHER_HOST:$PORT" --stream "$STREAM" --version "$VERSION" \
  --cache "$NODE_CACHE" --checkpoints "$NODE_CHECKPOINTS" \
  --disk-budget-bytes "$CHECKPOINT_BUDGET" --result "$PREPARED_JSON"
```

同目标 FULL 的新建/修复需要存储 owner 的 canonical 材料。SharedFS 消费者可直接读取既有完好 FULL；需要 owner 补包时，使用持有 canonical_root 的 owner 实例/Transport。生产者退出后的离线读取使用预先封存的材料。

Python 调用示例：

```python
handle = publisher.submit(update_id, named_tensors)
published = publisher.finalize_pending(handle)
prepared = consumer.prepare(published.stream, published.version)
# 接下来由安装协调者完成真实 Loader、读回与业务准入；此处尚不表示 ACTIVE。
```

发布调用者必须在捕获期间维持训练安全边界，不得让模型同时被 optimizer 修改。CPU worker 不参与训练 collective。finalize 超时不代表 worker 已取消，需使用同一 handle 确认最终结果；close 会等待最后候选结束。

prepare 失败时保留引用及已校验内容以便诊断和恢复。`consumer.last_metrics['reference']` 给出本次引用。成功且不再需要其准备引用时调用 `consumer.release(prepared)`，同时释放远端和本地 pin；正在服务的版本需要由未来安装协调者交接独立 serving 引用后再释放准备引用。`release` CLI 是存储管理操作，只影响指定 store 中的引用。

持续消费 API 为 `register_consumer`、`acknowledge` 和 `unregister_consumer`。acknowledge 只允许下一个连续版本，并幂等核对安装身份和 root；调用者负责仅在真实 ACTIVE 后发送确认。本模块不通过文件重建结果推断推理已就绪。

## 验证与验收边界

测试位于 `tests/cpu/distributed/weight_sync/`。新增用例针对位模式损坏、错误 base、发布可见性、GC 与获取并发、损坏后的恢复、配额死锁及 TCP 截断等真实失败机制；没有为不相关组件新增 mock 或扩大现有测试清单。少量新进程测试用于证明离线读取与 Python 进程内状态解耦，单进程超时限制为 20 秒。

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/cpu/distributed/weight_sync
pre-commit run --all-files --show-diff-on-failure
```

CPU 验证包含 SharedFS 和 TCP 各 20 个连续合成版本、周期 anchor、全部支持 dtype 的位精确比较、全量回退、新消费进程、source/base 与 delivery 损坏、损坏本地 checkpoint 修复、真实并发 GC、截止时间、慢连接 worker 释放、元数据/数据配额、同一 update_id 冲突和不完整 checkpoint 拒绝。这些合成轨迹不等价于真实训练序列，也不证明传输量降低 50%。

尚未执行多节点 GPU 集成：没有获得本任务专用的 RAY_ADDRESS、GPU 拓扑和远端验证授权。没有提交训练 Job、使用其他任务资源或把当前平台状态写成验收通过。

## RFC 剩余开发

| 阶段 | 当前状态与下一步 |
| --- | --- |
| M0 | 未完成。接通 optimizer 安全边界的 Megatron 完整导出、独立 SGLang 加载、设备权重对照和图文探针，建立真实 FULL 基线。 |
| M1 | 编码/存储/SharedFS/重建核心已实现；仍缺训练 API 接线、固定成员安装 journal、业务请求准入、排空、进程身份 fencing、实际权重读回、探针和 ACTIVE 回执。 |
| M2 | TCP 核心和本机真实 socket 故障回归已实现；仍需跨节点部署、完整恢复/安装集成和重传链路账本。 |
| M3a | 未完成。Qwen3-VL Dense/MoE、TP/PP/EP 与不同 GPU 需逐布局导出与实际分片对照。 |
| M3b | 未实现。SourceMap、转换依赖组、REUSE/PATCH/FULL 源端汇聚及通信计量。 |
| M4 | 未执行。两条 fully-async 20 版回归及四组 FULL+100 次真实更新、完整链路总字节与资源验收。 |

当前接口审查确认：Direct HF iterator 含 PP/EP/TP/Gloo collective，所有源 rank 必须同序遍历；Bridge 还有分布 owner 和专家快速路径，不能默认 rank 0 收到完整 HF 清单。源端 capture/导出错误必须在 collective 安全边界聚合，不能让一个 rank 提前返回。fully-async 的独立 rollout 发布也不能吞掉 ActorFwd 的原有更新。

后续接线计划：通过默认关闭的 `--weight-sync-config` 传入统一配置；Actor 在首次访问 RolloutManager 之前选择独立发布；消费侧固定成员并共享拓扑更新锁，先关闭并排空业务入口，再暂停/完整加载/实际读回；业务入口保持关闭时恢复后端并运行探针，全体通过且安装决定持久化后才能 ACTIVE。旧 RPC 未确认结束前不能重装或重新开放。

仓库 AGENTS.md 的 Ask First 指定参数解析和 Service 逻辑需确认。已提交该接入范围的确认请求，尚未据此修改 `relax/utils/arguments.py`、Controller、Service 或 Launcher。独立核心的开发不依赖这项确认。当前阶段没有 push、创建 PR、发 GitHub comment 或外部文档写入。
