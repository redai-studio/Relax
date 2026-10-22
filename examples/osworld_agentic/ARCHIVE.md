# CUA 成功轨迹库

默认关闭。归档独立于 reward、advantage、训练样本过滤和 SessionOutput：每次完整任务尝试是一条候选轨迹，OSWorld 在真实模型请求、环境动作和 evaluator 边界旁路采集。其他 recipe 不会自动启用归档，也不会自动启动服务或后台线程。

## 部署与开启

第一版使用独立的单写入 HTTP 进程。所有训练节点的 Agent 连接同一服务，容量是该服务管理的**整个库**上限，不是每节点上限。无需改训练参数解析或 Ray 服务。服务不创建沙箱、不调用模型。

在一台可被训练节点访问的机器上，选择专用、本地持久磁盘目录。下面的值是示例，使用实际路径和监听端口：

```bash
export CUA_ARCHIVE_TOKEN='替换为私有访问令牌'
python -m relax.agentic.archive.server \
  --archive-dir /local-disk/cua-library \
  --max-bytes 10737418240 \
  --per-task-limit 5 \
  --host 0.0.0.0 --port 8765
```

`max-bytes` 为必填容量预算，上例 10 GiB；`per-task-limit` 默认 5。绑定非回环地址要求令牌。HTTP 适用于受信任的私有网络；跨不可信网络应使用 HTTPS 反向代理。不要把令牌写进 Git 或日志。

在每个节点可读取的 OSWorld `env.local.sh` 中显式添加：

```bash
export CUA_ARCHIVE_ENABLE=1
export CUA_ARCHIVE_URL='http://实际归档主机:8765'
export CUA_ARCHIVE_TOKEN='与服务一致的私有令牌'
```

仍使用现有 OSWorld 训练入口。`run_agent_app.sh` 已读取 `env.local.sh`，无需把令牌放入会打印的 Ray runtime-env。关闭时删除 `CUA_ARCHIVE_ENABLE` 或设为 `0`。

服务接受多个节点/进程的请求，但串行修改文件并在写入前预留本次事件及新资源的全部字节。HTTP 请求超时为 5 秒，无归档自动重试；归档服务不可达、数据不支持或磁盘错误会停止该候选的归档并记录原因，不改训练结果。同步归档会增加少量边界延迟，不能承诺零延迟。单次请求限制为 min(预算的两倍, 64 MiB)，超限候选明确放弃。

支持 Linux 本地 ext2/3/4、XFS、Btrfs、tmpfs、overlay 文件系统；启动按 mountinfo 检查，拒绝 NFS、FUSE 共享盘及未知类型。tmpfs/容器临时层只适合测试，不具备持久性保证。多个节点不得直接打开同一目录。不能在不同主机各启一个库后将它们的独立预算当作全局预算。

## 内容、筛选与容量

公共实现位于 `relax/agentic/archive/`，不导入 OSWorld。OSWorld 映射位于 `app/archive.py`。

- `archive.json`：schema、库容量及任务配额；重启不能静默改变这些设置。
- `staging/<trajectory_id>.jsonl`：逐步记录 trajectory、observation、model_call、call_result/call_parse、action/action_result、outcome。未完成记录不可导出。
- `trajectories/<trajectory_id>.jsonl`：通过校验后由 staging 原子 rename 提交的正式记录。
- `assets/<sha256>`：独立图片内容，按原始字节 hash 去重；文件无扩展名，PIL 按文件内容解码。

轨迹有稳定 trajectory/session/run ID。观察、调用、动作使用各自 ID 关联；一次 step 可以有多次请求，一次调用可以有多个顺序动作。调用保存实际 payload（messages、tools、模板参数等）和原始 HTTP 回复，观察保存截图及外部环境实际返回的其他字段。模型请求中的 base64 图片单独提取保存，包括裁剪/缩放后的模型输入，不用原始桌面截图替代。调用的 `observation_ids` 指向当时的环境观察，`model_input_asset_ids` 列出实际请求中包括历史图像在内的资源；无法推断裁剪图像对应哪张原图时不伪造映射。`asset://<hash>` 可以还原为图片字节或 data URI。

任务配额键为 `(task_source, task_version, task_id)`。OSWorld 的 task_version 是完整任务配置的 SHA-256。仅明确 success、无 evaluator 异常、关联完整且资源可校验的轨迹可以入库。公共层不使用奖励阈值；OSWorld 归档采用保守规则：evaluator 标量恰好为 `1` 才是 success，`0` 为 failure，`(0,1)` 为 partial，其他值、缺失或异常为 unknown。此规则不改变训练原有 reward/成功率口径。advantage 不参与筛选。

完全重复按任务键、顺序动作类型/完整参数、动作前后截图 hash 及观察字段判断，不包含 ID、时间、分数或推理文字。同路径保留更优版本；排序优先：已知执行错误数、环境步数、动作数、模型调用数、执行状态未知的动作数、trajectory ID。无语义相似度模型。

容量按库内所有文件的逻辑字节数统计，包含正式记录、staging、唯一资源和配置，不等同于磁盘分配块、文件系统元数据或实际剩余磁盘空间。重复图片只有一个文件。写入前同时计入本次 JSON 和新图片，多个客户端不能超额争抢预算。

空间不足时，优先从保有多条轨迹的任务中淘汰多余的低优先级记录，优先保留任务覆盖；不因新候选挤掉某任务的唯一代表。不足则放弃候选，日志说明原因。staging 不会为别的候选被自动淘汰。资源 GC 同时检查正式记录和 staging 引用；共享图片仍被引用就不删除。

容量满且只剩唯一代表或活跃 staging 时，新任务也可能无法进入；这是硬预算下的明确取舍。空间预留可能先淘汰旧的多余记录，随后候选写盘失败，不保证淘汰回滚。文件系统 ENOSPC 仍会被单独捕获。

## 崩溃恢复

归档服务持有本地独占 owner 锁；第二个服务和离线导出不能同时打开库。服务重启保留 staging 并继续计费，不按心跳超时猜测 Session 已死。训练/所有 Agent 和归档服务都停止后，才能明确清理残留：

```bash
python -m relax.agentic.archive.export recover \
  --archive-dir /local-disk/cua-library --confirm-no-active-sessions
```

该命令持有独占锁，只删除本库 staging 并回收无引用资源。标志是操作者对所有 Agent 已停止的确认；不要对仍在运行但暂时未写入的 Session 使用。导出始终忽略 staging，不要求先清理。

## 离线导出与 SFT

训练结束并停止归档服务后：

```bash
python -m relax.agentic.archive.export export \
  --archive-dir /local-disk/cua-library --output-dir /data/cua-export-001
```

目标目录必须不存在。导出全程独占库，防止淘汰造成中途缺图。输出 `manifest.json`、`trajectories.jsonl`、`samples.jsonl` 和实际引用的 `assets/`。manifest 记录 schema/导出版本、来源、入选/跳过/重复/缺失数量、原因和监督策略。未完整完成的导出没有 manifest；失败后改用新目录，不静默覆盖。

完整轨迹保留解析重试和纠错。SFT 保守排除有未选中/未确认解析调用、已知执行失败的整条轨迹；只选择关联了实际执行动作的回复，不把强制终止等 harness 动作作为目标。OSWorld 对 RPC 正常返回但无法确认语义成功的动作记录 unknown + rpc_completed；在最终 evaluator 成功、无已知执行错误及重试时允许用于 SFT，并不宣称每个动作语义都已验证。

`samples.jsonl` 使用现有 Relax SFT messages 格式：所有历史消息 `learn=false`，仅最后一条当前 assistant 回复 `learn=true`；保留原历史内容、tools、`metadata.apply_chat_template_kwargs`。需要 `tool_call_id/name`、developer 角色等当前 loader 无法无损表示的消息会被跳过并计数，完整轨迹不丢弃这些字段。

导出图片路径相对导出目录，便于复制。现有 loader 按进程 CWD 解析图片，因此部署到所有训练节点同路径可访问的目录后，先生成本地绝对路径版本：

```bash
python -m relax.agentic.archive.export convert-sft \
  --samples /data/cua-export-001/samples.jsonl \
  --output /data/cua-export-001/sft.local.jsonl
```

在已有、与图像模型匹配的 SFT 启动脚本中使用以下数据参数（不运行此处的示例）：

```bash
--loss-type sft \
--prompt-data /data/cua-export-001/sft.local.jsonl \
--input-key messages --metadata-key metadata --tool-key tools \
--multimodal-keys '{"image":"images"}'
```

不设置 `--label-key`，不注入额外 system prompt；内嵌 image_url 直接提供图片，无需再映射重复的 images 列。移动目录后重新转换，不能继续使用旧绝对路径。需自行选择匹配模型的 processor、chat template 和资源配置；本功能不自动回灌训练。

## 验证与限制

使用临时目录、mock 模型/环境及本地 HTTP 验证容量、去重、恢复、多进程并发写入、完整性、写盘中断、淘汰时的图片引用保护和 CLI 导出。导出要求完整提交标记，损坏记录会跳过并计数；CLI 测试确认活跃 owner 锁阻止导出。现有 SFT 加载器测试确认仅当前回复保留监督标记。没有运行真实沙箱、多节点训练或模型训练。

外部 OSWorld fork 未在本仓库实现。环境版本、策略权重版本若没有可靠元数据则记录 unknown，不推测 DesktopEnv/Nex 属性；结果文件没有可靠获取接口时标记不可用。这里只保存实际请求中的图像字节，不保存 processor 图像张量。HTTP/本地文件形式的模型图片无法证明与当时输入一致，第一版拒绝这类候选而不事后下载。环境内部未暴露的细粒度失败无法凭 RPC 正常推断。
