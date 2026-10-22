# CUA 成功轨迹与图片保存

OSWorld 正式训练脚本默认将完整成功轨迹和图片直接写入本轮训练输出目录，无需独立归档服务、URL 或令牌。采集不改变 reward、advantage、训练过滤或 SessionOutput。其他 recipe 不会自动启用。

## 训练时配置

在原有 MODEL_DIR、DATA_DIR、SAVE_DIR 和私有 Nex 配置准备好后运行：

```bash
bash examples/osworld_agentic/scripts/run-qwen3vl-8B-osworld-8xgpu.sh \
  --save-trajectories \
  --trajectory-max-bytes 10737418240 \
  --trajectory-per-task-limit 5 \
  --trajectory-task-limits /path/to/task-limits.json
```

`--save-trajectories` 默认开启，可用 `--no-save-trajectories` 关闭。容量默认 10 GiB，每任务默认最多保留 5 条。按任务覆盖文件是可选的 JSON 对象，例如 `{"task-id-a": 2, "task-id-b": 0}`；键是实际任务 ID，值为非负整数，0 表示不采集该任务。默认配额必须至少为 1，容量至少为 1024 字节。这些是 OSWorld 脚本参数，不传给框架训练解析器。

每轮输出下新增：

```text
rollout_result/<run_tag>/
├── train/*.jsonl
├── reward_dump.jsonl
└── sessions/
    ├── archive.json
    ├── staging/<trajectory_id>.jsonl
    ├── trajectories/<trajectory_id>.jsonl
    └── assets/<task_id>/<trajectory_id>/
        ├── screenshots/000000-<sha256>.png
        ├── screenshots/000001-<sha256>.png
        └── model_inputs/000000/000000-<sha256>.png
```

轨迹文件头包含 `session_id`、`task_id` 和完整任务配置，可与 `reward_dump.jsonl` 对应。新建实验将图片按任务和轨迹分目录，文件自带图片扩展名，可以直接打开查看。任务 ID 中的特殊字符会转义为安全目录名，过长或特殊的 ID 使用哈希目录名；原 ID 保留在轨迹头。

`screenshots/` 的六位序号按实际观察的先后递增：初始截图为 `000000`，后续包含每个实际动作后的观察。一轮模型回答可以触发多个动作，因此截图序号不等于环境轮号。动作事件的 `before_observation_id` 和 `after_observation_id` 仍明确关联操作前后图片，`outcome.final_observation_id` 指向最终截图。

`model_inputs/000000/` 表示第一个实际模型请求，后续请求、解析重试和传输重试依次编号。目录内图片按请求遍历顺序编号，包括历史图像，不表示桌面观察时间顺序。它们保留模型实际收到的裁剪或缩放版本，不能用桌面原图替代。图片保留原始字节，扩展名来自实际解码格式；PNG 保存为 `.png`，JPEG 保存为 `.jpg`。其他事件附带的图片保存于 `events/<事件序号>/`。

布局记录在 `archive.json` 的 `asset_layout` 字段。新建实验使用 `task`；旧库没有该字段时继续使用原来的 `assets/<sha256>` 布局，可读取和导出，不会在正在运行的实验中重排历史图片。

**长期保留规则仍是完整且 evaluator 恰好为 1 的成功轨迹。** 失败、部分成功、不完整、重复或低优先级候选会清理；它们的临时图片不会成为永久训练截图日志。关闭保存只影响新启动的任务。已启动 Agent 已放弃的历史图片不会补存。

单节点可使用本地输出目录。多节点必须同路径访问同一个输出目录，文件系统需要提供可靠的跨节点 `flock`、原子重命名和关闭后可见性；不能使用各节点独立本地目录却将预算解释成全局预算。所有 Agent 每次写入时短暂获取同一个 `.owner.lock`，在锁内核算整个 `sessions/` 的容量、去重和配额，模型调用期间不持锁。锁等待最多 30 秒；超时、磁盘错误或不支持的图片会放弃当前候选并继续训练。文件系统 I/O 本身不受锁等待计时器限制。共享盘同步写入会增加延迟，需按实际并发和文件系统性能评估。

容量和配额固定在每轮的 `archive.json`。同目录不能由不同 Agent 使用不一致的配置，也不能在运行中修改。新 run_tag 使用新的目录和预算；预算不包含其他 run、训练 JSONL 或手动导出的副本。训练脚本配置优先于 Agent 再次读取的旧 `env.local.sh`，旧 URL 和令牌不会用于直接写入。

## 独立 HTTP writer（兼容入口）

公共 HTTP 接口仍可使用，但正式 OSWorld 训练脚本已切换为直接写入。自行调用 Agent 时，可显式设置 `CUA_ARCHIVE_BACKEND=http`、`CUA_ARCHIVE_ENABLE=1`、`CUA_ARCHIVE_URL` 和 `CUA_ARCHIVE_TOKEN`。HTTP writer 仍只支持本地 Linux ext2/3/4、XFS、Btrfs、tmpfs 或 overlay 目录，不支持共享盘；非回环监听需要令牌。

```bash
python -m relax.agentic.archive.server \
  --archive-dir /local-disk/cua-library \
  --max-bytes 10737418240 --per-task-limit 5 \
  --host 0.0.0.0 --port 8765
```

调用前从私有配置设置令牌。HTTP 请求超时为 5 秒，无自动重试；单次请求限制为 min(预算的两倍, 64 MiB)。不能让 HTTP writer 和直接写入的 Agent 同时管理同一个库。

## 内容、筛选与容量

公共实现位于 `relax/agentic/archive/`，不导入 OSWorld。OSWorld 映射位于 `app/archive.py`。

- `archive.json`：schema、库容量及任务配额；重启不能静默改变这些设置。
- `staging/<trajectory_id>.jsonl`：逐步记录 trajectory、observation、model_call、call_result/call_parse、action/action_result、outcome。未完成记录不可导出。
- `trajectories/<trajectory_id>.jsonl`：通过校验后由 staging 原子 rename 提交的正式记录。
- `assets/<task_id>/<trajectory_id>/`：新建训练实验按顺序组织的桌面截图和模型输入图片；旧库及默认 HTTP 库仍使用 `assets/<sha256>`。

轨迹有稳定 trajectory/session/run ID。观察、调用、动作使用各自 ID 关联；一次 step 可以有多次请求，一次调用可以有多个顺序动作。调用保存实际 payload（messages、tools、模板参数等）和原始 HTTP 回复，观察保存截图及外部环境实际返回的其他字段。模型请求中的 base64 图片单独提取保存，包括裁剪/缩放后的模型输入，不用原始桌面截图替代。调用的 `observation_ids` 指向当时的环境观察，`model_input_asset_ids` 列出实际请求中包括历史图像在内的资源；无法推断裁剪图像对应哪张原图时不伪造映射。`asset://<相对路径>` 可以定位到 `assets/` 下的图片，再还原为图片字节或 data URI；旧的 hash 引用仍可读取。

任务配额键为 `(task_source, task_version, task_id)`。OSWorld 的 task_version 是完整任务配置的 SHA-256。仅明确 success、无 evaluator 异常、关联完整且资源可校验的轨迹可以入库。公共层不使用奖励阈值；OSWorld 归档采用保守规则：evaluator 标量恰好为 `1` 才是 success，`0` 为 failure，`(0,1)` 为 partial，其他值、缺失或异常为 unknown。此规则不改变训练原有 reward/成功率口径。advantage 不参与筛选。

完全重复按任务键、顺序动作类型/完整参数、动作前后截图 hash 及观察字段判断，不包含 ID、时间、分数或推理文字。同路径保留更优版本；排序优先：已知执行错误数、环境步数、动作数、模型调用数、执行状态未知的动作数、trajectory ID。无语义相似度模型。

容量按库内所有文件的逻辑字节数统计，包含正式记录、staging、图片文件和配置，不等同于磁盘分配块、文件系统元数据或实际剩余磁盘空间。新布局为每次出现的图片保留顺序文件，不再跨轨迹共享图片文件；相同图片的多个文件均计入预算。旧的平铺布局仍按内容去重。完整轨迹的行为去重仍依据图片内容哈希，不受任务目录、轨迹 ID 或文件序号影响。写入前同时计入本次 JSON 和新图片，多个客户端不能超额争抢预算。

空间不足时，优先从保有多条轨迹的任务中淘汰多余的低优先级记录，优先保留任务覆盖；不因新候选挤掉某任务的唯一代表。不足则放弃候选，日志说明原因。staging 不会为别的候选被自动淘汰。资源 GC 同时检查正式记录和 staging 引用；共享图片仍被引用就不删除。

容量满且只剩唯一代表或活跃 staging 时，新任务也可能无法进入；这是硬预算下的明确取舍。空间预留可能先淘汰旧的多余记录，随后候选写盘失败，不保证淘汰回滚。文件系统 ENOSPC 仍会被单独捕获。

## 崩溃恢复

直接写入在每次操作期间持有 owner 锁，HTTP writer 在服务生命周期内持有该锁。离线导出会全程独占库；它会阻止 Agent 写入，不能在训练期间执行。进程重启保留 staging 并继续计费，不按心跳超时猜测 Session 已死。训练/所有 Agent 和可能使用的 HTTP writer 都停止后，才能明确清理残留：

```bash
python -m relax.agentic.archive.export recover \
  --archive-dir /data/outputs/rollout_result/example-run/sessions \
  --shared-filesystem --confirm-no-active-sessions
```

该命令持有独占锁，只删除本库 staging 并回收无引用资源。标志是操作者对所有 Agent 已停止的确认；不要对仍在运行但暂时未写入的 Session 使用。导出始终忽略 staging，不要求先清理。

## 离线导出与 SFT

训练结束且所有 Agent 和可能使用的 HTTP writer 停止后（命令中的路径需替换为实际值）：

```bash
python -m relax.agentic.archive.export export \
  --archive-dir /data/outputs/rollout_result/example-run/sessions \
  --shared-filesystem --output-dir /data/cua-export-001
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
