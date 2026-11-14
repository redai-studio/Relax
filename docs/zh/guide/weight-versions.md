# 独立权重发布与消费

权重版本通路将训练端导出与推理端加载分开。训练端沿现有 Megatron 转换器导出逻辑权重，发布带内容摘要的完整版本或差量包；消费者通过 SharedFS 或 TCP 获取版本，在本地重建完整 safetensors checkpoint，再交给 SGLang Loader 安装。

安装期间暂时关闭该引擎的生成入口。所有参与进程完成加载、物理权重核对、缓存刷新与内部生成探针后，消费者开放请求并确认消费进度。下载和重建在这段暂停之前完成。

## 启用方式

源端文件操作使用 `source_io_timeout`（默认 3600 秒）。训练主线程持续交换各进程的完成状态，后台线程只处理 CPU 张量和文件；设备拷贝与训练通信仍在主线程。超时后先等待正在执行的文件操作退出，再释放临时目录，并报告重试同一目标所需的状态。所有参与进程均已成功完成的文件操作返回实际结果，同时记录耗时提示；文件操作报错仍按失败处理。

发布候选、导出临时文件和分片缓存按流及进程隔离并持有独占锁。重启时保留发布日志指向的候选，回收同一所有者留下的未完成副本；删除失败记录在 `cleanup-*.json` 中，下次启动重试。早期版本中没有所有权标记的目录需要单独核实归属。

TCP 的 `timeout` 限制连接及帧传输；控制操作另有有限的 `control_timeout`（默认 120 秒），覆盖容量准入和持久化确认，同一操作的重试共用该预算。消费者在 transport 配置中设置，源端通过 `serve --control-timeout` 设置。容量扫描使用有界的元数据并发，硬链接内容仍只计一次。`full_timeout`（默认 600 秒）限制同目标 FULL 修复。服务端通过 `serve --full-timeout` 配置对应预算。并发的同目标请求共享正在进行的修复；后续请求仍重新核验已有 FULL 对象。消费者在关闭生成入口之前，按运行时 TP/PP/EP 布局逐项检查 checkpoint 的名称、形状和数据类型，包括整数缓冲区；安装后继续逐位核对设备权重。单次校验最多保留四个 safetensors 读取器，避免反复解析大型 MoE 文件头；成功或失败后均关闭，下一次校验重新打开。

此功能通过两个参数显式启用；省略参数时沿用现有同步路径。

```bash
python -m relax.entrypoints.train \
  --weight-version-config /path/to/publisher.json \
  --weight-consumer-config /path/to/consumer.json \
  ...
```

`publisher.json` 示例：

```json
{
  "store": "/path/to/published",
  "scratch": "/path/to/publication-scratch",
  "stream": "policy",
  "disk_budget": 34359738368,
  "max_weight_bytes": 8589934592,
  "anchor_interval": 20,
  "compression": false,
  "full_only": false,
  "source_delta": false
}
```

容量字段以字节为单位，上面的数值仅说明配置格式。`disk_budget` 约束版本仓库；源快照、发布临时目录、消费者仓库和重建 checkpoint 各占自己的空间。按实际模型大小、保留历史及部署位置分别配置和测量。

完整逻辑权重默认保存在仓库的 `materialized/` 下，与交付包共用 `disk_budget`。在首次发布前设置 `canonical_root` 和 `canonical_budget`，可以将它们放到独立持久卷，并单独限制这个仓库的占用。该卷按仓库划分子目录；实际位置保存在本地仓库元数据中，重启后继续使用，交付 manifest 不包含这些路径。回收仍由同一版本引用和消费游标决定，同时清理该版本的 canonical 数据。已有仓库保持原位置；迁移数据须在停止发布、消费和回收后单独处理。

例如，在上述发布配置中增加 `"canonical_root":"/path/to/local-canonical"` 和 `"canonical_budget":34359738368`，可将交付包放在共享目录、完整逻辑权重保存在发布 owner 的持久盘。TCP 服务在 owner 上使用记录的位置生成同目标 FULL。SharedFS 使用方可在 `transport` 中配置 `"repair_owner":{"host":"publisher-host","port":9010}`，由 owner 处理 FULL 修复请求，继续从共享目录读取对象。直接在 SharedFS 端生成 FULL 时，该进程需要访问 canonical 卷。发布进程和提供 FULL 的服务采用相同预算。

`source_delta=false` 使用完整转换器导出。`source_delta=true` 在 raw/direct 和 Bridge 导出路径中，按确认过的本地分片和转换映射选择 REUSE、PATCH 或 FULL。Bridge 先在原始 Megatron 汇聚坐标中应用元素替换，再调用现有 BridgeConverter 转换为逻辑权重；QKV、gate/up、专家和视觉参数沿用原转换器。源缓存缺失、未注册的映射及密集变化使用完整转换器；源副本摘要冲突会给出对应参数和分片，供训练运行时修复后重新捕获。

源端消融可设 `source_patch=false`：保留未变化组复用，变化组走完整转换器。`source_cache_root` 可将可重建的完整导出、逐 rank 分片基线和自适应转换缓存放到独立本地文件系统，全量与自适应导出使用相同配置位置；发布候选、日志和保留的 canonical 权重仍使用 `scratch` 与 store 配置。源缓存丢失后重新完整转换，已提交目标保持可读。容量准入保守检查两个文件系统。

同一主机上的独立 Ray 集群可设置不同的 `RELAX_SERVE_HTTP_PORT`，默认值为 `8000`。解析后的端口在 driver、服务客户端、worker 和 Controller 重启时保持一致。验证入口通过 `--serve-port` 设置；每个运行仍需独立分配 GPU、store 与消费目录。

PATCH 选择计入远程帧载荷、计划交换的填充字节和逐帧确认通信，与完整导出中 TP/PP/EP 发送量的下界比较。路径回执同时保留这些预算和实测计数。单 rank 且无导出通信时，变化张量使用 FULL，不变张量使用 REUSE。

`consumer.json` 示例：

```json
{
  "directory": "/path/to/node-local-consumers",
  "consumer_prefix": "rollout",
  "stream": "policy",
  "cache_budget": 34359738368,
  "checkpoint_budget": 34359738368,
  "max_weight_bytes": 8589934592,
  "start_version": 0,
  "mode": "all_versions",
  "transport": {
    "type": "sharedfs",
    "store": "/path/to/published",
    "disk_budget": 34359738368
  }
}
```

消费目录应位于引擎及其 worker 可见的同一节点文件系统。同一节点的引擎共享已校验对象和重建 checkpoint，各自保留安装状态与消费身份；最后一个使用者释放后才删除 checkpoint。节点内依次执行安装，避免多个 Loader 同时占用临时内存。托管 RolloutManager 在生成或评估开始前安装可用版本；重新启动或卸载过内存的引擎先恢复已确认版本，再处理后续版本。

TCP 传输配置使用 `{"type":"tcp","host":"publisher-host","port":9010}`。传输服务由独立进程启动，发布训练进程结束后仍可提供保留的版本：

```bash
python -m relax.distributed.weight_sync \
  --store /path/to/published --disk-budget 34359738368 \
  --max-weight-bytes 8589934592 \
  serve --bind 127.0.0.1 --port 9010
```

TCP 对象服务用于可信部署网络。跨主机访问时，由部署层提供访问控制及加密通道，并将绑定地址改为该通道可达的地址。

发布器默认使用 `encode_workers=4`，可设为正整数。CPU 块读取、编码、哈希和不可变内容写入采用有界线程池。内容锁避免重复写入，容量预留串行核算，所有写入完成后由单一负责人提交有序 manifest。发布 owner 按配置的线程数、其余源 rank 按单个工作窗口核算编码缓冲，每个窗口预留 64 MiB。成对耗时比较应使用相同线程数。

## 发布、安装和恢复

| 状态或结果 | 含义与操作 |
| --- | --- |
| 发布句柄 | 已接受候选；后台编码仍可能进行中。 |
| `STORE_COMMITTED` | 交付内容已落盘，消费者可发现和重建；源端继续确认快照基准。 |
| `SOURCE_CONFIRMED` | 训练侧已完成快照基准确认，返回 PublishedVersion。 |
| `ACTIVE` | 本次安装和生成探针完成，可接受业务请求。 |
| `SUSPENDED` | 推理内存已卸载；恢复内存后重新安装已确认版本。 |
| `WAITING_ACTION` | 记录失败阶段、目标版本和恢复操作；查看消费结果中的错误及操作提示。 |

发现对象缺失或内容校验失败时，消费者在当前操作期限内请求一次同目标 FULL；显式 FULL 请求仍可用于恢复操作。原有 FULL 的损坏对象由 owner 根据保留的 canonical 内容重建。交付包身份允许不同，重建内容摘要保持一致。

模型配置、逻辑参数清单与 dtype 在关闭生成入口前检查。安装请求绑定安装身份、worker 身份及操作内容；完成加载、核对和探针后，提交决定与对应 checkpoint 写入同一条持久状态。提交之后恢复该 checkpoint，提交之前可以恢复上一份可用 checkpoint。消费确认可以幂等重试。

`all_versions` 订阅的持久游标保护尚未确认消费的历史。确认在安装达到 ACTIVE 后写入，重复确认幂等。配额不足时发布返回容量问题，部署者可以扩容、调整已完成订阅或明确注销弃用订阅后重试。

`mode="latest"` 在生成边界固定最新目标，直接安装该目标，获取过程中仍重建全部必要依赖。该订阅保留当前已确认版本及其依赖，常规 anchor 保留策略负责较新的历史；需要逐版观察训练更新时使用 `all_versions`。同一消费身份的模式保持固定，切换模式时创建新的消费身份。

消费配置的 `operation_timeout`（秒，默认 1800）覆盖一次目标版本的准备、安装与确认；追赶多个版本时，每个目标分别使用该期限。TCP 请求、worker RPC、生成探针及各文件块之间共同使用剩余时间。准备阶段超时保留原来可用的引擎，已校验对象留在缓存中供重试；安装阶段超时关闭入口，返回目标版本和恢复操作。释放引用及关闭入口另有最多 5 秒的清理预算。文件系统调用及设备调用由底层运行时完成或报错后交回控制；如果运行时失去响应，按状态提示重启该引擎，再从保留的 checkpoint 恢复。

### 源端容量与计量

发布配置可增加 `source_limits`，其中 `ram_bytes` 是每台源主机的同步额外内存上限，`gpu_bytes` 是每个源 rank 的同步额外显存上限，`scratch_bytes` 是所有源 rank 临时目录的合计占用上限。预检在冻结快照前汇总各 rank，计算冻结权重、最大转换组、编码窗口、源分片基准、替换帧、逻辑导出、发布候选和 Bridge 原始汇聚缓存的同时占用；已有文件和仓库保留量也参与检查。

可用 RAM 与显存是采样时的观测值。估算需求超过观测值时记录告警，发布继续执行；显式配置的源端上限和存储预算仍按配置检查。实际分配或 I/O 失败时报告所在阶段，并保留原目标供重试。容量报告与源端路由等辅助诊断使用有界写入，诊断延迟不会阻塞发布。

主机内存同时检查实际可用量和 Slurm/cgroup 上层限制；显存不足时先释放本进程未使用的 PyTorch 缓存，再读取驱动可用量。同一磁盘或共享挂载上的临时文件与仓库写入合并核算。容量回执保存在 `scratch/capacity-*.json`；不足时返回具体资源、需求、余量和恢复操作，扩容或释放已不用的缓存后可重试相同更新。

RAM 估算计入内核可回收的干净、非活跃磁盘文件缓存；共享内存、活跃缓存、脏页和待完成写回仍按占用保留。估算同时受各级 cgroup 限额和宿主 `MemAvailable` 约束；缓存统计缺失时使用原始剩余额度。

Bridge 原始汇聚缓存按 Megatron 参数的实际形状和 TP 分片数计算，逻辑导出按 checkpoint 上限预留。词表填充、裁剪和共享权重转换导致两者大小不同时，各自保留对应的空间预算。

`scratch/metrics-v*-rank-*.json` 和 `publisher-metrics-*.json` 保存源汇聚、文件写入、后台发布、控制消息及重试记录。`aggregate_events` 按事件 ID 去重，分别汇总 collective 载荷、文件写入和 socket 字节。版本重试的新事件保留；同一回执重复采集只计一次。

消费者配置 `delivery_stage` 为 `X`（默认）表示第一个目标节点，为 `T` 表示其他目标节点；同一节点的消费者使用相同归属。多个托管节点共用配置时，可用 `first_target_host` 指定第一个目标节点，其他节点自动归入 T。TCP 的对象和 manifest 正文归入交付阶段，帧头及命令归入 C。准备失败时，已发生的传输保存在 `checkpoint_root/failed-metrics/`。这些计数与服务端发送账本逐方向核对。HTTP 安装控制将实际序列化请求和响应正文记入 `audit/control-metrics-*.json`，包含错误响应，保留为独立的 `http_body` 口径。

采集还保留加载前后的进程 I/O、实际存储类型和 NFS 客户端挂载 RPC 计数。设置 `RELAX_WEIGHT_METRIC_INTERFACES` 为逗号分隔的已核实网卡名，可增加节点网卡观测。挂载和网卡计数包含并发流量，应与隔离校准区间及存储端记录对应。应用字节、文件写入、CPU/GPU 搬运与网络发送量分开报告；完整通信比较使用同一序列、同一采集层级的 G/W/X/T/C 范围和证据。

观测记录包含主机、进程、起止时间和原始计数，供同一挂载的重叠区间对账。内存文件系统上的临时文件与发布内容同时计入主机 RAM 预算。TCP 服务增加 `serve --metrics-output /path/to/service-metrics.json` 后，可记录 socket 计数和按需 FULL 补包的写入；正常关闭时保存结果。服务异常退出时，缺失区间使用外部链路观测核对。

## 接入范围与验证

### 封存离线版本集合

选定版本发布完成后，可以封存其交付文件：

```bash
python -m relax.distributed.weight_sync \
  --store /path/to/published --disk-budget 34359738368 \
  --max-weight-bytes 8589934592 \
  seal --name evaluation --stream policy --versions 1 3
```

命令逐块验证所选版本及其依赖的编码对象和重建内容，通过后原子写入 `seals/evaluation.json`。清单记录目标版本、依赖版本、内容摘要、交付包身份及编码体积；模型配置由内容摘要绑定。回收操作保留整个集合，直到执行 `release-seal --name evaluation`。重复封存同一集合时重新验证文件；封存其他集合时使用新的名称。

封存成功后，训练生产者可以退出。制作独立离线目录时，复制清单、其中列出的 `versions/<stream>/<version>.json` 和它们引用的 `objects/` 文件，即可重建权重。源端缓存和放置在外部磁盘的 canonical 权重无需随附。离线目录应保留写权限，用于记录读取期间的保留引用。新消费者配置自己的仓库和容量后，可执行 `fetch --source-store /path/to/archive --stream policy --version 3 --checkpoint-output /path/to/checkpoint`；模型配置随发布内容恢复，推理引擎安装另行执行。

中断或失败的封存不会产生完成清单。恢复缺失或损坏的交付文件后，可以重新执行命令。解除封存只移除该集合的保留引用，版本删除仍由显式 `collect` 操作完成。

### 块复用、发布来源和节点预算

发布器和托管消费者默认启用 `canonical_blocks=true`：每个解码后的逻辑块按内容摘要保存，不变块由多个版本共同引用。一个张量的局部变化只增加发生变化的块；写标准 checkpoint 时依次读取这些块。源端冻结快照及标准 checkpoint 仍是完整内容，各自纳入临时空间预算。格式标记随仓库保留，服务重启或先于发布器启动时均按该标记读取。

manifest 的 `source` 记录训练运行标识、发布进程代次、捕获序号、rollout、优化器尝试次数和成功更新次数。初始化与第一次更新具有不同的 `update_id`；优化器跳过或 BF16 舍入导致内容相同时，内容摘要可以相同，来源记录仍各自保留。发布过程中各源 rank 定期交换等待状态，存储写入时间与单次 collective 的超时分开处理。发布重试沿用候选身份，持久发布记录中的 `attempt` 单独递增。计数从本次训练进程启动开始，与模型 checkpoint 中的全局训练步数分别解释。

同节点消费者在准备阶段共享 checkpoint 缓存与文件锁。配置相同的 `node_budget` 后，分别预留本组准备与安装的资源，合计容量允许时并发推进；同组操作保持串行。未配置节点预算时，沿用共享目录和节点安装锁串行执行。

`node_budget` 按部署实测的驻留量和操作峰值预留容量：`limits` 给出各资源上限，`resident` 包含同节点模型、KV cache 和 CUDA Graph 的驻留量，`operation` 给出本次准备及安装所需的额外峰值。三个字段都使用资源名称到字节数的映射，例如 `host` 和节点级 GPU 标识；同节点消费者共享相同的上限与驻留配置。

独立消费者通过 `node_directory` 指定节点共享目录；同节点的多个独立进程应指向同一目录，供安装锁和资源预留共同使用。默认目录是消费者缓存的父目录。容量不足时，操作状态记录所需容量、当前上限和恢复操作，已安装版本继续服务；恢复未完成的安装或显式扩大容量后可以重试同一目标。

内容相同的块使用 COPY；变化块在 SPARSE 和 RAW 中比较 payload 与 manifest 中编码元数据的合计字节数，保留 BF16 的原始位模式。`compression: true` 额外比较 zlib 和 XOR-zlib；相同大小时优先使用解码步骤较少的表示。部署前可用完整冻结序列的诊断结果权衡编码时间与传输量。按需新建的 FULL 恢复包沿用发布时的压缩选择，包括 DELTA 只含 COPY 块的情况；未记录该策略的旧 manifest 默认使用原始块。已有恢复包按其记录的编码重建，保持交付包身份。

安装中断后保留预留，待原消费者完成设备恢复再释放。节点内存采样、磁盘配额与设备核对分别提供验证依据。

### 模型和运行验证

当前 worker 扩展以 SGLang 0.5.17 为适配基线，使用仓库安装时注册的插件入口。推理验证器覆盖非量化 Qwen3 / Qwen3-VL Dense 与 MoE 的逐参数布局，安装适配器使用单节点。小模型 GPU 回归覆盖 Dense TP2、PP2 和 MoE TP2/EP2；每组比较同一六版输入的 FULL/DELTA 安装、图文生成及恢复结果。专家并行使用静态专家分配，动态专家重排和量化布局需要相应的物理验证适配。

使用与 checkpoint 相同的模型配置及 dtype 启动引擎。配置变化时创建新的引擎，再安装目标版本。物理权重核对由各 worker 分块读取设备参数，并与本地 checkpoint 的预期布局逐位比较；协调者汇总全部成员的覆盖范围。共享 embedding/head 同时检查逻辑别名一致性。全量设备快照与文件比较保留为集成测试的独立参照。

CPU 回归覆盖版本重建、发布重试、消费游标、安装恢复及通信计量：

```bash
python -m pytest -q tests/cpu/distributed/weight_sync
python -m pytest -q tests/cpu/backends/sglang/test_weight_version_verification.py
```

[FULL checkpoint 指南](./weight-checkpoints.md) 提供真实训练导出及独立 Loader 的 GPU 运行入口。端到端验收按同一冻结训练序列记录汇聚、发布写入、读取、目标分发及控制通信，同时记录同步耗时、推理暂停和吞吐。

### 诊断与运行状态

内存采样、I/O 计数和独立诊断文件使用有界等待。单次观测默认等待至多 0.2 秒；延迟或失败会记录告警和缺失原因，权重更新继续执行。诊断工作线程数量有上限，磁盘卡住时后续请求记录观测缺失。`memory_sampling.status` 与 `errors` 说明采样覆盖，已存在的采样文件原样保留。完整性能验收需要补齐相应观测记录。

生成请求在安装期间返回 503。响应包含 `status`、`reason`、`closed_seconds` 和 `action`；控制请求失败后会提示检查消费状态并恢复 checkpoint。向 `/relax_weight_version` 发送 `{"action":"status"}` 可查询当前阶段、开放状态与恢复提示，状态查询独立于安装控制锁。独立消费者等待源端版本时定期记录目标版本及剩余等待时间。

权重完整性、版本身份、安装回执和实际容量仍在功能路径中核验。已开始的文件写入保留输入直到完成；长时间等待会输出阶段及排查提示。诊断文件与这些持久化协议记录分别保存。

状态查询直接返回已有消费者的状态；尚未创建时返回 NOT_INITIALIZED，不创建会话或暂停引擎。已启用并确认的版本返回成功，旧 checkpoint 清理失败单独返回 cleanup=PENDING，并在后续消费中重试。operation.json 仅用于展示，写入延迟或失败不改变安装回执与消费游标。

版本模式在启动引擎时自动加入所需插件，并保留已有插件配置；进程启动后恢复调用方的环境。SGLang 0.5.17 是适配基线，入口按实际接口和权重布局校验兼容性。普通暂停、继续生成和 LoRA 接口沿用引擎行为；安装期间的冲突操作返回当前原因，基座权重更新经安装适配器完成。

独立消费者的 wait_seconds 限制等待每个目标发布的时间，可设为任意有限非负值。安装 worker、生成探针和文件锁使用当前操作的剩余期限。配置 node_budget 时，不同消费组在合计容量允许的情况下并发推进，每组分别串行安装；未配置时保留同节点串行安装。
