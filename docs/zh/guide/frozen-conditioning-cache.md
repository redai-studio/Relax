# 冻结条件特征缓存

图像生成 SFT 遇到相同的有效输入时，可以复用冻结条件编码器的确定性输出。Relax 将这些张量保存到有容量上限的节点本地文件缓存，同一节点的 worker 可共享 CPU 映射，避免每个进程长期保留一份独立副本。

当前接入包括 SenseNova 图像编辑 LoRA SFT 和 Qwen-Image 文生图 LoRA SFT。SenseNova 复用参考图与文本联合前缀的 KV 张量及图像位置索引；Qwen-Image 复用冻结文本特征，两者共用公共缓存。SenseNova 全参训练的条件参数可训练，因此仍在线计算。其他模型适配和 RL 路径需要单独接入；本功能不负责从 SGLang 或 RL rollout 引擎向训练端传递条件特征。

下文启动示例与分支分组优化针对 SenseNova。Qwen-Image 的数据格式、独立编码器边界和启动脚本见[文生图 SFT](./t2i-sft.md)。

## 启用 SFT 示例

先按[图像编辑 SFT](./i2i-sft.md) 准备已有 Ray 集群、模型、配对图片 metadata，以及训练 worker 可访问的存储。为实际负载设置 `MODEL_PATH`、`DATA_ROOT`、`METADATA_PATH`、`DATASET_SIZE` 和 `SAVE_DIR`，其中 `DATASET_SIZE` 必须等于 metadata 行数。另外设置 Ray Jobs 地址、可写的节点本地 `CACHE_DIR`，以及用于收集测量文件的全新共享目录 `RUN_DIR`。

在仓库根目录运行对照示例：

```bash
: "${RAY_DASHBOARD_ADDRESS:?设置 Ray Jobs HTTP 地址}"
: "${MODEL_PATH:?设置训练 worker 可访问的模型目录}"
: "${DATA_ROOT:?设置训练 worker 可访问的图片根目录}"
: "${METADATA_PATH:?设置配对图片 metadata 路径}"
: "${DATASET_SIZE:?设置准确的 metadata 行数}"
: "${SAVE_DIR:?设置 checkpoint 输出根目录}"
export RAY_DASHBOARD_ADDRESS MODEL_PATH DATA_ROOT METADATA_PATH DATASET_SIZE SAVE_DIR

RELAX_CONDITION_CACHE=1 \
RELAX_CONDITION_CACHE_MAX_BYTES=4294967296 \
RELAX_CONDITION_CACHE_DIR="${CACHE_DIR:?设置可写的节点本地缓存根目录}" \
RELAX_CONDITION_CACHE_RUN_DIR="${RUN_DIR:?设置全新的共享指标目录}" \
USE_CLEARML=0 \
bash scripts/entrypoint/ray-job.sh \
    scripts/training/diffusion/run-sensenova-condition-cache.sh
```

示例为每个节点选择 4 GiB 的 payload 容量。应先为模型、数据准备和 staging 缓冲留出空间，再根据 host 或 tmpfs 余量设置容量。脚本默认使用 8 张 GPU、有效全局 batch 8、64 个 optimizer steps，以及一张条件图。`GPUS` 可取 1、2、4、8，必须整除固定的有效 batch；多节点时，`GPUS_PER_NODE` 还必须整除 `GPUS`。缓存只在节点内共享，不跨节点共享。

建立在线基线时，用相同命令设置 `RELAX_CONDITION_CACHE=0`，改用全新的 `RUN_DIR` 和独立实验输出目录。保持数据顺序、分辨率、GPU 布局、optimizer、预取和训练配置不变。`DRY_RUN=1` 可打印提交命令而不启动任务。

## 配置

以下变量由图像 SFT runtime 读取。对照脚本会将其传入 Ray worker；自定义启动脚本也应将这些变量加入 worker 的运行时环境。

| 环境变量 | Runtime 默认值 | 作用 |
|---|---|---|
| `RELAX_CONDITION_CACHE` | `0` | 启用冻结条件复用；设为 `0` 时在线编码。 |
| `RELAX_CONDITION_PROFILE` | `0` | 缓存关闭时也启用条件阶段测量。对照脚本默认设为 `1`。 |
| `RELAX_CONDITION_CACHE_MAX_BYTES` | `8589934592` | 每个节点的 payload 字节上限。启用缓存时必须为正值。 |
| `RELAX_CONDITION_CACHE_DIR` | `/dev/shm/relax-conditioning` | 同一节点训练 worker 共用的可写本地根目录。建议使用绝对路径，也可选择本地磁盘。 |
| `RELAX_CONDITION_CACHE_RUN_DIR` | 未设置 | 写入 `events-rank*.jsonl` 的全新 worker 可见目录；设置该变量也会启用测量。脚本默认使用 `${SAVE_DIR}/${EXP_NAME}/metrics`。 |
| `RELAX_CONDITION_VERIFY` | `0` | 对首个有效训练 batch 比较在线与缓存路径，要求启用缓存并设置 run 目录。应与吞吐测量分开运行。 |
| `RELAX_CONDITION_VERIFY_RTOL` | `0.01` | loss、累积梯度及 FP32 参数更新量的相对误差阈值。脚本接受 `0` 到 `0.01`。 |
| `RELAX_CONDITION_VERIFY_SNAPSHOT` | `0` | 验证通过后保存参考在线路径的逐 rank 梯度和 FP32 更新量，供跨配置比较；仅用于数值验证。 |
| `RELAX_CONDITION_PREFETCH_DEPTH` | `0` | 预取未来条件的最大条目数；为 0 时关闭。对照脚本要求同时开启图像预取。 |
| `RELAX_CONDITION_PREFETCH_MAX_BYTES` | `536870912` | 每 worker 的预取 lease 字节上限，包含正在查找的预留和消费中的借用。 |

缓存关闭且未设置测量变量时，SFT 使用原有在线路径。启用缓存也会启用相关测量；`RELAX_CONDITION_PROFILE=0` 不会移除缓存所需的协调和统计。

若设置了 HTTP 代理，应通过 `NO_PROXY` 和 `no_proxy` 将 localhost、Ray head 及节点内网地址排除，确保 Ray Jobs 和节点发现请求直接访问集群。

对照脚本默认采用 BF16 计算、FP32 optimizer master，以及 rank/alpha 均为 32 的 LoRA。其他默认值和提交参数见 [`run-sensenova-condition-cache.sh`](../../../scripts/training/diffusion/run-sensenova-condition-cache.sh)。这些是示例配方的选择，并非全局参数解析器默认值。

对照脚本还提供两个独立的训练执行开关。`PREFIX_GRADIENT_CHECKPOINTING=0` 关闭模型的激活重算；虽然名称包含 prefix，它也控制生成 decoder。显存开销需在实际负载上测量。`FSDP_BRANCH_GROUPS=1` 将 `fsdp_branch_groups: true` 写入 SFT 配置，把每层的理解和生成模块分成独立 FSDP 通信组，使未执行的分支不随整层聚合权重。该开关默认关闭，只允许条件分支全部冻结；full 或理解侧可训练的 LoRA 会在分片前明确报错。启用后不能在同一个 FSDP 实例中解冻条件分支，需关闭开关并重新创建训练任务。公共 FSDP runtime 消费适配层的分组声明，不依赖 SenseNova 层名。

上述开关必须同时应用于在线和缓存对照，分别报告训练配置收益与缓存收益。不要把配置优化后的缓存路径与旧配置在线路径直接作为缓存增益对照。

可选参考快照位于 run 目录的 `reference-snapshots/`。它保存完整有效 batch 归约后的梯度、真实 FP32 `theta_after - theta_before`，以及初始参数、optimizer、输入和 RNG 指纹。所有 rank 文件原子写入并校验后才发布 manifest；失败会恢复验证前的训练状态。快照是保留的验证产物，不计入缓存 payload 配额，也不会由缓存关闭逻辑删除；不应在吞吐实验中开启。

## 复用边界与失效

缓存键由 checkpoint/tokenizer 文件内容指纹、适配层声明的 schema，以及有效输入依赖共同生成。SenseNova 的依赖包括 prompt、按顺序排列的参考图内容与尺寸、目标尺寸、实际 token ID、特殊 token ID、位置索引、图像 grid、提示词模板、预处理参数、attention 设置和 dtype。样本重命名不会使其自动可复用；同一文件名下的内容变化也不会沿用原命中。

每次查找都在 CPU 上准备实际编码输入，在线计算直接消费同一份结果，避免构键后再次 tokenize 产生差异。词表或 tokenizer 设置变化只要影响有效 token 或索引就会失效；若这些设置变化后，编码器的全部有效输入仍完全相同，冻结确定性输出可以继续复用。无需每次序列化整个词表。数值验证的在线参考单独执行原有 tokenize 与 device 索引路径，检查这次准备流程的数值一致性。

Runtime 在启动时计算 checkpoint 与 tokenizer 文件指纹。每个样本开始前，SenseNova 还会在 FSDP forward 外检查条件参数的身份、版本计数（含 DTensor 本地存储）、可训练性、dtype、shape 和 stride。还会检查条件分支的注册 buffer 及实际 RoPE、归一化、attention、视觉和激活配置。SenseNova 在确定性构造延迟初始化的位置张量时记录 metadata；正常首次初始化及受支持的设备迁移保留复用，外部替换或原地修改则停用缓存。观察到标准 `copy_` 等原地更新、参数替换或条件参数变为可训练后，会对当前模型实例永久停用复用，所有 rank 继续通过协调后的在线路径计算。即使恢复原参数或重新挂载 resolver，也不会重新启用缓存，必须启动新的训练任务。运行时不会为变更后的编码器权重自动重建缓存。

该防护无法识别 `.data` 等绕过版本计数器的非标准写入。运行期间仍应保持 checkpoint/tokenizer 文件不变，因为启动时的文件指纹不会刷新。有效 tokenizer 与预处理设置继续参与每次查找的依赖键；修改适配层编码算法或特征 schema 时，也应更新声明中的 schema 版本。

SenseNova 根据实际条件参数的 `requires_grad` 状态判断能否复用。条件计算包含可训练参数时，会绕过缓存读写并保留 autograd 图。生成侧 LoRA 权重继续更新时，不依赖这些权重的冻结条件仍可复用。

LoRA 训练中，适配层为视觉编码器和文本 embedding 设置独立的 FSDP 边界，有效命中会跳过对应调用及权重 all-gather。未参与这条训练路径的语言模型 head 也有独立边界，因此训练步不会为它执行权重 all-gather。在线与缓存路径使用相同边界，测量复用收益时应将这项策略保留在基线中。

图片解码和样本准备仍发生在缓存查找之前。有效命中会跳过重复的条件预处理、输入搬运和编码，但不会缓存监督目标的训练计算、采样噪声、timestep、生成轨迹或奖励。若适配层包含随机裁剪、增强或 caption dropout，必须保留原有执行顺序，并对产生的有效输入构造键，不能悄悄沿用上一次随机选择。本次接入不会为 SenseNova 新增这些随机增强。Qwen 可选扩展只复用确定性的 VAE 后验分布参数，采样本身仍逐次在线执行，见[文生图 SFT](./t2i-sft.md)。

## 存储、所有权与释放

缓存通过原子发布写入不可变的 safetensors 文件，每次查找都计算完整文件的 SHA256 校验，CPU 张量保持原 dtype。查找返回的 lease 会阻止使用中的条目被淘汰。支持的对齐布局只搬运一次文件数据区，并从同一 GPU storage 建立原 dtype 的张量视图；其他布局逐张量复制。SFT runtime 等待当前复制 stream 的完成事件后再释放 lease，包括搬运中途异常的情况，并为当前 forward 创建新的模型专属包装对象；缓存本身不长期持有 GPU 特征。

Payload 上限同时覆盖已发布文件、正在借用的条目和待发布文件写入。LRU 只淘汰未被借用的条目；条目无法容纳时会 bypass，训练继续在线编码。序列化另有独立 staging 上限，在 CPU 复制前检查。SFT runtime 将该上限设为 `RELAX_CONDITION_CACHE_MAX_BYTES` 的相同数值，因此 8 GiB 的 payload 配置还允许最多 8 GiB 的 writer staging 预留。公共缓存 API 为自定义接入提供 `max_staging_bytes`。条目准入使用保守缓冲估计，因此文件大小能放下，也可能因 staging 不足而 bypass。

Payload 与 staging 上限不是进程 RSS 总上限。训练输入、GPU 副本、Python metadata 和每个 reader 的小型校验缓冲也占用内存。JSON 索引与 client 标记数量受缓存条目及活跃 worker 数量约束，详细资源快照会单独报告 metadata。使用 tmpfs 时，payload 文件以共享内存存储占用 host 内存；使用本地磁盘时，相同 payload 上限约束文件内容，操作系统页缓存仍可能占用 host 内存。

Worker 按训练 attempt 和节点共享 namespace。正常结束、取消或进入异常清理时，每个 worker 释放自己的 lease 和 client，最后一个活跃 client 删除 namespace。`SIGKILL`、`os._exit` 或 worker 丢失后，操作系统会释放锁；下一次缓存初始化确认所有 client 和 payload 锁均已释放后，回收遗留 namespace。本功能没有后台清理 daemon。后续任务会建立自己的缓存，不把这些文件当作可恢复的训练状态。

容量与 staging 预算作用于每个节点上的单个训练 namespace；并发训练任务的预算需要相加，不存在跨任务的全局内存配额。性能对照应避免其他任务争用同一节点资源。

## 适配层接口与分布式执行

多个 writer 共用 staging 总预算，在短临界区内预留容量，锁外执行 CPU 搬运、序列化和写文件，再短锁原子发布。独立 owner 锁保护在途写入，恢复逻辑回收失效 writer 的临时文件和预留；关闭 client 会等待读写结束。

可选条件预取将未来样本的文件读取和 SHA 校验与当前反向/更新重叠。它只借用图像队列中已经解码的样本，依赖准备仍在训练线程执行，实际 forward 重新构造内容键并检查冻结状态。单个后台 reader 的预算按完整文件字节限制，消费中的 lease 直到 GPU 复制完成才归还。它不建立 GPU 常驻缓存或额外 pinned pool，GPU 搬运仍在训练线程执行；更长的 lease 可能影响容量不足时的淘汰。

滑动窗口保留的同一预取游标只准备一次提示；离开窗口、异常或关闭后立即丢弃。提示数量受预取深度限制，仅用于安排 I/O，不能按 sample ID 授权特征复用；实际 forward 仍独立检查输入与冻结状态。

消费预取结果前会检查文件身份、大小和时间戳，变化时回退同步查找；每次新查找仍做完整 SHA。退出先停止并等待后台 reader，再关闭缓存；后台工作和清理计入生命周期。`prefetch_hit` 统计实际消费数，`prefetch_stats` 单独报告后台读取、等待、丢弃和预算峰值，不能替代有效样本命中率。

公共声明位于 [`relax/models/generative_sft.py`](../../../relax/models/generative_sft.py)。适配层提供 `ConditioningSpec(name, dependencies, cacheable)`，并在模型正常的 FSDP forward 边界内调用 `ConditioningResolver.resolve(spec, compute, restore)`。

- `name` 表示特征 schema 版本；`dependencies` 使用可表示为 JSON 的内容、预处理和 dtype 信息，runtime 补充权重指纹。
- `cacheable` 必须反映实际冻结参数边界。为 false 时，在线结果保留梯度，不得存储或 detach。
- `compute` 返回扁平张量字典；`restore` 检查字段名、shape、dtype 和 device，并重建当前 forward 所需的模型对象。它不得修改借用张量，也不能在 lease 结束后继续持有 CPU 映射。

SenseNova 额外提供 `conditioning_state()`，供状态变更防护使用。可选的 `semantic` 字段只接受 CPU 侧 JSON 配置，公共 runtime 保存其不可变规范化摘要；配置变化或声明无效时停用复用。适配层也必须将影响输出的配置加入每次查找的依赖键。SenseNova 对此使用 `sensenova.prefix.v3` schema。进程内的参数身份只用于检测替换，不进入共享缓存键。自定义适配提供这一 hook 时，应在 FSDP forward 外检查分片 master 的 metadata，不能将 FSDP 临时展开的完整参数视图误判为编码器被替换。

不同 rank 可以处理不同输入。跳过编码器前，runtime 要求整个 FSDP 参数分片组中的每个 rank 都已取得并校验自己输入所需的条件。任一 rank 未命中或恢复失败，该 microstep 的所有 rank 都执行在线编码，保持 FSDP collective 一致。因此，本地命中也可能转为强制重算，分析收益时应看有效命中率。

完整冻结副本可以使用独立的 rank-local 决策：适配器声明参与条件计算的模块，后端确认它们
均由原生 FSDP 排除分片，且没有嵌套 FSDP 边界或分布式参数。适配器还须保证 compute/restore
没有显式 collective。runtime 在初始化时校验各 rank 策略一致，并在整个生命周期中保持不变。
默认仍是全组一致决策；内容键、参数失效检查、容量与释放规则在两种策略下相同。

文件缺失、checksum 失败或适配层发现 schema/shape 错误时，都会退回在线计算。失效只影响相应 key，不清空其他条目；仍在借用的条目延后删除。存储层不依赖 SenseNova 的层名，其他 SFT 适配可以共用 resolver 和缓存约定。未来接入 RL 时，仍需定义对应依赖、所有权和分布式跳过边界；当前没有实现 RL 特征传递。

## 衡量实际负载

确定性条件重复充分、原有编码开销高于哈希、查找、协调与搬运开销时，复用才可能带来收益。低复用数据也可能变慢。条件特征可能比原始输入更大，因此命中并不意味着搬运字节一定减少。

为重复在线/缓存对照使用独立 run 目录，统计包含数据准备、前向、反向和 optimizer 更新的完整训练循环。单独报告启动与缓存构建成本，再比较稳态吞吐。冷启动窗口应根据数据量和预期复用方式确定，不要直接套用汇总工具默认的 16 步。

```bash
python scripts/tools/benchmark_condition_cache.py \
    --online "${ONLINE_RUN_DIR:?设置在线路径 run 目录}" \
    --cached "${CACHED_RUN_DIR:?设置缓存路径 run 目录}" \
    --cold-steps "${COLD_STEPS:?设置冷启动窗口步数}" \
    --output "${REPORT_JSON:?设置汇总 JSON 输出路径}"
```

工具支持每组传入多个 run 目录，并用 `--low-reuse-online`、`--low-reuse-cached` 单独比较低复用负载。它比较相匹配的完整任务，采用每步最慢 rank 的时间，全局 batch 只计一次。局部编码耗时仅用于诊断，不能代替端到端吞吐。完整 Ray job 时长还需要另行收集 `job.json`；trainer 事件本身不包含模型与 Ray 启动。

应结合 `encoding_calls`、`local_hit`、`effective_hit`、`forced_recompute`、输入与特征搬运字节，以及缓存查找/写入时间分析结果。每步记录 CPU RSS、GPU allocated/reserved 和节点整体共享内存用量。大部分步骤调用 `stats_light()`，以 `cache_storage_scope=index_accounting` 标记：条目数、resident 字节和预留量来自加锁索引，不扫描 payload。该快照没有测量 `leased_bytes`、`metadata_bytes`、`total_bytes`，因此这些值为 `null`，表示未知，不能视作零或沿用上次详细读数。

首个实际执行的步骤、step 索引为 16 倍数的步骤以及末步调用完整 `stats()`；成功的文件系统快照标记为 `cache_storage_scope=filesystem`、`cache_stats_detailed=true`。Close 也会在释放 client 后请求详细统计，但已关闭缓存返回的是 `last_detailed_snapshot`，不代表实时资源，runtime 会将当前缓存与磁盘占用保留为未知。应结合 namespace 的文件系统状态及外部资源释放前后快照核实清理。同一节点共享的缓存字节不能按 rank 重复求和。有效命中率较低时，先检查输入或键变化、staging bypass、容量压力和其他 rank 的 miss，再考虑增大缓存。

数据与恢复行为见[图像编辑 SFT](./i2i-sft.md)，负载测量建议见[性能调优](./performance-tuning.md)。

可复用的计划执行、来源校验、重复实验汇总及 FP32 快照比较工具见[实验工具说明](../../../scripts/benchmarks/conditioning_cache/README.md)。
