# TransferQueue host-RDMA 使用与运维

## 范围与配置

Relax 默认使用 TransferQueue SimpleStorage。首期 RDMA 支持只接入 MooncakeStore/host-RDMA，不改变 payload 形状或数据分发语义，也不支持 GDR。生产路径只有 host-RDMA 和 SimpleStorage；Mooncake/TCP 仅作为跨节点 benchmark 的 C1 对照。

| 参数 | 语义 |
|---|---|
| `--tq-rdma-mode=off` | 默认值；保持 SimpleStorage 数据路径，不执行 Mooncake 检查或集群 handshake；worker 的 TQ attach 仍有 60 秒边界 |
| `--tq-rdma-mode=auto` | 尝试 host-RDMA；任一检查或节点 attach 失败时，完成清理后统一回退 SimpleStorage |
| `--tq-rdma-mode=required` | 要求 host-RDMA；不可用时完成清理并终止启动，不允许降级 |
| `--tq-rdma-device=<device>` | 指定 RDMA 设备；空值由 Mooncake 原生逻辑选择，多 HCA 环境建议显式指定 |

Relax 固定 Mooncake `use_gdr=false`。GDR 若有实际需求，应由独立 PR 实现并做专项验证。

## 部署前提

首期只支持**单任务独占 Ray 集群**：同一个 Ray cluster 在初始化和运行期间只能有一个 Relax job，不支持 concurrent initializer、多 job admission 或复用其他作业的 TransferQueue controller。

Mooncake master 由部署环境管理，Relax 不启动、重启或停止它。driver 必须设置外部 endpoint：

```bash
export MC_MASTER_ADDRESS=master.example:50051
```

driver 将该 endpoint 写入 job-level TQ config，owner 和 worker attach 复用已存储配置，因此 worker 节点不要求重复设置该环境变量；但所有节点都必须能访问同一个 endpoint。Relax 只检查 `host:port` 格式；DNS、路由、防火墙和 master 健康状态由真实初始化与 attach 验证。

## 启动、回退与清理

`off` 直接初始化 SimpleStorage，不创建 owner actor 或执行 Mooncake handshake；各 worker 仍通过有界 helper attach 到 SimpleStorage，以避免旧版无界等待。启动时还会回收不可用的半初始化 controller，并因首期独占集群约束拒绝复用健康的既有 controller。`auto` 和 `required` 按以下顺序执行：

1. 校验 mode、TransferQueue/Mooncake correctness contract、master 格式和 segment 容量。
2. 在独立 owner actor 中有界执行 Mooncake `tq.init`。
3. 在每个 ALIVE Ray 节点运行一次性 worker（`max_calls=1`、`max_retries=0`），真实 attach 并确认 `MooncakeStorageManager` 和 `protocol=rdma`，随后 detach。
4. 全部节点通过后启用 host-RDMA；任一失败则等待 worker 终止，并清理 owner、controller 和 segment。
5. 清理可确认时，`auto` 初始化 SimpleStorage，`required` 报错；清理无法确认时两种模式都 fail closed。

Relax 不再维护 `/sys` 启发式能力探测。真实 attach 是运行时能力判据，但它只证明 manager、配置 protocol 和 setup 成功；线路是否真正传输 RDMA 数据必须由 benchmark counter 的 wire proof 证明。

owner 初始化和 worker attach 默认各有 60 秒边界；worker attach 可通过 `RELAX_TQ_ATTACH_TIMEOUT_SECONDS` 调整。RDMA 不可用时，`auto` 需要完成真实初始化、全节点 attach 和清理，启动可能达到分钟级。已知无需 RDMA 的作业应使用 `off`。

健康的既有 controller 不会被接管或关闭，启动会直接失败；半初始化、超时或已死亡的 controller 才允许回收。全局 `tq.close()` 只由 owner 调用，普通 worker 只 detach 本地 client，Mooncake master 始终不由 Relax 管理。

## Segment 与容量

真实 handshake 会在每个 ALIVE 节点（包括 CPU-only head）创建 Mooncake client，瞬时挂载并注册完整 client segment，结束后立即 detach。默认配置为每 client 4 GiB global segment 和 1 GiB local buffer；实际 RSS、锁页内存及注册资源由 Mooncake 实现决定。内存或 `memlock` 不足会表现为 attach 失败。

global segment 可按部署容量调整：

```bash
export RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB=8
```

启动前容量预检采用保守上界：文本按 `seq_length × 32 B`；多模态按当前支持的最大 transported tensor layout（16×16 spatial patch、temporal patch 2、RGB、2×2 spatial merge、float32）另加 `seq_length × 24,576 B`。因此 8,192 token 的单样本多模态 tensor 上界约 192 MiB。最终再乘 rollout batch、`n_samples_per_prompt` 和 `max_staleness + 1`。容量不足时 `auto` 回退 SimpleStorage，`required` 失败。增大 segment 时必须同步规划每个节点的瞬时内存与 `memlock` 足迹。

异常退出时 client segment 可能继续注册到 master，直到 Mooncake `client_ttl`（默认 30 秒）到期。

## Correctness 与依赖 gate

当前 Relax pin 为 TransferQueue `9784ad0d8a5f0db93ff8326379ab2a6d06e640a0`，并要求 `MOONCAKE_CORRECTNESS_CONTRACT_VERSION >= 1`。Contract version 1 保证：

- `batch_upsert_from` 和 `batch_get_into` 的每次 batch/retry 都校验返回结果与请求 key 等长；
- `batch_remove` 的非幂等失败向调用方传播；
- `NOTIFY_DATA_UPDATE_ACK` 验证 positive ACK，controller 拒绝时 producer 不会按成功结束。

这些修复应在 TransferQueue 上游实现；Relax 不使用 monkey patch 替代依赖修复。

测试确认 `mooncake-transfer-engine==0.3.10.post2` 的 TCP memcpy 路径存在静默截断风险。Relax 当前会统一强制 `MC_STORE_MEMCPY=0`，显式设置不安全值会拒绝启动；只有在能够可靠识别已修复 build 后，才重新评估是否允许 memcpy 路径。

每次验收必须记录 Relax commit SHA、TransferQueue commit 和 Mooncake 版本，不能只记录分支名。

## 跨节点验收

唯一保留的 benchmark 是 `scripts/benchmarks/tq_cross_node_bench.py`：

| 档位 | 后端 | 用途 |
|---|---|---|
| C0 | SimpleStorage | 默认路径基线 |
| C1 | Mooncake/TCP | benchmark 对照，不是生产配置或 fallback |
| C2 | Mooncake/host-RDMA | 生产 RDMA candidate |

每个 protocol 必须使用全新 Python 进程和独立 CSV。示例仅展示 C2；C1 改用 `--protocol tcp` 并删除 `--device`/`--rdma-port`，C0 改用 `--protocol simple` 并额外删除 `--master`：

```bash
python -u scripts/benchmarks/tq_cross_node_bench.py \
  --protocol rdma \
  --master master.example:50051 \
  --consumer-node-id <consumer-node-id> \
  --device <rdma-device> \
  --rdma-port 1 \
  --tcp-device <network-interface> \
  --payload-profiles synthetic multimodal \
  --payload-mib 256 1024 2048 4096 \
  --repeats 5 \
  --csv c2-rdma.csv
```

所有档位都必须 byte-exact。C2 只读取明确指定的 HCA/port，要求一次完整 `put → get` round 的 idle-adjusted IB receive bytes 至少覆盖 raw payload 的 80%；完整 round 可以覆盖对象随机落在 producer、consumer 或 owner segment 的情况。C1 要求 idle-adjusted TCP receive bytes 至少覆盖 20%，C0 只要求观测到跨节点 TCP，因为 SimpleStorage unit 可能位于 consumer 本地。端口级 counter 不是 per-flow 指标，正式验收必须使用静默或独占的数据端口；CSV 同时记录 raw delta、紧邻 round 的 idle rate 和扣除后的 proof bytes，不能在共享端口有显著背景流量时宣称 wire proof。

C1 会在 driver 及其 Ray worker runtime 中设置 `MC_TCP_ENABLE_CONNECTION_POOL=1`，这是当前 Mooncake/TCP correctness baseline 的组成部分，必须随结果一并记录。benchmark 通过一次性 owner lifecycle 有界初始化 TQ，producer/consumer 均有界 attach；任一失败都非零退出，不执行 backend fallback。

CSV 使用 exclusive-create，不覆盖既有文件。每个 warmup/测量 round 在清理 partition 前写入并 flush 一条 `phase=measurement` 记录，清理结束后追加 `phase=final` 记录。两条记录通过 `protocol/profile/payload_mib/run` 关联；统计最终结果时只使用 `phase=final` 行。测量通过但尚未清理时 `status=pending`，只有校验与清理均成功才记录最终 `status=pass`。`cleanup_status` 和 `cleanup_error_kind` 单独记录清理结果；只有 measurement 行表示该轮未完成，不能算验收通过。校验失败使用稳定的 `error_kind`，同时发生清理失败时仍保留校验错误为主异常。每行还记录 Relax SHA、安装的 TransferQueue VCS commit 和 Mooncake package version，且 Relax tracked worktree 不干净时拒绝作为正式验收运行。

真实多模态 fixture、原始 CSV、版本信息和性能分布属于 PR 验收附件，不在仓库文档维护生成教程或易过期的性能数字。没有双节点 RDMA 环境时必须明确记录“真机验收未执行”，不能用 mock 结果替代。

## 排障

| 现象 | 检查与处理 |
|---|---|
| correctness contract 不满足 | 核对各节点 TransferQueue/Mooncake 版本和 capability marker；不要绕过 gate |
| master 缺失、格式错误或不可达 | 检查 driver 的 `MC_MASTER_ADDRESS`，并检查所有节点到该 endpoint 的 DNS、路由、防火墙和 master 服务 |
| segment capacity insufficient | 减少 batch、采样数或 staleness，或增大 `RELAX_TQ_GLOBAL_SEGMENT_SIZE_GB` 并重新核算资源 |
| attach handshake 失败 | 在失败节点检查 HCA port、GID、内存、`memlock` 和到 master 的连接；CPU-only head 也必须满足 attach 条件 |
| manager/protocol 不符 | 清理前一个作业；首期要求单任务独占且不会接管健康的旧 controller |
| `Connection refused` 指向旧 segment | 等待 `client_ttl` 过期，再确认旧 client/segment 已从 master 清理 |
| C1 尾部全零或 SIGSEGV | 确认未显式启用 `MC_STORE_MEMCPY`；C1 仅用于 benchmark |
| 多 HCA 环境建连失败 | 显式设置 `--tq-rdma-device`，不要依赖自动选卡 |

### 手工核验 HCA、GID、memlock、线路与 master

```bash
export TQ_RDMA_DEVICE=<rdma-device>
export TQ_RDMA_PORT=1
export TQ_TCP_DEVICE=<network-interface>

cat "/sys/class/infiniband/${TQ_RDMA_DEVICE}/ports/${TQ_RDMA_PORT}/state"
cat "/sys/class/infiniband/${TQ_RDMA_DEVICE}/ports/${TQ_RDMA_PORT}/gids/0"
cat "/sys/class/infiniband/${TQ_RDMA_DEVICE}/ports/${TQ_RDMA_PORT}/rate"
ulimit -l

# 在数据传输前后取差值；IB counter 单位为 4-byte words。
cat "/sys/class/infiniband/${TQ_RDMA_DEVICE}/ports/${TQ_RDMA_PORT}/counters/port_rcv_data"
cat "/sys/class/net/${TQ_TCP_DEVICE}/statistics/rx_bytes"
# 使用通用占位 endpoint 检查 DNS 与 TCP。
export TQ_MASTER_HOST=master.example TQ_MASTER_PORT=50051
getent hosts "${TQ_MASTER_HOST}"
nc -vz "${TQ_MASTER_HOST}" "${TQ_MASTER_PORT}"
```

master 可达性可用部署环境已有的 DNS/TCP 工具检查；不要把真实 endpoint、hostname 或本地路径写入提交、公开日志和 PR 文档。

## 已知边界

- 首期不支持 GDR、多 job 或 concurrent initializer。
- Mooncake 传输层自身的 timeout 不由 Relax 控制。
- attach 成功不等于 wire proof；C2 合入前仍需真实双节点 byte-exact、wire-proof 和 fully-async smoke。
- mock/CPU CI 不能替代真实 RDMA 验收，真实测试结果必须关联准确的代码和依赖 SHA。
