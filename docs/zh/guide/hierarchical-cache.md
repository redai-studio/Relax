# CPU 与 SSD 多级缓存

多轮编程任务执行工具后，会再次发送不断增长的对话。SGLang HiCache 将可复用的前缀状态保存在 CPU 内存中，也可以继续备份到本地 SSD，让后续请求恢复已经从 GPU 淘汰的前缀。它扩大了缓存容量；实际收益取决于前缀复用、内存压力、数据搬运和工具执行时间。

[Mini SWE 多级缓存入口](../../../examples/mini_swe_agent/run_mini_swe_agent_hicache.sh) 沿用 [PD 示例](./pd-disaggregation.md) 的 R2E-Gym 任务与 GRPO 训练配方。两种缓存配置都使用 **单机 8 卡、2 个 Prefill 引擎和 2 个 Decode 引擎，每个引擎 TP2**。训练与 rollout 通过 `--colocate` 分时复用这些 GPU。只有 P 引擎开启 HiCache；D 引擎关闭 CPU/SSD 缓存及 decode radix cache。

本示例面向 Qwen3.5-35B-A3B 的 Unified Full KV 与 Mamba 缓存，使用单节点、DP1、由 Relax 原生管理的 SGLang 引擎，以及仓库配套的 SGLang v0.5.17 patch。其他模型或缓存组合、跨节点文件存储、运行时动态挂载/卸载存储、Session 专用 KV 控制不属于本示例范围。

## 缓存如何流转

| 层级 | 位置 | 后续请求如何使用 |
| --- | --- | --- |
| L1 | P 引擎 GPU 显存 | 直接复用仍驻留的前缀 |
| L2 | P 引擎 CPU 内存 | GPU 前缀淘汰后，将已备份的状态恢复到 GPU |
| L3，可选 | 本地 SSD 文件 | 先将已存储的前缀读入 CPU，再恢复到 GPU |

示例采用 `write_through`：符合条件的前缀从 GPU 复制到 CPU；开启 file backend 后，再异步备份到文件。原生淘汰机制按需释放 GPU 或 host 容量。存储读取由到来的请求触发，不依赖 Session hints、Session 主动预取或显式 demote 接口。

Qwen3.5 需要 **同一前缀、同一权重** 的 Full KV 页面与 Mamba checkpoint。只有 Full KV 文件并不足以恢复；所有 TP rank 必须具备兼容的 checkpoint。缺失或不完整的状态回退到正常 prefill 计算。对于 chunked prefill，即使最后一个短 chunk 没有产生新的 Mamba checkpoint，也会保留之前有效 checkpoint 的备份机会。请求存在公共前缀时，Mamba 可能优先记录分支点而非 chunk 末尾，因此 8192 token 的 chunk 不保证产生同样长度的可复用 checkpoint。

## 准备环境并启动

先完成 [PD 环境准备](./pd-disaggregation.md#准备环境) 和 [Mini SWE 数据准备](../../../examples/mini_swe_agent/README.md)。使用专用单节点 Ray 集群，在该节点提交。统一 Ray 入口会清理集群残留，因此提交前应确认集群可供本次任务使用。

设置模型、本地 Mini SWE venv、准备好的数据与 SIF 镜像，以及新的输出路径：

```bash
cd /path/to/Relax

export MINI_SWE_AGENT_VENV=/local/path/mini-swe-agent-venv
export MODEL_DIR=/path/to/models
export R2E_DATA_PATH=/path/to/r2e_data
export R2E_SIF_DIR=/path/to/r2e_sifs
export RUN_ID=mini-swe-cache-$(date +%Y%m%d-%H%M%S)
export SAVE_DIR=/path/to/checkpoints/${RUN_ID}
export EXP_ROOT=/path/to/experiments/${RUN_ID}
export RAY_ADDRESS=<ray-head-ip>:6379
export RAY_JOB_ADDRESS=http://<ray-head-ip>:8265
export NCCL_SOCKET_IFNAME=<network-interface>
```

使用 GPU+CPU 缓存：

```bash
HICACHE_TIER=cpu NUM_ROLLOUT=3 ROLLOUT_BATCH_SIZE=1 RAY_NO_WAIT=1 \
  bash scripts/entrypoint/ray-job.sh \
  examples/mini_swe_agent/run_mini_swe_agent_hicache.sh
```

使用 GPU+CPU+SSD 缓存时，指定一个 Ray worker 可访问的 **本地 SSD 绝对路径，且必须是新的空目录**：

```bash
export RELAX_HICACHE_FILE_ROOT=/local/ssd/relax-cache/${RUN_ID}

HICACHE_TIER=ssd NUM_ROLLOUT=3 ROLLOUT_BATCH_SIZE=1 RAY_NO_WAIT=1 \
  bash scripts/entrypoint/ray-job.sh \
  examples/mini_swe_agent/run_mini_swe_agent_hicache.sh
```

不同运行应使用不同的 `RUN_ID`、`SAVE_DIR` 和 `EXP_ROOT`。默认层级是 `ssd`。包装脚本检查 SSD 根目录，并将其写入 Ray runtime 环境；仅在提交端 export 并不能配置 worker。随后使用已有 PD driver 管理 agent server，直到训练结束。`NUM_ROLLOUT=3`、`ROLLOUT_BATCH_SIZE=1` 表示运行三轮训练，每轮一个任务 Group、八个 sample，仍执行正常工具循环与评分。

原始 `run_mini_swe_agent.sh` 和纯 PD 的 `run_mini_swe_agent_pd.sh` 保留各自入口。新脚本通过 `--sglang-config` 选择 [CPU 配置](../../../examples/mini_swe_agent/pd-hicache-cpu-2p2d-tp2.yaml) 或 [SSD 配置](../../../examples/mini_swe_agent/pd-hicache-ssd-2p2d-tp2.yaml)。

## 容量与存储布局

下表是 **示例配置值**，不是 SGLang 的通用默认值：

| 配置 | 值 | 作用范围与含义 |
| --- | --- | --- |
| `max_total_tokens` | `196608` | 每个引擎的 GPU token pool 容量；各 TP rank 保存自己的分片 |
| `max_mamba_cache_size` | `666` | P 引擎 GPU Mamba cache 槽位数 |
| `page_size` | `64` | Full KV 页的 token 数 |
| `mamba_track_interval` | `256` | Mamba checkpoint 的 token 间隔 |
| `chunked_prefill_size` | `8192` | 每个 prefill chunk 的最大 token 数 |
| `hicache_ratio` / `hicache_size` | `2.0` / `0` | 按对应 device pool 的比例分配 host pool，不覆盖为固定大小 |
| `hicache_write_policy` | `write_through` | 备份符合条件的驻留前缀 |
| `hicache_io_backend` / `hicache_mem_layout` | `direct` / `page_first_direct` | GPU↔host 搬运方式及 host 布局 |
| SSD prefetch policy | `wait_complete` | 等待原生存储操作完成后使用结果 |
| File `max_size` | `34359738368` 字节（32 GiB） | **每个 TP rank** 的文件配额，不是每个引擎或整次运行的配额 |
| File `min_free_space` | `68719476736` 字节（64 GiB） | 底层文件系统要求保留的空闲空间 |
| File `eviction_ratio` | `0.9` | LRU 淘汰后的配额目标比例 |
| File `enable_metadata_cache` | `false` | 直接检查文件是否存在 |

按示例的 BF16 Qwen3.5 配置，这两类 host pool 每个 P 侧 TP rank 约需 47 GB，四个 rank 合计约 175 GiB。更换模型或 dtype 时，应以实际启动分配量重新核算。

CPU 内存预算必须包含 **Full KV 与 Mamba 两类 host pool**，乘以四个 P 侧 TP rank，再加上权重、训练 offload、Ray、sandbox 和操作系统的用量。`hicache_ratio=2` 是 pool 容量比例，不是 2 GB，也不是整机上限。Mamba 状态可能占据主要内存；增大容量前应核对启动日志中的分配量和容器内存限制，不能只看宿主机空闲内存。

Relax 按引擎逻辑 rank，在运行根目录下创建 `prefill-engine-0000`、`prefill-engine-0001` 等子目录。同一引擎的两个 TP rank 共用该子目录，通过文件名和配额统计区分 rank。按当前配置，两个 P 引擎的文件配额合计 **128 GiB**，另需考虑临时写入和文件系统开销。64 GiB 空闲空间阈值作用于整个文件系统，并不是每个 rank 额外预留一份空间。

存储目录应放在空间和 IO 能力足够的本地 SSD。共享文件系统会改变延迟与竞争情况。`direct` 配置控制 GPU/CPU 搬运，**不代表** SSD 文件读取绕过 OS page cache；一次快速的重复文件读取不能证明物理 SSD 吞吐。

## 权重更新、取消与恢复

每个 P 引擎启动时创建新的 file cache incarnation。更新权重前，引擎等待在途缓存操作完成，停止存储 worker，让 TP ranks 一致推进 weight epoch，并在修改权重前清空内存缓存。所有 TP rank 共同确认 namespace 与更新结果。一次分片权重同步可能多次推进文件 epoch；这个后端计数器与训练端的 weight_version 分开计数。如果更新失败或只完成一部分，引擎会失败退出以便重建，不能继续以混合权重返回成功服务。

新请求不能命中旧 weight epoch 或旧引擎 incarnation 的文件。旧文件可以继续占据磁盘，并计入原生文件配额，直到被淘汰；namespace 隔离不等于立即删除文件。引擎恢复后沿用原目录，但新 incarnation 从冷缓存开始。没有修改权重的普通 offload/onload 保留 namespace，不过这不保证某个 L1/L2 条目始终保留。

存储 reader 在 IO 完成前始终持有其 buffer，包括请求取消和读取异常的情况。scheduler 暂停时仍推进缓存完成事件。文件写入失败后会释放备份持有的资源，已有 CPU 缓存仍可使用；读取失败或状态不完整则重新计算前缀。因此，存储 miss 应改变计算开销，而不改变预期模型状态。

任务及其 worker 停止后，再按存储策略归档或移除该次运行的缓存目录。不要通过删除正在使用的缓存文件释放空间，也不要让新运行复用非空根目录；启动脚本会拒绝非空 SSD 根目录。任务进度、日志、agent server 清理和停止命令沿用 [PD 指南](./pd-disaggregation.md#查看进度和停止任务)。

## 检查命中与排查问题

检查各 P 引擎的实际参数和启动日志：HiCache 已开启、host 分配量符合预期；SSD 模式使用 `file`，且目录属于对应引擎。D 引擎应保持 HiCache 关闭。Ray 接受任务、服务 ready、目录中出现文件，都只能证明其中一部分链路。

重复前缀验证应区分 GPU 命中、CPU 恢复和存储恢复。启用缓存统计时，原生生成响应可在 `cached_tokens_details` 中报告 `device`、`host`、`storage`。结合存储与搬运指标，确认恢复后的前缀被完成的请求实际使用；混合模型必须同时检查 Full KV 与 Mamba。比较不同层级时，保持 GPU/host 容量、模型、请求、并发和训练配置一致；使用原纯 PD 配置作对照前，也应先对齐容量设置。

| 现象 | 排查方向 |
| --- | --- |
| 没有 L2/L3 命中 | 前缀可能仍在 GPU、长度不足 checkpoint、刚因权重变化失效，或被路由到另一个 P 引擎 |
| 有文件但 storage 命中为零 | 检查 namespace、各 TP rank 的 Full KV/Mamba 完整性，以及 L1/L2 是否已满足请求 |
| host 分配失败 | 汇总所有 P rank 的 Full KV 和 Mamba 用量，检查容器限制与训练 offload |
| 文件写入失败或停止 | 检查目录权限、每 rank 配额、文件系统空闲空间与存储 worker 错误 |
| 存储恢复比重算更慢 | 检查前缀长度、文件 IO、OS cache 状态、搬运开销和 `wait_complete` 的等待时间 |
| 启动时报未知参数 | 使用仓库配套 SGLang patch 和示例 YAML，避免复制研究版 Session 专用选项 |

任务并发和导出契约见 [Agentic Rollout](./agentic-rollout.md)，路由布局与恢复见 [PD 分离](./pd-disaggregation.md)。
