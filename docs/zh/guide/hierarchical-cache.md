# CPU 与 Mooncake 多级缓存

CPU 缓存将可复用的 prompt 状态从 GPU 显存扩展到主机内存。SGLang HiCache 让后续请求恢复已从 GPU 淘汰的前缀，减少重复 prefill。多轮对话、Agent 等反复提交长前缀的任务可以评估这一能力；实际收益取决于前缀复用程度和搬运开销。

| 层级 | 位置 | 请求如何使用 |
| --- | --- | --- |
| L1 | GPU 显存 | 直接复用驻留的前缀状态 |
| L2 | CPU 内存 | GPU 前缀淘汰后，将已备份的状态恢复到 GPU |

**CPU cache 可独立用于集中式部署，也可与 [PD 分离](./pd-disaggregation.md)组合。** 集中式部署中，一个 regular 引擎完成 prefill 和 decode；PD 部署把两阶段交给不同引擎。缓存位置和引擎分工是两个独立配置维度。同一引擎也可选用 Mooncake Store L3，缓存层级与是否启用 PD 相互独立。

## 集中式部署

使用[安装指南](./installation.md)中的运行环境和配套 SGLang patch，并确认模型的缓存实现支持所选 HiCache 布局与并行配置。Relax 通过 `--sglang-*` 参数透传 SGLang 设置。

所有 regular 引擎使用相同缓存配置时，可在已有训练配置中加入：

```bash
--sglang-enable-hierarchical-cache \
--sglang-hicache-ratio 2.0 \
--sglang-hicache-write-policy write_through
```

保持 `hicache_storage_backend` 未设置或为 `null`。这些参数不要求 prefill/decode 分组、PD router 或存储目录。`write_through` 将符合条件的前缀备份到 CPU；`hicache_ratio` 按对应 device pool 的比例分配 host pool。

也可以通过 `--sglang-config` 为 regular 引擎配置 CPU cache：

```yaml
sglang:
  - name: default
    num_gpus_per_engine: 2
    engine_groups:
      - worker_type: regular
        num_gpus: 8
        overrides:
          enable_hierarchical_cache: true
          hicache_ratio: 2.0
          hicache_write_policy: write_through
          hicache_storage_backend: null
```

在 PP1、DP1 下，上例得到 4 个 TP2 regular 引擎。组级 `overrides` 使用 SGLang 字段名，覆盖对应命令行参数。GPU 数量和 TP2 是示例选择，模型所需的其他缓存设置应另行补充。

## PD 分离部署

PD 部署可以单独配置 P/D 两侧。以下示例只为 prefill 组开启 CPU cache：

```yaml
sglang:
  - name: default
    num_gpus_per_engine: 2
    engine_groups:
      - worker_type: prefill
        num_gpus: 4
        overrides:
          enable_hierarchical_cache: true
          hicache_ratio: 2.0
          hicache_write_policy: write_through
          hicache_storage_backend: null
      - worker_type: decode
        num_gpus: 4
        overrides:
          enable_hierarchical_cache: false
          hicache_storage_backend: null
          disaggregation_decode_enable_radix_cache: false
```

保存 YAML 后，通过 `--sglang-config /path/to/cpu-cache.yaml` 指定路径。上例得到 2 个 prefill 和 2 个 decode 引擎，每个引擎 TP2。decode 侧关闭缓存是这份 PD 配方的设置，不限制 regular 引擎。托管 PD 的资源布局、路由及并行限制见 [PD 指南](./pd-disaggregation.md#路由与限制)。

## 缓存兼容性与容量

请求在选定引擎的本地前缀树中匹配：GPU 命中直接复用，CPU 命中先恢复到 GPU，缺失或不兼容的状态通过 prefill 重算。混合模型可能同时需要 attention KV 页面与 Mamba checkpoint 等递归状态；所有参与 rank 必须具备同一前缀、同一权重下的兼容状态。prefill chunk 的长度不保证其末尾存在可复用 checkpoint。

每个 DP 副本具有独立的前缀树和 GPU/CPU pool。同一副本内的 TP rank 持有对应分片并协调缓存操作；Attention DP 同样按 attention-DP 组划分本地状态。请求路由到另一个副本时，不会自动继承原副本的缓存命中。Mini SWE 配方包含集中式 TP2 引擎、2P2D TP2，以及 Attention DP2 × TP2、EP4 的 1P1D；这些是具体布局，不是 CPU cache 的通用并行限制。

DP2 的 PD 配方需要 [Mini SWE 示例](../../../examples/mini_swe_agent/README.md#train-with-cpu-caching)中说明的配套 router 修复，并通过稳定的会话 key 在 P/D 两侧执行 dp-aware 一致性哈希。P/D 选中的 rank 可以不同：router 须分别指定两侧 rank，并告知 decode 从哪个 prefill rank 获取 KV。验收时应检查两侧实际 rank，以及每个 prefill DP rank 的 host 恢复；只有 decode rank 稳定不足以证明 prefill 缓存亲和性。CPU cache 本身不要求 PD 或这一特定路由策略。

| 配置 | 作用 |
| --- | --- |
| `hicache_ratio` | host pool 相对对应 device pool 的容量比例 |
| `hicache_size` | 以 GB 指定固定 host pool 大小；正值覆盖比例设置 |
| `hicache_write_policy` | 符合条件的前缀何时备份，例如 `write_through` |
| `hicache_io_backend` / `hicache_mem_layout` | GPU↔CPU 搬运方式与 host 布局，需选择模型支持的组合 |

`max_total_tokens` 是可选容量上限。不设置时，SGLang 根据显存预算推导 GPU token pool 容量；固定容量对比时可显式设置。它不是启用 CPU cache 的必要参数，也不是单次请求的输出长度限制。

主机内存预算应汇总所有启用缓存的引擎、所有 rank 的 KV 和递归状态池，并计入训练 offload、权重、Ray 和操作系统用量。`hicache_ratio=2` 是容量比例，不是 2 GB 或整机上限。相同每 rank 容量下，4 个 TP2 regular 引擎的 host pool 总预算是上述 2P2D 配方的两倍。检查实际启动分配日志及容器内存限制。

## RL 权重更新与暂停

CPU 缓存内容依赖模型权重。Relax 的权重同步流程会先暂停引擎并清空 GPU/CPU 前缀缓存，再更新权重；新权重不能复用旧前缀状态。引擎重启也从空缓存开始。训练中的 offload/onload 应通过框架生命周期执行。

暂停生成不代表 GPU↔CPU 搬运已完成。配套补丁在 regular 和 prefill 引擎暂停期间继续推进缓存完成事件，确保在途 buffer 安全释放后才执行 flush。对于使用 chunked prefill 的 Full KV+Mamba 模型，补丁保留最后短尾之前已受保护 checkpoint 的 CPU 备份，不伪造不存在的 Mamba 状态。

## 检查命中与排查问题

检查实际引擎角色、`enable_hierarchical_cache`、host 分配量，以及 `hicache_storage_backend=null`。原生生成响应可通过 `cached_tokens_details` 区分 `device` 与 `host` 命中；CPU 配置的 `storage` 应为零。零命中时详情可能为 `null`。仅看到 host pool 分配，不代表请求发生了 CPU 恢复，应结合 H2D 指标及输出正确性检查。

| 现象 | 排查方向 |
| --- | --- |
| 没有 CPU 命中 | 前缀仍在 GPU、长度不足 checkpoint、因权重变化失效，或请求进入另一个引擎 |
| host 分配失败 | 汇总各引擎、rank 的各类 pool，并检查训练 offload 与容器限制 |
| flush 等待 | 检查是否仍有请求或 GPU/CPU 搬运及未处理的完成事件 |
| 恢复比重算慢 | 比较前缀长度、搬运成本和完整 rollout 耗时 |

完整配方见 [Mini SWE 示例](../../../examples/mini_swe_agent/README.md#train-with-cpu-caching)，包含集中式和 PD 的启动命令及 Qwen3.5 专属 YAML。任务并发与导出契约见 [Agentic Rollout](./agentic-rollout.md)。功能验收不代表性能提升；性能对比需要保持模型、请求、并发和缓存容量一致，并执行独立重复实验。

## Mooncake Store L3

Mooncake Store 将缓存扩展为 **GPU → CPU → Store**，读取时按相反方向回迁。L3 表示层级位置；使用内存容量的 Store 不能称为 SSD 缓存。PD 的 Mooncake 传输与 Mooncake Store 是两个独立功能，开启 PD 传输不会自动开启 L3。集中式 regular 引擎同样可以使用 L3。

使用 `docker/Dockerfile` 构建的 CUDA 12 镜像（应用 SGLang 0.5.17 补丁，并固定 `mooncake-transfer-engine==0.3.12.post1`）；CUDA 13 镜像使用 SGLang 0.5.21，不包含这些 L3 生命周期修改。先启动独立的 Store master 与存储 client，再启动推理引擎。单机 TCP 示例先选择空闲端口，设置 `MOONCAKE_RPC_PORT`、`MOONCAKE_METADATA_PORT`、`MOONCAKE_METRICS_PORT`、`MOONCAKE_CLIENT_PORT`：

```bash
mooncake_master --rpc_address=127.0.0.1 --port="$MOONCAKE_RPC_PORT" \
  --enable_http_metadata_server=true --http_metadata_server_host=127.0.0.1 \
  --http_metadata_server_port="$MOONCAKE_METADATA_PORT" --metrics_port="$MOONCAKE_METRICS_PORT"

# 在另一个进程中运行，并保持到验证结束。
mooncake_client --host=127.0.0.1 --port="$MOONCAKE_CLIENT_PORT" \
  --master_server_address="127.0.0.1:$MOONCAKE_RPC_PORT" \
  --metadata_server="http://127.0.0.1:$MOONCAKE_METADATA_PORT/metadata" \
  --protocol=tcp --global_segment_size="16 GB"
```

创建连接 JSON，并替换端口占位符：

```json
{
  "local_hostname": "127.0.0.1",
  "master_server_address": "127.0.0.1:<RPC port>",
  "metadata_server": "http://127.0.0.1:<metadata port>/metadata",
  "master_metrics_port": 0,
  "check_server": false,
  "protocol": "tcp",
  "device_name": "",
  "global_segment_size": 0,
  "standalone_storage": false
}
```

`global_segment_size: 0` 表示引擎使用独立存储 client 提供的容量。正值会让每个启用缓存的 rank 按 attention TP 大小均分该值后贡献容量：DP2 × TP2 的 P 引擎合计为配置值的两倍，还需另算 CPU HiCache 池。Store 容量由各 namespace 共享；旧 epoch 对象可能占用空间，直到 Store 淘汰。本示例不启用 SSD offload、跨节点存储或 standalone dummy-client 分配。

按 [Mini SWE 示例](../../../examples/mini_swe_agent/README.md#train-with-cpu-caching) 准备数据和输出目录后运行：

```bash
HICACHE_TIER=mooncake HICACHE_DEPLOYMENT=regular \
HICACHE_MOONCAKE_CONFIG=/shared/path/mooncake.json \
  bash scripts/entrypoint/ray-job.sh examples/mini_swe_agent/run_mini_swe_agent_hicache.sh
```

`HICACHE_DEPLOYMENT=pd` 选择 2P2D TP2；`pd-dp2` 选择 1P1D attention DP2 × TP2 / EP4，并启用 DP-aware 一致性哈希。入口将连接配置内嵌到 `${EXP_ROOT}/hicache-mooncake.yaml`；Ray worker 需要访问共享输出目录，不依赖提交端原 JSON 路径或环境变量。regular/P 使用 `hicache_storage_backend: mooncake` 与 `wait_complete`，D 仍关闭 HiCache。默认 `HICACHE_TIER=cpu`。

本次运行时支持面向 Qwen3.5 Unified Full KV + Mamba。所有 attention TP rank 必须同时具备同一 prefix、owner、权重对应的 KV 和 Mamba。每个 owner 启动时生成新 incarnation。每次权重修改，包括不携带版本号的 bucket 与 post-process，都会排空并停止存储线程、推进 epoch、清空本地状态，再修改权重。失败状态在 global TP 内达成一致，部分更新失败会让引擎退出；namespace 本身只在同一 owner 内比较。其他缓存实现没有这个 Unified namespace，使用 Mooncake 时会在在线权重修改之前拒绝操作。

启用 `enable_lora` 时 Mooncake L3 会被拒绝。LoRA adapter 更新不会执行权重修改前的排空流程，否则新 adapter 可能恢复旧权重计算出的 KV/Mamba 状态。需要 LoRA 时请使用仅 CPU 的 HiCache，或等待两条 adapter 更新入口都执行存储排空与 namespace 失效后再启用。

普通 GPU/CPU flush、没有修改权重的 offload/onload 保留 L3 namespace；重启使用新 incarnation。共享 Store 的全量 clear 被禁止，避免影响其他引擎。缺失或截短的 KV/Mamba 对象触发重算；取消后，后台 reader 退出前继续持有 host buffer。存储线程停止有等待上限，超时失败而不释放仍被 IO 使用的内存；这不承诺能中断阻塞的 C++ 传输调用。

验证 L3 时，先完成请求及异步备份，再只清空 GPU/CPU 缓存，并将相同 prompt 路由回同一 owner。要求 `cached_tokens_details.storage > 0`、真实 H2D 搬运及与冷计算基线一致的输出。DP2 配置须覆盖两个 P owner。对象数量、GPU 命中或 CPU 命中不能代替 L3 回迁证据。比较延迟须固定输入与容量，并单独衡量完整 RL；功能通过本身不能证明端到端加速。
