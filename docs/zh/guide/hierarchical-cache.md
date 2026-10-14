# 多级缓存

多级缓存将可复用的 prompt 状态从 GPU 显存扩展到 CPU 内存，并可通过存储后端继续扩容。SGLang HiCache 让后续请求恢复已经从 GPU 淘汰的前缀，减少重复 prefill。多轮对话、Agent 等反复提交长前缀的任务可以评估这一能力；实际收益取决于前缀复用程度和数据搬运开销。

| 层级 | 位置 | 请求如何使用 |
| --- | --- | --- |
| L1 | GPU 显存 | 直接复用驻留的前缀状态 |
| L2 | CPU 内存 | GPU 前缀淘汰后，将已备份的状态恢复到 GPU |
| L3，可选 | 存储后端，例如本地 SSD 或分布式存储 | 先恢复到 CPU，再加载到 GPU |

多级缓存决定前缀状态保存在什么位置；[PD 分离](./pd-disaggregation.md)决定哪些引擎处理 prompt、哪些引擎生成 token。两者可以组合配置，不依赖某一种 Agent 框架或训练算法。

## 启用 CPU 缓存

使用[安装指南](./installation.md)中的运行环境和配套 SGLang patch，并确认模型的缓存实现支持所选 HiCache 布局与并行配置。Relax 通过 `--sglang-*` 参数透传 SGLang 设置。

所有引擎使用相同缓存配置时，可在已有训练配置中加入：

```bash
--sglang-enable-hierarchical-cache \
--sglang-hicache-ratio 2.0 \
--sglang-hicache-write-policy write_through
```

不设置 `hicache_storage_backend` 时，这组参数启用 GPU+CPU 缓存。`write_through` 将符合条件的前缀备份到 CPU，`hicache_ratio` 按对应 device pool 的比例分配 host pool。容量与模型状态的区别见下文。

使用 PD 时，通过 `--sglang-config` 分别配置 P/D 两侧。以下示例在 [PD 指南的资源布局](./pd-disaggregation.md#使用-yaml-配置-engine-groups)上，仅为 prefill 组开启 CPU 缓存：

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

保存 YAML 后，通过 `--sglang-config /path/to/pd-cache.yaml` 指定路径。在 PP1、DP1 下，上例得到 2 个 prefill 和 2 个 decode 引擎，每个引擎 TP2。8 卡仅用于说明配置方式，请按模型与实际资源调整分组。组级 `overrides` 使用 SGLang 字段名，覆盖对应命令行参数；模型所需的其他缓存设置应另行补充。

## 增加存储层

`hicache_storage_backend` 在 CPU 缓存之外选择 L3。SGLang 提供 `file`、`mooncake`、`hf3fs`、`nixl` 等后端，也支持动态加载。不同后端的依赖、缓存组件支持和共享语义不同，配置时应参考所用 SGLang 版本的 [HiCache 文档](https://docs.sglang.io/docs/advanced_features/hicache)。

使用本地文件后端时，将 prefill 组中的存储配置替换为：

```yaml
hicache_storage_backend: file
hicache_storage_prefetch_policy: wait_complete
hicache_storage_backend_extra_config: '{"enable_metadata_cache": false, "max_size": 34359738368, "min_free_space": 68719476736, "eviction_ratio": 0.9}'
```

这里给出的是示例值：`max_size` 为每个负责存储的 rank 设置 32 GiB 配额，`min_free_space` 要求底层文件系统保留 64 GiB 空闲空间，`eviction_ratio` 将淘汰目标设为配额的 90%。这些值不是整次任务的总预算。`wait_complete` 等待存储预取结束，有利于复用已存储前缀，但 IO 慢时会增加等待时间。

对于 Relax 管理的原生 prefill 引擎，在本地 SSD 创建专用的绝对路径目录，并通过 `RELAX_HICACHE_FILE_ROOT` 传给 **Ray worker**。将以下配置合并到任务的 runtime environment，路径替换为自己的目录：

```yaml
env_vars:
  RELAX_HICACHE_FILE_ROOT: /local/ssd/relax-cache/run-001
```

根目录必须已经存在。Relax 按引擎逻辑 rank 创建 `prefill-engine-0000`、`prefill-engine-0001` 等子目录。只在提交端设置环境变量不能配置远端 worker。这个目录管理入口仅适用于使用 `file` 的原生 prefill 引擎，目前要求引擎为单节点、`dp_size=1`；它不管理 regular、external 或自定义引擎。

原生 file 后端在各进程内维护 LRU 和配额账本。每个引擎使用独立目录，可以明确淘汰与配额归属。同一引擎的 TP rank 通过原生文件名后缀区分分片；MLA 的重复数据由单个存储 owner 管理。按上面的配额，2 个保存独立分片的 TP2 prefill 引擎合计允许 128 GiB 文件缓存。64 GiB 空闲空间阈值作用于整个文件系统，不是每个 rank 额外预留一份。

引擎间共享 L3 需要合适的后端和兼容的缓存 key。把多个 file 后端实例指向同一个目录，并不会建立统一的淘汰或配额管理，当前按引擎划分的权重 namespace 也不会因此跨引擎复用。清理缓存目录前，应先停止任务及其 worker。

## 请求流程与缓存兼容性

1. 请求进入选定引擎，在该引擎的本地前缀缓存中匹配。
2. GPU 命中直接复用；匹配状态只在 CPU 时，先恢复到 GPU。
3. 配置存储后端后，可从 L3 补回缺失的 host 状态，再加载到 GPU。
4. 缺失或不兼容的状态通过 prefill 重新计算。

混合模型的缓存可能同时包含 attention KV 页面和 Mamba checkpoint 等递归状态。恢复要求所有参与 rank 都具备同一前缀、同一权重下的兼容状态。只有 KV 文件可能不足以恢复，prefill chunk 的长度也不保证其末尾恰好存在可复用 checkpoint。

每个 DP 副本有独立的前缀树和 GPU/CPU pool。同一副本内的 TP rank 持有对应分片，并协调缓存操作；Attention DP 同样按 attention-DP 组划分本地状态。请求被路由到另一个副本时，不会自动继承原副本的 L1/L2 命中，跨副本复用需要路由配合或共享存储层。

DP1 是上述目录管理入口和 [PD 指南中托管静态启动路径](./pd-disaggregation.md#路由与限制)的限制，不是多级缓存的通用限制。仅删除这项检查，不会自动补齐每 DP 的存储账本与 DP-aware PD 路由。

## 容量与权重更新

| 配置 | 作用 |
| --- | --- |
| `hicache_ratio` | host pool 相对对应 device pool 的容量比例 |
| `hicache_size` | 以 GB 指定固定 host pool 大小；正值覆盖比例设置 |
| `hicache_write_policy` | 符合条件的前缀何时备份，例如 `write_through` |
| `hicache_io_backend` / `hicache_mem_layout` | GPU↔CPU 搬运方式与 host 布局，需选择模型支持的组合 |
| `hicache_storage_prefetch_policy` | 何时等待或停止存储预取 |
| `hicache_storage_backend_extra_config` | 后端专属配置，包括 file 配额 |

`max_total_tokens` 是可选的容量上限。不设置时，SGLang 根据显存预算推导 GPU token pool 容量；需要固定容量对比等场景时再显式设置。它不是启用 HiCache 的必要参数，也不是单次请求的输出长度限制。按比例分配的 host pool 会跟随 device 容量变化，因此调整 GPU pool 也可能改变 CPU 内存用量。

主机内存预算应汇总所有引擎、所有 rank 的各类缓存组件。混合模型除 KV 外，还可能分配较大的递归状态池。再计入训练 offload、权重、Ray 和操作系统用量，并检查启动分配日志及容器限制。`hicache_ratio=2` 是容量比例，不是 2 GB 或整机上限。`direct` IO 选项控制 GPU/CPU 搬运，不代表文件读取绕过 OS page cache。

RL 在 rollout 轮次之间更新模型权重，缓存状态必须随权重变化失效。针对 file L3，仓库配套的 SGLang v0.5.17 patch 在引擎启动时建立新 namespace，并在权重更新前排空缓存 IO、推进 namespace。旧文件不能再被新请求命中，但可能留在磁盘上直到淘汰。引擎重启后从新的冷 namespace 开始；未改变权重的普通 offload/onload 保留 namespace。其他存储后端不会自动获得这套 file 专属的 namespace 管理。

取消请求时也需要保持 buffer 所有权：请求已经中止，存储读取仍可能进行。其 buffer 要保留到 IO 完成，之后才能排空缓存操作并执行重置或权重更新。

## 检查命中与排查问题

检查各引擎的实际参数、host 分配量和存储目录。区分 GPU 命中、CPU 恢复和 L3 恢复；目录中出现文件不等于请求已经使用缓存。原生生成响应可在支持的配置下通过 `cached_tokens_details` 报告 `device`、`host`、`storage` 计数，再结合 IO 指标和完整 rollout 耗时判断效果。

| 现象 | 排查方向 |
| --- | --- |
| 没有 CPU 或存储命中 | 前缀可能仍在 GPU、长度不足 checkpoint、因权重变化失效，或请求进入了另一个引擎 |
| 有文件但 storage 命中为零 | 检查 namespace、各 rank 缓存组件的完整性，以及 L1/L2 是否已满足请求 |
| host 分配失败 | 汇总所有引擎、rank 的各类 pool，再检查训练 offload 与容器限制 |
| 文件写入失败或停止 | 检查权限、配额、文件系统空闲空间和存储 worker 错误 |
| 恢复比重算更慢 | 比较前缀长度、IO 与搬运成本，以及预取等待时间 |

性能对比应保持模型、请求、并发和 GPU/CPU 容量一致。扩大缓存可以减少重复 prefill，但存储延迟或工具执行时间也可能主导完整 rollout 的耗时。

完整的 Agent 训练配方见 [Mini SWE 示例](../../../examples/mini_swe_agent/README.md#train-with-cpu-and-ssd-caching)，其中包含启动命令和模型专属 YAML。任务并发与导出契约见 [Agentic Rollout](./agentic-rollout.md)。
