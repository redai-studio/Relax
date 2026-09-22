# S3 模型加载加速

Relax 支持直接从 S3 兼容对象存储加载 Hugging Face checkpoint。为 `--hf-checkpoint` 传入 `s3://` URI 后，实际消费策略模型的节点会在启动阶段将 checkpoint 下载到共享内存（`/dev/shm`），之后按普通本地路径读取。该能力不需要独立开关，传入本地路径即不走这条链路。

### 适用场景

S3 加载替换的是从原有文件系统读取 checkpoint 这一步，是否更快取决于两者的读带宽对比：

| 环境 | 建议 |
|---|---|
| 对象存储读带宽高于存放 checkpoint 的共享文件系统（如 NFS） | 使用 `s3://`，这是该能力的设计场景 |
| 各节点本地已有 checkpoint 且位于高速 NVMe | 使用本地路径，本地读通常快于网络下载 |
| 对象存储带宽受限或距集群较远 | 建议先实测，每节点拉取数十 GB 的开销可能超过收益 |

对于 S3 模型源，Relax 会先分配各 Service 的 Placement Group，再在实际消费策略 checkpoint 的每个 GPU 节点上启动一个预取任务。预取进行时 Ray Serve 会继续部署；如果 consumer 更早走到模型加载位置，它会等待同一个节点缓存锁。无论一个节点承载多少个 rank，每个节点都只下载一次，纯 CPU Ray head 不下载模型文件。本地模型路径继续使用原有的 Service 构造即部署流程，也不会执行 S3 缓存探测或清理。

### 前置条件

- checkpoint 可通过 `s3://` URI 访问
- 各节点的 `/dev/shm` 在启动阶段能容纳完整 checkpoint
- 使用 SGLang 流式加载模式时，镜像中的 SGLang 需提供 `runai_streamer`

______________________________________________________________________

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│  S3 Object Store        s3://bucket/model-prefix/               │
└────────────────────────────┬────────────────────────────────────┘
                             │  parallel download, once per node
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Node shared memory (/dev/shm)                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  full checkpoint = weights + config + tokenizer           │  │
│  └───────────────┬───────────────────────┬───────────────────┘  │
│                  │                       │                      │
│                  ▼                       ▼                      │
│  ┌────────────────────────────┐  ┌─────────────────────────┐    │
│  │  Megatron Actor            │  │  SGLang Rollout Engine  │    │
│  │  always loads full weights │  │  dummy / auto / stream  │    │
│  └────────────────────────────┘  └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

训练 Actor 始终加载完整权重；可调整的只有 rollout 引擎，通过 `--sglang-load-format` 控制。

______________________________________________________________________

## 快速开始

```bash
python3 -m relax.entrypoints.train \
    --hf-checkpoint s3://bucket/model-prefix/ \
    --sglang-load-format dummy \
    # ... 其他训练参数
```

使用自建或 S3 兼容存储时，通过标准 AWS 环境变量配置连接信息，不需要把 transport 参数写进训练命令：

```bash
export AWS_ENDPOINT_URL_S3=http://s3.example
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

`AWS_ENDPOINT_URL_S3` 的优先级高于 `AWS_ENDPOINT_URL`。凭证使用标准 boto3 credential chain，因此 profile、session token 和 workload credential 均无需 Relax 私有参数即可生效。

使用 `runai_streamer` 时，Relax 仅在用户没有设置 `RUNAI_STREAMER_S3_ENDPOINT` 时，将上述 AWS endpoint 桥接给 RunAI Streamer，因此用户显式设置的 RunAI endpoint 优先。网关要求 path-style addressing 时，可以直接设置 RunAI Streamer 原生变量 `RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0`。

### 为什么用 `dummy`

RL 训练中，Actor 会在首次 rollout 前将权重同步给 rollout 引擎，SGLang 启动时加载的权重随即被覆盖，因此在启动阶段加载真实权重没有意义。`dummy` 使 SGLang 只读取模型结构和 metadata，真实权重由 Actor 后续提供。

该模式仅适用于 policy rollout 引擎，它是 Actor 的同步对象；其他角色仍加载真实权重，见下方警告。

______________________________________________________________________

## 加载模式与推荐搭配

`--sglang-load-format` 只控制 rollout 引擎，默认值为 `auto`。按场景选择：

| 场景 | 配置 | 效果 |
|---|---|---|
| **常规 RL 训练**（推荐） | `--sglang-load-format dummy` | rollout 引擎只读 metadata，启动最快。Actor 在首次 rollout 前推送真实权重，之后释放 SHM |
| **引擎启动顺序不确定** | `--sglang-load-format auto` | SHM 副本就绪时直接复用，否则从 S3 流式加载。SHM 会被占用到任务结束 |
| **rollout 先于训练启动，或独立 SGLang** | `--sglang-load-format runai_streamer` | 权重直接从 S3 流式加载，rollout 完全不占 SHM，Actor 首次同步后释放 SHM |
| **排障，需要保留权重文件** | 追加 `--disable-s3-model-cleanup` | 完整 checkpoint 全程保留在 `/dev/shm` |
| **关闭该能力** | `--disable-s3-model-download` | 同时关闭 Relax 通用 S3 下载和已注册的模型来源 provider，之后由普通 loader 直接处理 `--hf-checkpoint` |

::: warning
`dummy` 只适用于 policy rollout 引擎，因为 Actor 会在首次请求前覆盖它的权重。GenRM、teacher、evaluation 和独立推理服务始终加载真实权重，即使它们指向同一个模型。
:::

checkpoint 越大，`dummy` 省下的启动时间越多；但模式选择本身与模型规模无关，按上表的场景选即可。

______________________________________________________________________

## 启动时序

```
提交任务
  → Placement Group 就绪
  → 清除前一个串行任务遗留的、由 Relax 管理的模型缓存
  → 策略模型 consumer 所在 GPU 节点开始预取 checkpoint 到 /dev/shm
  → 预取进行时 Ray Serve 同步开始部署
       consumer 需要文件但缓存未就绪时，等待节点本地缓存锁
  → SGLang rollout 引擎初始化
       dummy           只读 metadata，最快就绪
       auto            直接复用 /dev/shm，不再重复下载
       runai_streamer  从 S3 流式加载，不占用 SHM
  → Controller 确认所有已提交的预取任务完成
  → Actor 执行首次权重同步（dummy 引擎在此拿到真实权重）
  → 自动释放 /dev/shm 中的权重分片
  → 开始训练
```

Relax 不会恢复、续传或复用前一个任务的完整或部分下载。新的 S3 预取开始前，会在逐缓存加锁后清除前一个串行任务遗留的 Relax 缓存。清理范围严格限制在 Relax 自己的哈希命名空间内，不会删除 `/dev/shm` 中的其他文件；因此长期存活的 Ray 集群串行运行多个 Relax 任务时，也不会持续堆积旧模型。

______________________________________________________________________

## 内存规划

`/dev/shm` 占用的是真实 Pod 内存，且启动阶段必须放得下完整 checkpoint。Pod 内存按下式预留：

```
Pod 内存  ≥  checkpoint 大小  +  RL 运行峰值内存  +  余量
```

其中 checkpoint 大小等于 S3 prefix 下所有文件之和。完整 checkpoint 必须放得进 `/dev/shm`，否则任务在启动阶段就会失败。

首次同步之后，Relax 会释放权重分片，只保留 config、tokenizer 和 processor 文件——但仅限于后续不会再读取权重的模式：

| 配置 | 训练开始后是否释放 SHM 权重 |
|---|---|
| `--sglang-load-format dummy` | 是 |
| `--sglang-load-format runai_streamer` | 是 |
| `--sglang-load-format auto` | 否 |
| 没有 rollout 服务的任务（例如 SFT） | 是 |
| `--disable-s3-model-cleanup` | 否 |

使用外部 rollout 引擎或 per-engine SGLang 配置时，同样会保留 checkpoint。

______________________________________________________________________

## 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--hf-checkpoint` | — | 传 `s3://` URI 启用 S3 加载；本地路径则不走该链路 |
| `--sglang-load-format` | `auto` | rollout 引擎加载模式，见上表 |
| `--s3-model-shm-root` | `/dev/shm` | 下载使用的共享内存目录 |
| `--s3-model-download-workers` | `20` | 下载并发数 |
| `--disable-s3-model-download` | 关闭 | 同时关闭通用 S3 下载和已注册的模型来源 provider，再把 `--hf-checkpoint` 交给普通 loader |
| `--disable-s3-model-cleanup` | 关闭 | 全程保留共享内存中的权重 |

Relax 不再通过私有命令行参数传递 endpoint 和凭据。普通 S3 使用 `AWS_ENDPOINT_URL_S3`（其次为 `AWS_ENDPOINT_URL`）以及标准 boto3 credential chain。模型来源 provider 仍可携带只作用于该来源的 transport 配置，不会改变普通 S3 模型使用的环境。

______________________________________________________________________

## 常见问题

| 问题 | 原因 | 解决方案 |
|---|---|---|
| 启动失败，提示共享内存容量不足 | checkpoint 放不进 `/dev/shm` | 提高 Pod 内存和 `/dev/shm` 规格，参考[内存规划](#内存规划) |
| 启动失败，提示共享内存目录不存在 | 某些节点没有挂载 `/dev/shm` | 检查每个节点的容器挂载，而不只是 rank 0 |
| SGLang 报错不识别 `runai_streamer` | 安装的 SGLang 不提供该 loader | 升级到 SGLang 支持 `runai_streamer` 的镜像 |
| GenRM 或 teacher 输出异常 | 误以为 `dummy` 对它们生效 | `dummy` 只作用于 policy rollout 引擎，其他角色加载真实权重 |
| 训练开始后内存没有释放 | 使用了 `auto`、外部 rollout 或 `--disable-s3-model-cleanup` | 希望回收内存则改用 `dummy` |
| 启动比使用本地路径更慢 | 对象存储读带宽低于本地文件系统 | 对比两者带宽，本地路径可能更适合当前集群 |

______________________________________________________________________

## 延伸阅读

- [配置](./configuration.md) —— 完整 CLI 参数参考
- [权重更新流程](./update-weights-pipeline.md) —— Actor 如何向 `dummy` 加载的引擎推送真实权重
