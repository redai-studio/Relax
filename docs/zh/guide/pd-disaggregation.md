# Prefill–Decode 分离

Prefill–decode（PD）分离将 prompt 处理和逐 token 生成交给不同的 SGLang engine 池，使两个阶段可以按各自的负载分配资源。

| 阶段 | 工作内容 | 常见瓶颈 |
| --- | --- | --- |
| Prefill | 处理 prompt，构建 KV 状态 | 计算能力 |
| Decode | 使用已有 KV 状态逐 token 生成 | 显存带宽 |

PD 分离的是 rollout 推理内部的两个阶段。训练与 rollout 的调度方式单独配置；如需将训练和推理部署到独立 GPU 池，请参阅 [Fully Async 训练](./fully-async-training.md)。

## 启用 PD

使用[安装指南](./installation.md)中的运行环境及配套 SGLang patch，并确认模型和传输后端支持所选并行配置下的 PD。使用 Mooncake 时，按 [SGLang PD 文档](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)准备依赖和通信网卡。

当所有 engine 使用相同数量的 GPU 时，在已有训练配置中加入以下参数即可划分 P/D 资源：

```bash
--rollout-num-gpus 8 \
--rollout-num-gpus-per-engine 2 \
--prefill-num-servers 2 \
--sglang-router-prefill-policy consistent_hashing \
--sglang-router-decode-policy power_of_two
```

`--prefill-num-servers` 指定的是 **engine 数量**。Relax 将指定数量的 engine 分配给 prefill，其余 rollout GPU 用于 decode。在 PP1、DP1 配置下，上例得到：

| 角色 | Engine 数量 | 每个 engine 的 GPU 数 | GPU 总数 |
| --- | ---: | ---: | ---: |
| Prefill | 2 | 2（TP2） | 4 |
| Decode | 2 | 2（TP2） | 4 |

8 卡仅用于说明资源分配方式。实际配置应保证每个 engine 能容纳模型和 KV 状态，并为两个阶段都保留 GPU。使用 `--colocate` 时，还需与训练共享的 GPU 数量对齐：在没有独立奖励模型或 teacher 的常规配置下，Relax 会将 rollout GPU 数量设为 Actor 的 GPU 总数。

`--prefill-num-servers` 与 `--sglang-config` 互斥，二者均不能与 `--rollout-external` 同时使用。上例的 `power_of_two` 还需满足[路由与限制](#路由与限制)中的拓扑要求。

## 使用 YAML 配置 engine groups

需要显式分配 P/D 资源或为不同组设置 SGLang 参数时，使用 `--sglang-config`。将以下内容保存为训练进程可访问的 YAML 文件：

```yaml
sglang:
  - name: default
    num_gpus_per_engine: 2
    engine_groups:
      - worker_type: prefill
        num_gpus: 4
      - worker_type: decode
        num_gpus: 4
```

将上一示例中的 `--prefill-num-servers 2` 替换为该文件的路径：

```bash
--sglang-config /path/to/pd-config.yaml
```

这份 YAML 对应同样的 2 个 prefill、2 个 decode engine。上述路径是占位符，请替换为自己保存的文件。

- `num_gpus` 是该组的 GPU 总量，必须能被该组的 `num_gpus_per_engine` 整除。
- 组内的 `num_gpus_per_engine` 优先于模型级配置；模型级未设置时，使用 `--rollout-num-gpus-per-engine`。P/D 两侧使用不同 engine 大小时，仍需确认模型和传输后端兼容。
- 所有组的 `num_gpus` 之和必须等于最终生效的 `--rollout-num-gpus`。
- 可选的组级 `overrides` 使用 SGLang `ServerArgs` 字段名，例如 `mem_fraction_static`，不加 `sglang_` 前缀；其值会覆盖对应的 `--sglang-*` 参数。

## 请求流程

1. SGLang router 为请求选择 prefill engine 和 decode engine。
2. Prefill engine 处理 prompt，SGLang 通过传输后端将生成的 KV 状态传给 decode engine。
3. Decode engine 逐 token 生成，router 将结果返回给调用方。

Relax 根据 engine groups 配置 P/D worker 角色、bootstrap port 和 router 注册信息。训练中的权重更新同时覆盖 prefill 和 decode engines。

## 调整 P/D 配比

长 prompt 和并发生成场景适合评估 PD：拆分两个阶段有助于减少 prompt 处理对持续 decode 的干扰。短 prompt 或低并发下，路由和 KV 传输开销可能抵消收益。

对比非 PD 配置时，保持 rollout GPU 总数、模型、请求长度和并发量一致。除排队时间、首 token 延迟和 decode 吞吐外，也要观察完整 rollout 的耗时，并根据瓶颈调整资源：

| 现象 | 可以评估的调整 |
| --- | --- |
| Prefill 排队增加，decode 仍有余量 | 增加 prefill engine 数量 |
| Decode 排队增加，prefill 仍有余量 | 增加 decode engine 数量 |
| 两侧持续满载 | 增加总资源或降低请求并发 |
| KV 传输耗时占比高 | 先检查传输后端、通信网卡和带宽，再调整 P/D 比例 |

固定 GPU 总量时，增加一侧资源会减少另一侧资源。请在两次运行之间调整分配；目前不支持运行时 PD 动态扩容。

## 路由与限制

上例为 prefill 显式选择 `consistent_hashing`，为 decode 选择 `power_of_two`。如需 decode 也使用 consistent hashing，替换 decode 路由参数：

```bash
--sglang-router-decode-policy consistent_hashing
```

当 router 由 Relax 管理、decode 使用 `power_of_two` 时，Relax 先启动 engines，再使用其地址启动 router。确认 worker 角色、prefill bootstrap port 和健康注册状态正确后，才将部署交给后续流程使用。这条启动路径要求：

- 使用 Relax 启动的原生 SGLang engines，至少包含 1 个 prefill engine 和 2 个 decode engines。
- P/D 两侧均为 `dp_size=1`，且不开启 DP-aware routing。
- P/D worker groups 中不混入 `regular` workers；这条路径不支持自定义 engine 类。

上述托管启动流程适用于显式设置 `--sglang-router-decode-policy power_of_two` 的配置。Decode 改用 `consistent_hashing` 时走动态 worker 注册路径；动态注册不代表支持运行时调整 P/D 池的规模。

PD 使用 SGLang router；`--use-slime-router` 不支持 PD。

## 故障排查

| 现象 | 检查方向 |
| --- | --- |
| GPU 分配报错 | YAML 总量是否与实际 rollout GPU 数一致，各组 engine 大小是否匹配 |
| Router 启动或注册失败 | P/D engine 数量、worker 角色、DP 设置及 prefill bootstrap 连通性 |
| KV 传输超时 | 模型与后端兼容性、传输依赖及通信网卡 |
| GPU OOM | 先降低请求并发或上下文长度，再检查每个 engine 的显存配置 |
