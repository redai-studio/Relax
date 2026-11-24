# Colocate 模式 update_weights 流水线优化

## 概述

在 Colocate 模式下，Actor 训练完成后需要将 Megatron 格式的权重转换为 HF 格式，并通过 CUDA IPC 传输给 SGLang 推理引擎。由于权重按 chunk 分批传输，原始实现中 chunk 间严格串行（发送 → 阻塞等待 → 下一个），导致 `update_weights_time` 耗时较大。

本优化通过 **chunk 间流水线化**（延迟 `ray.get`，使 IPC 传输与下一个 chunk 的 HF 转换重叠）和 **移除不必要的 barrier**，将耗时降低约一倍。

______________________________________________________________________

## 自动优化 MoE 权重同步

在 CUDA 上使用 Megatron、Bridge 导出、训练状态卸载和共卡模式时，Relax 会自动保留权重导出使用的 PP/EP/TP/ETP 通信组。同步阶段只恢复各 rank 共同确认的这些组，训练阶段仍恢复全部训练通信组。量化权重或专家张量并行需要走原导出路径时，也可以使用这项通信组优化。LoRA、外部或混合远端 rollout 引擎，以及无法识别的迭代器或通信组布局，保留原有生命周期。

BF16 专家路由单独协商：每个完整专家只发送给持有它的 rollout EP rank，同时保留 Bridge 转换和 SGLang HF loader。启用依据是映射及实际运行布局，不依赖模型名：

| 位置 | 条件 |
|---|---|
| 训练端 | Bridge、BF16 专家、ETP1、EP 大于 1；TP/PP 可以大于 1。专家 DP 副本的参数元数据必须一致，再选出一个真实 owner。 |
| 推理端 | CUDA BF16、静态连续 EP、TP=EP、PP1、MoE DP1；没有量化、EPLB、冗余专家、弹性重映射或共享专家融合。没有共享专家的模型无需调整融合设置。 |
| 转换 | 可识别逐专家的 `gate/up/down_proj` 或 `w1/w2/w3` HF 权重，支持 `model.language_model` 前缀及 Bridge 已处理的融合映射。MTP、塔模型和无法识别的专家张量保留原导出路径。 |

支持使用同一 checkpoint 的 NEXTN/EAGLE。独立 draft checkpoint 或无法确认的模型覆盖配置会回退原导出路径；专家路由被拒绝时，满足条件的通信组优化仍然生效。已有 Kimi K3 MXFP4 路由选项继续使用其原有协议和保守的暂存预算。

无需新增开关。启动日志中的 `Weight sync communicator plan` 和 `Weight sync BF16 expert routing` 会报告选择结果。接收端能力查询失败或并行布局不支持时，继续使用已有 direct/gather/HF 选择逻辑。

`--update-weight-buffer-size` 仍以字节指定暂存上限，默认值仍为 512 MiB。显存充足时可以增加分桶大小，减少同步轮次，例如 `--update-weight-buffer-size 8589934592` 表示 8 GiB。常驻通信组会在阶段切换期间占用显存，而且多个暂存 buffer 可能同时存在，因此应按实际显存余量选择分桶大小。同步耗时取决于模型、并行布局和硬件。

## 问题背景

### 场景描述

在多机多卡环境下训练大规模 MoE 模型时，`perf/update_weights_time` 相比预期偏高约一倍。

### 权重更新调用链

```
MegatronTrainRayActor.update_weights()
  └─ UpdateWeightFromTensor.update_weights()
       ├─ rank 0: pause_generation + flush_cache
       ├─ dist.barrier(gloo)                          ← 必要
       ├─ weights_getter()                            ← 获取 Megatron 本地权重
       │
       ├─ for chunk in get_hf_weight_chunks():        ← 多个 chunk
       │     ├─ _send_hf_params(chunk)                  HF 转换 + 序列化 + Gloo gather
       │     └─ ray.get(refs)                         ← 阻塞等待！
       │
       └─ rank 0: continue_generation
```

核心瓶颈在于调用链中的 `ray.get(refs)` —— 每个 chunk 必须等待 IPC 传输完成后才开始下一个，多个 chunk 严格串行，占据了总耗时的大部分。

______________________________________________________________________

## 优化方案

### Chunk 间流水线化

**核心思路**：将 `ray.get` 延迟到下一个 chunk 发送完成后再执行，使当前 chunk 的 IPC 传输与下一个 chunk 的 HF 转换重叠。

**优化前**（严格串行）：

```
Chunk 0: [HF转换][序列化+gather][IPC传输][等待]
Chunk 1:                                       [HF转换][序列化+gather][IPC传输][等待]
Chunk 2:                                                                             [HF转换]...
```

**优化后**（流水线重叠）：

```
Chunk 0: [HF转换][序列化+gather][IPC传输]
Chunk 1:                        [HF转换][序列化+gather][等待prev][IPC传输]
Chunk 2:                                               [HF转换][序列化+gather][等待prev][IPC传输]
                                                                                              [等待last]
```

优化后的代码：

```python
prev_refs: list[ObjectRef] = []
prev_long_lived_tensors = None
for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
    refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
    if prev_refs:
        ray.get(prev_refs)
    del prev_long_lived_tensors
    prev_refs = refs
    prev_long_lived_tensors = long_lived_tensors
if prev_refs:
    ray.get(prev_refs)
del prev_long_lived_tensors
```


______________________________________________________________________

## 预期收益

- **chunk 循环耗时**：流水线重叠显著缩短
- **总 update_weights_time**：整体降低约一倍

### 资源影响

| 资源 | 影响 |
|------|------|
| GPU 显存 | 峰值略有增加（额外一个 chunk 的 `long_lived_tensors`） |
| CPU 内存 | 无变化 |
| 网络带宽 | 无变化（总传输量不变，只是时序重叠） |

### 适用范围

此优化对以下场景收益最大：

- **MoE / 大参数模型**：chunk 数量多，流水线重叠收益显著
- **Colocate 模式**：CUDA IPC 路径下 IPC 传输与 HF 转换可真正并行
- **多节点部署**：跨节点引擎时延更高，流水线重叠收益更明显


______________________________________________________________________

## 下一步

- [Distributed Checkpoint](./distributed-checkpoint.md) — 非 Colocate 部署中使用的 DCS 权重同步机制
