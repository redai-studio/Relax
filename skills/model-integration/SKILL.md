---
name: model-integration
description: Guide for integrating a new model into Relax. Use when adding a model architecture, validating SFT chat templates, tool boundaries or loss masks, writing weight converters, adapting TP sharding, or debugging weight synchronization. Covers Megatron bridge and raw modes and the FSDP backend.
---

# Model Integration Guide

## Overview

Relax 支持两种训练后端和两种执行模式，新模型接入需根据组合适配不同的模块。

### 后端 x 模式矩阵

|              | Colocate (共卡)                             | Fully-Async (全异步)                            |
|--------------|---------------------------------------------|-------------------------------------------------|
| **Megatron** | Bridge 自动转换；Actor/Rollout 时分共享 GPU | Bridge 或 raw 转换；独立 GPU 集群，DCS 权重同步 |
| **FSDP**     | HF 原生权重名，DTensor 自动 redistribute    | 暂不支持                                        |

### 决策树

```
新模型接入
  ├─ Megatron 后端？
  │    ├─ Bridge 支持？ → [快速路径] 仅需 Step 1-2
  │    └─ Bridge 不支持 → [完整路径] Step 1-5
  └─ FSDP 后端？ → [FSDP 路径] Step 6
```

## Megatron Backend

### 权重同步 Pipeline

训练完成后，权重需从 Megatron 内部格式同步到 Rollout 引擎（SGLang）：

```
Training Step Complete
        │
        ├─── [bridge 模式] ─────────────────────┐
        │    AutoBridge.export_hf_weights()     │
        │    自动处理 Megatron→HF 转换          │
        │                                       ▼
        ├─── [raw 模式] ────────────────────────┐
        │    all_gather_param (common.py)       │  TP-sharded → full
        │            │                          │
        │            ▼                          │
        │    convert_to_hf (__init__.py)        │  Megatron name → HF name
        │            │                          │
        │            ▼                          │
        ├────────────┴──────────────────────────┘
        │
        ├─── [共卡] UpdateWeightFromTensor
        │    GPU→CPU serialize → Gloo gather → Ray IPC
        │
        └─── [全异步] UpdateWeightFromDistributed / DeviceDirectBackend
             NCCL broadcast / HTTP push
                     │
                     ▼
             chunk_param (checkpoint_engine/utils.py)
             full → TP-sharded (逆操作，仅全异步)
```

**共卡 vs 全异步选择逻辑**（`actor.py:176`）：
```python
update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed
```

### Step 1: Model Provider (Bridge 模式 — 快速路径)

**前提**：模型已有 [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) 支持。

Bridge 模式下，`model_provider.py` 自动完成：
- **模型构建**：`AutoBridge.from_hf_pretrained()` → `bridge.to_megatron_provider()`
- **HF→Megatron 加载**：`checkpoint.py` 中 `bridge.load_hf_weights(model)` 自动转换权重
- **Megatron→HF 导出**：`hf_weight_iterator_bridge.py` 中 `bridge.export_hf_weights()` 自动导出

**你需要做的**：
1. 确认 `AutoBridge.from_hf_pretrained(hf_checkpoint)` 能正确加载你的模型
2. 如果模型有非标准 TP 分片（参见 Step 5），仍需添加自定义 all_gather/chunk 逻辑

### Step 2: Model Config & Launch Script

**File**: `scripts/models/<model_name>.sh` — 模型架构参数

从 HF config 提取对应的 Megatron 命令行参数。具体样例见 `references/examples.md#model-config-script`。

**File**: `scripts/training/<category>/run-<model>-<size>-<gpus>gpu[-async].sh` — 训练启动脚本

关键区分参数：
- 共卡：无特殊 flag（默认即共卡）
- 全异步：`--fully-async --resource '...' --max-staleness N`
- 转换模式：`--megatron-to-hf-mode bridge`（推荐）或 `raw`

具体样例见 `references/examples.md#launch-script-colocate` 和 `references/examples.md#launch-script-fully-async`。

> **Bridge 模式到此结束**。以下 Step 3-4 仅在 `--megatron-to-hf-mode raw` 时需要。

### Step 3: Megatron-to-HF Converter (raw 模式)

**File**: `slime/backends/megatron_utils/megatron_to_hf/<model_name>.py`

转换器是纯函数，将单个 Megatron 参数映射为一或多个 HF 参数：

```python
def convert_<model>_to_hf(
    args, name: str, param: torch.Tensor
) -> list[tuple[str, torch.Tensor]]:
```

**必须覆盖的参数映射**：

| Megatron Name Pattern | HF Target | Notes |
|---|---|---|
| `embedding.word_embeddings.weight` | `model.embed_tokens.weight` | |
| `output_layer.weight` | `lm_head.weight` | |
| `decoder.final_layernorm.weight` | `model.norm.weight` | |
| `self_attention.linear_qkv.weight` | `q/k/v_proj.weight` | 按 GQA 拆分 |
| `self_attention.linear_proj.weight` | `o_proj.weight` | |
| `mlp.linear_fc1.weight` | `gate_proj` + `up_proj` | GLU: chunk(2) |
| `mlp.linear_fc2.weight` | `down_proj.weight` | |
| `*.layer_norm_weight` | `*_layernorm.weight` | fused norm |

MoE、多模态、MTP 等扩展映射见 `references/examples.md`。

**参考**：简单模型看 `qwen2.py`(71行)，复杂模型看 `qwen3_5.py`(345行)。

### Step 4: Register Converter (raw 模式)

**File**: `slime/backends/megatron_utils/megatron_to_hf/__init__.py`

```python
# 1. 顶部添加 import
from .<model_name> import convert_<model>_to_hf

# 2. 在 _convert_to_hf_core() 添加路由（具体名字在前）
if "<model_name>" in model_name:
    converted_named_tensors = convert_<model>_to_hf(args, name, param)
```

`model_name` 来自 `type(AutoConfig.from_pretrained(hf_checkpoint)).__name__.lower()`。注意路由顺序：具体名字（如 `qwen3_5`）必须在通用名字（如 `qwen3`）之前。

### Step 5: Custom TP Sharding (bridge 和 raw 均适用)

大多数标准层的 TP 分片由框架通用逻辑处理。**仅当模型有非标准 TP 分片方式时**才需要此步骤。

需要同时修改两个互为严格逆操作的函数：

| 函数 | 文件 | 方向 | 场景 |
|---|---|---|---|
| `all_gather_param` | `slime/.../update_weight/common.py` | TP-sharded → full | 训练侧发送权重 |
| `chunk_param` | `relax/checkpoint_engine/utils.py` | full → TP-sharded | Rollout 侧加载权重(全异步) |

**核心不变量**：`chunk_param(all_gather_param(p))[tp_rank] == p`

在通用逻辑前按参数名 pattern 插入模型特有分支。具体代码见 `references/examples.md#custom-tp-all-gather`。

## FSDP Backend

### Step 6: FSDP Model Adaptation

FSDP 后端直接使用 HuggingFace 模型，权重名就是 HF 原生名称，**不需要写转换器**。

**权重同步**（`slime/backends/fsdp_utils/update_weight_utils.py`）：
- `model.state_dict()` 遍历所有参数
- DTensor 通过 `redistribute(Replicate)` 自动 all-gather
- 按 buffer 大小分桶，序列化推送到 Rollout

这套逻辑对所有模型通用，无需模型特化。

**需要模型特化的场景**：仅 MoE 模型需要自定义 expert dispatch 和 fused kernel（如 `slime/backends/fsdp_utils/models/qwen3_moe.py`），放在 `slime/backends/fsdp_utils/models/` 下。

## Common Pitfalls

### 1. module. prefix 不一致
不同模型 prefix 深度可能不同。如 Qwen3.5 多了 `language_model` 层，需要在转换器开头归一化：
```python
if name.startswith("module.module.language_model."):
    name = "module.module." + name[len("module.module.language_model."):]
```

### 2. 多模态子模型绕过 TP
非 Megatron 的视觉/音频编码器没有 `tensor_model_parallel` 属性，需在 `all_gather_param` 中跳过：
```python
if ".vision_model." in name:
    if not hasattr(param, "tensor_model_parallel"):
        return param
```

### 3. ZeroCenteredRMSNorm 偏移
Megatron 的 ZeroCenteredRMSNorm 权重中心为 0，HF 的 RMSNorm 中心为 1：
```python
return [(f"{prefix}.norm.weight", param + 1)]
```

### 4. GLU linear_fc1 layout
框架已内置 SwiGLU 的交错 layout 处理，不需要额外代码。

### 5. Bridge 模式仍需自定义 TP 逻辑
Bridge 自动处理名称转换，但 `all_gather_param`/`chunk_param` 中的自定义 TP 逻辑仍然需要——TP 分片/还原发生在转换之前/之后。

### 6. _convert_to_hf_core 路由顺序
`in` 匹配 model_name 时，具体名字必须在前：`qwen3_5` → `qwen3vl` → `qwen3`。

### 7. HF Config 动态读取
模型特有参数通过 `get_hf_config()` 获取（`@lru_cache`，只加载一次）：
```python
from slime.utils.misc import get_hf_config
config = get_hf_config(args.hf_checkpoint)  # .text_config.xxx
```

### 8. 共卡模式下 named_params_and_buffers 的 convert_to_global_name
共卡模式使用 `UpdateWeightFromTensor`，内部通过 `HfWeightIteratorBase.create()` 根据 `megatron_to_hf_mode` 选择迭代器。raw 模式需要 `convert_to_global_name=True`（将 VP/PP 局部层索引转换为全局索引），bridge 模式不需要：
```python
# actor.py:154
convert_to_global_name=args.megatron_to_hf_mode == "raw"
```

## Validation Checklist

1. **转换器覆盖率** (raw 模式)：遍历所有参数名，确认无 `ValueError`
2. **all_gather/chunk 对称性**：构造随机 tensor → 分片 → all_gather → chunk → 验证一致
3. **端到端**：小规模跑 1-2 iter，检查 log 无 `Unknown parameter name`，Rollout 引擎成功加载权重
4. **对话 SFT**：加载目标 checkpoint 的真实模板和停止配置，使用真实训练样本与协议边界样本跑 Relax 数据集。按 [对话 SFT 对齐验收](references/sft-chat-alignment.md) 逐 token 核对训练标签、loss mask 和推理前缀，记录样例覆盖和未验证项。

## File Reference Map

| File | Role | When to Modify |
|---|---|---|
| `slime/backends/megatron_utils/megatron_to_hf/<model>.py` | Megatron→HF 映射 (raw) | raw 模式 |
| `slime/backends/megatron_utils/megatron_to_hf/__init__.py` | model_name 路由 (raw) | raw 模式 |
| `slime/backends/megatron_utils/update_weight/common.py` | TP all_gather | 非标准 TP 时 |
| `relax/checkpoint_engine/utils.py` | TP chunk (逆操作) | 非标准 TP 时 |
| `slime/backends/megatron_utils/model_provider.py` | 模型构建 (raw/bridge) | raw 自定义时 |
| `slime/backends/megatron_utils/checkpoint.py` | HF→Megatron 加载 (bridge) | Rarely |
| `slime/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py` | bridge 权重迭代 | Rarely |
| `slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py` | raw 权重迭代 | Rarely |
| `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py` | 共卡模式权重同步 | Rarely |
| `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py` | 全异步权重同步 | Rarely |
| `relax/checkpoint_engine/backends/device_direct.py` | 全异步 DCS 通信 | Rarely |
| `slime/backends/fsdp_utils/update_weight_utils.py` | FSDP 权重同步 (通用) | Rarely |
| `slime/backends/fsdp_utils/models/` | FSDP 模型特化 (MoE) | MoE 模型时 |
| `scripts/models/<model>.sh` | 模型架构参数 | Always |
| `scripts/training/<category>/run-<model>-*.sh` | 训练启动脚本 | Always |
| `slime/utils/misc.py :: get_hf_config()` | HF config 缓存 | Never |

## Bundled Resources

- `references/examples.md` — 完整代码样例：简单/复杂转换器、自定义 TP 逻辑、注册、脚本模板
- `references/sft-chat-alignment.md` — 对话和工具轨迹的模板、训练标签与推理前缀验收
