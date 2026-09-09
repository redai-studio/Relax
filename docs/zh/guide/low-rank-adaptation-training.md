# LoRA RL（参数高效 RL 后训练）

Relax 支持使用 **LoRA**（Low-Rank Adaptation，低秩适配）进行 RL 后训练：不再更新全部权重，而是冻结基座模型、只训练小规模的低秩适配矩阵。这样可以大幅缩减优化器状态和每步权重同步的数据量，从而在相同 GPU 预算下容纳更大的模型。

Relax 内置了两条端到端的 LoRA rollout 路径，二者的区别仅在于**训练好的适配器如何送达 rollout 引擎**：

| 模式            | 每步同步到 rollout 的内容                   | rollout 提供的模型      | 参考启动脚本                                                          |
| --------------- | ------------------------------------------ | ----------------------- | ------------------------------------------------------------------- |
| **Merge 模式**  | 完整模型（适配器已折叠进基座）             | 一个已合并的模型        | `scripts/training/text/run-qwen3-4B-lora-merge-8xgpu.sh`             |
| **Adapter 模式**| 基座仅同步一次，之后只推送 LoRA 适配器     | 基座 + 运行时适配器     | `scripts/training/text/run-qwen3-4B-lora-adapter-8xgpu-async.sh`     |

两种模式**互斥**。如果启用了 LoRA（`--lora-rank > 0`）却没有指定模式，Relax 会强制使用 **Merge 模式**（默认、兼容性最广的路径）。

两种模式都支持**稠密和 MoE（grouped-expert）**模型，也都支持 **colocate 和 fully-async**。更大规模的参考配方：

| 配方                     | 脚本                                                                     |
| ------------------------ | ------------------------------------------------------------------------ |
| MoE、Adapter 模式、2 节点 | `scripts/training/text/run-qwen36-35B-A3B-lora-adapter-16xgpu-async.sh`   |
| MoE + VL、Merge 模式      | `scripts/training/multimodal/run-qwen36-35B-A3B-lora-8xgpu-image.sh`      |

## 概述

训练侧通过 Megatron-Bridge 的 PEFT 集成来应用 LoRA（`relax/backends/megatron/model_provider.py`）：基座模型被冻结，只有注入的适配器参数会接收梯度。两种模式的差异完全体现在**权重同步路径**（`relax/backends/megatron/weight_update/`）上：

- **Merge 模式**在导出时把每个适配器折叠进对应的基座权重（`LoRAMerge`），rollout 引擎因此加载的是单个已合并的模型——推理侧无需感知 LoRA。这条路径复用标准的全量权重同步流程，所以每步带宽与全参数训练相同。
- **Adapter 模式**只同步一次冻结的基座，之后每步仅通过 SGLang 的运行时 LoRA API 推送**小规模的适配器**。rollout 请求通过 `lora_path` 选中它。这样以少量 rollout 侧的 LoRA 开销，换取每步同步带宽的大幅下降。

::: tip 该选哪种模式？
- **Merge 模式**——最简单，兼容最广的模型和两种部署布局（包括分布式 / NCCL rollout 引擎）。rollout 推理就是普通的全模型推理。代价是每步都要做一次全量权重同步。
- **Adapter 模式**——当每步权重同步带宽成为瓶颈时（基座大、适配器小）最合适。附带一些约束（`--sglang-dp-size 1`、不支持分布式 rollout 引擎、VL / omni 模型必须用 `--lora-scope language`）。rollout 走 SGLang 的 LoRA 内核。
:::

## 架构

两种模式共享同样的训练侧；区别只在同步箭头上传输的数据。

```
        ┌────────────────────────────────────────────────────────┐
        │                  Training side (Actor)                 │
        │  Megatron-LM + Megatron-Bridge PEFT                    │
        │  base weights FROZEN, only LoRA adapter has gradients  │
        │  (model_provider.wrap_model_provider_with_lora)        │
        └───────────────────────────┬────────────────────────────┘
                                     │
             ┌───────────────────────┴───────────────────────┐
             │                                               │
     MERGE mode: fold adapter                      ADAPTER mode: base once,
     into base, sync FULL model                    then push ADAPTER only
     every step (NCCL / IPC)                       (SGLang runtime LoRA API)
             │                                               │
             ▼                                               ▼
        ┌─────────────────────────┐              ┌─────────────────────────┐
        │  Rollout (SGLang)       │              │  Rollout (SGLang)       │
        │  one merged model       │              │  base + adapter         │
        │  plain inference        │              │  lora_path=             │
        │                         │              │   relax_policy_lora     │
        └─────────────────────────┘              └─────────────────────────┘
```

权重同步后端会根据 LoRA 模式进行分派：

- **Colocate（同置）**——`UpdateWeightFromTensor`（`weight_update/update_weight_from_tensor.py`）。Merge 模式在 HF 导出时折叠适配器；Adapter 模式走 `_update_weights_adapter_mode`（基座同步一次 + 每步通过 `load_lora_adapter_from_tensors` 在内存中推送适配器）。
- **Fully-async（全异步）**——`DeviceDirectBackend`（`distributed/checkpoint_service/backends/device_direct.py`）。Merge 模式在 NCCL 广播前折叠适配器；Adapter 模式每步由 rank 0 通过 NCCL 广播适配器张量（按 `--update-weight-buffer-size` 分桶），再向每个引擎扇出只带元数据的 HTTP `/update_lora_from_distributed`。

与模式无关的适配器导出/聚合逻辑（构建 HF 导出 bridge、以 SGLang 命名导出本地适配器、PP 聚合、集合式 delta-skip 决策、生成适配器 config dict）都收敛在共享的 `LoraAdapterSync` 辅助类（`weight_update/lora_adapter_sync.py`）中，两个后端以组合方式复用它。整条路径纯内存——运行时的适配器不落盘。

## 配置

所有 LoRA 参数都定义在 `relax/utils/arguments.py`。

| 参数                     | 类型       | 默认值                        | 说明                                                                                                 |
| ------------------------ | ---------- | ----------------------------- | ---------------------------------------------------------------------------------------------------- |
| `--lora-rank`            | int        | `0`                           | LoRA 秩。`0` 表示禁用 LoRA；任何 `> 0` 的值都会启用它。                                               |
| `--lora-alpha`           | int        | `32`                          | LoRA alpha 缩放因子。                                                                                 |
| `--lora-target-modules`  | str（列表）| `linear_qkv linear_proj`      | 要适配的 Megatron 风格模块名（见下方映射表）。以空格分隔。                                            |
| `--lora-scope`           | str        | `all`                         | 适配器注入到哪个模型区域：`all` / `language` / `vision`。纯文本模型无影响。                           |
| `--lora-dropout`         | float      | `0.0`                         | LoRA 层内部的 dropout 概率。                                                                          |
| `--lora-merge-mode`      | flag       | `False`                       | 同步前把适配器折叠进基座权重。与 `--lora-adapter-mode` 互斥。                                         |
| `--lora-adapter-mode`    | flag       | `False`                       | 基座只同步一次，之后每步只推送适配器。与 `--lora-merge-mode` 互斥。                                   |

### 目标模块

`--lora-target-modules` 接收 **Megatron 风格**的名字（Megatron-Bridge 的 LoRA 匹配器直接遍历的规范形式）。导出适配器时会自动扩展成 HF 风格名字（`relax/utils/megatron_peft_utils.py` 中的 `convert_megatron_to_hf_target_modules`）：

| Megatron 名字          | HF 投影                                                    |
| ---------------------- | ---------------------------------------------------------- |
| `linear_qkv`           | `q_proj`、`k_proj`、`v_proj`                               |
| `linear_proj`          | `o_proj`                                                   |
| `linear_fc1`           | `gate_proj`、`up_proj`                                     |
| `linear_fc2`           | `down_proj`                                                |
| `router`               | `gate`                                                     |
| `in_proj`（GDN）       | `in_proj_qkv`、`in_proj_z`、`in_proj_b`、`in_proj_a`       |
| `out_proj`（GDN）      | `out_proj`                                                 |

（完整映射表，包括拆分 QKV 和 MLA 变体，见 `MEGATRON_TO_HF_MODULES`。）

推送给 rollout 引擎的适配器使用的命名略有不同（`convert_megatron_to_sglang_target_modules`，差异项见 `MEGATRON_TO_SGLANG_MODULES`）：SGLang 把 GDN 的输入投影融成一个模块，所以 `in_proj` 映射到它实际包装的 `in_proj_qkvz`。其余名字与 HF 相同。写进 **checkpoint** 的适配器仍保留 HF-PEFT 名字，以便被 `peft` 直接加载。

### LoRA 作用域（VL / omni 模型）

Megatron-Bridge 的 LoRA 匹配器按**叶子名**匹配，因此一个裸的 `linear_qkv` 会同时注入语言主干和视觉塔。`--lora-scope` 会把请求的区域解析到具体的模块树上，改用全路径名去包装（`scope_target_modules_to_region`）：

- `all`（默认）——所有匹配到的模块，含视觉塔 / projector / 音频编码器。
- `language`——排除视觉塔 / projector / 音频编码器。纯语言 RL 的常用配置，也是 Adapter 模式在带 `vision_config` / `audio_config` 的模型上的**强制要求**。
- `vision`——只包装这些区域。

它控制的是适配器**注入**，不是基座权重的冻结。

### 校验规则

`lora_rank > 0` 时在 `relax/utils/arguments.py` 中强制执行：

- `--lora-merge-mode` 与 `--lora-adapter-mode` **互斥**——只能选其一。
- `--lora-adapter-mode` 要求 `--sglang-dp-size 1`（SGLang 的动态 LoRA 加载不支持 DP attention）。
- 如果模型的 HF config 声明了 `vision_config` 或 `audio_config`，`--lora-adapter-mode` 搭配 `--lora-scope all` 会被**拒绝**。SGLang 只在语言模型层上承载 LoRA，视觉适配器会被训练、却在送往 rollout 前被静默丢弃——actor 优化的是 rollout 永远不会跑的策略。请改用 `--lora-scope language`，或改用 `--lora-merge-mode`。
- `--lora-target-modules` 里含 `router` 时必须使用 `--lora-adapter-mode`。Merge 模式经由 `LoRAMerge` 折叠适配器，而它只处理 `LoRALinear`；router 用的是 `LoRATopKRouter`，其适配器会在同步时被丢弃。
- 若两个模式标志都未设置，Relax 会**强制启用 `--lora-merge-mode`** 并打印警告。

## Merge 模式配方

参考：`scripts/training/text/run-qwen3-4B-lora-merge-8xgpu.sh`（Qwen3-4B，8 卡 colocate，在 `dapo-math-17k` 上跑 GRPO）。

该脚本中的 LoRA 配置块：

```bash
LORA_ARGS=(
   --lora-rank 32
   --lora-alpha 64
   --lora-target-modules linear_qkv linear_proj
   --lora-dropout 0.0
   --lora-merge-mode
)
```

像任何 colocate 脚本一样启动：

```bash
bash scripts/training/text/run-qwen3-4B-lora-merge-8xgpu.sh
```

每步的工作方式：actor 导出自己的权重，`LoRAMerge` 在 HF 转换过程中把每个适配器折叠进配对的基座权重，然后合并后的完整模型通过正常的 IPC/NCCL 路径同步给 SGLang。rollout 引擎完全不感知 LoRA——它只是提供一个完整模型。

::: warning MoE 专家折叠要求 `--expert-tensor-parallel-size 1`
Grouped-expert LoRA 是在拥有该专家的 EP rank 上本地折叠进专家基座权重的（`tp_size=1`，不走 expert-TP 集合通信），ETP 不匹配会让单个 rank 独自进入集合通信而死锁。这一约束对 **Merge 模式始终生效**，对 **Adapter 模式则在存在 `actor_fwd` / reference 角色时（off-policy）生效**——该角色没有自己的适配器传输通道，它的专家 delta 是折叠进基座的。EP（expert-model-parallel）仍可 `> 1`；稠密模型不受影响。参考配方均已设置 `--expert-tensor-parallel-size 1`。
:::

## Adapter 模式配方

参考：`scripts/training/text/run-qwen3-4B-lora-adapter-8xgpu-async.sh`（Qwen3-4B，8 卡 fully-async，在 `dapo-math-17k` 上跑 GRPO）。

该脚本中的 LoRA 配置块：

```bash
LORA_ARGS=(
   --lora-rank 128
   --lora-alpha 64
   --lora-target-modules linear_qkv linear_proj
   --lora-dropout 0.0
   --lora-adapter-mode
)
```

注意它同时设置了 `--rollout-num-gpus-per-engine 1`（每个引擎一张卡 → `sglang_dp_size == 1`，这是 Adapter 模式的要求）。启动：

```bash
bash scripts/training/text/run-qwen3-4B-lora-adapter-8xgpu-async.sh
```

工作方式：

1. **首次同步**——把仅基座的权重推给引擎（适配器参数会从转换 bucket 中剥离，绝不合并），然后以固定名字 `relax_policy_lora` 注册 LoRA 适配器。
2. **之后每一步**——**只**刷新适配器。基座保留在引擎上不动。rollout 请求会自动带上 `lora_path=relax_policy_lora`（`relax/engine/rollout/sglang_rollout.py`），使生成走训练好的适配器。
3. **Delta-skip（增量跳过）**——如果任一 rank 上都没有适配器参数变化超过 `1e-6` 阈值，则整次推送被跳过。这是一个跨所有 rank 的集合决策（单 rank 提前返回会让聚合失去同步并挂起）。

适配器本身的传输方式因部署而异：

- **Colocate**——适配器被序列化后通过 `load_lora_adapter_from_tensors` 在内存中推送（SGLang ≥ 0.5.12，无磁盘 IO）。
- **Fully-async**——rank 0 汇总出完整适配器后，通过 NCCL 在权重更新组上 `broadcast`（`src=0`，与基座权重同一个组），并向每个引擎扇出只带元数据（张量名 / dtype / shape / 分桶边界 / 适配器配置）的 HTTP `/update_lora_from_distributed`。广播在收发两侧都按 `--update-weight-buffer-size`（默认 512 MiB）分桶，因此显存峰值是一个桶而不是整个适配器——A3B MoE 的适配器有数 GB。同样无磁盘 IO——适配器不会落到网络文件系统上。

::: warning Adapter 模式约束
- 要求 `--sglang-dp-size 1`（参数校验时强制）。
- 在 colocate 模式下**不支持分布式（非同置的 NCCL）rollout 引擎**——适配器只会推送给同置的 IPC 引擎。请使用 colocate 或 fully-async；如需分布式 rollout，请改用 Merge 模式。
- 在 **VL / omni** 模型上必须使用 `--lora-scope language`（参数校验时强制）：SGLang 只在语言模型层上承载 LoRA。若确实要训练视觉塔，请改用 Merge 模式。
- MoE（grouped-expert）LoRA 在两种部署下**均已支持**。当存在 `actor_fwd` / reference 角色时，需要 `--expert-tensor-parallel-size 1`（见上方警告）。
- 在 **GDN** 层上，`in_proj` 适配器的 `b`/`a`（beta/alpha 门控）行会被一个 forward hook 钉在零（`install_gdn_gate_mask_hooks`）：SGLang 只在融合后的 `in_proj_qkvz` 上承载适配器，门控切片没有对应的适配器，在那里学到的 delta 永远无法在 rollout 时复现。屏蔽的是适配器的*输出*，因此这些行拿到的梯度恒为零，训练与 rollout 在数值上保持一致。Merge 模式不需要这层处理——门控切片可以无损折叠进基座。
- Adapter 模式在 colocate 下会关闭下一步 rollout 预取（每步适配器更新约 1s，预取只会被重做）。
:::

## MoE 与 VL 配方

- **MoE、Adapter 模式、2 节点**——`scripts/training/text/run-qwen36-35B-A3B-lora-adapter-16xgpu-async.sh`（Qwen3.6-35B-A3B，16 卡 fully-async）。适配 attention、GDN 和两个 MLP 投影，并限定在语言主干：

  ```bash
  LORA_ARGS=(
     --lora-rank 32
     --lora-alpha 64
     --lora-scope language
     --lora-target-modules linear_qkv linear_proj in_proj out_proj linear_fc1 linear_fc2
     --lora-dropout 0.0
     --lora-adapter-mode
  )
  ```

- **MoE + VL、Merge 模式**——`scripts/training/multimodal/run-qwen36-35B-A3B-lora-8xgpu-image.sh`。用 `--lora-scope all` 连视觉塔一起训练，因此必须使用 `--lora-merge-mode`（SGLang 无法承载视觉适配器）。

两者都设置了 `--expert-tensor-parallel-size 1`。

## Checkpoint 与导出

默认情况下，`--save` 写入常规的完整 Megatron 分布式 checkpoint。增加 `--save-lora-only` 后，改为写入更小、可续训的 actor checkpoint：其中包含 LoRA 张量以及 optimizer、scheduler、RNG、iteration 和 common state；恢复时，冻结的基座权重从 `--hf-checkpoint` 重新加载。

`--save-lora-only` 与全量 checkpoint 导出互斥：它要求同步 BF16 `torch_dist` checkpoint，不能与 `--async-save`、`--rotate-ckpt`、`--no-save-optim`、`--no-save-rng`、`--fp16`、`--fp8` 或 `--save-hf` 同时使用。如果存在任何非 LoRA 的可训练模型参数，也会直接报错，防止生成无法完整续训的产物。

::: tip
轻量 checkpoint 是分片的 Megatron 续训产物，不是可移植的 HF-PEFT adapter。显式使用 `--save-hf` 时仍可导出包含 `adapter_config.json`、`adapter_model.safetensors` 和 `relax_lora_meta.json` 的 `lora_adapter/`；该目录仅供外部或推理使用，不是续训来源。
:::

### 离线合并为独立 HF checkpoint

`scripts/tools/merge_lora_adapter_to_hf.py` 可以把导出的 `lora_adapter/` 折回基座 HF 模型，写出一个独立的合并 checkpoint——不需要 Ray / Megatron / GPU 集群：

```bash
python scripts/tools/merge_lora_adapter_to_hf.py \
    --base-hf-dir /path/to/Qwen3.6-35B-A3B \
    --adapter-dir /path/to/save/iter_0000100/lora_adapter \
    --output-dir  /path/to/Qwen3.6-35B-A3B-merged
```

它刻意在**张量**层面合并，而不是用 `peft.merge_and_unload()`：`AutoModelForCausalLM` 会把多模态 checkpoint 静默解析成它的纯文本类（重新保存时丢掉 `vision_config` 和视觉塔），而 PEFT 按*模块*名匹配，看不到以 3 维 `nn.Parameter` 存储的 grouped MoE 专家——那恰恰是 A3B MoE 上训练容量的大头。该工具逐 shard 流式处理（内存上限是最大的单个 shard），原样拷贝 `config.json` 和各类辅助文件，并把「适配器张量配不上基座张量」视为硬错误而非静默丢弃。

## 故障排除

| 现象                                                                          | 可能原因 / 解决办法                                                                                                                          |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `--lora-merge-mode and --lora-adapter-mode are mutually exclusive`             | 同时设置了两个标志。只保留其一。                                                                                                            |
| `--lora-adapter-mode requires --sglang-dp-size 1`                             | Adapter 模式遇到了 DP attention。设置 `--rollout-num-gpus-per-engine 1`（或以其他方式使 `sglang_dp_size == 1`）。                            |
| `--lora-adapter-mode does not yet support distributed ... rollout engines`     | Adapter 模式遇到非同置 rollout 引擎。改用 colocate，或用 `--lora-merge-mode` 做分布式 rollout。                                             |
| `--lora-adapter-mode with --lora-scope all is not supported on a model that has a vision/audio encoder` | VL / omni 模型用了 Adapter 模式。改用 `--lora-scope language`，或改用 `--lora-merge-mode` 以训练视觉塔。          |
| `LoRA on 'router' is only supported in --lora-adapter-mode`                    | Merge 模式下 `--lora-target-modules` 含 `router`。去掉 `router`，或改用 Adapter 模式。                                                      |
| `MoE LoRA expert folding requires --expert-tensor-parallel-size 1`             | MoE 且 ETP > 1。设置 `--expert-tensor-parallel-size 1`（EP 可保持 > 1）。只有完全 on-policy（无 `actor_fwd`）的 Adapter 模式才不受此约束。   |
| `[lora-merge] NO adapter tensors in backup dict`                             | `weights_getter()` 输出里缺少适配器参数——检查 `--lora-target-modules` 名字，以及 LoRA 是否真的在建模时挂上。                                |
| Adapter 模式下 rollout 质量像是基座模型                                        | 适配器推送被跳过，或 `lora_path` 未生效。确认请求带有 `lora_path=relax_policy_lora`，并检查日志里是否有适配器推送记录（delta-skip 会跳过无变化的推送）。   |
