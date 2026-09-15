# 低精度训练 (FP8 & INT4)

Relax 在两条路径上支持低精度 RL 后训练：**FP8 训练**（Megatron-LM 原生 FP8 前向）与 **INT4 fake-QAT**（BF16 主权重 + MoE expert 层 INT4 假量化）。两种模式都驱动 **真实的低精度 rollout**（SGLang 端真实低精度推理），并在每个训练 step 后通过 NCCL 同步权重。

## 概述

仓库中提供两条端到端配方：

| 模式               | 训练侧                                                                                  | Rollout 侧                                                              | 参考启动脚本                                                  |
| ------------------ | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------- |
| **FP8**            | Megatron-LM 原生 FP8（`e4m3`、blockwise）                                                | SGLang FP8 推理（真实 FP8 权重）                                         | `scripts/training/text/run-qwen3-30B-A3B-fp8-8xgpu.sh`        |
| **INT4 fake-QAT**  | BF16 前向 + `TEGroupedLinear` 上的 STE INT4 假量化（仅 MoE expert，对称）                  | SGLang W4A16 推理（compressed-tensors，**symmetric**，group_size=128）    | `scripts/training/text/run-qwen3-30B-A3B-int4-8xgpu.sh`       |

配套的离线工具有四个：

- `scripts/tools/convert_hf_to_fp8.py` — 把 BF16/FP16 的 HF checkpoint 量化为 FP8。
- `scripts/tools/convert_fp8_to_bf16.py` — 把 block 量化的 FP8 HF checkpoint 反量化回 BF16（`convert_hf_to_fp8.py` 的逆操作；当你拿到一个预量化的 FP8 发布版，但下游链路需要纯 BF16 HF 时使用）。
- `scripts/tools/convert_hf_to_int4.py` — 把 BF16 的 HF checkpoint 量化为 W4A16（compressed-tensors）。
- `scripts/tools/convert_moe_int4_to_bf16.py` — 把 W4A16 的 HF checkpoint 反量化回 BF16（当你拿到一个预量化的 W4A16 发布版，但下游链路（非 bridge 模式或其他工具）需要纯 BF16 HF 时使用）。

## 架构

两种模式都采用标准的 `--colocate` 部署：actor 与 rollout 共享同一组 GPU 时分复用，低精度链路只改变它们之间流动的数据格式。

```
                 ┌──────────────────────────────────────────────────────┐
                 │                  Training side (Actor)               │
                 │  Megatron-LM, transformer_engine, --bf16             │
                 │                                                      │
                 │  FP8 mode:   real FP8 forward (TE blockwise e4m3)    │
                 │  INT4 mode:  BF16 forward + fake-int4 STE on         │
                 │              TEGroupedLinear._get_weight_tensors()   │
                 └────────────────────────┬─────────────────────────────┘
                                          │
                            per-step weight sync via NCCL
                                          │
                                          ▼
                 ┌─────────────────────────────────────────────────────┐
                 │                  Rollout side (SGLang)              │
                 │                                                     │
                 │  FP8 mode:   real FP8 weights                       │
                 │              quantizer_fp8.quantize_params_fp8      │
                 │  INT4 mode:  real W4A16 (AWQ pack)                  │
                 │              quantizer_compressed_tensors           │
                 │                  .quantize_params_compressed_tensors│
                 └─────────────────────────────────────────────────────┘
```

权重更新流水线（`relax/backends/megatron/weight_update/`）根据 `--hf-checkpoint/config.json` 中的 `quantization_config.quant_method` 做分发：

- `quant_method == "fp8"` → `quantize_params_fp8`（`weight_conversion/processors/quantizer_fp8.py`）
- `quant_method == "compressed-tensors"` → `quantize_params_compressed_tensors`（`weight_conversion/processors/quantizer_compressed_tensors.py`）

## 离线量化工具

### `convert_hf_to_fp8.py`

把 BF16/FP16 的 HF safetensors checkpoint 量化为 FP8。

```bash
python scripts/tools/convert_hf_to_fp8.py \
  --model-dir /path/to/Qwen3-30B-A3B \
  --save-dir  /path/to/Qwen3-30B-A3B-FP8 \
  --strategy  block \
  --block-size 128 128 \
  --max-workers 4
```

| 参数             | 默认值  | 说明                                                                                                  |
| ---------------- | ------- | ----------------------------------------------------------------------------------------------------- |
| `--model-dir`    | —       | 源 HF safetensors 目录。                                                                              |
| `--save-dir`     | —       | 输出目录。                                                                                            |
| `--strategy`     | `block` | `block` / `channel` / `tensor` 三选一。`block` 写 `fp8` 布局；`channel` 写 `compressed-tensors` 布局。  |
| `--block-size`   | —       | `--strategy=block` 时必填两个整数（例如 `128 128`）。                                                  |
| `--max-workers`  | `1`     | shard 级并行的线程池大小。                                                                            |
| `--scale-fmt`    | `None`  | 可选，设为 `ue8m0` 表示输出 UE8M0 scale。                                                              |

跳过的模块（保持原 dtype 写出）：`layernorm`、`embed`、`router`、`lm_head`、`mlp.gate.*`、`norm`、`eh_proj`、`weights_proj`、`conv1d`、`A_log`、`dt_bias`、`in_proj_a`、`in_proj_b`。该过滤规则硬编码在脚本中。

输出：

- 量化后的 `*.safetensors` 分片（FP8 权重 + `weight_scale_inv` / `weight_scale`）。
- 改写后的 `config.json`，包含 `quantization_config` 块。`block`/`tensor` 时是 `{"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic", "weight_block_size": [...], "modules_to_not_convert": [...]}`，`channel` 时遵循 compressed-tensors schema。
- 更新后的 `model.safetensors.index.json`。

### `convert_fp8_to_bf16.py`

把 block 量化的 FP8 HF checkpoint 反量化回 BF16。适用于起点是预量化的 FP8 发布版、但下游链路需要纯 BF16 HF 的场景。

```bash
python scripts/tools/convert_fp8_to_bf16.py \
  --model-dir /path/to/Qwen3-30B-A3B-FP8 \
  --save-dir  /path/to/Qwen3-30B-A3B-bf16 \
  --max-workers 4
```

| 参数             | 默认值  | 说明                                                                              |
| ---------------- | ------- | --------------------------------------------------------------------------------- |
| `--model-dir`    | —       | 源 FP8 HF safetensors 目录。                                                       |
| `--save-dir`     | —       | 输出目录。                                                                         |
| `--max-workers`  | `1`     | shard 级并行的线程池大小。                                                          |

每个 FP8 `weight` 与其 `weight_scale_inv` 配对，并通过 Triton kernel（`weight_dequant_kernel`）反量化；shard 级并行处理，若所需的 scale 张量位于其他 shard，则通过 `safetensors.safe_open` 按需读取。`element_size() > 1` 的张量（本身就不是 FP8）原样拷贝；找不到配对 `_scale_inv` 的 FP8 张量会保留原样并打 warning。

输出：

- BF16 的 `*.safetensors` 分片（反量化后的 FP8 权重；`_scale_inv` 张量被丢弃）。
- 移除 `quantization_config` 块的 `config.json`，避免下游加载器对已反量化的权重再做一次反量化。
- 重写后的 `model.safetensors.index.json`，不再包含已废弃的 `_scale_inv` 条目。

::: tip
FP8 训练工作流下通常 **不需要** 这个脚本 — bridge 模式（`--megatron-to-hf-mode bridge`）会直接读取 FP8 HF。此工具用于离线转换：当你需要把 FP8 checkpoint 转回 BF16 HF 作为其他流水线的输入时（例如作为另一份配方的 `--ref-load`，或喂给 `convert_hf_to_int4.py`）。
:::

### `convert_hf_to_int4.py` {#convert-hf-to-int4}

把 BF16 的 HF checkpoint 量化为 W4A16（compressed-tensors）。依赖 `fake_int4_quant_cuda` kernel，需先编译（见 [编译 int4_qat kernel](#build-int4-qat-kernel)）。

```bash
python scripts/tools/convert_hf_to_int4.py \
  --model-dir /path/to/Qwen3-30B-A3B \
  --save-dir  /path/to/Qwen3-30B-A3B-int4 \
  --group-size 128 \
  --is-symmetric \
  --max-workers 4
```

| 参数              | 默认值                                                                                                                                                              | 说明                                                                                                                |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `--model-dir`     | —                                                                                                                                                                   | 源 HF safetensors 目录。                                                                                            |
| `--save-dir`      | —                                                                                                                                                                   | 输出目录。                                                                                                          |
| `--group-size`    | `32`                                                                                                                                                                | INT4 group size；训练脚本里用 `128`。                                                                                |
| `--is-symmetric`  | CLI 默认 `false` — **跑 INT4 fake-QAT 训练时必须带上这个 flag**                                                                                                       | 启用对称量化。必须与训练侧 STE（硬编码对称）保持一致；不带则 train/rollout 分布不一致。                              |
| `--ignore-rules`  | `re:.*lm_head.*`、`re:.*norm.*`、`re:.*embed.*`、`re:.*self_attn.*`、`re:.*shared_experts.*`、`re:.*mlp\.(gate|up|gate_up|down)_proj.*`、`re:.*mlp\.gate\.*`        | 跳过量化的 key 规则（支持 `re:` 前缀正则或字面前缀匹配）。默认只量化 MoE expert 的 `linear_fc1` / `linear_fc2`。      |
| `--max-workers`   | `1`                                                                                                                                                                 | 线程池大小。                                                                                                        |

::: warning
默认的 `--ignore-rules` 是为只量化 **expert** 权重的 MoE 拓扑准备的。如果改动 ignore 列表，请务必与训练侧 fake-QAT 的作用范围（只覆盖 `TEGroupedLinear`，即 MoE expert）保持一致 — 否则 rollout 和 training 看到的量化模式会不一致。
:::

::: danger
**生成用于 INT4 fake-QAT 训练的 `--hf-checkpoint` 时，务必带上 `--is-symmetric`**。`docker/patch/megatron/20260506-85bced0ae.patch` 里的训练侧 STE 是硬编码对称（`q_max=7`，无 zero-point）。如果 W4A16 checkpoint 是非对称（CLI 默认），`pack_layer(sym=False)` 会在 rollout 端打包出带 zero-point 偏移的权重，**与训练侧 STE 所模拟的量化噪声不一致**，QAT 的核心假设就被破坏了。
:::

输出：

- 量化后的 `*.safetensors`，对每个被匹配的权重写出 `weight_packed`（int32 打包的 int4）、`weight_scale`、`weight_shape` 以及（asymmetric 时）`weight_zero_point` 三元组/四元组。
- 改写后的 `config.json`，写入 compressed-tensors 的 `quantization_config` 块。

### `convert_moe_int4_to_bf16.py`

把 W4A16 compressed-tensors HF checkpoint 反量化为 BF16。适用于起点是预量化的 W4A16 发布版的情况。

```bash
python scripts/tools/convert_moe_int4_to_bf16.py \
  --model-dir /path/to/Qwen3-30B-A3B-int4
  # 默认输出：/path/to/Qwen3-30B-A3B-int4_bf16
```

| 参数                          | 默认值                | 说明                                                                                       |
| ----------------------------- | --------------------- | ------------------------------------------------------------------------------------------ |
| `--model-dir`                 | —                     | 源 W4A16 HF checkpoint。                                                                   |
| `--output-dir`                | `<model-dir>_bf16`    | 输出目录。                                                                                 |
| `--files`                     | 全部 `*.safetensors`  | 限定只处理部分 shard（断点重跑时有用）。                                                    |
| `--config-path`               | `<model-dir>/config.json` | 覆盖读取 `group_size` 的配置文件路径。                                                  |
| `--overwrite`                 | `false`               | 即使输出文件已存在也重新处理。                                                              |
| `--keep-quantization-config`  | `false`               | 在输出 `config.json` 中保留 `quantization_config` 块，而不是剥除。                          |

输出：

- BF16 的 `*.safetensors` 分片（expert 的 `weight_packed` 三元组合并回 `.weight`；非 expert tensor 原样拷贝）。
- 默认从 `config.json` 中剥除 `quantization_config`（除非 `--keep-quantization-config`）。
- 旁路文件 `quantization_config.json`，保存被剥除的 quantization 配置块，并追加 `ignore` 列表（把那些有 `.weight` 但没有 `weight_packed` 的顶层命名空间，如 `vision_tower` / `mm_projector` 加进去）。

::: tip
INT4 fake-QAT 训练流程下通常 **不需要** 这个脚本 — bridge 模式（`--megatron-to-hf-mode bridge`）会经由 `megatron/bridge/models/qwen/qwen3_moe_bridge.py` 中被 patch 过的 `build_conversion_tasks` 直接加载 W4A16，patch 会为每组 `weight_packed` 三元组合成虚拟的 `.weight` key。
:::

## Qwen3-30B 训练工作流

Relax 在 Qwen3-30B-A3B（8 卡 colocate）上提供两条参考配方：**FP8 原生训练** 与 **INT4 fake-QAT**。两者共用同一份 Megatron patch 和 colocate 部署，差异只在权重路径与启动脚本。

### 共同前置条件

1. 一个 BF16 HF checkpoint（例如 `Qwen3-30B-A3B`）。
2. 应用 Megatron patch `docker/patch/megatron/20260506-85bced0ae.patch`（项目 Dockerfile 已自动应用）—— 该 patch 同时提供 FP8 配套的 override 与 INT4 假量化的 `_FakeInt4QuantizationSTE`（override 了 `TEGroupedLinear._get_weight_tensors()`）。

FP8 配方额外需要一个支持 FP8 blockwise scaling 的 TransformerEngine 构建；INT4 配方额外需要编译 `fake_int4_quant_cuda` CUDA 扩展，见下文 [编译 int4_qat kernel](#build-int4-qat-kernel)。

### FP8 低精度训练

#### 步骤

1. **把 HF checkpoint 量化为 FP8：**

   ```bash
   python scripts/tools/convert_hf_to_fp8.py \
     --model-dir ${MODEL_DIR}/Qwen3-30B-A3B \
     --save-dir  ${MODEL_DIR}/Qwen3-30B-A3B-FP8 \
     --strategy  block --block-size 128 128
   ```

2. **配置启动脚本里的路径** (`scripts/training/text/run-qwen3-30B-A3B-fp8-8xgpu.sh`)：

   | 路径项               | 应指向                                                                                                                                                |
   | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
   | `--hf-checkpoint`    | 步骤 1 生成的 **FP8 HF 目录**（例如 `${MODEL_DIR}/Qwen3-30B-A3B-FP8`）。驱动 SGLang 初始化与 push 侧 `quantize_params_fp8` 的配置读取。                |
   | `--ref-load`         | **BF16 HF 目录**（例如 `${EXP_DIR}/Qwen3-30B-A3B`，未量化的原始 HF）。                                      |
   | `--load` / `--save`  | **BF16 Megatron checkpoint 目录**，用于 resume / save（与普通 BF16 训练一致；冷启动时可不填）。                                                          |

3. **启动训练：**

   ```bash
   bash scripts/training/text/run-qwen3-30B-A3B-fp8-8xgpu.sh
   ``` 

### INT4 低精度训练

#### 编译 int4_qat kernel {#build-int4-qat-kernel}

```bash
cd relax/backends/megatron/kernels/int4_qat
pip install -e . --no-build-isolation
```

编译产物 `fake_int4_quant_cuda.cpython-<py>-x86_64-linux-gnu.so` 落在同目录。Rollout 侧 `quantizer_compressed_tensors.py` 与 `convert_hf_to_int4.py` 都通过 `import fake_int4_quant_cuda` 引用该 kernel。

#### 步骤

1. **（可选）把 HF checkpoint 量化为 W4A16 — 必须用对称量化：**

   ```bash
   python scripts/tools/convert_hf_to_int4.py \
     --model-dir ${MODEL_DIR}/Qwen3-30B-A3B \
     --save-dir  ${MODEL_DIR}/Qwen3-30B-A3B-int4 \
     --group-size 128 \
     --is-symmetric
   ```

   `--is-symmetric` 是必须项，用来对齐训练侧 STE。如果已有 W4A16 发布版，请打开它的 `config.json` 确认 `config_groups.group_0.weights.symmetric == true`；如果是 `false`，请重新生成（或先用 `convert_moe_int4_to_bf16.py` 反量化回 BF16，再带 `--is-symmetric` 重新量化）。

2. **配置启动脚本里的路径** (`scripts/training/text/run-qwen3-30B-A3B-int4-8xgpu.sh`) —— 两个 HF 路径承担**不同**角色，不要指向同一个目录：

   | 路径项               | 应指向                                                                                                                                                                                                |
   | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | `--hf-checkpoint`    | **W4A16 INT4 HF 目录**（例如 `${EXP_DIR}/Qwen3-30B-A3B-int4`）。其 `config.json` 中的 `quantization_config.quant_method == "compressed-tensors"` 正是把每个 step 的 push 路由到 `quantize_params_compressed_tensors` 的关键。 |
   | `--ref-load`         | **BF16 HF 目录**（例如 `${EXP_DIR}/Qwen3-30B-A3B`，未量化的原始 HF）。STE 在 forward 上叠加 INT4 量化噪声，需要在真实的 BF16 权重底子上做 —— 不能用 W4A16。                                              |
   | `--load` / `--save`  | **BF16 Megatron checkpoint 目录**，用于 resume / save（冷启动时可不填，Megatron 会从 `--ref-load` 初始化）。                                                                                              |

3. **启动训练：**

   ```bash
   bash scripts/training/text/run-qwen3-30B-A3B-int4-8xgpu.sh
   ```

## Kimi-K2.6 256xGPU INT4 QAT（文本 & 多模态）

对于 Kimi-K2.6 这种超大 MoE 模型 —— HF 端可用的发布版本本身就已经是 W4A16 —— Relax 给出了一个略有差异的 INT4 fake-QAT 配方：**两个独立的 checkpoint**（一个 INT4 用于 SGLang 推理，一个 BF16 cast 用于 Megatron 训练），而不是单一的 W4A16 HF 同时驱动训练和推理两侧。训练侧仍然是 BF16 前向 + MoE expert 上的 STE INT4 假量化，但**推理侧直接原样加载 W4A16 发布版**（其 param dict 会注册 `weight_packed` / `weight_scale` / `weight_shape`），从而避免在 init 阶段对万亿参数重新量化一次。

提供了两个启动脚本，覆盖文本与多模态：

| 启动脚本                                                       | 数据集                              | 算法 | 奖励      |
| -------------------------------------------------------------- | ----------------------------------- | ---- | --------- |
| `scripts/training/text/run-kimi-k2.6-256xgpu-int4.sh`          | `dapo-math-17k`                     | GRPO | `math`    |
| `scripts/training/multimodal/run-kimi-k2.6-256xgpu-int4.sh`    | `multimodal-open-r1-8k-verified`    | GRPO | `openr1mm`|

### 前置条件

**一次性**把原始的 W4A16 发布版 cast 成 BF16 HF —— Megatron bridge 需要真实的 BF16 权重来加载，STE 只在 forward 路径上叠加量化噪声：

```bash
python -m relax.utils.quant_cast.convert_moe_int4_to_bf16 \
    --model-dir  ${MODEL_DIR}/Kimi-K2.6 \
    --output-dir ${MODEL_DIR}/Kimi-K2.6_bf16
```

### 双 checkpoint 布局

```bash
HF_INT4="${MODEL_DIR}/Kimi-K2.6/"        # 原始 compressed-tensors W4A16 发布版
HF_BF16="${MODEL_DIR}/Kimi-K2.6_bf16/"   # 由上面的前置步骤生成

CKPT_ARGS=(
   --hf-checkpoint        ${HF_INT4}   # AutoConfig → quant_method/group_size → push 自动走 compressed-tensors
   --sglang-hf-checkpoint ${HF_INT4}   # SGLang 原样加载 W4A16（param dict = weight_packed/scale/shape）
   --ref-load             ${HF_BF16}   # Megatron bridge 加载 BF16；STE 在每次 forward 上把权重 round 到 INT4 grid
   --megatron-to-hf-mode  bridge
)
```

三个 flag 分别承担不同的角色：

| 参数                      | Checkpoint   | 由谁读取                              | 作用                                                                                                                                       |
| ------------------------- | ------------ | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `--hf-checkpoint`         | INT4 (W4A16) | `AutoConfig`（push 侧 dispatcher）    | 让 `hf_config.quantization_config.quant_method == "compressed-tensors"`，从而每个 step 的 push 自动路由到 `quantize_params_compressed_tensors`。 |
| `--sglang-hf-checkpoint`  | INT4 (W4A16) | SGLang 引擎初始化                     | **必须是 INT4 目录**，不能是 BF16 cast —— 否则 SGLang 的 param dict 注册的就是 `.weight`（BF16），所有 push 都会被静默丢弃，报 `X.weight_packed not found in params_dict`。 |
| `--ref-load`              | BF16         | Megatron bridge loader                | 真实 BF16 working/master 权重；STE 在每次 forward 上叠加 INT4 量化噪声。                                                                    |

### 启动

```bash
# 纯文本
bash scripts/entrypoint/ray-job.sh scripts/training/text/run-kimi-k2.6-256xgpu-int4.sh
# 多模态
bash scripts/entrypoint/ray-job.sh scripts/training/multimodal/run-kimi-k2.6-256xgpu-int4.sh
```

两个脚本共享一致的并行配置和 INT4 链路：

| 配置项                                              | 取值                                                                                       | 说明                                                                                                  |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| 并行布局                                            | TP=8、PP=8、CP=4、EP=32、ETP=1                                                              | 共 256 GPU。INT4 QAT 只影响权重更新路径，并行布局保持不变。                                            |
| `OPEN_TRAINING_INT4_FAKE_QAT_FLAG`                  | `1`                                                                                        | 启用 `TEGroupedLinear._get_weight_tensors()` 中的 `_FakeInt4QuantizationSTE`。                        |
| `OPEN_TRAINING_INT4_GROUP_SIZE`                     | `32`                                                                                       | 与 W4A16 发布版的 per-group scale 布局保持一致（Kimi 使用 **32**，而不是 Qwen3-30B 配方里的 128）。     |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`    | `256`                                                                                      | DeepEP 低延迟 dispatch 缓冲；默认 128 会与 bs=128 时的 cuda_graph capture 冲突。                       |
| `--rollout-num-gpus-per-engine`                     | `16`                                                                                       | 每个 SGLang 引擎占 16 GPU → 256 GPU 总共 16 个引擎。                                                   |
| `--sglang-{dp-size,ep-size}`                        | 都是 `16`                                                                                  | 每个 16 GPU 的引擎内启用 DP-attention + EP。                                                           |
| `--sglang-mem-fraction-static`                      | `0.7`                                                                                      | 在此规模下为权重更新缓冲预留显存。                                                                     |
| Optimizer                                           | Adam + `--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer` | 1T 参数量下必须开启，用以放下 fp32 master 权重。                                              |
| Recompute                                           | `--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`         | 全量激活重计算 —— 在这个规模下是必须的。                                                                |

两个脚本仅在数据、奖励和少量算法超参上有差异：

- **文本** (`run-kimi-k2.6-256xgpu-int4.sh`)：`dapo-math-17k`，`--rm-type math`，`--rollout-max-response-len 16384`，`--global-batch-size 256`，`--lr 1e-6`，并附带一段 `EVAL_ARGS`（AIME-2024，`--eval-interval 20`）。
- **多模态** (`scripts/training/multimodal/` 下的 `run-kimi-k2.6-256xgpu-int4.sh`)：`multimodal-open-r1-8k-verified`，`--rm-type openr1mm`，`--multimodal-keys '{"image":"image"}'`，`--image-max-token-num 256`，`--rollout-max-prompt-len 2048` / `--rollout-max-response-len 4096`，`--global-batch-size 512`，`--lr 5e-6`。多模态脚本额外设置 `--vision-dp-when-tp` 与 `--decoder-first-pipeline-num-layers 1 --decoder-last-pipeline-num-layers 6`，以便把 vision tower 装进 PP-8 布局。

::: warning
不要为了"保持一致"把 `--sglang-hf-checkpoint` 换成 BF16 cast。SGLang 的参数注册只在 init 阶段做一次；如果注册的是 `.weight`（BF16），而 push 推送的是 `.weight_packed`（INT4），每次 push 都会被静默丢弃，训练会一直用过期的 rollout 权重。
:::

::: tip
这个配方假设 W4A16 发布版是用**对称量化**生成的（与训练侧 STE 对齐）。如果你从 BF16 出发用 `convert_hf_to_int4.py` 重新生成 W4A16，必须带上 `--is-symmetric` —— 详见上文 [离线量化工具](#convert-hf-to-int4) 的 warning。
:::
