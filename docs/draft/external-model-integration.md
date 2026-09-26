# 外部模型接入最佳实践（草稿）

本文当前保存第 1-7 章。

## 1. 接入目标与基本原则

外部模型接入 Relax 时，需要同时满足两条路径：

- Rollout 路径：SGLang 能加载该模型，并能正确处理文本或多模态输入。
- Training 路径：Megatron 能构建同构模型，并能通过 Megatron Bridge 完成 HF ↔ Megatron 权重转换。

这两条路径必须使用同一份 HuggingFace checkpoint 作为结构来源。Relax 中推荐把 HF checkpoint 作为架构、tokenizer、processor 和初始权重的共同入口，然后通过 `--megatron-to-hf-mode bridge` 让 Megatron Bridge 负责训练侧加载和训练后导出。

接入时优先遵循以下原则：

- 优先使用 Megatron Bridge，而不是手写 raw 权重转换器。
- SGLang 侧优先通过外部模型包注册模型和多模态 processor。
- 启动脚本中显式拆分模型结构参数、checkpoint 参数、rollout 参数和并行参数。
- 对齐时先做单样本前向一致性，再验证 packed / CP / rollout 训练路径。
- 不在训练脚本里隐藏模型特例；模型特例应沉淀到 `relax/models/<model_name>/` 下。

dots.mocr 的完整接入示例见第 6 章。

## 2. SGLang 侧如何接入外部模型

SGLang 侧的目标是让 Rollout 引擎能够加载外部模型，并把 Relax 的 rollout 请求转换成模型可执行的输入。对于标准 HuggingFace 架构，通常无需额外适配；对于 SGLang 未原生支持、或多模态 token 规则不同的模型，需要提供外部模型包。

在 Relax 中，外部模型包通过启动参数注册：

```bash
--sglang-external-model-package relax.models.dots_ocr.sglang
```

该参数会在 SGLang 进程启动前设置以下环境变量：

```bash
SGLANG_EXTERNAL_MODEL_PACKAGE=relax.models.dots_ocr.sglang
SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=relax.models.dots_ocr.sglang
SGLANG_EXTERNAL_MM_MODEL_ARCH=<auto resolved arch>
```

外部模型包通常需要包含两类对象：

- 模型实现：提供 SGLang 可加载的 `nn.Module`，并在模块中暴露 `EntryClass`。
- 多模态 processor：继承 SGLang 的 `BaseMultimodalProcessor`，负责把图片、视频等输入转成模型期望的 token 与 feature。

以 dots.mocr 为例，`relax.models.dots_ocr.sglang` 提供 `DotsOCRForCausalLM` 和 `DotsOCRImageProcessor`，并使用 dots 原生的 `<|img|><|imgpad|><|endofimg|>` token 规则。

接入外部 SGLang 模型时，建议按这个顺序实现：

1. 先确认模型能被 HF config 和 tokenizer 正常加载。
2. 编写 SGLang model class，保证 `load_weights()` 能处理 HF 权重名。
3. 如果是多模态模型，编写 processor，并确认占位 token 与 HF processor 完全一致。
4. 在 package 中暴露 `EntryClass`，让 Relax 自动解析模型架构名。
5. 在训练脚本中加入 `--sglang-external-model-package <package>`。
6. 单独启动或通过 Relax 启动 SGLang，先用一条文本样本和一条多模态样本验证 `/generate`。

SGLang 侧最容易出错的是 processor。多模态 token 规则不一致时，模型可能正常启动，但视觉 embedding 会插入到错误位置，导致输出异常或 logprob 对齐失败。因此 processor 应优先复刻模型原生 HF processor 或 SGLang 上游 processor。

## 3. Megatron 侧如何接入训练模型

Megatron 侧的目标是构建一个与 HF checkpoint 结构等价的训练模型，并支持从 HF 加载初始权重、训练后再导出给 SGLang。Relax 推荐优先使用 Megatron Bridge 接入新模型：

```bash
--megatron-to-hf-mode bridge
```

Bridge 模式下，Relax 会在训练侧通过 `AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)` 创建 bridge，再通过 `bridge.to_megatron_provider(load_weights=False)` 构建 Megatron provider。加载 HF checkpoint 时，Relax 会调用 `bridge.load_hf_weights(ddp_model)`；更新 SGLang 权重时，会通过 bridge mapping 把 Megatron 参数导出回 HF 命名空间。

一个完整的 Megatron Bridge 接入通常包含三部分：

1. Megatron model：定义训练时实际 forward 的模型结构。
2. Model provider：根据 HF config 和 CLI 参数创建 Megatron model。
3. Bridge：注册 HF 架构名到 Megatron model，并定义 HF ↔ Megatron 参数映射。

以 dots.mocr 为例：

- `relax/models/dots_ocr/megatron/model.py` 定义 `DotsOCRModel`。
- `relax/models/dots_ocr/megatron/provider.py` 定义 `DotsOCRModelProvider`。
- `relax/models/dots_ocr/megatron/bridge.py` 定义 `DotsOCRBridge`。
- `relax/models/__init__.py` 导入 `relax.models.dots_ocr.megatron`，触发 `@MegatronModelBridge.register_bridge(...)` 注册。

Bridge 注册关系如下：

```python
@MegatronModelBridge.register_bridge(
    source="DotsOCRForCausalLM",
    target=DotsOCRModel,
)
class DotsOCRBridge(MegatronModelBridge):
    ...
```

`source` 必须匹配 HF checkpoint 中的模型架构名。否则 `AutoBridge.from_hf_pretrained(...)` 找不到对应 bridge，训练侧模型无法构建或会错误落到其他 bridge。

Megatron provider 需要从 HF config 中提取训练所需结构参数，例如：

- `num_hidden_layers`
- `hidden_size`
- `intermediate_size`
- `num_attention_heads`
- `num_key_value_heads`
- `rms_norm_eps`
- `rope_theta`
- `vocab_size`
- 多模态模型的 `vision_config`
- 模型特有 token，例如 `image_token_id`

参数映射建议全部收敛到 bridge 的 `mapping_registry()` 中。dots.mocr 当前使用的映射包括：

- 词表 embedding：`language_model.embedding.word_embeddings.weight` ↔ `model.embed_tokens.weight`
- 输出层：`language_model.output_layer.weight` ↔ `lm_head.weight`
- final norm：`language_model.decoder.final_layernorm.weight` ↔ `model.norm.weight`
- attention qkv：Megatron fused `linear_qkv` ↔ HF `q_proj` / `k_proj` / `v_proj`
- MLP gate/up：Megatron fused `linear_fc1` ↔ HF `gate_proj` / `up_proj`
- vision tower：`vision_model.**` ↔ `vision_tower.**`

Megatron 侧接入后，建议先验证三个点：

1. `AutoBridge.from_hf_pretrained(...)` 能找到自定义 bridge。
2. `bridge.load_hf_weights(...)` 能完整加载 HF checkpoint，没有 missing 或 unexpected 权重。
3. `bridge.export_hf_weights(...)` 或 Relax 权重更新路径能导出 SGLang 可加载的 HF 参数名。

只有当 Megatron Bridge 无法覆盖模型，或模型需要绕过 Bridge 的权重转换机制时，才考虑 `--megatron-to-hf-mode raw`。raw 模式需要在 `relax/backends/megatron/weight_conversion/` 下手写转换器，并在 `relax/backends/megatron/weight_conversion/__init__.py` 中按模型名路由；这条路径维护成本更高，建议作为兜底方案。

## 4. 启动配置如何写

外部模型接入后，启动脚本建议拆成独立参数块。这样便于检查 SGLang、Megatron、数据、并行策略是否一致，也便于后续从 colocate 切换到 fully async 或 hybrid。

推荐结构如下：

```bash
source "${MODEL_CONFIG_DIR}/<model>.sh"

CKPT_ARGS=(...)
ROLLOUT_ARGS=(...)
PERF_ARGS=(...)
SGLANG_ARGS=(...)
MISC_ARGS=(...)

ray job submit ... -- python3 -m relax.entrypoints.train \
  --resource '...' \
  <execution mode flags> \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}"
```

### 4.1 模型结构参数

模型结构参数放在 `scripts/models/<model>.sh` 中。它描述 Megatron 训练模型的结构，应从 HF config 中提取，并与 Bridge provider 保持一致。

dots.mocr 示例位于：

```bash
scripts/models/dotsocr2.sh
```

其中包含：

```bash
MODEL_ARGS=(
   --swiglu
   --num-layers 28
   --hidden-size 1536
   --ffn-hidden-size 8960
   --num-attention-heads 12
   --group-query-attention
   --num-query-groups 2
   --use-rotary-position-embeddings
   --disable-bias-linear
   --add-qkv-bias
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 1000000
   --vocab-size 151936
   --kv-channels 128
   --untie-embeddings-and-output-weights
)
```

## 5. SGLang 与 Megatron 如何对齐

外部模型接入后，不建议直接启动训练。应先验证 HF、SGLang、Megatron 三条路径在同一输入上的行为一致。对齐的目标不是逐层完全相同，而是先确认输入格式、权重映射、位置编码、多模态 embedding 插入位置和 logprob 计算没有系统性偏差。

推荐按这个顺序排查：

1. HF 单卡前向：确认 checkpoint、tokenizer、processor 本身可用。
2. SGLang 单引擎前向：确认外部 model 和 processor 能正确加载并生成。
3. Megatron 单样本前向：确认 Bridge 能构建模型并加载权重。
4. HF / SGLang / Megatron logprob 对齐：同一 prompt、同一图片、同一 target tokens 下比较 token logprob。
5. Packed / CP 路径对齐：打开训练实际使用的 packed sequence、context parallel、multimodal batch 形态再比较。
6. 小步训练验证：跑 1-2 个 rollout / train step，确认权重更新后 SGLang 仍可生成。

对 dots.mocr，当前分支提供了两个调试入口：

```bash
# 单样本 / 单进程为主，用于快速比较 HF、SGLang、Megatron
scripts/debug/run-compare-dotsocr.sh

# packed 输入路径，分 stage 比较 HF + SGLang 与分布式 Megatron
scripts/debug/run-compare-dotsocr-packed.sh
```

这两个脚本默认使用：

```bash
${MODEL_DIR}/rednote-hilab/dots.mocr
```

也可以通过环境变量覆盖：

```bash
HF_CKPT=/path/to/rednote-hilab/dots.mocr \
DUMP_DIR=/tmp/relax_dotsocr_debug \
bash scripts/debug/run-compare-dotsocr.sh <image_path_or_url> "<prompt>"
```

对齐时重点看以下几类问题：

### 5.1 Processor token 规则

多模态模型最常见的问题是图片占位 token 不一致。dots.mocr 使用：

```text
<|img|><|imgpad|><|endofimg|>
```

如果误用了 Qwen-VL 风格 token：

```text
<|vision_start|><|image_pad|><|vision_end|>
```

SGLang 可能仍能启动，但图片 embedding 会插入到错误位置，logprob 会明显偏离。

### 5.2 RoPE 与 position ids

Megatron 侧必须使用与 HF 一致的 RoPE 配置。dots.mocr 的 Megatron provider 显式设置了 `position_embedding_type="rope"`，启动配置中也关闭了 RoPE fusion：

```bash
--no-rope-fusion
```

如果 RoPE 参数、position ids 或 packed sequence 的位置计算不一致，通常表现为文本样本也无法对齐。

### 5.3 权重命名与 fused 参数映射

SGLang 侧通常加载 HF 命名空间参数；Megatron 侧训练时使用 fused qkv、fused gate/up 等结构。Bridge 必须正确处理：

- `q_proj` / `k_proj` / `v_proj` ↔ `linear_qkv`
- `gate_proj` / `up_proj` ↔ `linear_fc1`
- `vision_tower.**` ↔ `vision_model.**`

如果文本路径对齐但图片路径不对齐，优先检查 vision tower 和图片 embedding 插入逻辑；如果所有 token 都不对齐，优先检查语言模型权重映射、RoPE 和 tokenizer。

### 5.4 Packed 与 CP 路径

训练脚本可能启用 packed sequence、dynamic batch、context parallel 等路径。dots.mocr 的 sync 脚本中包含：

```bash
--context-parallel-size 2
--vision-dp-when-cp
--use-dynamic-batch-size
--max-tokens-per-gpu 8192
```

因此单样本对齐通过后，还需要验证 packed / CP 路径。否则单卡前向正常，训练时仍可能因为 sequence 切分、attention mask、vision embedding gather 不一致而出问题。

### 5.5 权重更新后再次验证 SGLang

训练开始后，SGLang 不只加载初始 HF checkpoint，还会接收 Megatron 导出的新权重。因此对齐不应只验证初始加载，还应在至少一次 train step 后检查：

- SGLang 权重更新没有报错。
- SGLang 仍能处理文本和多模态请求。
- rollout 返回的 `tokens`、`rollout_log_probs`、`loss_mask` 长度一致。
- reward 和 loss 没有出现明显异常值。

建议每个新模型至少保留一个固定 checkpoint、输入样本、prompt、dump 目录和关键并行参数的 compare 脚本，便于后续回归。

## 6. 以 dots.mocr 为例

dots.mocr 的接入可以作为外部多模态模型的参考实现。它覆盖了 SGLang 外部模型包、Megatron Bridge、启动脚本和对齐脚本四个部分。

模型 checkpoint 使用 HuggingFace 上的：

```text
rednote-hilab/dots.mocr
```

链接：

```text
https://huggingface.co/rednote-hilab/dots.mocr
```

### 6.1 目录结构

当前分支中，dots.mocr 相关代码集中在：

```text
relax/models/dots_ocr/
├── configuration.py
├── vision.py
├── sglang/
│   ├── __init__.py
│   ├── model.py
│   └── processor.py
└── megatron/
    ├── __init__.py
    ├── bridge.py
    ├── model.py
    └── provider.py
```

其中：

- `configuration.py` 定义 `DotsOCRConfig` 和 `DotsVisionConfig`。
- `vision.py` 定义 dots vision tower。
- `sglang/model.py` 定义 SGLang 可加载的 `DotsOCRForCausalLM`。
- `sglang/processor.py` 定义 dots 专用的图片 processor。
- `megatron/model.py` 定义训练侧 `DotsOCRModel`。
- `megatron/provider.py` 定义 `DotsOCRModelProvider`。
- `megatron/bridge.py` 定义 `DotsOCRBridge` 和 HF ↔ Megatron 参数映射。

### 6.2 SGLang 接入点

dots.mocr 不是直接依赖 SGLang 内置模型路径，而是通过外部模型包注册：

```bash
--sglang-external-model-package relax.models.dots_ocr.sglang
```

该 package 暴露：

```python
EntryClass = [DotsOCRForCausalLM]
```

Relax 在启动 SGLang 进程前设置 `SGLANG_EXTERNAL_MODEL_PACKAGE` 和 `SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE`，SGLang 子进程会从这个 package 中发现模型和多模态 processor。

dots.mocr 的 processor 不能复用 Qwen-VL processor，因为图片 token 规则不同。dots 使用：

```text
<|img|><|imgpad|><|endofimg|>
```

而不是：

```text
<|vision_start|><|image_pad|><|vision_end|>
```

这是 dots.mocr 接入中最关键的 SGLang 侧差异。

### 6.3 Megatron 接入点

Megatron 侧通过 Bridge 注册：

```python
@MegatronModelBridge.register_bridge(
    source="DotsOCRForCausalLM",
    target=DotsOCRModel,
)
class DotsOCRBridge(MegatronModelBridge):
    ...
```

`relax/models/__init__.py` 会导入 `relax.models.dots_ocr.megatron`，从而触发 bridge 注册。训练启动后，Relax 的 Megatron backend 会调用 `AutoBridge.from_hf_pretrained(...)`，根据 HF checkpoint 的架构名找到 `DotsOCRBridge`。

`DotsOCRBridge` 负责两件事：

- 从 HF config 构造 `DotsOCRModelProvider`。
- 在 `mapping_registry()` 中声明参数映射。

dots.mocr 的语言模型部分是 Qwen2-like 结构，Megatron 侧使用 fused qkv 和 fused MLP，因此 bridge 中需要把 HF 的 `q_proj` / `k_proj` / `v_proj` 映射到 Megatron 的 `linear_qkv`，把 `gate_proj` / `up_proj` 映射到 `linear_fc1`。

vision tower 使用 replicated mapping：

```text
vision_model.** ↔ vision_tower.**
```

这表示 vision tower 参数按 HF 命名空间整体映射，不走语言模型的 fused qkv / MLP 映射规则。

### 6.4 启动脚本

dots.mocr 的模型结构参数位于：

```bash
scripts/models/dotsocr2.sh
```

训练脚本位于：

```bash
scripts/training/multimodal/run-dotsocr2-8xgpu.sh
```

核心 checkpoint 配置是：

```bash
--hf-checkpoint ${MODEL_DIR}/rednote-hilab/dots.mocr
--ref-load ${MODEL_DIR}/rednote-hilab/dots.mocr
--megatron-to-hf-mode bridge
```

核心多模态数据配置是：

```bash
--multimodal-keys '{"image":"image"}'
```

这要求数据集中存在 `image` 字段，并且 prompt 中的图片占位与 processor 规则一致。

### 6.5 对齐脚本

dots.mocr 已提供两类对齐脚本，具体使用建议见第 5 章：

```bash
scripts/debug/run-compare-dotsocr.sh
scripts/debug/run-compare-dotsocr-packed.sh
```

### 6.6 README 中的支持项

正式发布文档时，README 支持模型表建议增加 dots.mocr：

```markdown
| **[dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr)** | 3B | Vision + Language | OCR, document understanding, multimodal reasoning | Megatron |
```

中文 README 建议增加：

```markdown
| **[dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr)** | 3B | 视觉 + 语言 | OCR、文档理解、多模态推理 | Megatron |
```

模型名链接到：

```markdown
[rednote-hilab/dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr)
```

## 7. 常见问题与检查清单

外部模型接入失败通常来自 SGLang、Megatron、processor、checkpoint 或启动参数之间的不一致。优先按下面几类问题排查。

### 7.1 常见问题

**SGLang 能启动，但多模态输出异常**

优先检查 processor：图片占位 token、image token id、image grid、pixel values 形状必须与模型原生 processor 一致。dots.mocr 必须使用 `<|img|><|imgpad|><|endofimg|>`。

**Megatron 找不到 bridge**

检查 HF checkpoint 的 architecture 名称是否与 bridge 注册的 `source` 一致，并确认 `relax.models.__init__` 会导入对应 megatron package 以触发注册。

**HF 能加载，但 Megatron forward 不对齐**

先查语言模型配置、RoPE、position ids、attention mask、packed sequence 和 CP 切分。文本样本都不对齐时，先不要看 vision tower。

**文本对齐，但图片样本不对齐**

重点检查 SGLang processor、Megatron vision embedding 插入位置、`image_token_id`、vision tower 权重映射。

**训练一步后 SGLang 权重加载失败**

检查 Megatron → HF 导出路径。Bridge 的 `mapping_registry()` 要覆盖训练模型参数，SGLang `load_weights()` 也要能消费导出的 HF 参数名。

### 7.2 接入检查清单

SGLang 侧：

- [ ] 外部模型 package 可以 import，并暴露 `EntryClass`。
- [ ] `--sglang-external-model-package` 指向正确 package。
- [ ] 多模态 processor 使用模型原生 token 规则。
- [ ] `load_weights()` 能处理 HF checkpoint 中的参数名。
- [ ] 文本和多模态样本都能通过 `/generate`。

Megatron 侧：

- [ ] `AutoBridge.from_hf_pretrained(...)` 能找到自定义 bridge。
- [ ] bridge 的 `source` 与 HF architecture 名称一致。
- [ ] provider 从 HF config 读取了必要结构参数。
- [ ] `bridge.load_hf_weights(...)` 能完整加载 checkpoint。
- [ ] `mapping_registry()` 覆盖语言模型、vision tower 和输出层参数。

启动和验证：

- [ ] `scripts/models/<model>.sh` 中的结构参数与 HF config 一致。
- [ ] `--hf-checkpoint` 指向用于结构、tokenizer、processor 的 HF checkpoint。
- [ ] 使用 Bridge 模式时传入 `--megatron-to-hf-mode bridge`。
- [ ] 多模态数据通过 `--multimodal-keys` 显式声明。
- [ ] 执行模式只选择一种：`--colocate`、`--fully-async` 或 `--hybrid`。
- [ ] HF、SGLang、Megatron 固定样本 logprob 已对齐。
- [ ] packed / CP / dynamic batch 路径已单独验证。
- [ ] 至少完成一次 rollout、train、update weights 闭环。
