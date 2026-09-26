# 外部模型接入

本文说明如何把外部模型接入 Relax 的 SGLang rollout 和 Megatron 训练路径。当前 dots.mocr 支持可作为参考实现。

## 概述

外部模型需要同时打通两条路径：

- **Rollout**：SGLang 能加载模型，并处理文本或多模态请求。
- **Training**：Megatron 能构建同构模型，并通过 Megatron Bridge 加载和导出权重。

建议使用同一份 HuggingFace checkpoint 作为结构、tokenizer、processor 和初始权重来源，并优先使用 Bridge 模式：

```bash
--megatron-to-hf-mode bridge
```

模型特例放在 `relax/models/<model_name>/` 下；启动脚本只保留配置。

## SGLang 接入

对 SGLang 未原生支持的模型，通过外部模型包注册：

```bash
--sglang-external-model-package relax.models.dots_ocr.sglang
```

Relax 会在启动 SGLang 子进程前设置外部模型和多模态 processor 环境变量。这个 package 通常需要提供：

- 通过 `EntryClass` 暴露的 SGLang 模型类。
- 多模态模型所需的 processor。
- 能消费 HF 参数名的 `load_weights()` 路径。

dots.mocr 中，`relax.models.dots_ocr.sglang` 提供 `DotsOCRForCausalLM` 和 `DotsOCRImageProcessor`。它的图片 token 规则是：

```text
<|img|><|imgpad|><|endofimg|>
```

不要复用看起来相似的 VLM processor，除非特殊 token 和 feature 布局完全一致。

## Megatron 接入

Megatron 侧通常包含：

- 训练模型，例如 `relax/models/dots_ocr/megatron/model.py`。
- provider，例如 `relax/models/dots_ocr/megatron/provider.py`。
- Bridge 适配，例如 `relax/models/dots_ocr/megatron/bridge.py`。

Bridge 注册 HF 架构名：

```python
@MegatronModelBridge.register_bridge(
    source="DotsOCRForCausalLM",
    target=DotsOCRModel,
)
class DotsOCRBridge(MegatronModelBridge):
    ...
```

`source` 必须匹配 HF checkpoint 的 architecture。还需要在 `relax/models/__init__.py` 导入对应 Megatron package，确保 `AutoBridge.from_hf_pretrained(...)` 前完成注册。

在 `mapping_registry()` 中声明 HF 与 Megatron 的参数映射。常见映射包括：

- `model.embed_tokens.weight` 到 `language_model.embedding.word_embeddings.weight`
- `lm_head.weight` 到 `language_model.output_layer.weight`
- HF `q_proj` / `k_proj` / `v_proj` 到 Megatron `linear_qkv`
- HF `gate_proj` / `up_proj` 到 Megatron `linear_fc1`
- 多模态 tower，例如 `vision_tower.**` 到 `vision_model.**`

只有 Bridge 无法覆盖模型时才使用 raw 转换。raw 模式需要在 `relax/backends/megatron/weight_conversion/` 下维护自定义转换器。

## 启动配置

启动脚本建议拆成清晰参数块：

```bash
source "${MODEL_CONFIG_DIR}/<model>.sh"

CKPT_ARGS=(...)
ROLLOUT_ARGS=(...)
PERF_ARGS=(...)
SGLANG_ARGS=(...)
MISC_ARGS=(...)
```

关键 checkpoint 参数：

```bash
--hf-checkpoint ${MODEL_DIR}/rednote-hilab/dots.mocr
--ref-load ${MODEL_DIR}/rednote-hilab/dots.mocr
--megatron-to-hf-mode bridge
```

多模态 rollout 参数：

```bash
--multimodal-keys '{"image":"image"}'
```

外部 SGLang 模型参数：

```bash
--sglang-external-model-package relax.models.dots_ocr.sglang
```

执行模式只选择一种：

- `--colocate`
- `--fully-async`
- `--hybrid`

全异步结合 `--use-dynamic-batch-size` 时，动态 batch 路径会自动在 DP rank 间做 token 均衡。`--balance-data` 仍适用于静态/序列长度均衡路径；在动态 batch 路径上传入该 flag 是允许的，但不会额外改变行为。

## 对齐验证

不要直接启动完整训练。建议按顺序验证：

1. HF 单样本 forward 可运行。
2. SGLang 单引擎文本和多模态生成可运行。
3. Megatron 能通过 Bridge 构建模型并加载 HF 权重。
4. 固定样本下 HF / SGLang / Megatron token logprob 对齐。
5. packed sequence、context parallel、dynamic batch 路径已验证。
6. 一个小规模 rollout → train → update weights 闭环可运行。

dots.mocr 可使用：

```bash
scripts/debug/run-compare-dotsocr.sh
scripts/debug/run-compare-dotsocr-packed.sh
```

常见对齐问题：

- 多模态特殊 token 或 processor 行为不一致。
- RoPE / position id 不一致。
- Bridge 缺少 fused qkv、fused MLP 或 vision tower 映射。
- SGLang `load_weights()` 无法消费导出的 HF 参数名。

## dots.mocr 参考

模型 checkpoint：

[rednote-hilab/dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr)

相关文件：

```text
relax/models/dots_ocr/
├── configuration.py
├── vision.py
├── sglang/
│   ├── model.py
│   └── processor.py
└── megatron/
    ├── bridge.py
    ├── model.py
    └── provider.py
```

启动文件：

```bash
scripts/models/dotsocr2.sh
scripts/training/multimodal/run-dotsocr2-8xgpu.sh
scripts/training/multimodal/run-dotsocr2-8xgpu-hybrid.sh
```

## 检查清单

完成接入前确认：

- [ ] SGLang 外部 package 可 import，并暴露 `EntryClass`。
- [ ] 多模态 processor 使用模型原生 token 规则。
- [ ] `AutoBridge.from_hf_pretrained(...)` 能找到自定义 bridge。
- [ ] `bridge.load_hf_weights(...)` 能加载 checkpoint。
- [ ] `mapping_registry()` 覆盖语言模型、输出层和多模态 tower 权重。
- [ ] 启动脚本使用 `--megatron-to-hf-mode bridge`。
- [ ] `--multimodal-keys` 与数据集字段一致。
- [ ] 固定样本下 HF / SGLang / Megatron logprob 已对齐。
- [ ] 小规模 rollout → train → update weights 闭环可运行。

## 下一步

- [自定义训练](./customize-training.md)
- [性能调优](./performance-tuning.md)
- [调试指南](./debugging.md)
