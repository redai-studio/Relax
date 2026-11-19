# External Model Integration

This guide covers the minimum path for adding an external model to Relax with SGLang rollout and Megatron training. Use the current dots.mocr support as the reference implementation.

## Overview

An external model must work through two compatible paths:

- **Rollout**: SGLang can load the model and process text or multimodal requests.
- **Training**: Megatron can build the same architecture and load / export weights through Megatron Bridge.

Use one HuggingFace checkpoint as the source of architecture, tokenizer, processor, and initial weights. Prefer Bridge mode:

```bash
--megatron-to-hf-mode bridge
```

Keep model-specific logic under `relax/models/<model_name>/`; keep launch scripts as configuration only.

## SGLang Integration

For models not natively supported by SGLang, register an external model package:

```bash
--sglang-external-model-package relax.models.dots_ocr.sglang
```

Relax sets the SGLang external model and multimodal processor environment variables before spawning SGLang. The package should provide:

- A SGLang-loadable model class exposed through `EntryClass`.
- A multimodal processor when the model has image, video, or audio inputs.
- A `load_weights()` path that accepts HF-format parameter names.

For dots.mocr, `relax.models.dots_ocr.sglang` provides `DotsOCRForCausalLM` and `DotsOCRImageProcessor`. Its image token rule is:

```text
<|img|><|imgpad|><|endofimg|>
```

Do not reuse a similar VLM processor unless the special tokens and feature layout match exactly.

## Megatron Integration

Megatron integration normally consists of:

- A training model, such as `relax/models/dots_ocr/megatron/model.py`.
- A provider, such as `relax/models/dots_ocr/megatron/provider.py`.
- A Bridge adapter, such as `relax/models/dots_ocr/megatron/bridge.py`.

The bridge registers the HF architecture name:

```python
@MegatronModelBridge.register_bridge(
    source="DotsOCRForCausalLM",
    target=DotsOCRModel,
)
class DotsOCRBridge(MegatronModelBridge):
    ...
```

The `source` value must match the HF checkpoint architecture. Import the Megatron package from `relax/models/__init__.py` so the registration runs before `AutoBridge.from_hf_pretrained(...)`.

In `mapping_registry()`, map HF names to Megatron names. Common mappings include:

- `model.embed_tokens.weight` to `language_model.embedding.word_embeddings.weight`
- `lm_head.weight` to `language_model.output_layer.weight`
- HF `q_proj` / `k_proj` / `v_proj` to Megatron `linear_qkv`
- HF `gate_proj` / `up_proj` to Megatron `linear_fc1`
- Multimodal towers, such as `vision_tower.**` to `vision_model.**`

Use raw conversion only when Bridge cannot cover the model. Raw mode requires custom converters under `relax/backends/megatron/weight_conversion/`.

## Launch Configuration

Split launch scripts into clear argument blocks:

```bash
source "${MODEL_CONFIG_DIR}/<model>.sh"

CKPT_ARGS=(...)
ROLLOUT_ARGS=(...)
PERF_ARGS=(...)
SGLANG_ARGS=(...)
MISC_ARGS=(...)
```

Key checkpoint options:

```bash
--hf-checkpoint ${MODEL_DIR}/rednote-hilab/dots.mocr
--ref-load ${MODEL_DIR}/rednote-hilab/dots.mocr
--megatron-to-hf-mode bridge
```

Key rollout options for multimodal data:

```bash
--multimodal-keys '{"image":"image"}'
```

Key SGLang option for external models:

```bash
--sglang-external-model-package relax.models.dots_ocr.sglang
```

Choose exactly one execution mode:

- `--colocate`
- `--fully-async`
- `--hybrid`

In fully async with `--use-dynamic-batch-size`, the dynamic batch path automatically balances tokens across DP ranks. `--balance-data` remains useful on the static/seqlen-balanced path and is accepted without extra effect on dynamic batching.

## Alignment

Do not start full training before alignment. Check in this order:

1. HF single-sample forward works.
2. SGLang single-engine text and multimodal generation works.
3. Megatron can build the model and load HF weights through Bridge.
4. HF / SGLang / Megatron token logprobs match on a fixed sample.
5. Packed sequence, context parallel, and dynamic batch paths are verified.
6. One small rollout → train → update weights loop succeeds.

For dots.mocr, use:

```bash
scripts/debug/run-compare-dotsocr.sh
scripts/debug/run-compare-dotsocr-packed.sh
```

Common alignment failures:

- Wrong multimodal special tokens or processor behavior.
- RoPE / position id mismatch.
- Missing Bridge mappings for fused qkv, fused MLP, or vision tower weights.
- SGLang `load_weights()` not accepting the exported HF names.

## dots.mocr Reference

Model checkpoint:

[rednote-hilab/dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr)

Relevant files:

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

Launch files:

```bash
scripts/models/dotsocr2.sh
scripts/training/multimodal/run-dotsocr2-8xgpu.sh
scripts/training/multimodal/run-dotsocr2-8xgpu-hybrid.sh
```

## Checklist

Before treating a model as integrated:

- [ ] The SGLang external package imports and exposes `EntryClass`.
- [ ] The multimodal processor uses the model's native token rules.
- [ ] `AutoBridge.from_hf_pretrained(...)` finds the custom bridge.
- [ ] `bridge.load_hf_weights(...)` loads the checkpoint.
- [ ] `mapping_registry()` covers language, output, and multimodal tower weights.
- [ ] The launch script uses `--megatron-to-hf-mode bridge`.
- [ ] `--multimodal-keys` matches the dataset fields.
- [ ] Fixed-sample HF / SGLang / Megatron logprobs are aligned.
- [ ] A small rollout → train → update weights loop succeeds.

## Next Steps

- [Customize Training](./customize-training.md)
- [Performance Tuning](./performance-tuning.md)
- [Debugging Guide](./debugging.md)
