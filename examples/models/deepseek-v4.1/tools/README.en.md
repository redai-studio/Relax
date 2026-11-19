# DeepSeek V4.1 checkpoint preparation

[中文](./README.md)

[convert_to_bf16.py](./convert_to_bf16.py) converts the official DeepSeek-V4.1-Flash HF checkpoint, which contains mixed FP8/FP4 weights, into the BF16 HF format used by Relax for text-only training and rollout. SFT, RL, BSHD and THD use the same format; switching to THD does not require another conversion.

## Source and conversion scope

The previously prepared `DeepSeek-V4.1-bf16` came from the official `deepseek-ai/DeepSeek-V4.1-Flash` model. Its local download record identifies revision `8b3cec70c307624fb098c3a04bdbfc118e7abda9`. The archived `convert_deepseek_v4_1_to_bf16-6f52e0b.py` is identical to the script in [Miles commit `6f52e0b`](https://github.com/radixark/miles/blob/6f52e0b3021484e8d6ad0d43c1074100adc8e4b9/tools/convert_deepseek_v4_1_to_bf16.py); the accompanying `convert-v41-bf16.log` records a completed conversion.

The `dsv41_cut_cast.py` mentioned in the [Miles documentation](https://github.com/radixark/miles/blob/fa13e2ec42622f6e77aabd8e0c295ab8b4347eac/docs/models/deepseek/deepseek-v4-1-flash.md#L44-L53) is not a runnable file in the current main branch. The historical commit above provides the model-specific converter. This tool retains its conversion semantics while removing the Miles package dependency, processing one tensor at a time, streaming unchanged data, and adding preflight checks and explicit resume support.

| Source content | Output |
| --- | --- |
| Dense FP8 weights with 32×32 E8M0 scales | Dequantize to BF16 and remove the corresponding ordinary scales |
| Routed experts stored as packed FP4 with group-32 E8M0 scales | Dequantize to BF16 and remove the corresponding ordinary scales |
| Existing BF16 weights and FP32 auxiliary weights | Copy unchanged, preserving their dtypes |
| Frozen Engram `embed.weight` / `embed.scale` | Preserve the original FP8/E8M0 bytes; do not convert to BF16 |
| `mtp.*`, `vision.*`, `aligner.*`, `image_*`, `.ffn.gate.bias_vl` | Drop |
| Configuration | Flatten `text_config`, remove quantization and vision configuration and `dspark_*` fields, and set `num_nextn_predict_layers=0` |

The tool preserves all text layers and their original indices. It does not select or remap layers. It also removes the vision head, so its output is unsuitable for multimodal training with a frozen vision encoder. A four-layer model that is already in BF16 does not need this conversion.

## Prepare the checkpoint

Run these commands from the repository root in a Relax CU13 environment that matches the repository. Conversion uses PyTorch and Relax's existing logging module; Miles is not required. It runs on CPU by default and does not require Ray.

Start with a complete source model downloaded at a fixed revision. The revision below is the base release used previously, not `Flash-1017`:

```bash
export MODEL_DIR=/path/to/models
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --revision 8b3cec70c307624fb098c3a04bdbfc118e7abda9 \
  --local-dir "$MODEL_DIR/DeepSeek-V4.1-Flash"
```

Skip the download if you already have the complete source directory. Keep the source and output directories separate. Use a new output name for a new conversion to avoid interfering with a BF16 directory used by training.

First, inspect the configuration, index and all shard headers to check quantization formats, weight/scale pairs, required text weights and Engram tensors, and estimate the output size:

```bash
python examples/models/deepseek-v4.1/tools/convert_to_bf16.py \
  --model-dir "$MODEL_DIR/DeepSeek-V4.1-Flash" \
  --save-dir "$MODEL_DIR/DeepSeek-V4.1-bf16-new" \
  --check-only
```

`--check-only` does not read tensor payloads or create the output directory. For the source revision above, the expected output is 46,968 tensors in 44 shards, with 1,306,598,355,408 bytes of weight data—approximately 1.31 TB / 1.19 TiB. Reserve this space, plus a small allowance for metadata, in addition to the space occupied by the source model.

After checking the plan, run the conversion. Keep the log outside the output directory:

```bash
set -o pipefail
python examples/models/deepseek-v4.1/tools/convert_to_bf16.py \
  --model-dir "$MODEL_DIR/DeepSeek-V4.1-Flash" \
  --save-dir "$MODEL_DIR/DeepSeek-V4.1-bf16-new" \
  --device cpu \
  2>&1 | tee convert-deepseek-v4.1-bf16.log
```

The converter processes one tensor at a time. Data that does not require dequantization, including Engram tables, is copied in 64 MiB blocks. CPU memory must still accommodate the tensor being dequantized and its expanded temporary data. To convert on a GPU, use `--device cuda` and select an available GPU with `CUDA_VISIBLE_DEVICES`. A multi-GPU training setup is not required.

To resume an interrupted conversion, keep the source snapshot, script version and output directory unchanged, then add `--resume` to the same command. Resume checks the source path, configuration/index hashes and script hash recorded in `conversion.json`, along with the headers and sizes of completed shards. It skips completed shards and rebuilds incomplete `.tmp` shards. Do not run concurrent writers against the same output directory. These checks do not checksum all tensor payloads; keep the source snapshot read-only.

The final index is published only after every shard is complete, and `conversion.json` is then marked with `complete: true`. The output also includes the flattened `config.json`, tokenizer, chat template, generation configuration and original model license when those assets are present in the source directory. After conversion, point the training script to the output:

```bash
export HF_CKPT="$MODEL_DIR/DeepSeek-V4.1-bf16-new"
```

Training loads the HF weights directly; you do not need to create a Megatron `torch_dist` checkpoint first. Retain this BF16 base: resuming training and any future standalone deployment export need its frozen Engram data.

## Exporting trained checkpoints to HF

This tool only handles **original quantized HF → BF16 HF** conversion. Megatron `torch_dist` checkpoints saved during training can be used to resume training. Exporting a standalone HF model for inference is a separate operation.

As of Miles commit `fa13e2e`, Miles provides a generic [DCP → HF exporter](https://github.com/radixark/miles/blob/fa13e2ec42622f6e77aabd8e0c295ab8b4347eac/tools/convert_torch_dist_to_hf.py) and [V4.1 parameter mappings](https://github.com/radixark/miles/blob/fa13e2ec42622f6e77aabd8e0c295ab8b4347eac/miles/backends/megatron_utils/megatron_to_hf/deepseekv4_1.py), but no complete V4.1 deployment export recipe. The mapping skips frozen Engram tables and scales, and the exporter's asset-copying step skips the original safetensors files. Running that generic exporter alone therefore does not produce a complete standalone model with its frozen tables.

Relax has not yet implemented this complete export path either. Its [V4.1 Bridge](../../../../relax/models/deepseekv41/bridge.py) explicitly sets `SUPPORTS_HF_PRETRAINED_EXPORT=False`, and the existing CPU exporter has no V4.1 model mapping. Generic export commands for other models cannot be used directly.

A complete export must map trained checkpoint parameters back to V4.1 HF names, copy the four frozen Engram tensors unchanged from **the BF16 base used for that training run**, preserve required FP32 parameters, rebuild the index/configuration/tokenizer assets, and verify numerical behavior after reloading. Successful online weight synchronization to SGLang does not establish that these offline export steps are implemented.

A BF16 HF export also differs from the original FP8/FP4 deployment package. Returning to low-precision deployment weights requires a separate requantization step and inference accuracy checks; adding the original `quantization_config` back is insufficient.
