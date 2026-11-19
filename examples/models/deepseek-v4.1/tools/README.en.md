# DeepSeek V4.1 checkpoint preparation

[中文](./README.md)

[convert_to_bf16.py](./convert_to_bf16.py) converts the official DeepSeek-V4.1-Flash HF checkpoint, which contains mixed FP8/FP4 weights, into a BF16 HF base. Use `--keep-vision` to prepare a composite model for frozen-vision SFT on Flash-1017. The default produces the existing text-only training and rollout format.

## Source and conversion scope

The previously prepared `DeepSeek-V4.1-bf16` came from the official `deepseek-ai/DeepSeek-V4.1-Flash` model at revision `8b3cec70c307624fb098c3a04bdbfc118e7abda9`.

| Source content | Default text-only output |
| --- | --- |
| Dense FP8 weights with 32×32 E8M0 scales | Dequantize to BF16 and remove the corresponding ordinary scales |
| Routed experts stored as packed FP4 with group-32 E8M0 scales | Dequantize to BF16 and remove the corresponding ordinary scales |
| Existing BF16 weights and FP32 auxiliary weights | Copy unchanged, preserving their dtypes |
| Frozen Engram `embed.weight` / `embed.scale` | Preserve the original FP8/E8M0 bytes; do not convert to BF16 |
| `mtp.*`, `vision.*`, `aligner.*`, `image_*`, `.ffn.gate.bias_vl` | Drop |
| Configuration | Flatten `text_config`, remove quantization and vision configuration and `dspark_*` fields, and set `num_nextn_predict_layers=0` |

## Flash-1017 with frozen vision

Use the complete `deepseek-ai/DeepSeek-V4.1-Flash-1017` snapshot. The downloaded revision is `08c8657f02291e11b6f6b63e0e16014f8e748518`. It adds `chat_template.jinja` to revision `1bae0e1088993b9b82de08f59de5be5a34e42170` with identical weights and configuration. Run from the Relax repository root in the training environment. Set `SOURCE_1017` to the downloaded snapshot and `BASE_1017` to a new directory on storage accessible to every rank.

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SOURCE_1017=/path/to/DeepSeek-V4.1-Flash-1017
export BASE_1017=/path/to/DeepSeek-V4.1-Flash-1017-bf16
convert=(python examples/models/deepseek-v4.1/tools/convert_to_bf16.py
  --model-dir "$SOURCE_1017" --save-dir "$BASE_1017" --keep-vision)
"${convert[@]}" --check-only
"${convert[@]}"
```

The multimodal base retains the native `vision.*`, `aligner.*`, `image_start`, `image_end`, `image_newline`, and each retained decoder layer’s `ffn.gate.bias_vl` with the original bytes and dtypes. It includes the composite config, tokenizer, official prompt encoder, image processor and vision reference module. Draft MTP/DSpark weights are excluded. The converter records its mode, source shard sizes/timestamps, available HF download revisions and etags, and asset digests in `conversion.json`. To continue an interrupted conversion, rerun the same command with `--resume`.

For a smaller diagnostic model, add `--decoder-layers 4` to the command array **before both commands** and choose a separate output directory. It keeps native layers 0–3, all 384 experts per layer, the entire vision tower, and the full original Engram table and scale at layer 1. KV and index source 2 retain their original numbering. Candidate filtering is disabled because its source is outside the prefix. Check disk capacity with `--check-only` on the actual source revision.

## Prepare the default text-only checkpoint

Run these commands from the repository root in the matching Relax CU13 environment. The converter uses PyTorch and the Relax logging module and runs independently on CPU by default.

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

Use the CPU exporter from the repository root. `DCP_ITER` selects the saved training iteration and `BASE_1017` must point to the exact read-only BF16 directory used by that run. An iteration records its base identity in `v41-frozen-base.json`. The exporter and checkpoint resume compare asset digests and shard names, sizes and modification times against that record. Preserve file timestamps when relocating the base. Allow enough space for a complete HF export in a fresh directory.

```bash
export DCP_ITER=/path/to/checkpoints/iter_0000100
export HF_EXPORT=/path/to/new-hf-export
convert=(python scripts/tools/convert_checkpoint_to_hf_cpu.py
  --input-dir "$DCP_ITER" --origin-hf-dir "$BASE_1017" --output-dir "$HF_EXPORT"
  --allow-base-fallback vision --allow-base-fallback engram)
"${convert[@]}" --dry-run
"${convert[@]}"
```

The exporter reconstructs language weights from the global DCP shards, including routed/shared experts, mHC and trainable Engram projections. The explicit fallback groups permit byte copies of frozen vision assets, image router biases and frozen Engram tables only. The resulting directory includes its own index, processor, tokenizer, original composite configuration and trained weights in their saved floating-point dtypes. `conversion-result.json` lists the tensors copied from the base and any DCP/base dtype differences. For a legacy flat text-only model, use `--allow-base-fallback engram` alone.

To check an export, compare its mean negative log-likelihood in a fresh SGLang process with Megatron at the same training step. Use identical images, histories and supervised assistant token positions. To measure the effect of switching the full model from native quantization to BF16, evaluate the original 40-layer 1017 and the untrained 40-layer BF16 base on the same protocol. See the [eight-GPU validation results](../../../../docs/draft/deepseek-v41-1017-frozen-vision-thd-plan.md) for the four-layer measurements.
