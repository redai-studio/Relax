# Weight versions and FULL checkpoints

A training update can be exported as a self-contained weight package. Tensors use Hugging Face logical names, dtypes and shapes. A consumer reconstructs a safetensors checkpoint from shared storage and loads it with SGLang's existing Loader.

## Export from training

As the final distributed operation in the active training step, after an optimizer step and immediately before any sleep/offload, call `export_weight_checkpoint` on every `MegatronTrainRayActor` in the training group. The method does not wake an already-offloaded model. Keep training weights stable until the synchronous call returns. Every rank receives the same receipt.

LoRA adapter mode is not supported by this FULL export API. All training ranks reject the call before conversion or publication, even if only one rank requests that mode. Use the existing adapter synchronization path for adapter-mode training; this API does not export a separate adapter artifact. The first-PR acceptance covers non-LoRA models.

```python
receipt = actor.export_weight_checkpoint(
    store_path="/shared/weights",
    stream="training-run",
    version=0,
    update_id="optimizer-step-0",
    scratch="/local/weight-export",
    disk_budget=200 * 1024**3,
    max_weight_bytes=32 * 1024**3,
    metrics_dir="/local/weight-metrics",
)
```

The capacities above are examples; configure them for the model and retained versions. `version` identifies the weights within a stream, while `update_id` identifies the caller's training update. The receipt contains the logical content digest `root` and the package identifier `artifact_id`.

Export reuses Relax's Direct or Bridge converter. Training ranks gather source parameters and convert them into logical tensors; rank zero persists and commits the FULL package. Ranks coordinate file progress and receive file errors together, allowing the same target to be retried. Consumers can read committed packages after the training processes exit.

## Reconstruct and load

With the shared directory mounted on the consumer:

```bash
python -m relax.distributed.weight_sync.checkpoint   --store /shared/weights --stream training-run --version 0   --output /local/checkpoint-0   --disk-budget 214748364800 --max-weight-bytes 34359738368

python -m sglang.launch_server   --model-path /local/checkpoint-0   --tokenizer-path /models/original-model
```

Reconstruction produces `model.safetensors`, `config.json` and `weight-version.json`. Weights and model configuration are versioned together; tokenizer and processor assets come from the original model directory. SGLang loads the complete checkpoint using the consumer's parallel configuration.

## Format and measurements

The manifest binds the tensor directory, block content and model configuration. Storage provides FULL/delta block representations, immutable versions, retention references and complete reconstruction. The training-side `export_weight_checkpoint` method always publishes a FULL package.

Per-rank measurement files record conversion time, application-level sent bytes, file writes and process memory. Collective payloads use sender accounting; file writes are separate. Interface and filesystem observations retain their individual scopes.

## Validation

CPU tests cover bitwise reconstruction, version identity, checkpoint format, concurrent capacity accounting, parallel file operations, and two training participants returning and retrying together after a file-write failure.

```bash
python -m pytest -q tests/cpu/distributed/weight_sync

torchrun --standalone --nproc-per-node=1   tests/distributed/weight_sync/run_full_export.py --root /tmp/full-direct
torchrun --standalone --nproc-per-node=1   tests/distributed/weight_sync/run_full_export.py --root /tmp/full-bridge --bridge
```

The GPU drivers use a small Qwen3 model, exporting initialization and five real Megatron-Core optimizer updates. Transformers loads every checkpoint and compares forward outputs with the source model.

`tests/distributed/weight_sync/run_full_checkpoint_loader.py` reconstructs a selected version in an independent process, starts the SGLang Loader and runs generation. `--probe` accepts a text or image/text request.
