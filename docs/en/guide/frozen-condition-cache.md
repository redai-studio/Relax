# Frozen Condition Cache

Image-edit SFT often sees the same instruction and source image across epochs. When the condition encoder is frozen, Relax can reuse its output while continuing to train the generation model. This avoids repeated condition encoding without caching training noise or model predictions.

The cache is opt-in and shared by two model adapters:

| Adapter | Cached features | Training path |
|---|---|---|
| `relax.models.sensenova.adapter.SenseNovaSFTAdapter` | Frozen prompt/image prefix KV tensors and indexing metadata | Generation-only LoRA; full-parameter training keeps the online path |
| `relax.models.qwen_image.sft_adapter.QwenImageEditSFTAdapter` | Frozen text/vision embeddings and attention mask | LoRA with one conditioning image |

The Qwen adapter uses the conditioning and latent geometry from the [Diffusers `QwenImageEditPipeline`](https://github.com/huggingface/diffusers/blob/d791c5c024014784df4a8dac2601e19fb4d300fc/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py). It trains with a uniform-sigma flow-matching loss.

## Enable reuse

Set the following variables in each training worker's environment before loading the model. The budget is **per process**, so account for every FSDP rank when choosing a host-memory limit.

```bash
export RELAX_CONDITION_CACHE_BYTES=68719476736
export RELAX_CONDITION_CACHE_DEVICE=cpu
export RELAX_CONDITION_CACHE_IDENTITY="<verified-model-manifest-sha256>"
```

Replace the identity placeholder with the SHA256 identity of a verified manifest covering the model files. A model directory name is not a content identity. The external validation repository generates this identity after validating its pinned assets. See the link below for that fixed comparison.

| Setting | Default | Meaning |
|---|---|---|
| `RELAX_CONDITION_CACHE_BYTES` | `0` | Byte budget. Zero disables reuse. |
| `RELAX_CONDITION_CACHE_DEVICE` | `cpu` | Tensor storage on `cpu` or `cuda`. CUDA storage consumes device memory. |
| `RELAX_CONDITION_CACHE_IDENTITY` | empty | Verified model-content identity. Qwen cache runs require it; SenseNova can hash the local model files when it is omitted. |

SenseNova accepts a SHA256 identity supplied through this variable. Both adapters expect the supplied identity to describe the files actually loaded. After replacing model files, generate a new identity and reload the model.

To return to online encoding, start the next run with:

```bash
export RELAX_CONDITION_CACHE_BYTES=0
```

Variables are read when the model is created. Changing them does not resize an existing cache. For the standard SenseNova training recipe and metadata format, see [Image-to-Image SFT](./i2i-sft.md). The Qwen adapter loads a local Diffusers model directory; pass `relax.models.qwen_image.sft_adapter.QwenImageEditSFTAdapter` as `--model-adapter-path` in the I2I training path.

## Capacity and cache keys

`ConditionFeatureCache` is an in-process LRU with a byte budget and a default limit of 4,096 entries. A new entry evicts the least recently used entries until both limits fit. An entry larger than the entire budget is not stored; the sample still uses its online result.

Keys use the effective prompt and image pixels after preprocessing, together with model identity, feature schema, preprocessing/adapter version, dtype and relevant attention settings. SenseNova also binds the target geometry. Qwen includes its live prompt template and processor configuration. Two filenames with identical contents can reuse features; changing the processed condition creates a different key.

The byte counter covers stored tensor payloads and tree metadata. It is not a limit on total process RSS or CUDA allocator reservation. In a low-reuse dataset, key construction, copying and eviction can cost more than encoding saved. Watch `global_hits`, `global_misses`, `bytes`, `evictions` and `oversized` through `model.condition_cache_stats()` before increasing the budget.

## Training semantics and lifecycle

The adapters decide whether a feature is eligible for reuse. Trainable condition encoders bypass the cache. They clear stored entries after loading state or observing changed frozen-parameter versions; SenseNova also bypasses a prefix with active dropout. Qwen holds its frozen text encoder and VAE in evaluation mode.

Cache reads preserve tensor dtype and return private copies. SenseNova reconstructs a fresh `DynamicCache` for each use, so generation cannot append to another sample's prefix. Noise initialization, flow timesteps/noise and target processing keep their online sampling positions. Qwen's VAE encodes also remain online.

Each FSDP rank owns its cache. Ranks agree on eligibility and hits before taking a branch around a sharded encoder; a rank-local hit cannot skip a collective that another rank needs. Changing data assignment between ranks may therefore reduce reuse.

The model exposes `clear_condition_cache()`, `close_condition_cache()` and `condition_cache_stats()`. `clear` drops entries while leaving the cache usable; `close` releases entries and ends its use. The FSDP trainer's cleanup method closes the model cache, including when an update fails or is interrupted. Custom training loops should call cleanup in `finally` on completion, failure or cancellation. Entries are not written to disk, shared memory or checkpoints, and a resumed process rebuilds them.

## Validation and benchmark evidence

The [separate validation repository](https://github.com/tryu-ai/relax-condition-cache-validation) contains the fixed model/data locks, numerical and performance comparison tools, and recorded H200/RTX 3090 evidence. It documents its own source revision and environment; these tools are not part of the Relax runtime or required to use the cache.

Use the standard [Image-to-Image SFT](./i2i-sft.md) training path for everyday runs, with the cache variables above. The external comparison covers repeated and low-reuse workloads, including cold-cache cost; its results do not guarantee a speedup on other data or hardware.
