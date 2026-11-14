# Frozen Condition Cache

Image-edit SFT often sees the same instruction and images across epochs. When the condition encoders are frozen, Relax can reuse their outputs while continuing to train the generation model. The cache is opt-in; training noise, timesteps and model predictions remain outside it.

Two model adapters use the same storage interface:

| Adapter | Cached features | Training path |
|---|---|---|
| `relax.models.sensenova.adapter.SenseNovaSFTAdapter` | Frozen prompt/image prefix KV tensors and indexing metadata | Generation-only LoRA; full-parameter training keeps the online path |
| `relax.models.qwen_image.sft_adapter.QwenImageEditSFTAdapter` | Frozen text/vision embeddings, attention mask, and deterministic VAE mode latents of the target and conditioning images | LoRA with one conditioning image |

The Qwen adapter follows the conditioning and latent geometry of the [Diffusers `QwenImageEditPipeline`](https://github.com/huggingface/diffusers/blob/d791c5c024014784df4a8dac2601e19fb4d300fc/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py), including posterior-mode image latents. It trains with a uniform-sigma flow-matching loss.

## Enable reuse

Set the following variables in every training worker's environment before loading the model. For workers on the same node, use the same node-local cache root. Keep this directory dedicated to Relax condition caches and ensure it has enough free space for the budget and temporary writes.

```bash
export RELAX_CONDITION_CACHE_BYTES=68719476736
export RELAX_CONDITION_CACHE_DEVICE=shm
export RELAX_CONDITION_CACHE_DIR=/dev/shm/relax-condition-cache
export RELAX_CONDITION_CACHE_IDENTITY="<verified-model-manifest-sha256>"
```

In `shm` mode, each run has a separate directory shared by its ranks on a node. A sample can reuse an entry even when the sampler assigns it to another rank in a later epoch. Each node has its own capacity budget; this is not a cache shared between nodes or training runs.

Replace the identity placeholder with the SHA256 identity of a verified manifest covering the model files actually loaded. A directory name is not a content identity. The external validation tools generate this identity after checking pinned assets. After replacing model files, regenerate the identity and reload the model.

| Setting | Default | Meaning |
|---|---|---|
| `RELAX_CONDITION_CACHE_BYTES` | `0` | Byte budget; zero disables reuse. In `shm` mode it bounds committed entry files per run per node. In `cpu`/`cuda` mode it is a per-process budget. |
| `RELAX_CONDITION_CACHE_DEVICE` | `shm` | Node-shared files under the cache root, or a private in-process LRU on `cpu` or `cuda`. CUDA storage consumes device memory. |
| `RELAX_CONDITION_CACHE_DIR` | `/dev/shm/relax-condition-cache` | Node-local root used by `shm` storage. All ranks on that node must see the same directory; do not use a network-shared filesystem. |
| `RELAX_CONDITION_CACHE_MAX_ENTRIES` | `4096` | Maximum committed entries per shared run, or per private process. |
| `RELAX_CONDITION_CACHE_STAGING_BYTES` | `2147483648` | Per-rank budget for pending shared-cache writes. Writes that exceed it are dropped and counted; training uses the online result. |
| `RELAX_CONDITION_CACHE_IDENTITY` | empty | Optional verified model-content identity. When omitted, both adapters derive a run-local identity from small-file contents and large-weight relative paths, sizes and modification times. |

Set `RELAX_CONDITION_CACHE_DEVICE=cpu` or `cuda` when a private cache is appropriate. Account for every rank's budget in those modes. Variables are read when the model is created; changing them does not resize an existing cache. To use online encoding in the next run, set `RELAX_CONDITION_CACHE_BYTES=0`.

For the SenseNova recipe and metadata format, see [Image-to-Image SFT](./i2i-sft.md). The Qwen adapter loads a local Diffusers model directory; pass `relax.models.qwen_image.sft_adapter.QwenImageEditSFTAdapter` as `--model-adapter-path` in the I2I training path.

## Capacity and cache keys

`SharedConditionStore` writes detached features asynchronously through bounded host staging buffers. A background writer finishes the device-to-host copy, writes a temporary file, then publishes it by atomic rename. A process lock serializes eviction and publication. Least recently used committed entries are evicted until both byte and entry limits fit; an oversized entry is not stored.

The shared byte limit covers committed entry files, including their serialized headers and alignment. Temporary writes, private read buffers, each rank's staging buffers, filesystem allocation overhead and the model's own memory are additional. Leave free SHM and host memory beyond the configured budget. Write failures are counted and leave the online path usable.

Before ranks agree on a cache hit, each reader validates and stages its entry in private host memory. A later eviction cannot remove the staged value needed for that update. Missing or malformed entries are treated as misses and recomputed. Reads preserve dtype and give each consumer its own tensors; SenseNova rebuilds a fresh `DynamicCache` so generation cannot append to another sample's prefix.

Keys cover the effective prompt and post-preprocessing image pixels, model identity, feature schema, preprocessing/adapter version, dtype and relevant attention settings. SenseNova also binds target geometry and the encoder's device semantics; homogeneous shared-cache ranks can reuse an entry without tying it to a local CUDA ordinal. Qwen includes target pixels, its live prompt template, processor configuration and VAE settings. Different filenames with identical effective inputs can reuse features; changed crops, prompts or model settings select a different key.

In a low-reuse dataset, key construction, transfers, serialization and eviction can cost more than the encoding saved. Inspect `model.condition_cache_stats()` and the trainer's `cache/*` metrics: hits/misses, entries/bytes, evictions, pending bytes, dropped writes, write errors and corruptions. Branch and operation counters in rank-0 logs describe that rank; shared entry/byte readings describe the node's run directory.

## Training semantics and lifecycle

Adapters decide whether reuse is valid. Trainable condition encoders bypass the cache. Loading model state or changing frozen-parameter versions clears stored entries. SenseNova bypasses prefixes with active dropout; Qwen keeps its frozen text encoder and VAE in evaluation mode. Qwen caches deterministic posterior-mode latents, while flow timesteps and noise are sampled on every update. Effective image pixels are part of the key, so augmentation results are not identified only by a sample filename.

FSDP ranks agree on eligibility and hits before taking a branch around a sharded encoder. A hit on one rank cannot skip a collective another rank still needs. The adapter binds shared storage to the training process group; custom integrations must keep the same collective participation.

The model exposes `clear_condition_cache()`, `close_condition_cache()` and `condition_cache_stats()`. `clear` invalidates entries while keeping the store usable; `close` ends the owner's use and releases pending work. The last owner on a node removes the run directory. A persistent root lock serializes joining, closing and stale-run cleanup; a zero-byte lifecycle lock may remain after entries are gone.

The FSDP trainer closes the cache after its last configured update and in its failure/cleanup paths. Custom training loops should close it in `finally` on completion, failure or cancellation. Abruptly killed processes cannot execute cleanup; directories whose owners are no longer alive are reclaimed on the next cache start. Do not manually remove a cache root while its ranks are active. Cached features are not checkpoint state, and a resumed run rebuilds them.

## Validation and benchmark evidence

The [separate validation repository](https://github.com/tryu-ai/relax-condition-cache-validation) contains pinned model/data manifests, numerical and performance tools, raw logs and their source identities. It keeps measurements from different implementation versions and hardware separate. These tools are not part of the Relax runtime or required to enable the cache.

Use the standard [Image-to-Image SFT](./i2i-sft.md) training path for everyday runs. The external comparisons cover repeated and low-reuse workloads, cold-cache costs and full training-loop timing; their results describe those workloads rather than every dataset or machine.
