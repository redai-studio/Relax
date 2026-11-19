# Detached text prefixes for Qwen3.5 (2026-10-29)

## Status and scope

The optional actor path is connected to Relax's Megatron `train_one_step`.
Default `baseline` retains the existing VCS and frozen-vision cache behavior.
This is approximate training: historical prefix computation has no gradient,
while suffix computation and its reads of fixed prefix states remain trainable.
It is not gradient-equivalent to baseline.

GPU verification uses the actual Qwen3.5 hybrid VLM provider, TransformerEngine
projections, FLA GatedDeltaNet and the existing visual feature cache, with reduced
model dimensions and synthetic packed inputs. It does **not** establish full
Qwen3.5-9B CUA training performance or task success rate. No Ray training or
evaluation job has been launched by these tests.

## Configuration

The mobile script forwards these variables to Ray workers:

```bash
# Existing behavior; default
RELAX_TRAIN_PREFIX_MODE=baseline bash scripts/run-qwen35-9B-mobilegym-train-fxs.sh
# Rebuild detached text prefix for every window
RELAX_TRAIN_PREFIX_MODE=detached_prefix bash scripts/run-qwen35-9B-mobilegym-train-fxs.sh
# Reuse/extend exactly matching prefixes within one optimizer update
RELAX_TRAIN_PREFIX_MODE=detached_prefix_reuse \
RELAX_TRAIN_PREFIX_MAX_BYTES=268435456 \
bash scripts/run-qwen35-9B-mobilegym-train-fxs.sh
```

These commands run from the CUA plugin checkout, with the normal model/data/cluster
configuration already set. The byte budget limits retained prefix entries per
worker; temporary detached copies and tensors held for suffix backward are additional.
The separate vision-cache and VCS switches keep their existing meanings.

## Implementation and semantics

- `relax/utils/visual_context.py` prepares VCS windows, including cropped images
  and segment markers. `relax/agentic/session/state.py` exports authoritative
  tokens and new-response loss masks. Neither was changed.
- `relax/utils/data/stream_dataloader.py` records host token IDs, logical sample
  identity and prompt boundary before CUDA transfer. No prompt reconstruction.
  Independent reference/old-policy logprob fetches need no prefix metadata.
- `relax/utils/training/prefix_runtime.py` wraps only actor-update forwards, after
  the original VLM has prepared image embeddings and actual mRoPE positions.
  Prefix ends before the first remaining image/video/start marker, no later than
  the token preceding the first response target, aligned for sequence parallel.
  This is conservative: later eligible text is not detached if an image remains.
- `qwen35_prefix_state.py` preserves original projections and output gates.
  Full attention stores rotated local-head K/V; GDN stores local convolution and
  recurrent state. The suffix receives detached copies. Full attention uses SDPA
  with explicit offset causal masking; no custom CUDA kernel or rollout KV.
- `qwen35_prefix_decoder.py` runs original TransformerLayers, captures immutable
  per-layer states in checkpoint closures, and preserves full recomputation.
- `detached_prefix.py` owns a bounded LRU with exact tokens/positions/semantics,
  separate trajectory keys, copy isolation and explicit invalidation.
- `relax/backends/megatron/model.py` creates one runtime per gradient accumulation
  update and clears it in `finally` before optimizer.step. No cache survives to
  weight loading, actor/reference switching or the next update. Old logprobs,
  reference forward, rewards, advantages and original loss normalization stay on
  their existing paths. Physical windows are not new reward episodes.

TP projections and SP gather/scatter retain their original process groups. Each
rank caches its own state; DP workers share no cache. Packed rows execute separately
and the full original output layout is restored before the original loss code.
Only the first packed row currently uses cross-window reuse: its image-free
position origin is verified exactly against the prepared rotary tensor on GPU.
Other rows use detached recomputation, including in reuse mode, and count rebuilds.

Exact prefix matching deliberately includes the changing VCS segment marker.
If that marker changes near the start, a longer cache pool cannot create a hit.
The implementation must miss rather than silently omit the marker or reuse a
still-image-bearing state for cropped text.

## Capability limits

The adapter targets the installed dense Qwen3.5 hybrid VLM Megatron interface,
BF16, causal attention, zero dropout, PP1 and CP1. TP is not whitelisted; the
original local projection interfaces are used. PP needs pipeline state scheduling
and CP needs distributed prefix-state handling, so those paths explicitly fail.
No parallel configuration is silently changed. Nonempty deepstack injection,
selective recomputation, adapters, MoE, MTP, quantization/offloading variants and
nonstandard attention are rejected. SFT CPU-prepacked data has no prefix metadata
and is not supported. These are interface limits, not claims of general model
incompatibility. Capability checks run at the actor forward before custom execution.

## Reproducible validation

From the Relax checkout, using existing dependencies and idle GPUs:

```bash
python -m pytest -p no:cacheprovider \
  tests/utils/test_detached_prefix.py tests/utils/test_prefix_runtime.py -q

MEGATRON_ROOT=/path/to/Megatron-LM PREFIX_TEST_IMAGES=1 PREFIX_BENCHMARK=1 \
CUDA_VISIBLE_DEVICES=1,3,4,6 CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE=0 \
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=4 \
  tests/experimental/check_prefix_actor.py
```

Use one visible GPU and `python` for TP1 without SP; use `torchrun` with two
processes for TP2 with SP. GPU identifiers above are examples: check occupancy.
The test performs three accumulated microbatches followed by AdamW, compares
supervised logits/logprobs across modes and gradients/updated parameters across
both detached modes. BF16 tolerances: outputs atol=0.01, rtol=0.02;
gradients/updates atol=0.001, rtol=0.02. CPU float64 prototype tests use 1e-10.
CPU tests also cover mutation isolation, cache limits, identity/position/semantic
mismatches, invalidation, and unchanged training metadata (8 passed).

Benchmark mode warms up one update, then times three complete updates with CUDA
synchronization, including prefix builds/extensions/copies and optimizer updates.
It reports forward/backward/total wall time, peak allocated memory and cache counters.
It is a reduced synthetic benchmark, not a real CUA trajectory benchmark.

## Remaining validation

Full 9B weights, real CUA trajectory replay, Megatron distributed optimizer/DDP
fusion, TP4/DP2 end-to-end Ray execution, task success and long-history behavior
still require validation. Reduced-model timing must not be extrapolated to 9B.
No success-rate preservation claim is made. Real-trajectory exact-hit rates may be
low because of VCS segment markers and the first-packed-row reuse restriction.
The explicit SDPA mask and per-row execution can cost more than the work saved,
especially for short contexts. The feature remains default-off.

## Observed GPU checks and timing

TP1 (SP off), TP2 and TP4 (SP on), PP1/CP1, full recomputation all passed
three-mode logits and detached-versus-reuse gradients/AdamW checks with images
and the existing vision cache enabled. TP1 and TP4 additionally passed the
TP-vocabulary cross-entropy/logprob check. These are reduced architecture tests,
not full distributed optimizer or GRPO training tests.

Latest TP4 BF16 cross-entropy benchmark, median of three rank-0 observations:

| Mode | Forward (s) | Backward (s) | Update incl. cleanup/optimizer (s) | Process peak allocated (MiB) |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.0912 | 0.1739 | 0.2689 | 123.06 |
| detached_prefix | 0.2445 | 0.2222 | 0.4682 | 123.21 |
| detached_prefix_reuse | 0.2275 | 0.2174 | 0.4488 | 123.21 |

The test has only four language layers, hidden size 128, two 64-token packed
rows and three microbatches. Prefix demand is 224 tokens across six windows;
reuse computes 152 (two hits, four builds, one extension), retaining 56,832 bytes
per rank. The vision cache is enabled in all modes. Peak memory includes test
weight/gradient snapshots, so this is not a production memory estimate.

Reuse was about 4% faster than detached rebuild in this short measurement,
but both were slower than baseline. Extra per-row/chunk execution and communication
outweigh saved prefix work for this small input. This is not evidence of a 9B
training speedup. Raw per-rank measurements and package versions are in
`detached-prefix-benchmark.json`; no task-success measurements were taken.

## Full-size validation finding (not yet resolved)

The complete local Qwen3.5-9B architecture was exercised using `PREFIX_FULL_CONFIG`
with random initialization and short synthetic image/text input on TP4. All three
forward/backward/AdamW paths executed, but baseline-versus-detached logits failed
the declared tolerance (maximum absolute error about 0.52). This run is **not a
correctness pass**; do not treat the reduced-model passes as 9B validation.
Layer-level diagnosis is ongoing.

A separate reduced-model real Megatron DDP test with fused main_grad accumulation
and zero-length prefixes passed baseline-versus-split gradients. The adapter
forces accumulation for each decoder chunk instead of allowing TE's first-chunk
flag to overwrite gradients from other chunks. The transpose-cache hint is
restored after each layer call; quantized paths remain unsupported.
