# Replay Manager

Replay Manager carries discrete model decisions from SGLang rollout into Megatron training. It has three built-in kinds: `moe` for expert routing (R3), `indexer` for sparse attention selections such as DSA and QSA, and `mm_embedding` for frozen visual outputs. Replaying a selection avoids choosing a different set in training; QSA can also skip indexer scoring for rows covered by rollout.

This page focuses on **reference delivery**: samples carry small descriptors, and immutable replay bodies live in TransferQueue (TQ). This is separate from [Trajectory Replay](./trajectory-replay.md), which records offline loss/advantage reproduction bundles.

## Architecture and ownership

```text
SGLang model / capture hooks
        |
        v
ProducerPublication -- encoded bodies --> existing TQ storage
        |                                       ^
        v                                       |
ReplayRef in rollout sample              ConsumerLayerReader
        |                                       ^
        v                                       |
Sample TQ partition --> Megatron layout --> layer / microbatch channel
                                                |
                                                v
                                     model forward / recomputation

ReplayInventory: run identity, sealed records, capacity, partition holds, leases
```

The orchestration and model semantics are separate:

| Component | Responsibility |
| --- | --- |
| `ReplayManager`, `ReplayKind` | Register producer/consumer adapters and process-local engine hooks; manage run and channel state |
| `ProducerPublication` | Publish bounded row chunks and layer bodies; expose a reference only after writes complete |
| `ReplayRef` | Identify the run, owner, record, kind, dimensions, codec and token digest; carries no dense top-k array |
| `ReplayInventory`, `ReplayRegion` | Track capacity, queue-partition ownership and reader leases; coordinate reclamation |
| `ConsumerBatchLayout`, `ConsumerLayerReader` | Map sample/token coordinates to the actual packed TP/SP/CP layout and read the owning layers |
| `RoutingReplay`, `IndexerReplay` | Apply decisions at model call sites and validate forward/backward consumption order |

Replay reuses Relax's resolved TQ connection; it does not start a second TQ cluster. Body data has a run-specific partition. The inventory is hosted by a Ray actor; model workers keep their own channels, caches and read sessions.

A producer seals a record after its body writes finish. Queue publication attaches a partition hold, then transfers producer ownership after a successful sample write. Consumers acquire leases before reading. Retiring a sample partition does not delete a body still held by a reader. Releases drain pending reads before dropping leases; bodies become reclaimable after publisher, partition and reader ownership are gone. Global recovery creates a new replay run identity so stale references cannot cross into the restarted generation.

Token identity, kind, codec, layer and partition checks reject incompatible references. Channel cursors also detect missing or extra consumption during forward and recomputation. The manager owns these contracts; each model adapter owns capture placement, block coordinates and parallel-layout semantics.

## Enabling replay

Add the following to a supported online, colocated Megatron recipe:

```bash
--use-rollout-indexer-replay
```

This flag automatically enables `RELAX_REPLAY_REFERENCES=1`, allocates a run ID when needed, and forwards the reference environment to workers. There is no second reference-mode switch to configure.

MoE replay is independently enabled with `--use-rollout-routing-replay`; `--use-routing-replay` controls training-side routing replay. `--optimize-routing-replay` selects the existing routing replay optimizations. These flags do not enable QSA. When indexer replay and rollout routing replay are combined, both kinds use the same reference infrastructure.

### Frozen multimodal embedding replay

```bash
--use-rollout-mm-embedding-replay
```

This flag also selects reference delivery automatically. The Qwen3.8 adapter requires online colocated Megatron training, CP1, BF16, and a fully frozen vision encoder **including the merger/projection**. The checkpoint must produce 2560-wide visual embeddings without DeepStack. Critic-based algorithms, MTP, speculative decoding, partial rollout, hierarchical cache and custom generation paths are not supported by this adapter. Existing indexer-replay restrictions on multimodal input remain in place; this flag does not enable indexer replay.

Capture occurs after the visual merger and before visual embeddings are combined with text embeddings. Newly computed features, embedding-cache hits, precomputed features and chunked prefill all refer to the complete media body. The deduplication key is the engine's image hash: the first published body is reused for that hash, without comparing subsequent feature values. Conflicting grid or modality metadata is rejected.

The `bf16-raw-v1` codec transports the original BF16 bits as bytes, with no quantization. A body uses `N_visual × 2560 × 2` bytes: 1024 visual tokens occupy 5 MiB. Multiple responses reference the same body instead of each carrying a copy. The sample queue carries only references and layout metadata; raw pixel tensors are removed after their grids are validated. Existing TQ storage, reader leases and partition retirement manage ownership.

Training loads the body on the input pipeline stage, reads once per tensor-parallel group and broadcasts to its peers. A bounded cache reuses embeddings for log-probability and training forwards; the original placeholder scatter, MRoPE and language-model gradients are preserved. Disabling the flag leaves the original visual computation path in use.

### Supported indexer layouts

The current validator recognizes these model families. These are adapter constraints, not claims that every topology has been benchmarked.

| Model | Training layout | Rollout constraints / representation |
| --- | --- | --- |
| Qwen3.8 (`qwen4_exp`, `qwen4_exp_text`) | CP1; zigzag partition mode; no allgather-CP; mixed backend and TP selector; TP/SP supported | No DP attention; R4, 512 complete blocks; `qsa-r4-bitmap-v1`; capture context at most 262144 tokens |
| DeepSeek-V4 (`deepseek_v4`) | CP>1, TP1, no SP; contiguous partition mode and allgather-CP | DP attention enabled; captures configured C4 layers; current generic publication uses `int32` |
| Keye DSA (`KeyeVL2`) | CP>1, TP1, no SP, PP1; zigzag mode, no allgather-CP | No DP attention; disable overlap scheduling and radix cache; top-k 2048; `keye-bitmap-v1`, capture context at most 32768 tokens |

Indexer replay currently rejects offline training, non-colocated, fully async/hybrid, multimodal, agentic/custom rollout paths, partial rollout, dynamic CP, MTP/speculative decoding and hierarchical cache. Indexer loss must be zero; full activation recomputation requires the uniform method. Combined-1f1b MoE overlap is also rejected. Startup validation reports unsupported combinations.

Use a compatible Relax training image with the SGLang capture and Megatron integration patches. Runtime adapters install process-local hooks; source changes alone cannot supply missing engine APIs. There is no additional Megatron source edit for the bitmap attention operator itself.

## Compression, reads and communication

Model adapters select the wire codec automatically; bitmap compression requires no additional user switch.

Keye encodes selected token sets as bitmap words before D2H, avoiding a persistent dense int32 top-2048 host cache. Fully visible causal prefixes use a header-only representation. QSA applies the same idea to **document-relative complete four-token blocks**: 2048 selected tokens mean at most 512 block IDs. Its codec excludes the incomplete current block; attention reconstructs the causal tail separately. Bitmap membership preserves the selected set, not the original top-k ordering.

QSA publication stores only the visible word range for each chunk. With an 8192-token document, a full-domain QSA bitmap row is approximately 260 bytes (header plus 64 words), versus 2048 bytes for 512 int32 block IDs, or 8204 bytes for an expanded 2051-token row. Early full-prefix chunks need only the header. These are representation sizes, not measured network compression ratios; bitmap cost grows with context length.

Both bitmap consumers batch heterogeneous encoded reads within a 32 MiB/128-body bound and share a bounded GPU body cache. The CPU path caches immutable routing metadata, avoids SimpleStorage's nested-batch repacking, and packs misses into pinned host staging for H2D. QSA additionally reads ahead within the current consumer batch; this is not cross-batch prefetch or a guarantee of complete communication overlap.

With TQ `MooncakeStore` and RDMA configured, CUDA consumers of encoded bitmap bodies select GDR reception on a copied connection. This automatic path does not apply to the generic int32 DSA reader. Registered staging is bounded to 8 MiB, with one compressed D2D unpack into owned tensors. CPU producers retain host transport. SimpleStorage uses CPU reception and H2D. Enabling replay does not by itself configure RDMA or prove that a run used it.

`RELAX_REPLAY_CAPACITY_MB` bounds inventoried body storage; its default is 16384 MiB. The consumer materialization/body-cache budget defaults to 256 MiB per consumer cache. Raising storage capacity does not enlarge that GPU cache. Choose storage capacity for the in-flight rollout volume; do not confuse these independent budgets.

## QSA compute path and dedicated attention

For replay-covered rows, the selector skips query/key indexer projections, score computation and top-k work when no fallback rows require them. Missing rows are computed through the ordinary indexer, with TP communication restricted to missing selections. When projections are needed under SP, local projection plus compact raw-key exchange avoids gathering the full hidden activation. Invocation-local checkpoint tapes reuse selections and retain consumption checks; Q/K/V attention gradients remain enabled.

The bitmap remains compressed through the consumer and into mixed attention. The dedicated Triton implementation reads a 32-bit selection word for a key tile, applies document-local causal masking, and performs attention without constructing expanded routes or a Flex BlockMask. It folds GQA heads into matrix rows to reuse K/V loads. Backward computes Q gradients and reduces K/V gradients within each owning CTA, avoiding per-query-head FP32 K/V scratch and a separate reduction/cast.

Under the default **Grouped mixed** settings, dispatch is:

| Per-document length | Without rollout replay: selected IDs | With QSA bitmap replay |
| --- | --- | --- |
| Up to 2051 | Causal Flash Attention | Causal Flash Attention |
| 2052–16384 | Existing FlexAttention | Dedicated bitmap attention |
| 16385–32768 | Existing Grouped attention | Dedicated bitmap attention |
| Above 32768 | Existing Grouped attention | Decode routes only for Grouped attention |

The thresholds apply to each document, not the concatenated token count on a GPU. A 128K-token pack of sixteen 8K documents therefore uses the dedicated operator. Disabling rollout indexer replay keeps the selected-ID path; it does not implicitly convert native selections into bitmaps or select this operator.

The table describes the default Grouped mixed backend. Explicit model-backend routing settings still apply. The dedicated operator requires CUDA BF16, D256, integer GQA ratio at most 64, and exact CPU document boundaries. Model backend configuration is covered separately in [Qwen3.8 QSA backends](./qwen38-qsa.md).

## Performance observations

The November 2026 measurements below used a single Blackwell GPU, BF16, D256 and local 6Q/1KV, with GPU-resident selections and paired timing rounds. Forward+backward includes all Q/K/V gradients. These measurements exclude rollout capture, TQ transfer, indexer computation and distributed/full-model training.

| Workload and comparison | Before | Dedicated operator | Time reduction |
| --- | ---: | ---: | ---: |
| 16 × 8192 tokens, forward+backward, direct-bitmap Flex reference | 139.44 ms | 40.71 ms | 70.8% |
| One 16K document, forward+backward, expanded-route Grouped path | 37.54 ms | 11.37 ms | 69.7% |
| One 32K document, same comparison | 75.65 ms | 42.76 ms | 43.5% |

For the 16 × 8K case, additional peak PyTorch allocated memory above the call's inputs fell from 1.648 GiB (Flex) to 0.881 GiB. This is not total device memory. At 64K per document the dedicated implementation was about 10.4% slower than the old Grouped path, which is why the default cutoff remains 32K.

In an earlier isolated QSA chain, indexer compute accounted for about 5.8% of execution. Eliminating that computation alone has limited headroom, and exposed replay reads or synchronization can offset the savings.

### End-to-end training runs

The following are logged observations from the 32-GPU TP4/PP4/CP1/EP8 runs, with a 128K per-GPU dynamic microbatch token budget and TP4 rollout engines. Run IDs in parentheses are the first eight characters of the Ray submission ID. Steps are zero-based. The table excludes cold-start step 0 and records early steps 1–2 rather than a long-run average; kernel-v1 step 2 is excluded because profiling interfered with that sample.

`perf/train_time` covers the training region, including log-probability computation, actor training and associated work. `perf/actor_train_time` measures the actor update alone. `perf/step_time` is defined in code as `perf/train_wait_time + perf/train_time`; its wait component includes time outside the training region, such as rollout, weight updates and colocate transitions. It is not just attention execution, nor simply rollout time plus actor forward/backward.

| Experiment / run | Step | Training tokens (million) | `train_time` (s) | `step_time` (s) |
| --- | ---: | ---: | ---: | ---: |
| Baseline, no rollout replay (`f1849893`) | 1 | 2.978 | 72.13 | 480.56 |
| Baseline, no rollout replay (`f1849893`) | 2 | 3.301 | 68.18 | 427.43 |
| QSA replay only (`e9168b14`) | 1 | 3.163 | 81.41 | 506.51 |
| QSA replay only (`e9168b14`) | 2 | 3.346 | 72.00 | 427.08 |
| QSA + R3, direct bitmap (`59942491`) | 1 | 2.972 | 81.79 | 517.90 |
| QSA + R3, direct bitmap (`59942491`) | 2 | 3.010 | 72.56 | 465.41 |
| QSA + R3, bitmap kernel v1 (`b96d38fe`) | 1 | 3.444 | 82.53 | 512.26 |
| QSA + R3, bitmap kernel v2 (`f6ee7d39`) | 1 | 3.164 | 80.31 | 534.96 |
| QSA + R3, bitmap kernel v2 (`f6ee7d39`) | 2 | 3.093 | 63.18 | 442.27 |

The bitmap-v2 step-2 training window was 63.18 seconds, versus 68.18 seconds for the baseline, while the corresponding step times were 442.27 and 427.43 seconds. Token counts and rollout lengths differ, so these observations do not establish a controlled replay or kernel speedup. The single-operator result above does not translate directly into step-time savings. QSA replay alone did not materially lower the measured mismatch.

### Frozen visual embeddings: OpenR1MM GRPO

The 2026-11-18 Qwen3.8-Flash-Next OpenR1MM comparison used 32 Blackwell GPUs (280 GB each), TP4/PP4/CP1/EP8 training, four TP8/EP8 rollout engines, and a per-GPU dynamic microbatch token budget of 16384. The prompt batch was 128 with eight responses per prompt and a global batch of 1024. The vision encoder and merger/projection were frozen. Each run completed ten steps without checkpoint saving; only `--use-rollout-mm-embedding-replay` was toggled. Neither run enabled QSA indexer replay or R3.

| Median duration (steps 1–9, excluding step 0) | Native vision path | MM embedding replay | Reduction |
| --- | ---: | ---: | ---: |
| `perf/train_time` | 81.01 s | 74.53 s | 8.0% |
| `perf/step_time` | 313.68 s | 302.81 s | 3.5% |

Over the same interval, total training tokens divided by total `train_time` increased from 20.05K to 24.40K tokens/s (about 21.7%). Generated responses and lengths were not token-identical, and the baseline also encountered shape compilation after the first step. These are observations for this configuration; the throughput change cannot be attributed entirely to skipping visual computation.

On a representative step, one TP0 rank on the input pipeline stage read about 126 MB, spending approximately 0.9–1.1 seconds in CPU-side reads and 0.02 seconds in TP broadcast calls. It recorded about 900 cache hits and 120 misses, without eviction. CPU durations include calls and waits; they do not measure GPU completion through CUDA synchronization, and the byte count is not a global communication total.

Training/rollout consistency improved modestly: across the first ten steps, mean `train/mismatch_kl` fell from 0.002208 to 0.002012 (about 8.9%), and absolute log-probability difference fell from 0.02814 to 0.02703 (about 4.0%). A subsequent 200-step run had completed 39 steps at observation time; its first-ten-step mismatch KL was about 6.5% below the baseline. This is not a completed long-run result or, by itself, proof of a causal mismatch improvement from replay.

## Inspecting a run

Worker logs emit `replay_consumer` with TP/PP/CP/DP coordinates. Useful counters include `replay/encoded_read_bytes`, `replay/encoded_h2d_bytes`, `replay/encoded_rdma_bytes`, `replay/gpu_cache_peak_bytes`, `replay/encoded_prefetch_hits` and `replay/encoded_prefetch_wait_seconds`. The tracking backend receives one representative training rank's metrics, not a cross-rank maximum.

Use these together with actor/log-probability timing: bytes show whether the compressed path is used, H2D versus RDMA counters distinguish reception paths, and prefetch hit/wait counters expose foreground stalls. A prefetch hit alone does not prove that the transfer was fully hidden.

Implementation entry points are under `relax/distributed/replay_manager/`; the QSA adapter is `backend/qwen38.py`, and the dedicated kernel and mixed dispatch are under `relax/models/qwen3_8/ops/qsa_mixed_ops/`.
