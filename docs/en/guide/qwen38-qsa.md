# Qwen3.8 QSA backends and parallelism

This guide covers training-side QSA in the Megatron image, with source baseline `97210aaf5b40db54eb63e80748b5f34b3433fb78` (PR1). Local kernel geometry, Relax adapter acceptance, and a validated training topology are separate contracts. The NV recipes later in this guide are configuration examples, not a guarantee for every backend or parallel layout.

## Execution layers

The selector chooses historical **complete blocks**; the attention kernel consumes that selection and adds the incomplete causal tail. TP/SP/CP adapt projections, token layouts and communication around these operations. HybridEP handles MoE communication independently.

```text
Model projections / TP-SP-CP layout
  -> selector -> QSASelection -> QSACoreAttention
       -> all-selected causal shortcut (unless forced sparse / mixed)
       -> dense_masked -> PyTorch SDPA with an explicit mask
       -> flex -> optional fused candidate -> Triton / triton_v2 / triton_v3
               -> otherwise native Flex / optimized BlockMask Flex
       -> id_sparse -> native selected-ID Triton
                    -> community=mixed -> Flash / Flex / Grouped
                                       -> Flash / cuDNN PR1765
```

Fused attention replaces the Flex calculation; it does not replace the selector. Mixed wraps the selector and replaces the selected-ID attention entry. A fused candidate that fails `supports()` returns to Flex; mixed NV does **not** fall back to Grouped. Native all-selected CP1 inputs can use the configured dense core; CP>1 has an explicit gathered causal path. Mixed binds `all_selected=False` to preserve its own short-document route. See [native core][native], [dispatch][dispatch], [operator integration][integration] and [mixed entry][mixed].

### Selecting a route

Defaults below refer to the framework, before model/recipe scripts export overrides. Configure driver and training actors consistently. The patched Megatron implementation is supplied by [Docker build][docker], rather than by an arbitrary installed Megatron checkout.

| Route / layer | Configuration | Entry and default |
| --- | --- | --- |
| Native dense reference | Empty `RELAX_QWEN38_QSA_COMMUNITY_BACKEND`; `MCORE_QSA_SPARSE_BACKEND=dense_masked` | `QSACoreAttention._dense_masked_forward`; opt-in |
| Native Flex | Empty community; `MCORE_QSA_SPARSE_BACKEND=flex`; fused attention off | `_flex_forward`; native backend default is `flex` |
| Optimized BlockMask builder | Native Flex plus `RELAX_QWEN38_QSA_SPARSE_BLOCKMASK=1` | `qsa_block_mask._sparse_flex_forward`; framework default off; changes mask construction |
| Fused attention implementations | Native Flex plus `RELAX_QWEN38_QSA_FUSED_ATTENTION=triton`, `triton_v2`, or `triton_v3` | `qsa_fused.backend`; framework default off; `1`/`true` also select Triton v1 |
| Native selected-ID kernel | Empty community; `MCORE_QSA_SPARSE_BACKEND=id_sparse` | `_relax_native_id_sparse_forward`; opt-in |
| Mixed composition | `RELAX_QWEN38_QSA_COMMUNITY_BACKEND=mixed`, fused attention off, native backend `id_sparse` | `qsa_community` and `qsa_mixed_ops`; nonempty community selects `id_sparse` by default and rejects a conflicting explicit backend |

The [shared model script][model-script] defaults BlockMask, local-query, replay, fused selector and fused attention **on**; these differ from [environment registry][env] defaults. It disables fused selector when community is nonempty, but does not also disable fused attention. Consequently, setting only community to `mixed` while inheriting fused attention `1` triggers the mixed producer's explicit rejection. Use the complete mixed recipe, or explicitly set the combination below before model construction:

```bash
export RELAX_QWEN38_QSA_COMMUNITY_BACKEND=mixed
export MCORE_QSA_SPARSE_BACKEND=id_sparse
export RELAX_QWEN38_QSA_FUSED_ATTENTION=0
export RELAX_QWEN38_QSA_FUSED_SELECTOR=0
export RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=grouped
```

This selects the route only; model, frozen-indexer, layout and parallel prerequisites still apply.

### Attention input contracts

`Hq/Hkv/D` below describe **attention** heads. They do not describe the selector's separate projection. Native adapters receive THD `[T,H,D]` or SBHD `[S,B,H,D]` and convert them for their kernel. Check [native methods][native], [fused adapter][fused], [fused kernel][fused-kernel], and [mixed entry][mixed] before changing a shape.

| Attention implementation | Local geometry / selection | Execution constraints |
| --- | --- | --- |
| Native `dense_masked` | GQA by K/V replication; bitset selection; no Relax 12/1/D256 specialization | Explicit `[B,S,S]` boolean mask plus SDPA; quadratic mask memory. Device/dtype depend on the chosen PyTorch SDPA implementation; the Qwen recipe uses CUDA BF16 |
| Native Flex / BlockMask Flex | GQA via `enable_gqa`; bitset with document IDs and positions; no Relax 12/1 specialization | PyTorch FlexAttention/Torch compilation support required. Optimized BlockMask requires CUDA, nonempty R4 bitset; otherwise native Flex. Small (`S<=4096`) Dynamo compile failures can use dense-masked fallback |
| Native `id_sparse` Triton | Integer GQA, power-of-two `D<=256`; int32 IDs/positions; positive top-k; ratio divides 32 and is at most 32 | Contiguous CUDA Q/K/V, matching shapes. IDs must be distinct visible complete blocks, fill `min(visible,K)` slots, and use `-1` padding. No explicit dtype whitelist in this prototype; this is not arbitrary-dtype numerical validation |
| Fused Triton v1 / v2 / v3 | Shared `supports`: CUDA, native `flex`, one selection row, nonempty R4 bitset, `D` in `{64,128,256}`, integer GQA | THD kernel input; non-THD adapter accepts batch 1, otherwise returns to Flex. Plan carries forward/reverse 128-token tiles and optional global `query_ids`. `supports` has no dtype or Blackwell-only whitelist; compilation/numerical portability is separate |
| Mixed Flash / Flex / Grouped | R4 native IDs, expanded to tokens with document offsets and partial tail; geometry below | CUDA BF16, CP1, frozen Stage-1 indexer and zero indexer loss. Grouped uses BF16 products and FP32 K/V atomic accumulation; do not transfer fused deterministic-backward claims to it |
| Mixed Flash / cuDNN | R4 int32 IDs, exactly 512 columns per token (2048-token selection budget); geometry below | CUDA BF16; same mixed producer requirements. Long branch imports the optional pinned package; short branch still obeys NV adapter geometry. See NV runtime requirements below |

The [fused v2][v2]/[v3][v3] implementations reuse the base `supports()` and plan. SM103/BF16/D256 tile choices are specializations, not a universal Blackwell requirement. The v3 **shared-dS backward optimization** has narrower guards: SM103, BF16, D256, R4, no local `query_ids`, positive Q/K lengths at most 131072 and bounded int32 addressing. Its workspace policy caps shared scratch at 24 GiB; ineligible inputs or insufficient scratch retain the regular backward. These guards do not prohibit all CP use of v3. See `_can_share_ds` in [v3][v3] and its `shared_backward.py`.

### Selection, packing and training stage

[QSASelection][native] holds document-local `positions`, `doc_ids` and either a byte bitset (`selected_bits`) or compact complete-block IDs (`selected_ids`). A budget of 2048 tokens at R4 means at most 512 complete blocks, not 2048 block IDs. Causal masking adds at most three tokens of the current incomplete block; it must not cross a packed document boundary.

| Producer / adapter | Configuration and requirements | Boundary / stage contract |
| --- | --- | --- |
| Native selector | `QSAIndexer` parameters come from the provider's indexer configuration; supports bits or IDs according to attention backend | Reconstructs CP sequence and handles SP gather. Do not infer selector heads from attention's global 24Q/2KV |
| Fused selector postprocessing | `RELAX_QWEN38_QSA_FUSED_SELECTOR=1` (framework off); `qsa_select.select_blocks` | CUDA four-head selector, `bits` output, usable CPU offsets and other shape guards; ineligible cases use the ordinary selector. This is separate from fused attention |
| Mixed TP selector | `RELAX_QWEN38_QSA_TP_SELECTOR=1` (default `0`), mixed backend; optional `RELAX_QWEN38_QSA_SELECTOR_COMPILE=1` (default off) | CP1, frozen, IDs, selector **4Q/1KV/D128**, R4, positive budget divisible by 4 and at most 2048; zero-centered RMSNorm, BF16 CUDA `[S,1,hidden]`, RoPE, explicit TP/singleton-CP groups; gathered sequence nonempty and divisible by TP. Shards query work, all-gathers selected IDs |
| Mixed packed metadata | `qsa_mixed_ops.forward`, `bind_document_bounds` | Exact producer CPU boundaries (`qsa_physical_layout_cpu` binding or `qsa_select_offsets_cpu`); copied selection preserves boundaries during checkpoint replay. Nonpacked rows are independent documents |
| Stage-2 indexer loss | Native `id_sparse`, positive `qsa_indexer_loss_coeff` | Separate bounded KL path: gathered `S*B<=4096`, attention dropout zero, explicit TP group when TP>1; THD needs true-length `qsa_stage2_layout_cpu`. Mixed rejects this stage; frozen-indexer replay must not detach its projections |

Selector entry priority is mixed wrapper → fused-selector wrapper if enabled → replay → TP-selector → CPU-layout/native stages. “CPU layout” refers to metadata handling, not a CPU attention kernel. The branches are gated functions, not all executed for every batch. Sources: [dispatch][dispatch], [selector fusion][select], [TP selector][tp-selector], [mixed entry][mixed], [native loss attachment][native].

## Parallel layouts and independent guards

| Dimension | QSA/model behavior | Scope of the restriction |
| --- | --- | --- |
| TP | Global attention heads are partitioned; when TP exceeds two KV heads, KV groups are replicated and Q/gate slices are narrowed within each group | Model projection layout, not an attention-kernel topology test. See [projection patch][projection] and `qsa_routed_qkv.route_table` |
| CP | Native sparse core reconstructs full Q and differentiable K/V, then splits output back to the local slice | Baseline communication path; full activations still exist on each CP rank. Mixed producer explicitly requires **CP1** |
| CP local-query | `RELAX_QWEN38_QSA_LOCAL_QUERY=1` keeps Q local and gathers K/V with gradient summation | Only CP>1, THD, CUDA, Flex, nonempty R4 bitset, Stage-1/no differentiable indexer tensors, suitable sparse selection. Ineligible calls return to native core; it does not enable mixed CP>1 |
| SP | Native/TP selector gathers hidden states from the explicit TP group when sequence parallelism is active | Orthogonal to attention GQA. TP selector still requires gathered length divisible by TP; a script's `--sequence-parallel` is not a kernel-only capability |
| PP | QSA executes in its owning pipeline layer; MTP can add a QSA layer on the last stage | No general PP4 guard in the attention kernel. Layer partition, MTP and recomputation integration must be checked separately |
| EP / ETP | MoE communication is independent of attention head grouping | The selected dispatcher can impose physical topology and package constraints even if local QSA shapes pass |

For the global 24Q/2KV/D256 model, the projection code yields:

| TP | Local attention Q/KV | KV placement |
| --- | --- | --- |
| 1 | 24/2 | Both KV groups local |
| 2 | 12/1 | One KV group per rank |
| 4 | 6/1 | Each KV group shared by two TP ranks |
| 8 | 3/1 | Each KV group shared by four TP ranks |

The [projection patch][projection] gathers projection shards when query groups are fewer than TP ranks, selects the KV group by rank, and slices Q/gate within that group. The optional [routed QKV][routed] path replaces that gather with explicit TP all-to-all and sums shared-KV gradients in the adjoint. It is guarded to gated QSA, global 24/2/D256, TP4/8, CP1, BF16 or FP32 and no FP8/FP4/test mode; other cases use the original projection. This table describes **layout derivation**, not successful TP1/2/4/8 training. In particular, an NV adapter remains limited by the mixed geometry table below.

### MoE topology and recomputation are separate contracts

[HybridEP preflight][hybrid] runs only for the HybridEP dispatcher. Its Qwen3.8 layouts allow TP2 **or TP8**, PP4/CP1/EP8/ETP1/world32. The other listed TP1/PP2/CP4/EP4/world16 layout belongs to the Qwen3.6 recipe; it does not establish mixed QSA CP4 support. Preflight requires an EP×ETP group of distinct ranks on one hostname, the expected accessible-peer count on every member, an installed package matching `RELAX_HYBRID_EP_COMMIT` and module provenance. Forced accessible-rank overrides are rejected. Thus TP1/TP4 local geometry does not imply acceptance by this Qwen3.8 HybridEP configuration.

`RELAX_QWEN38_QSA_RECOMPUTE_REPLAY=1` caches frozen selection/mask/plan metadata on an **invocation-local** checkpoint tape. Trainable indexers use the ordinary path; K/V remain differentiable. It is distinct from `RELAX_QWEN38_RETAIN_QSA_ATTENTION=1`, which retains attention and checkpoints the MLP. The latter's [guard][retain] is deliberately narrow: BF16, full/uniform/one-layer recomputation, TP8/PP4/CP1/EP8/ETP1; no FP8/FP4, activation offload, distributed saved activations, VPP or CUDA graphs, with validated layer owners and frozen indexers. These are optimization guards, not universal QSA limits.

## Mixed geometry and document-length routing

The shared `geometry.validate_geometry` check in [geometry.py][geometry] is used by the public mixed entry, Grouped long adapter and direct NV adapter.

Mixed QSA requires CUDA BF16, D256, CP1, and a frozen Stage-1 indexer. Local head geometry depends on the selected long-document backend:

| Long backend | Accepted local geometry |
| --- | --- |
| `grouped` | Positive Q and KV head counts with Q divisible by KV, including 12Q/1KV, 24Q/2KV, 6Q/1KV, and 3Q/1KV; these examples are not a whitelist |
| `cudnn_pr1765` | The current Relax adapter accepts only 12Q/1KV/D256, including batches containing only short documents |

The framework default for `RELAX_QWEN38_QSA_MIXED_LONG_BACKEND` is `grouped`; the NV example scripts below select `cudnn_pr1765`. For layouts such as 24Q/2KV, 6Q/1KV, or 3Q/1KV, explicitly set `RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=grouped`. An unsupported NV layout raises an error before execution; it does not fall back to Grouped. Capabilities of the pinned NVIDIA package do not expand the geometry accepted by the current Relax adapter.

Passing the local geometry check does not establish GPU output/gradient correctness or validate a complete TP/CP/EP training topology. Megatron and HybridEP topology constraints still apply. Packed inputs require the producer's CPU document boundaries; token positions and the incomplete causal tail remain local to each document.


The framework defaults are `MIXED_LONG_BACKEND=grouped`, `MIXED_FLASH_MAX_LEN=-1`, and `MIXED_FLEX_MAX_LEN=16384` (all with prefix `RELAX_QWEN38_QSA_`). Let `K` be ID columns and `B=4*K` the complete-token budget after expansion:

| Mixed route | Short / intermediate / long documents | Configurable boundary |
| --- | --- | --- |
| Flash/Flex/Grouped | At production `K=512`: typically Flash through 2051, Flex from 2052 through 16384, Grouped above 16384 | Automatic Flash limit is **B+3**; Flex maximum is 0..16384. Explicit Flash limit must be 0..B+3 and requires Flex maximum 0 |
| Flash/cuDNN | At mandatory `K=512`: Flash through 2051, NV from 2052, with no Flex branch | Flash limit `-1` resolves to 2051; explicit range 0..2051. `NV_MAX_PACK_TOKENS=0` calls NV once per long document; 0..16384 permits bounded mini-packs |

These are per-document lengths, not concatenated pack sizes. The Grouped composition rebases selections and restores output order; Flex subsets can be padded to 16384 internally. NV constructs a document-local byte mask from native IDs and includes the partial tail. See [Grouped dispatcher][mixed-dispatch], `flex.attention`, [NV adapter][nv] and `routes.token_routes`. The common QSA training script sets Flex maximum to **0**, so even its Grouped recipe differs from the framework's three-way default. NV wrappers further set Flash maximum to 2051; the table later in this guide lists those recipe values.

## Evidence and validation boundaries

This table distinguishes implementation, test coverage and recorded execution at the source baseline above. “Cases” means tests exist, not that they ran. No source-version/hardware/config-bound GPU numerical report or complete distributed-training acceptance record was located in the repository audit; those columns remain **unconfirmed**. PR1's session-level CPU results are recorded in the change description, not promoted to GPU evidence here.

| Route / capability (implemented) | CPU contract coverage | GPU numerical / communication cases | Recorded GPU execution / distributed training |
| --- | --- | --- | --- |
| Native dense / Flex | Layout and dispatch contracts only; no CUDA numerical execution proof | `test_qsa_block_mask_patch.py`, `test_qsa_cp_patch.py` cover mask/Flex behavior and reference comparisons; not complete native dense/Flex dispatch or training coverage | Unconfirmed / unconfirmed |
| Native `id_sparse` | Community dispatch contracts | No standalone native ID-kernel numerical suite identified in this audit; mixed tests exercise its replacement, not this kernel | Unconfirmed / unconfirmed |
| Fused v1/v2/v3 | Switches, tile configuration, v3 workspace guards | `test_qsa_fused_attention.py`: outputs/gradients, v1/v2 geometry, v3 SM103 shared-dS and fallback cases | Unconfirmed / unconfirmed |
| Mixed Grouped | `test_qsa_mixed_{routes,threshold,checkpoint,cudnn}.py`: geometry, routing, boundaries/replay | `test_qsa_mixed.py`: packed layouts, 12/1, 24/2, 6/1, 3/1 and sparse budgets | Unconfirmed / unconfirmed |
| Mixed NV | `test_qsa_mixed_cudnn.py`: geometry rejection, metadata and mocked execution gradients | GPU file of the same name: SM100/SM103, pinned package, 12/1/D256/BF16, FP32 output/QKV-gradient reference | Unconfirmed / unconfirmed |
| Selector | Dynamic-kernel CPU interpreter/offline compilation and dispatch cases | `test_qsa_select_patch.py`; `test_qsa_tp_selector.py` includes a real two-rank NCCL TP2/CP1 test | Unconfirmed / unconfirmed |
| CP local-query / replay | Replay and layout contracts | CP query-ID tests include **single-GPU simulated rank slices**; these do not establish multi-rank communication | Unconfirmed / unconfirmed |
| Routed QKV / HybridEP | `test_qsa_routed_qkv.py`, `test_qwen38_hybridep.py`: CPU routing/adjoint and mocked transport/preflight | CPU mock tests are not NCCL or physical peer-discovery execution records | Unconfirmed / unconfirmed |

Test paths are under [CPU contracts][cpu-tests] and [GPU/backend tests][gpu-tests]. CUDA skips or missing-package skips are not passes. For future evidence, retain the source commit, image/package revisions, GPU architecture, actual TP/CP/SP/PP/EP/ETP layout, backend and selection budget, packed shapes, recomputation settings, numerical/gradient comparisons, and original logs. Distinguish a distributed numerical test from full training and optimizer/checkpoint validation.

The supplied performance chart describes a Blackwell GPU, TP2, local 12Q/1KV/D256/BF16 and explicitly excludes selector/TP from core timing. Its source commit and raw numerical/training logs are unavailable here. It can describe that external benchmark only. `mixed0` and `mixed16k` are not current community-backend values; do not infer their exact settings from their names. `pr7553` and `TileLang` are external benchmark labels, not enabled implementations in this checkout; community validation rejects the old `pr7553`/`tilelang7060` values. Flash is an internal short-document kernel, not an independent global QSA selector.

### Source map

These links identify repository implementations; function names above identify the relevant guards. Update this guide together with the source when a guard changes. Native QSA comes from the base model patch plus operator/projection integration, not from the base patch alone.

- [Native QSA, selected-ID kernel and Stage-2 loss][native]; [build-time operator integration][integration]; [TP/KV projection patch][projection].
- [Environment defaults][env]; [execution dispatch][dispatch]; [fused adapter][fused]; [base fused kernel][fused-kernel]; [v3 specialization][v3].
- [Mixed entry and geometry][mixed]; [Grouped composition][mixed-dispatch]; [NV adapter][nv]; [TP selector][tp-selector]; [selector postprocessing][select].
- [CP/replay][cp]; [routed QKV][routed]; [attention retention][retain]; [HybridEP preflight][hybrid]; [shared model defaults][model-script].

[native]: https://github.com/redai-studio/Relax/blob/main/docker/patch/megatron/20261016-qwen38-models.patch
[integration]: https://github.com/redai-studio/Relax/blob/main/docker/patch/megatron/20261020-model-operators.patch
[projection]: https://github.com/redai-studio/Relax/blob/main/docker/patch/megatron/20261020-qwen35-optimized.patch
[docker]: https://github.com/redai-studio/Relax/blob/main/docker/Dockerfile.cu13
[env]: https://github.com/redai-studio/Relax/blob/main/relax/utils/env.py
[dispatch]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/qsa_dispatch.py
[fused]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/qsa_fused.py
[fused-kernel]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/ops/qsa_fused_attention.py
[geometry]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/ops/qsa_mixed_ops/geometry.py
[v2]: https://github.com/redai-studio/Relax/tree/main/relax/models/qwen3_8/ops/qsa_triton_v2/
[v3]: https://github.com/redai-studio/Relax/tree/main/relax/models/qwen3_8/ops/qsa_triton_v3/
[mixed]: https://github.com/redai-studio/Relax/tree/main/relax/models/qwen3_8/ops/qsa_mixed_ops/
[mixed-dispatch]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/ops/qsa_mixed_ops/mixed.py
[nv]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/ops/qsa_mixed_ops/cudnn.py
[tp-selector]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/ops/qsa_mixed_ops/selector.py
[select]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/qsa_select.py
[cp]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/qsa_cp.py
[routed]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/qsa_routed_qkv.py
[retain]: https://github.com/redai-studio/Relax/blob/main/relax/models/qwen3_8/qsa_attention_recompute.py
[hybrid]: https://github.com/redai-studio/Relax/blob/main/relax/backends/megatron/hybridep_preflight.py
[model-script]: https://github.com/redai-studio/Relax/blob/main/scripts/models/qwen38-flash-next.sh
[cpu-tests]: https://github.com/redai-studio/Relax/tree/main/tests/cpu/backends/megatron/
[gpu-tests]: https://github.com/redai-studio/Relax/tree/main/tests/backends/megatron/

## NV runtime requirements

Use a compatible CUDA 13 Blackwell training environment, including Megatron/Bridge, HybridEP and Flash Attention. The optional dependency is [NVIDIA cuDNN Frontend PR1765](https://github.com/NVIDIA/cudnn-frontend/pull/1765), pinned to commit **627c23f62e860fe2f76d6b38d2efcc2ee84c70cd**, with CuTeDSL **4.8.1**.

The repository's CUDA 13 Blackwell image pins CuTeDSL to 4.6.2 through PIP_CONSTRAINT. A direct installation of 4.8.1 conflicts with that constraint, and creating a venv alone does not remove the inherited environment variable. Build a dedicated Qwen3.8 NV image instead of changing the running base environment. Save the following as Dockerfile.qsa:

~~~dockerfile
# syntax=docker/dockerfile:1.7
ARG RELAX_BASE_IMAGE
FROM ${RELAX_BASE_IMAGE}

RUN python - <<'PY'
import os
import re
from pathlib import Path

source = Path(os.environ["PIP_CONSTRAINT"])
replaced = re.compile(
    r"^\s*nvidia[-_.](?:cutlass[-_.]dsl|cudnn[-_.]frontend)\s*(?:[<=>!~@]|$)",
    re.IGNORECASE,
)
lines = [line for line in source.read_text().splitlines() if not replaced.match(line)]
lines.append("nvidia-cutlass-dsl==4.8.1")
Path("/etc/relax-qsa-runtime-constraints.txt").write_text("\n".join(lines) + "\n")
PY

ENV PIP_CONSTRAINT=/etc/relax-qsa-runtime-constraints.txt
RUN python -m pip install --no-cache-dir --upgrade \
      'nvidia-cutlass-dsl[cu13]==4.8.1' \
      'nvidia-cudnn-frontend @ git+https://github.com/NVIDIA/cudnn-frontend.git@627c23f62e860fe2f76d6b38d2efcc2ee84c70cd' && \
    python -c 'import importlib.metadata as m; assert m.version("nvidia-cutlass-dsl") == "4.8.1"; from cudnn.QSA.api import qsa_forward, qsa_backward'
~~~

This copies the base constraints and replaces only the CuTeDSL and frontend distribution pins: CuTeDSL has an explicit version, and the frontend has an explicit source commit in the installation command. All other constraints, including Torch, NumPy and Transformer Engine, remain in effect. Do not unset PIP_CONSTRAINT globally or edit the original constraints file.

~~~bash
docker build --build-arg RELAX_BASE_IMAGE='<your-cu13-blackwell-image>' \
  -f Dockerfile.qsa -t relax-qsa:pr1765 .
~~~

Use the derived image on every Ray node, including the head. Building the pinned frontend also requires its upstream build dependencies and a compatible cuDNN installation; follow the [upstream installation instructions](https://github.com/NVIDIA/cudnn-frontend/tree/627c23f62e860fe2f76d6b38d2efcc2ee84c70cd). A different package that only exports a module named cudnn is insufficient. Relax does not install or replace the dependency automatically. Grouped remains the framework default and does not import this optional backend.

The build and import checks cover dependency resolution, the selected CuTeDSL version and the QSA entry points. They do not establish GPU numerical correctness, TP/PP recomputation or compatibility of other kernels with the newer CuTeDSL. Validate those paths in the derived image before training; do not treat this Qwen3.8-specific variant as a validated replacement for every Blackwell recipe.

## Select the recipe

Both recipes use 4 nodes × 8 GPUs, 48 layers, global 24Q/2KV heads, TP2/PP4/CP1/EP8/ETP1 with sequence parallelism, BF16, GBS512/MB1, full uniform recomputation over two layers and one MTP layer.

| Recipe | Pipeline layer counts | LR decay iterations | Input |
| --- | --- | --- | --- |
| Pure text | 14 / 12 / 12 / 10 | 996 | Text messages and tool calls |
| Multimodal | 12 / 14 / 14 / 8 | 2023 | Messages with image references |

The text partition places three backbone QSA layers on each pipeline stage. The last stage also owns the MTP QSA layer and output head, so the total QSA counts are 3/3/3/4. The multimodal recipe keeps its separate partition; changing its PP layout is a separate tuning decision.

Run from the repository root. MODEL_DIR is the complete HF checkpoint directory, DATA_DIR is a JSONL file or directory, and SAVE_DIR is a persistent checkpoint root visible on all nodes:

~~~bash
export MODEL_DIR=/models/qwen38
export DATA_DIR=/datasets/text
export SAVE_DIR=/checkpoints/qwen38
export MEGATRON=/workspace/Megatron-LM
export RELAX_HYBRID_EP_COMMIT='<installed-hybridep-commit>'
export RAY_JOBS_ADDRESS='http://<ray-head>:8265'

EXP_NAME=qsa-text-nv NUM_STEPS=50 \
  bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-nv-text-sft-32xgpu-blackwell.sh
~~~

For image-plus-text SFT, supply a multimodal HF checkpoint and data, then use:

~~~bash
EXP_NAME=qsa-mm-nv NUM_STEPS=50 \
  bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-nv-sft-32xgpu-blackwell.sh
~~~

Both use messages and tools fields. The multimodal recipe additionally maps image to images, limits each image to 128 tokens and preprocesses images on training ranks. The text recipe omits those three multimodal options; it does not turn image data into text. Choose a text-only dataset for that recipe.

The scripts delegate cluster setup to scripts/entrypoint/ray-job.sh and then submit a Ray Job. They require an exclusive Ray Serve workspace and refuse an already-running job when invoked directly. They do not provide concurrent-job isolation. Submission blocks by default; set RAY_NO_WAIT=1 only when you want to monitor the returned job separately.

To inspect the effective command without Ray setup, network access, GPU imports or output-directory creation:

~~~bash
DRY_RUN=1 \
  bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-nv-text-sft-32xgpu-blackwell.sh
~~~

## Routing and controlled comparisons

Each recipe sets the same routing on the driver and training actors:

| Setting | Recipe value | Effect |
| --- | --- | --- |
| RELAX_QWEN38_QSA_COMMUNITY_BACKEND | mixed | Keep the existing selector and packed-input path |
| RELAX_QWEN38_QSA_MIXED_LONG_BACKEND | cudnn_pr1765 | Use NV for long documents |
| RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN | 2051 | Documents of at most 2051 tokens use Flash |
| RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN | 0 | Disable the intermediate Flex branch |
| RELAX_QWEN38_QSA_NV_MAX_PACK_TOKENS | 0 | Call NV separately for each long document |
| NVTE_USE_CUTLASS_GROUPED_GEMM | 0 | Keep the independent MoE GEMM option off |
| CUDA_DEVICE_MAX_CONNECTIONS | 1 | Match the TP/sequence-parallel runtime |

The 2051 boundary includes 512 complete four-token blocks and at most three partial-tail tokens. A document of length 2052 enters the sparse long branch. Use the producer's real document lengths, not the total concatenated pack length.

For a Grouped control with the same text recipe, set only the long backend and a distinct experiment identity:

~~~bash
RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=grouped EXP_NAME=qsa-text-grouped \
  bash examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-nv-text-sft-32xgpu-blackwell.sh
~~~

Use distinct EXP_NAME or SAVE_DIR values for separate arms so one does not resume the other's checkpoint. NVTE_USE_CUTLASS_GROUPED_GEMM=1 can be supplied independently for a MoE comparison; it does not select the QSA implementation or prove that a particular MoE kernel ran. The NV mini-pack setting is experimental and remains disabled in these recipes.

## Checkpoints and throughput

The recipes configure load and save at the stable SAVE_DIR/EXP_NAME path, save every SAVE_INTERVAL steps (default 50), and retain two Actor checkpoints. Keep that path stable to resume; do not append a fresh timestamp. ClearML task names and logs get a timestamp for each launch. The scripts configure checkpointing through the existing Megatron backend; a dry run only checks configuration, not successful optimizer/RNG restoration.

Both recipes disable custom record-function tags and autotune logging and do not enable the PyTorch profiler. They use a 120-minute producer queue timeout and zero automatic global restarts. See [Performance Tuning](./performance-tuning.md) for explicit diagnostic profiling.

Compare Actor throughput using **sum of logical training tokens / sum of Actor training seconds / GPU count** over the same step window. End-to-end throughput uses total step seconds instead. Preserve slow steps, list cold compilation and checkpoint saves separately, and compare equal data/topology/packing/recomputation settings. A faster QSA kernel does not imply the same speedup for the full training pipeline.
