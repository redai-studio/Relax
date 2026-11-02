# 280G Blackwell runtime migration into the CUDA 13 image

The 280G Blackwell runtime is built by `docker/Dockerfile.cu13`, target `relax`, with
`GPU_ARCH=blackwell`. Do not run the historical Ray-node installer or replace
`/root/Megatron-LM` with an archived directory after building the image.
The `train` target alone does not include the final runtime integration.

This is a source-validated candidate. The delivered archive has been inspected
as described below; this does not establish equivalence of archived native
binaries. A complete Docker build, 280G Blackwell numerical gates, and multi-node
save/resume remain required. CPU tests and successful patch application are not
GPU validation.

The installer is a reference delivery, not a build input: Docker builds fetch
pinned upstream sources and apply the checked-in patches. No `install.sh`,
`activate.sh`, archived site-packages or precompiled 280G Blackwell wheel is required.

Both Dockerfiles apply `20261020-model-operators.patch` after the existing model
patches. This replaces the former `megatron_patch` import-time installers with
explicit MCore/Bridge calls. Retained code lives in `relax/models/common`,
`relax/models/qwen3_5`, and `relax/models/qwen3_8`; model-specific kernels live
under each model's `ops/`. The TP4 hook is now
`relax.models.qwen3_5.recipe.before_step`. Use the migrated checkout together
with a rebuilt image: a checkout alone cannot install the new native entry
points into an older image. Existing feature gates and native fallbacks remain;
this does not establish binary or numerical compatibility across GPU families.

| Image                    | Shared Qwen model source patches | Optimized TP4 recipe                                | Qwen3.8 SGLang rollout                                                         |
| ------------------------ | -------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------ |
| CUDA 13 / 280G Blackwell | Included                         | Build configured; hardware acceptance pending       | Unified SGLang 0.5.21 source                                                   |
| CUDA 13 / Hopper         | Included                         | HybridEP build added; TP4-only dependencies omitted | Unified source; Hopper acceptance pending                                      |
| CUDA 12.9 / Hopper       | Included                         | Not enabled; no CUDA 13 libraries installed         | Existing SGLang 0.5.17 retained; not equivalent to the Qwen3.8 CUDA 13 rollout |

FlashQLA and DeepEP upstream support Hopper, but the delivered Relax recipe
checks CUDA 13 library paths, cuBLAS 13.3 and a specific TP4 topology. Installing
their CUDA 12 packages alone does not port that recipe. CUDA 12 / Blackwell is outside the requested scope. Qwen3.8 27B uses the
Qwen3.5 architecture; the newer Flash-Next/QSA rollout path is the part requiring
the unified SGLang stack. The CUDA 12 image keeps
TE 2.14.1 and its existing attention backends; it does not claim complete
functional or performance parity with either 280G Blackwell delivery.

## Dependency and patch mapping

| Source runtime (`4d9ba79a`)                              | Main image integration                                                                                                                                                                                                                                                                                                                    |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Archived merged Megatron tree                            | Keep Bridge `6862ee170a2569607e8ebc5dde91f119779093ca`, MCore `4a4de865795f123eda422ebf0a4c13d2418cf3cb`, main Megatron patch and bounded checkpoint staging patch. Both patches are applied with `--fuzz=0`.                                                                                                                             |
| HybridEP archive, including compiled extensions          | Build DeepEP's `hybrid-ep` source at `eb9cee7de5a24193bf09500668d3a619d3d3f3fb`, the revision selected by the pinned MCore CUDA 13 CI Dockerfile. Build both `deep_ep_cpp` and `hybrid_ep_cpp` against the image's Torch/CUDA. This is a candidate compatible upstream source, not an identification of the archived binary's provenance. |
| HybridEP variable-token padding                          | Already in the pinned MCore. Restore Relax provider forwarding and the launch flag; do not overlay an older dispatcher. CPU tests cover group MAX/alignment, empty local rows, growing/shrinking batches, combine trimming and gradients.                                                                                                 |
| GDN headwise CP, pre-rule fusion and convolution padding | Already in the pinned MCore. Keep current CP/MTP contracts and restore missing provider fields. No downgrade to old GDN monkey patches.                                                                                                                                                                                                   |
| FLA `0.4.2` archive                                      | Keep main's `fla-core==0.5.2` / `flash-linear-attention==0.5.2` and KDA backward precision patch. Install all three static convolution NB entries via a versioned patch.                                                                                                                                                                  |
| `audioread==3.1.0`, TE `2.18.0`                          | Already pinned in main's CUDA 13 Dockerfile. Validate required APIs; do not install runtime-local Python overlays.                                                                                                                                                                                                                        |
| Torch `2.11.0+cu130`, Triton `3.6.0` assertions          | The CUDA 13 build preserves packages from its SGLang base via build constraints; provide an immutable BASE_IMAGE digest for reproducible builds. Record actual versions in `/etc/relax-cu130-build.json`; do not independently replace Torch with old archive requirements.                                                               |
| Triton cache ZIPs tied to previous nodes                 | Do not ship historical machine caches. Keep the compiler available and allow the built runtime to create its own caches.                                                                                                                                                                                                                  |
| Runtime cuDNN/path overrides                             | Use the CUDA 13 image's libraries and build-time CCCL include path. Do not inject archived runtime paths or `/opt/relax/cudnn` overrides.                                                                                                                                                                                                 |

The HybridEP build uses `nvidia-nvshmem-cu13==3.4.5`, following the same pinned
MCore CI recipe. HybridEP's multinode extension is disabled: each ETP×EP group
must fit one NVLink domain. NCCL/EFA still carries other distributed collectives.
Ordinary DeepEP remains available from the same package. CUDA 13 Hopper builds the same HybridEP revision for SM90; CUDA 12
does not currently build this revision. The CUDA 12 image therefore does not
claim the reference Qwen3.8 HybridEP recipe is available.

The main patch's hybrid-optimizer hunk had insufficient trailing context for
strict GNU patch application. Only two context lines and the hunk counts were
corrected; the intended optimizer logic is unchanged.

## Build and inspect

From the repository root, build locally without publishing:

```bash
docker build -f docker/Dockerfile.cu13 --target relax \
  --build-arg GPU_ARCH=blackwell \
  -t relax:blackwell-runtime-candidate .
```

Both Dockerfiles install all three convolution entries in one image. There is
no `FLA_CONV_NB_MODE` build argument. Set `RELAX_FLA_CONV_NB_MODE` to `original`,
`pow2`, or `none` before starting each training process. `auto` (the default)
selects `none` when `RELAX_GPU_ARCH=blackwell`, otherwise `original`. Selection is
frozen on first FLA convolution import; restart workers to change it. Each entry owns its
autotuner, and kernels are JIT compiled only when used. Kernel arithmetic is
identical across entries. The optimized Qwen3.5 TP4 recipe requires `none`.

For example, set `RELAX_FLA_CONV_NB_MODE=pow2` in the training workers' environment
or pass `-e RELAX_FLA_CONV_NB_MODE=pow2` to `docker run`. With Ray, propagate it
through the workers' runtime environment; exporting it only in the submitting
shell may not reach existing workers. `RELAX_QWEN38_FLA_CONV_CACHE_REUSE=1` remains
available for the `original` entry and is also frozen at first import.

HybridEP and isolated DeepEP v2 are built in separate wheel stages based on the
same `TRAIN_IMAGE`. Changing application patches does not invalidate these
stages; changing the v2 namespace patch does not rebuild HybridEP. The shared CUDA 13 runtime stage mounts the HybridEP wheel, and the final
280G Blackwell stage mounts the isolated v2 wheel during installation without adding a second copy
to the image. Builder and runtime retain identical Torch/CUDA ABI and
NCCL/NVSHMEM installation paths because the extensions record those paths.
Do not reuse these wheels in the CUDA 12 image.

Use BuildKit and reuse a digest-pinned `TRAIN_IMAGE` built from this Dockerfile.
For CI, provide `--cache-from type=registry,ref=<cache-ref>` and
`--cache-to type=registry,ref=<cache-ref>,mode=max` to `docker buildx build`;
use separate cache references for CUDA and GPU variants. Pip cache mounts speed
up downloads on persistent builders; registry layer caches do not by themselves
transfer those mutable mounts. JIT kernel warmup remains a separate hardware
validation step, not a prerequisite for the image build.

The final build checks versions, compiled-extension presence, both DeepEP APIs,
MCore configuration fields, shifted-label MTP support and all three FLA entries.
It writes `/etc/relax-blackwell-runtime.json`; that report explicitly records
`gpu_validated: false`. `python /opt/relax/check_gpu_imports.py` then checks dynamic
CUDA imports when running the image on a GPU host.

## Tests and acceptance

CPU source tests take explicit source directories; none downloads dependencies or
starts Ray jobs. `BLACKWELL_FLA_SOURCE_DIR` is the extraction root containing `fla/` from
both pinned 0.5.2 wheels. `BLACKWELL_MEGATRON_SOURCE_DIR` is the assembled main source
with both patches applied. `BLACKWELL_HYBRIDEP_SOURCE_DIR` is the pinned DeepEP checkout.

```bash
PYTHONPATH=. pytest -q tests/tools/test_blackwell_runtime_build.py \
  tests/scripts/test_qwen_moe_runtime_preflight.py tests/scripts/test_qwen_moe_entries.py \
  tests/backends/megatron/test_model_provider_vpp.py
```

Inside the image on explicitly allocated 280G Blackwell hardware:

```bash
RELAX_RUN_BLACKWELL_GPU_TESTS=1 PYTHONPATH=. pytest -q \
  tests/integration/test_blackwell_runtime_numerics.py
```

Run those numerical gates separately for each FLA variant intended for deployment.
They compare packed convolution outputs/input/weight/bias gradients with an
independent PyTorch convolution and TE CE with native PyTorch CE.

Remaining distributed gates: ordinary DeepEP and HybridEP dispatch/combine and
backward; unequal rank token counts; a small, larger, then smaller microbatch to
exercise buffer growth; CP1/CP4 with MTP0/MTP1; then actual 280G Blackwell training and
checkpoint save/resume. The CPU buffer test verifies requested capacity updates,
not CUDA allocation or distributed synchronization correctness.

## Original archive source audit

The source manifest identifies these archives by SHA256:

- `megatron.tar.gz`: `5d780ccd3fd5ebd1310a75b50536a1d9d4397f6d90a9e89ca57d751a7f89b1f5`
- `fla.tar.gz`: `053c0d29df0a9546ece1961e667cde3a45a4e148d2d0585648c208a3fbc12bfb`
- `hybridep.tar.gz`: `d38671a0180269766eb07d9e972dd97aacc63771a4dab1dec6fc09e88ac942d7`

The source entries received in `qwen35-36-install-20261020.zip` were checked
against the bundle CRC/SHA256 manifest; `DELIVERY_COMMIT` matches `3fb0859060e5`.
MCore's eight protected source files were compared with the current patch stack.
Keep current checkpoint, RNG, routing and Qwen3.8 fixes. The necessary grouped-MLP
compatibility change is recipe-scoped: normalize expanded expert counts to CUDA
int64 without enabling the generic 256-token padding path.

The HybridEP archive identifies `1.2.1+10d4dd7`. Its Python sources match that
revision except for emptying the CUDA cache and logging memory on buffer growth.
The target recipe uses DeepEP v2 and rejects HybridEP; this inactive older change
is not ported over the newer canonical implementation. Binary-only private C++
changes cannot be excluded from the archived extension.

FLA 0.4.2's required interfaces were compared with 0.5.2. The local convolution
backward adapter now passes the three new `dy` strides. Keep the current KDA
precision patch, static none-NB entry and empty-segment chunk-index fix. These source
checks do not establish GPU numerical or performance equivalence.

## Optimized TP4 delivery (3fb0859060e5)

The same final 280G Blackwell image also builds these recipe dependencies:

| Dependency   | Fixed source/version                               | Coexistence                                                                  |
| ------------ | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| DeepEP 2.1.0 | `099d5f2bad488b9c534ea785062b12f2e91d1d41`         | `relax-deep-ep-v2` / `_relax_deep_ep_v2`; canonical DeepEP/HybridEP retained |
| FA4          | `56e48e3cc3557cfb9f45ec4763c5f971e17cbcbc`         | `_relax_fa4_pr2968.cute`; FA2 retained                                       |
| CUTLASS DSL  | `4.6.2`, CUDA 13 extra                             | 280G Blackwell image dependency; no worker-side alias mutation               |
| cuBLAS       | `nvidia-cublas==13.3.0.5`                          | Libraries in `/opt/relax/cublas13.3`; preloaded only by the optimized recipe |
| FlashQLA     | v0.1.3, `5538c35f0df33ef804eacd90c77d9d9087c1d4e4` | KKT TMA lowering disabled by static patch for TileLang 0.1.12                |

FlashQLA's exact revision comes from the received provenance manifest; its source
files match the archive byte for byte before applying the KKT patch. FA4's upstream
tree matches all 57 recorded hashes. All 51 isolated Python implementations match
the archive after AST normalization; cosmetic formatting is not copied. The build
checks those AST hashes in addition to the upstream hashes and records the results.
Neither source comparison replaces GPU validation.

MCore patches apply in this order: main version patch, bounded checkpoint staging,
Qwen3.8 model patch, then `20261020-qwen35-optimized.patch`. The last patch merges
MTP selection, adaptive recomputation, KV pair gather/QKV extraction, shared gate,
finite gradient norm, DDP delayed wgrad and DeepEP manager selection. These paths
are default-off and selected by the optimized recipe. CuTe API compatibility is
static. Existing Qwen3.8/CP/MTP mask and checkpoint fixes remain in the stack.

Instance-dependent bindings remain for attention/GDN kernels, grouped MLP,
projection overlap, rotary and expert recomputation. They validate topology before
installation; no AST/exec replacement, source-tree overlay or startup pip install
is used. `/etc/relax-qwen35-runtime.json` records source hashes and native versions.
Build checks allow only the base image's documented Torch/NCCL metadata conflict;
other missing or incompatible dependencies fail the build.
Copied FA4 sources have no pip distribution metadata, so the build also checks
their pinned `pyproject.toml` requirements explicitly. GPU import validation
loads legacy HybridEP, isolated DeepEP v2, FA4 and FlashQLA and verifies the
actual NCCL mapping. Legacy MCore imports only discover the isolated v2 package;
its NCCL/JIT initialization is deferred until an elastic buffer is requested.

Run the optimized recipe through the standalone shell entry:

```bash
MODEL_DIR=/shared/model DATA_DIR=/shared/data.jsonl SAVE_DIR=/shared/checkpoints \
  bash scripts/training/sft/run-qwen35-optimized-64xblackwell.sh
```

The script sources the shared model definition from `scripts/models/` and keeps
training parameters in shell arrays. There is no separate JSON preset or Python
launcher. It supplies the recipe hook and identical native-library environments
to the Ray driver and actors. Set `DRY_RUN=1` to inspect without submitting;
submission otherwise blocks unless `RAY_NO_WAIT` is nonempty. Additional training
arguments can be appended to the command.

`EXP_NAME` is stable: load/save both use `SAVE_DIR/EXP_NAME`, save interval is 50,
and at most five actor checkpoints are retained. Building the image alone does
not enable this optimized recipe for other training scripts.

The retained recipe requires TP4/PP1/CP1/EP8/ETP1, DP16 and GBS1600 before
selecting private binpacking and whole-pack exchange. Evaluation uses the
existing sampler. Async CPU prepacking is rejected; raw data prefetch is
supported. Other recipes retain their current packing behavior.

The optimized kernels require 280G Blackwell training validation: masked/empty MTP and main
loss gradients, dispatch/combine backward, short/oversize packs, throughput and
peak memory, and checkpoint replay. Repeat the existing Qwen3.8, 397B CPT and 35B
multimodal SFT acceptance runs against this final image. Hopper keeps its existing
dependency branch and also needs its normal smoke/regression run.

## Build matrix re-audit (2026-10-20)

The two remote branch heads above were fetched again and are unchanged. The ZIP
installer and activation/preflight scripts match the Qwen3.5 reference branch.
FA4 upstream source hashes, isolated Python AST hashes, FlashQLA source files,
and both cuBLAS library hashes match the delivery. CuTe 4.6.2, DeepEP 2.1.0,
NCCL 2.30.7, NVSHMEM 3.4.5 and TE 2.18.0 are accounted for.

The unified image uses TileLang 0.1.12, Quack 0.6.4 and FLA 0.5.2, whereas the
delivery was validated with 0.1.11, 0.6.3 and 0.4.2. Its historical numerical
acceptance does not transfer to this image. The FlashQLA patch now also checks
the actual compiled KKT source for unexpected TMA stores before launch, cached
per kernel specialization. CUDA 13 Hopper includes HybridEP but still requires
its own build and GPU acceptance; the 280G Blackwell TP4 recipe is not enabled there.

This audit environment has no Docker/BuildKit executable. Patch replay and CPU
checks cannot establish that all three image builds succeed. CUDA 12 Hopper
retains SGLang 0.5.17 and TE 2.14.1; complete Flash-Next/QSA rollout and HybridEP
parity remain unimplemented in that image.

ModelOpt 0.44.0 is installed with `--no-deps`; its `hf` extra declares
Transformers `<5.6`, which does not cover the unified 5.16.0 installation.
A successful `pip check` does not validate that optional integration. ModelOpt
HF functionality therefore requires separate compatibility validation.
