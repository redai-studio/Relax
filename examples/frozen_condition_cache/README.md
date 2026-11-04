# Frozen condition cache: SFT reproduction

Compare online conditioning with an exact, bounded CPU cache on SenseNova-U1 and Qwen-Image-Edit. Both paths use Relax's `FSDPSFTTrainer`: data preparation, forward/backward, full-batch gradient accumulation, reduction, clipping, AdamW and scheduler. The local actor supplies sample IDs without Ray/TransferQueue; measured time excludes that orchestration and checkpoint I/O.

Run the commands below from `examples/frozen_condition_cache/`. GPU selection and input/output directories are explicit; no cluster service is required.

## Environment

The measured recipe used Python 3.12, Torch 2.9.0/CUDA 12.8 and the versions in `requirements.txt`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Use GPUs with enough memory for the selected full model and LoRA training. The example does not require a particular GPU model or physical index. Default host cache capacity is 64 GiB; leave room for model loading, data and checkpoints. The public model/data download is about 97 GB.

## Prepare and validate

```bash
python -m runner.campaign prepare --assets /path/to/assets --workers 4
python -m runner.campaign validate --prepared /path/to/assets/prepared.json
```

`assets/manifest.json` pins model/data revisions, file sizes and SHA-256 values. Preparation verifies downloads and exports the original MagicBrush image bytes into disjoint `debug8.json`, `high128.json` and `low1024.json` subsets. `validate` reads back model and exported image hashes. Run `prepare` again after moving assets to regenerate the absolute paths in `prepared.json`.

## Run and summarize

```bash
python -m runner.campaign run \
  --prepared /path/to/assets/prepared.json \
  --work /path/to/results --devices 0 --world 1
python -m runner.campaign summarize --work /path/to/results
```

The default run is sequential: eight numerical trajectories, two comparisons and eighteen timing segments. Each model compares online A1/A2 with cold and warm cache B from the same initial LoRA weights, Adam state, scheduler, RNG and eight examples. Each performance workload has three A/B pairs in AB/BA/AB order, with 16 warm-up and 64 measured updates. Workloads cover repeated SenseNova and Qwen editing pairs and a SenseNova single-pass, low-reuse case. All repetitions and regressions appear in `summary.json`.

The recipe fixes global batch 8, LoRA rank/alpha 32, BF16 compute with FP32 trainable masters/reduction, seed 20261030, Torch SDPA, deterministic execution, and the same image preparation for A/B. A passes cache capacity 0; B uses 64 GiB divided by world size. The adapter reads these through `RELAX_CONDITION_CACHE_BYTES`, `RELAX_CONDITION_CACHE_DEVICE` and `RELAX_CONDITION_CACHE_IDENTITY`. Low reuse can cost more than online encoding; use its reported result when deciding whether to enable caching.

`--devices 0,1 --world 2` and world 4 are supported for explicit distributed runs; world 1 is the default comparison. Keep hardware, software and world size the same within A/B pairs. The default stage timeout is three hours and total run limit is 24 hours; both are configurable with `--stage-timeout-seconds` and `--wall-hours`.

Use the same command with `--resume` after an infrastructure interruption. Completed receipts are checked before skipping stages; unfinished stage directories are retained under `failed/`. Resume requires unchanged source, assets, recipe and devices, reuses the initial state, and keeps the original deadline. Use a new result directory for a changed experiment.

## Evidence

`stages/*/trajectory-rank*.pt` retains loss, complete accumulated gradients, FP32 parameter updates and state digests. The comparison requires per-tensor relative L2 error at most 1%, exact zero error for zero references, finite values, and matching RNG/sample/state digests. Components that were bitwise equal in A1/A2 must remain bitwise equal with caching. Cache payload dtype, shape and bytes are checked on actual warm-cache hits.

`result.json`, per-stage logs, resource/cache snapshots, native checkpoints, `plan.json`, `environment.json` and `summary.json` remain in the result directory. Numerical checks cover the registered batch and seed; they do not measure convergence or image quality. No result is uploaded automatically.

From the repository root, run the CPU regression suite with:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/cpu/examples/frozen_condition_cache -q
```
