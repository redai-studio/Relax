# FLOPs scheduling experiments

These tools compare the existing length-balanced, dynamically packed baseline
with `--balance-by-flops`. They use real OpenMathReasoning-mini examples and
separate numerical export, timed training, and diagnostic profiling.

Use an environment matching the repository's Dockerfile, including its pinned
Megatron/Bridge sources and TransferQueue `assignment_transform` patch. Record
the actual installed versions; a moving image tag is insufficient. The tools
use existing project dependencies and do not install packages or rent hardware.

For the older public CUDA 12 training image, `bootstrap-cu129.sh` updates the
pinned source and Python layers while constraining existing native ABI packages.
Use the immutable image
`ghcr.io/redai-studio/relaxrl@sha256:cd431e1094646c347aa97bc9cb8c5ac8315d4e72b0b6321f1cafd1ae12e680c6`.
Stage sources on a CPU host with `bash scripts/tools/flops_benchmark/bootstrap-cu129.sh /persistent/runtime --stage-only`; transfer that directory to the GPU host and
run the same command without `--stage-only`. Set `MEGATRON` to its `merged`
subdirectory and include it in `PYTHONPATH`. Set `RELAX_FLOPS_IMAGE_DIGEST` to the
image above. Install before starting the dedicated Ray cluster. This recipe
preserves CUDA, Torch, Triton, TE and FlashAttention; it is intended for Ampere,
Ada or Hopper with an appropriate CUDA 12.9 driver, and does not establish
Blackwell compatibility. Keep the generated source hashes, ABI inventory,
package freeze and bootstrap log with experiment evidence.

## Prepare the workloads

Set `MODEL_DIR`, `DATA_DIR`, and `SAVE_DIR` to persistent directories accessible
to every worker. Start from the repository root. Freeze the Dense workloads:

```bash
python -m scripts.tools.flops_benchmark.prepare \
    --model Qwen/Qwen3.5-4B \
    --data-dir "$DATA_DIR" \
    --recipe stratified
python -m scripts.tools.flops_benchmark.prepare \
    --model Qwen/Qwen3.5-4B \
    --data-dir "$DATA_DIR" \
    --recipe natural
python -m scripts.tools.flops_benchmark.prepare \
    --model Qwen/Qwen3.5-4B \
    --data-dir "$DATA_DIR" \
    --recipe natural \
    --batch-size 128 \
    --capacity 8192 \
    --steps 50
```

Prepare MoE with the first command, replacing the model with
`Qwen/Qwen3.5-35B-A3B`. Download each model into its corresponding directory
under `MODEL_DIR`, using the exact `model_revision` in the generated manifest.
For independent reproduction, pass both `--model-revision` and
`--dataset-revision` from that manifest to the preparation command.

The stratified recipe samples without replacement: one example of 10,000–11,200
tokens and fifteen of 1,750–2,250 tokens per batch. Natural sampling only filters
empty supervision and examples exceeding the specified capacity. Neither
recipe truncates examples. Lengths and masks come from `SFTStreamingDataset`.
The output file inverts the loader's epoch-zero permutation; preparation then
reads the file through that loader and verifies every batch's IDs, tokens and
masks. The manifest includes source revisions, file hashes, sampling seed,
length distribution and per-step effective token counts.

## Run on a dedicated cluster

Set `RAY_ADDRESS` to the confirmed GCS address and `RAY_JOB_ADDRESS` to the
confirmed Jobs HTTP address. The scripts use the existing `ray-job.sh` entrypoint
with broad cluster cleanup disabled. The experiment runner requires an idle,
dedicated cluster, cleans Serve state between its jobs, and retains submission
IDs, terminal status and full driver logs. It stops its own timed-out job.

```bash
python -m scripts.tools.flops_benchmark.run --model dense
# On the separate eight-GPU machine:
python -m scripts.tools.flops_benchmark.run --model moe
```

Dense uses TP1/DP4; MoE uses TP2/DP4/EP8/ETP1 with alltoall. Both use BF16,
FP32 gradient accumulation, recomputation and CPU optimizer offload. ClearML
defaults to offline mode so experiment logs require no account credentials.
The two launch scripts expose `FLOPS_BENCH_MODE`, `FLOPS_BENCH_ARM`,
`FLOPS_BENCH_LAYOUT`, `FLOPS_BENCH_WORKLOAD`, `FLOPS_BENCH_BUDGET`,
`FLOPS_BENCH_END_STEP` and `FLOPS_BENCH_LOAD_NAME`; training parameters remain
visible in their argument arrays. Submission is blocking by default; the runner
explicitly requests nonblocking submission and monitors completion.

The runner first trains a baseline for 20 steps and fingerprints its complete
native checkpoint. Numerical arms restore that checkpoint and execute step 20.
The native checkpoint view omits the SFT cursor, allowing the production replay
path to restore the same accepted batch when changing the regrouping budget.
Each arm's actual data-state batch hash must match the frozen manifest. Scheduler
state is restored with `--use-checkpoint-opt-param-scheduler`; warmup is the only
phase that writes a complete model checkpoint.

The numerical order is baseline, baseline repeat, baseline with budget 12,288,
then FLOPs. A failed control stops the sequence. `tolerances.json` is fixed before
capture and its hash is checked during comparison. Capture exports each actual
optimizer-owned gradient shard after reduction/unscale and before clipping, plus
the FP32 master-weight difference after one successful update. Comparison checks
all ranks, complete parameter coverage, identical initial master weights,
Adam moments/steps, optimizer settings, scheduler, RNG and full-batch loss.
The actual consumed token/mask hashes and sample multiplicity must match the
frozen batch on every model-parallel replica and virtual pipeline chunk.
It never substitutes BF16 weights converted to FP32 for
optimizer master weights. Keep the warm checkpoint immutable until all arms finish.

Main performance experiments use three pairs, alternating A/B and B/A order.
Each arm restores the same checkpoint, runs 30 steps, and discards the first ten.
The Dense suite also runs a 60-step natural-distribution pair and a 30-step
GBS128/8K pair. The primary throughput is summed supervised tokens divided by
summed training-loop time, including input wait. Every step's reported input
and supervised token counts must match the manifest. Profiling and numerical
capture are disabled during these runs.

Use `--suite numeric`, `performance`, `audit`, or `coverage` to run a particular
phase. Performance and coverage first require the main numerical gates to pass.
Completed jobs and numerical reports with matching inputs can be reused. A failed job requires
inspection; use `--name retry1` for a separate attempt after a fix. Never combine
results from different code, scripts, model/data versions or environments in a
performance pair. Dense coverage includes TP2, PP2, PP2/VPP2 and CP2, each with
DP2 and its own warm checkpoint. VPP enables overlap P2P; GDN CP uses headwise
zigzag packing and per-token normalization.

## Evidence and storage

The evidence directory under `SAVE_DIR` contains environment versions, GPU
topology, job metadata, raw logs, tensor comparison reports and paired throughput
summaries. Capture files contain full gradients and FP32 updates; a 35B model
requires roughly 280 GB per arm before checkpoint and temporary storage. The runner
compares files sequentially, preserves reports and shard manifests, then releases
each successfully compared candidate's tensors. It retains the baseline tensors.
Completed reports can be reused after checking their input hashes; a new comparison
requires every tensor export. Allow at least 1.2 TB for MoE checkpoints, weights,
the reference and one candidate, plus temporary storage headroom.

Diagnostic runs save executed microbatch indices, physical token counts, dummy
markers and rank timings, together with PyTorch profiler traces. These runs
include synchronization and logging overhead and must not enter throughput
summaries. Inspect collective waits in the traces alongside changes in pack
counts and rank workloads.

For the alignment ablation, use a separate checkout of `22678ed`, copy the same
benchmark tools and launch scripts into it, and run the Dense natural workload
with FLOPs enabled from the same warm checkpoint. Record both source revisions
and the instrumentation diff. This isolates the subsequent aligned repacking
fixes. Preserve all baseline and candidate results, including regressions.

The scheduling reference is [slime PR #2017](https://github.com/THUDM/slime/pull/2017).
Relax adapts global packing to a queue-carried per-step plan, checks token
capacity, repacks at the aligned count, and validates plans across training
groups. Its existing length-balanced baseline must remain enabled in comparisons.
