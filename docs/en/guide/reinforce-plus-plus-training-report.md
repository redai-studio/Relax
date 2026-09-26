# REINFORCE++ training and numerical validation report

This report accompanies the two algorithm definitions in
[REINFORCE++ and REINFORCE++-baseline](./reinforce-plus-plus.md). It records
the numerical tests and a frozen, equal-budget Qwen3-0.6B comparison against
GRPO. The experiment is a stability and implementation-validation study, not
a claim that one algorithm is statistically superior.

## Scope and evidence boundary

- Proposal: [Task 29 issue #192](https://github.com/redai-studio/Relax/issues/192)
- Sanitized reproducibility evidence:
  [logs, expanded commands, metrics and manifest](https://github.com/zheself/Relax/releases/tag/task29-reinforcepp-evidence-c72caf1)
- Experiment source commit: `5f7cd574372288391bb1c41ca0677422cd31e725`
- Experiment upstream base: `b095ba68ce95c7d98762cf128eab630878f394e6`
- Post-experiment rebase base: `0bc99af8dd39de8fd99c588a98b3f3a463bc818c`
- Model: Qwen3-0.6B
- Training data: GSM8K `train_clean.parquet`, 7,473 rows
- Algorithms: GRPO, REINFORCE++, and REINFORCE++-baseline
- Training repetitions: seeds 42, 1234, and 2026

The GPU results below remain attributed to the exact experiment commit. The
branch was later rebased onto the newer upstream base and the CPU numerical
and regression tests were rerun. No post-rebase GPU result is claimed.

The Git tree does not include checkpoints, data, credentials, cluster paths,
or complete raw logs. The public sanitized evidence release contains the nine
accepted training logs, nine accepted evaluation logs and summaries, all 18
expanded commands, GPU samples, report CSV/SVG files, and an internal SHA256
manifest. Its compressed archive SHA256 is
`0b52d8e8e6a85ff534e16569dd48c9dbef336402a2c80b6aa96b8bf2ffd7834f`.
The private local evidence bundle additionally retains raw TensorBoard event
files, evaluation JSONL files, and checkpoints; none is required to reproduce
the published tables from the released CSV files.

Machine-readable, path-free evidence used by the figures and tables is
published with the documentation:

- [all 450 training-step records](/reinforce-plus-plus/training_metrics_long.csv);
- [per-run training summaries](/reinforce-plus-plus/training_summary_by_run.csv)
  and [three-seed aggregates](/reinforce-plus-plus/training_summary_by_algorithm.csv);
- [per-run evaluation summaries](/reinforce-plus-plus/evaluation_summary_by_run.csv)
  and [three-seed aggregates](/reinforce-plus-plus/evaluation_summary_by_algorithm.csv);
- [paired reward differences](/reinforce-plus-plus/evaluation_paired_reward_differences.csv).
- [accepted job and evidence index](/reinforce-plus-plus/evidence_index.csv),
  including Slurm state, elapsed time, source commit, accepted step/response
  counts, and SHA256 identifiers for the retained primary artifacts.

## Algorithm contract

The comparison intentionally changes the advantage and KL contracts, while
holding the workload fixed:

| Algorithm | Advantage | Normalization | KL regularization |
|---|---|---|---|
| REINFORCE++ | terminal reward plus token k1 KL shaping, followed by reverse return with `gamma=1` | global population moments over all valid response tokens and DP ranks | k1 inside the shaped reward; no independent KL loss |
| REINFORCE++-baseline | reward minus the inclusive same-prompt group mean; no group-std division | the same global valid-token population normalization | separate k2 KL loss |
| GRPO | same-prompt group centering and group-std scaling | existing per-group normalization | separate k2 KL loss |

Both new variants use a token-level PPO clipped surrogate and response-mean
loss reduction. Padding and mask-zero tokens contribute to neither moments nor
loss. The dedicated distributed normalizer uses population variance
(`ddof=0`) and `rsqrt(max(variance, 1e-8))`. A zero-variance population returns
finite zeros; a globally empty mask triggers a coordinated device-side
asynchronous error without extracting a host scalar in the training hot path.

Because REINFORCE++ also changes where and how KL is injected, the comparison
between REINFORCE++ and the baseline variant is an algorithm-package
comparison, not a single-factor baseline ablation. The cleanest mechanism
comparison is baseline versus GRPO: both use independent k2 regularization,
but differ in group/global normalization.

## Environment and frozen workload

| Item | Frozen value |
|---|---|
| Image digest | `dbd4c122f11e2e83f955ceeeadf541573c46f6458c47d892ce74c03794ed317e` |
| Python / PyTorch | 3.12.3 / 2.11.0+cu129 |
| CUDA runtime | 12.9 |
| Ray / SGLang | 2.56.0 / 0.5.12.post1 |
| GPU | 1 x NVIDIA A40 48 GB per run |
| CPU / host memory | 16 CPU / 64 GiB per run |
| Steps | 50 Actor updates per run |
| Batch geometry | 4 prompts x 8 responses; global batch 32 |
| Response cap | 1,024 tokens |
| Dynamic token cap | 4,096 tokens per GPU |
| Learning rate / PPO clip | `1e-6` / `0.2` |
| Reward | Relax rule-based math reward |
| Sampling | temperature 1.0; identical seed set and common settings |

Online evaluation was disabled because repeating the full 1,319-prompt test
set every ten steps would have dominated the training-generation budget. Each
iteration-49 checkpoint was instead evaluated once on the same frozen 256-row
subset, with four responses per prompt and evaluation seed 29. This produces
1,024 responses per checkpoint.

The first materialized subset kept the original GSM8K rationale-form labels
and had SHA256 `9f13d3bb27995a3902a11d879b31b909b3aed6fa3bd5b7a14928a56c313c8db4`.
Relax's math reward expects the scalar target used by the training file, so the
questions and order were preserved while each label was deterministically
converted with `answer.rsplit("####", 1)[1].strip()`. The accepted clean-label
subset has SHA256
`b4a52290777ef180e5af2602e6dfc1614dda35b4b8534109acb13abbccfb4fce`.
The incompatible-label run is excluded and documented below.

## Reproduction entry point

The parameterized recipe supports all three compared algorithms. A portable
formal invocation is:

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
PROMPT_DATA=/path/to/gsm8k/main/train_clean.parquet \
OUTPUT_DIR=/path/to/output/<algorithm>-<seed> \
ADVANTAGE_ESTIMATOR=<grpo|reinforce_plus_plus|reinforce_plus_plus_baseline> \
NUM_ROLLOUT=50 \
SEED=<42|1234|2026> \
ROLLOUT_BATCH_SIZE=4 \
N_SAMPLES_PER_PROMPT=8 \
GLOBAL_BATCH_SIZE=32 \
ROLLOUT_MAX_RESPONSE_LEN=1024 \
MAX_TOKENS_PER_GPU=4096 \
REWARD_NUM_WORKERS=4 \
REWARD_MAX_CONCURRENCY=16 \
SGLANG_MEM_FRACTION_STATIC=0.40 \
USE_HEALTH_CHECK=1 \
bash examples/algorithms/run-qwen3-0.6B-1xgpu-reinforce-plus-plus.sh
```

The recipe supplies the variant-specific k1/k2, normalization, and validation
arguments. The cluster wrapper used one isolated Ray runtime, cache, temporary
directory, output directory, and port set per Slurm job. It set
`PYTHONNOUSERSITE=1` and bind-mounted the host checkout into the immutable
container.

The site-specific Slurm expansion for every formal training was:

```bash
sbatch \
  --partition=ShangHAI --account=hexm-shanghai \
  --gres=gpu:NVIDIAA40:1 --cpus-per-task=16 --mem=64G \
  --time=06:00:00 \
  --export=ALL,TASK29_ALGORITHM=<algorithm>,TASK29_SEED=<seed>,TASK29_NUM_ROLLOUT=50,TASK29_ROLLOUT_BATCH_SIZE=4,TASK29_N_SAMPLES_PER_PROMPT=8,TASK29_GLOBAL_BATCH_SIZE=32,TASK29_MAX_RESPONSE_LEN=1024,TASK29_MAX_TOKENS_PER_GPU=4096,TASK29_SGLANG_MEM_FRACTION=0.40,TASK29_REWARD_NUM_WORKERS=4,TASK29_REWARD_MAX_CONCURRENCY=16,TASK29_USE_HEALTH_CHECK=1,TASK29_USE_EVAL=0 \
  <job-isolated-wrapper.sbatch>
```

The wrapper verified the source commit, model/data/image inputs, runtime module
paths and visible A40 before invoking the portable recipe above. It gave each
job distinct Ray, Serve, cache, temporary and output roots. Site paths and the
wrapper itself are not a portable project API; the complete algorithm
invocation is the recipe command above. The path-free evidence index records
every accepted training and evaluation job, while the SHA256-verified local
bundle retains the expanded commands and raw logs for audit.

## Numerical and regression validation

The independent float64 test references do not call the production return,
advantage, normalization, policy-loss, KL-loss, or reduction functions. They
compare shaped reward, return, raw/normalized advantage, token loss, and
response-reduced loss element by element at `atol=rtol=1e-6`.

Coverage includes:

- response lengths `[1, 3, 5]`, right padding, and an internal mask hole;
- finite, `NaN`, and `Inf` sentinels outside the mask and exact mask-zero
  outputs;
- all-zero reward, nonzero KL, zero variance, and a single valid token;
- baseline `k=1` rejection;
- upper/lower PPO clipping and positive/negative advantages;
- independent k2 token and response-reduced losses;
- a real two-process Gloo process group with real `all_reduce` operations;
- unequal populations, one empty local rank, repartition invariance, one
  global token, zero variance, and a globally empty mask.

After rebasing onto upstream `0bc99af`, the focused suite reported:

```text
55 passed, 2 skipped in 20.37s
```

The two host skips require Megatron imports unavailable in the host Conda
environment. The broader host regression reported:

```text
204 passed, 12 skipped, 3 warnings in 18.61s
```

A post-rebase CPU-only Slurm run in the pinned container reported:

```text
57 passed, 1 skipped, 30 warnings in 22.65s
```

The sole skip is the newly upstreamed FP16 test module, whose standalone
container process cannot import the image's root-owned Megatron checkout. It
is not a Task 29 test. A `git range-diff` marks all five Task 29 commits as
patch-identical before and after the rebase, so the upstream FP16 change was
added without altering the trained Task 29 implementation.

A final pre-review audit hardened boolean masking against non-finite padding,
made a fully masked local response participate safely in global DP statistics,
and added a production-dispatch integration test. The updated focused host
suite reported `62 passed, 2 skipped`; the broader `tests/utils + tests/core`
regression reported `209 passed, 12 skipped, 3 warnings`. The two focused
skips remain upstream modules that require a full Megatron installation. These
post-experiment changes affect only masked/degenerate inputs and monitoring
scalar synchronization; the GPU results remain attributed to the frozen
experiment commit above.

Following maintainer review, the implementation was rebased again onto
upstream `ce650113` and narrowed without changing either algorithm's numerical
contract. Commit `0e1531b` replaces the per-batch `global_count.item()` check
with a device-side asynchronous assertion and a finite device-side denominator.
It also restores `relax/backends/megatron/cp_utils.py` exactly to upstream and
moves non-finite masked-token protection into a reducer wrapper selected only
for the two REINFORCE++ estimators. GRPO, GSPO, SAPO and the other existing
algorithms therefore retain the upstream shared-reducer behavior.

The review-fix validation reported:

- `51 passed` in the focused Task 29 suite, including the real two-process
  Gloo collective;
- `224 passed, 12 skipped` for `tests/utils + tests/core` in the pinned
  container;
- `168 passed, 4 skipped` for the complete Megatron backend suite in the
  pinned container;
- a successful VitePress 1.6.4 production build for both English and Chinese
  pages;
- 100 CUDA iterations under PyTorch synchronization debug mode without a
  host-synchronization error (`TASK29_CUDA_SYNC_DEBUG_OK`).

Two post-review three-step Qwen3-0.6B smoke runs exercised the narrowed
production dispatch on NVIDIA A40 GPUs: REINFORCE++ job `938288` and
REINFORCE++-baseline job `938293`. Both jobs completed all three Actor updates,
passed the structured finite-metric log validator, produced non-empty
TensorBoard, checkpoint, and rollout artifacts, and left an empty job-scoped
Ray cleanup list.

At the experiment commit, the pinned container passed 43 Task 29 tests and 14
metrics tests. All nine formal trainings produced 50 rollout records, 50 Actor
updates, finite TensorBoard metrics, and an iteration-49 checkpoint. All nine
accepted evaluations produced 1,024 responses, a non-empty summary and
TensorBoard file, the `TASK29_EVAL_OK` gate, and an empty job-scoped Ray cleanup
list.

## Training stability

Across-seed dispersion in this report is the sample standard deviation
(`ddof=1`, `n=3`). The stable-training table first averages each run across
steps 40--49, then computes the mean and sample standard deviation across the
three run-level values.

| Algorithm | Last-10 raw reward | Last-10 PG loss | Last-10 independent KL loss | Last-10 grad norm |
|---|---:|---:|---:|---:|
| GRPO | 0.5417 ± 0.0273 | -0.000000 ± 0.000000 | 0.006024 ± 0.000380 | 1.1674 ± 0.0396 |
| REINFORCE++ | 0.5469 ± 0.0188 | -0.111379 ± 0.015769 | N/A (k1 reward shaping) | 1.7015 ± 0.0244 |
| REINFORCE++-baseline | 0.5344 ± 0.0143 | -0.021490 ± 0.010483 | 0.006048 ± 0.002002 | 1.5847 ± 0.0499 |

`train/ppo_kl` is the response-reduced old-policy/current-policy log-prob
difference used by the PPO importance ratio. It is zero in all 450 published
rows for this frozen workload with one Actor update per rollout batch, but
that is not a
reference-policy KL measurement and does not imply that reference
regularization was disabled. REINFORCE++ instead folds its k1 reference term
into the return, while GRPO and REINFORCE++-baseline report an independent k2
term as `train/kl_loss`.

For a direct k1 activation check, the following table subtracts each
same-step TensorBoard `rollout/raw_reward` scalar from `rollout/returns`, then
averages those differences within the indicated interval. All three
REINFORCE++ commands used `--kl-coef 0.01 --kl-loss-type k1`. The consistently
negative, nonzero differences after policy movement show that reference KL
shaping reached the production return rather than remaining a recipe-only
setting.

| Seed / job | All-50 mean difference | Last-10 mean difference | Final-step difference |
|---|---:|---:|---:|
| 42 / 937653 | -0.007241 | -0.017225 | -0.019743 |
| 1234 / 937680 | -0.007396 | -0.018403 | -0.020091 |
| 2026 / 937689 | -0.006929 | -0.017273 | -0.014912 |

The baseline's separately optimized k2 loss is independently visible in the
stability table (`0.006048 ± 0.002002` over steps 40--49). The released
expanded commands and raw logs allow both checks to be recomputed without
access to the cluster.

![Training reward curves](../../public/reinforce-plus-plus/training_reward_curve.svg)

`train/loss`, `rollout/rewards`, and processed advantage magnitudes do not
have a common cross-algorithm meaning. In particular, REINFORCE++ has no
independent KL-loss term, whereas the GRPO and baseline total losses include
one. The next two figures are therefore optimization diagnostics rather than
algorithm-ranking metrics.

![Training total-loss curves](../../public/reinforce-plus-plus/training_loss_curve.svg)

![Independent k2 KL-loss curves](../../public/reinforce-plus-plus/training_kl_loss_curve.svg)

The normalized-advantage standard deviation was exactly 1 at all 150
REINFORCE++ steps. The baseline variant produced finite zero advantages on 7
of 150 steps where the global raw advantage population had zero variance; its
other steps had standard deviation 1. This is the intended degenerate-input
behavior, not a NaN or silent sample loss.

![Normalized advantage standard deviation](../../public/reinforce-plus-plus/training_advantage_std_curve.svg)

## Length, truncation, and efficiency

The all-50-step summaries are:

| Algorithm | Raw reward | Mean response length | Truncation | Response tok/s | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| GRPO | 0.5304 ± 0.0112 | 896.7 ± 6.1 | 0.5358 ± 0.0148 | 409.8 ± 0.7 | 35.33 ± 0.33 GiB |
| REINFORCE++ | 0.5144 ± 0.0087 | 905.9 ± 0.3 | 0.5608 ± 0.0150 | 405.3 ± 1.1 | 35.22 ± 0.09 GiB |
| REINFORCE++-baseline | 0.5346 ± 0.0113 | 896.9 ± 17.2 | 0.5396 ± 0.0328 | 409.2 ± 0.8 | 35.26 ± 0.09 GiB |

Mean Slurm elapsed time was 65.6 minutes for GRPO, 66.9 minutes for
REINFORCE++, and 65.7 minutes for the baseline variant. Throughput, elapsed
time, and peak memory are close; this experiment does not indicate a material
systems-cost difference among the algorithms.

![Mean response-length curves](../../public/reinforce-plus-plus/training_response_length_curve.svg)

![Truncation curves](../../public/reinforce-plus-plus/training_truncation_curve.svg)

![Response-token throughput curves](../../public/reinforce-plus-plus/training_throughput_curve.svg)

The response cap is an important limitation: roughly 54--56% of training
responses and 40--44% of evaluation responses were truncated. Length behavior
may affect the observed quality ordering and must not be hidden by reporting
accuracy alone.

## Fixed-subset final-checkpoint evaluation

Each row below is an independently trained checkpoint evaluated with the
identical prompt order, scalar labels, sample count, decoding parameters, and
evaluation seed.

| Algorithm | Seed | Reward / pass@1 | pass@2 | pass@4 | Truncation | Mean length |
|---|---:|---:|---:|---:|---:|---:|
| GRPO | 42 | 0.6006 | 0.7096 | 0.7773 | 0.3926 | 844.1 |
| GRPO | 1234 | 0.5996 | 0.6973 | 0.7617 | 0.4355 | 854.5 |
| GRPO | 2026 | 0.6230 | 0.7188 | 0.7930 | 0.3711 | 834.1 |
| REINFORCE++ | 42 | 0.6016 | 0.7135 | 0.7734 | 0.4385 | 867.3 |
| REINFORCE++ | 1234 | 0.5752 | 0.6960 | 0.7812 | 0.4443 | 859.6 |
| REINFORCE++ | 2026 | 0.5830 | 0.6927 | 0.7656 | 0.4316 | 855.0 |
| REINFORCE++-baseline | 42 | 0.5859 | 0.6908 | 0.7773 | 0.4639 | 866.6 |
| REINFORCE++-baseline | 1234 | 0.5986 | 0.6927 | 0.7695 | 0.4395 | 867.2 |
| REINFORCE++-baseline | 2026 | 0.6279 | 0.7396 | 0.8203 | 0.3594 | 824.2 |

Aggregate results:

| Algorithm | Reward / pass@1 | pass@2 | pass@4 | Truncation |
|---|---:|---:|---:|---:|
| GRPO | 0.6077 ± 0.0133 | 0.7086 ± 0.0108 | 0.7773 ± 0.0156 | 0.3997 ± 0.0328 |
| REINFORCE++ | 0.5866 ± 0.0135 | 0.7007 ± 0.0112 | 0.7734 ± 0.0078 | 0.4382 ± 0.0064 |
| REINFORCE++-baseline | 0.6042 ± 0.0215 | 0.7077 ± 0.0276 | 0.7891 ± 0.0273 | 0.4209 ± 0.0547 |

![Final-checkpoint evaluation reward](../../public/reinforce-plus-plus/evaluation_reward.svg)

The paired reward differences by training seed are:

| Comparison | Seed 42 | Seed 1234 | Seed 2026 | Mean ± sample SD | Exploratory 95% t-CI |
|---|---:|---:|---:|---:|---:|
| REINFORCE++ - GRPO | +0.0010 | -0.0244 | -0.0400 | -0.0212 ± 0.0207 | [-0.0726, 0.0303] |
| baseline - GRPO | -0.0146 | -0.0010 | +0.0049 | -0.0036 ± 0.0100 | [-0.0285, 0.0213] |
| REINFORCE++ - baseline | +0.0156 | -0.0234 | -0.0449 | -0.0176 ± 0.0307 | [-0.0938, 0.0587] |

These intervals use only three paired training seeds (`df=2`) and all cross
zero. GRPO has the highest mean reward, while the baseline variant has the
highest mean pass@4, but rankings reverse across seeds. The evidence supports
stable execution and an initial fixed-subset comparison, not a superiority or
statistical-significance claim.

## Incidents and resolutions

| Incident | Evidence boundary | Resolution |
|---|---|---|
| Container preflight lacked Ruff | CUDA/import checks passed; pytest had not run | kept Ruff as a host/static gate and reran container pytest successfully |
| Runtime-env JSON gained an extra brace | Ray started; training never entered parameter parsing | replaced Bash default expansion with an explicit empty-value branch and added regression coverage |
| 8/12 CPU smoke allocations could not schedule Rollout or RewardWorker actors | no optimizer step; excluded | formalized 16 CPU per run and bounded reward-worker concurrency |
| Training job 937654 exited `2:0` after all 50 updates | complete TensorBoard, rollout files, checkpoint, success marker, and empty cleanup are retained | replaced an unrestricted `loss...nan` regex that matched generated text containing `Nancy` with a structured log validator; Slurm state remains reported as failed |
| Eval-only startup used an invalid zero-step schedule and then a wrong config path | no accepted generation; excluded | retained a valid parser schedule without calling the training loop and corrected the container path |
| Eval job 937775 used rationale-form GSM8K labels and produced all-zero reward | validates weight sync, generation, metrics routing, and cleanup only; quality metrics excluded | deterministically normalized labels to scalar targets and reran all nine accepted evaluations |

No failed smoke, preflight, or incompatible-label result is mixed into the
formal comparison.

## Limitations and conclusion

1. Three training seeds are sufficient for a repeatability check, not a strong
   significance claim.
2. Evaluation covers one frozen 256-prompt subset and one decoding seed; it is
   not a full GSM8K benchmark.
3. There is no identically evaluated initial-model checkpoint, so this report
   cannot claim improvement over the base model.
4. Formal rollout generation was seeded but not fully deterministic; equal
   seeds fix common inputs and random-source configuration, not token-for-token
   identity across different policies.
5. The high response truncation rate may affect the quality ordering.

Within those limits, both new estimator names satisfy their documented
mathematical contracts, align element by element with independent references,
pass real cross-rank Gloo tests, and complete the same nine-run training and
evaluation budget as GRPO without NaN, Inf, OOM, or unexplained sample loss.
The baseline variant tracks GRPO closely in mean reward and has the highest
mean pass@4 in this small study; standard REINFORCE++ has longer responses and
more truncation. A larger follow-up should prioritize a longer response cap,
more training seeds, the full evaluation set, and an initial-checkpoint
control.
