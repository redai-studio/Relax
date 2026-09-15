# Trajectory Replay

> Status: the offline replay core (PR A) and production capture (PR B — `loss.policy` hot-path hook, async `CaptureManager` and GPU smoke) are implemented; the distributed layout (PR C) is landed. Capture is split into two bundle types: **rollout-level** (`reward.raw → reward.post_process → advantage.kl → advantage.estimate`, identity = `rollout_id`) and **step-level** (`loss.policy`, identity = `(rollout_id, step_id)`); `advantage.kl` uses rollout-vs-reference semantics and requires a `ref_log_probs` payload. Bundles are produced by the training side or built programmatically via `BundleWriter`.

Trajectory Replay is a verifiable offline reproduction tool for async training. It packs the data, stage outputs and identity information that one training step actually consumed into a **self-describing, verifiable replay bundle**, then — on a plain CPU host without Ray Serve, SGLang, a rollout worker or a GPU — recomputes the reward, advantage, return and loss stages with the same stage semantics and reports the **first verifiable divergence**.

Model forward, backward, gradient and optimizer steps are outside the replay boundary. The system stores the compact post-forward statistics the loss actually consumes (current/old log-probabilities, entropy, values, masks, advantages, returns), not full logits or a checkpoint.

## Why it exists

In synchronous training a rollout step and an Actor update are roughly one-to-one; in fully-async / hybrid modes this no longer holds:

- rollout can lead the actor by several weight versions;
- one rollout partition may be split into multiple TransferQueue consumer batches;
- dynamic batching changes micro-batch boundaries;
- one Actor update may consume samples from multiple rollout partitions.

Replay therefore treats "generated together", "normalized together" and "consumed together" as independent identities. The historical PR #65 bug (streamed prompt groups normalized against the physical batch) is the canonical case of "physical batch ≠ semantic cohort": the final loss changes, but the **first faulty stage is `reward.post_process`**. Replay exists to locate that kind of error, not just to compare the final scalar.

## Bundle layout

A bundle is a directory:

```text
<bundle>/
├── manifest.json    # versions, producer, stage contracts, payload specs + checksums, comparison policy
├── index.json       # identity + per-sample records + recompute config
├── expected.json    # JSON expected outputs per stage (scalars / lists)
├── payloads/        # tensor payloads (inputs and expected tensors)
│   ├── old_log_probs.pt
│   ├── log_probs.pt
│   ├── entropy.pt
│   ├── kl.pt
│   └── advantages.pt
└── COMPLETE         # completion sentinel (COMPLETE.<rank> + final COMPLETE for multi-rank)
```

- **Text goes through JSON**: `index.json` holds only JSON-compatible metadata (sample_id, group_index, lengths, loss_mask, raw_reward, ...). Raw prompt/response text is replaced by hash/truncation per the redaction policy.
- **Tensors use `weights_only=True`**: `payloads/` accepts only `torch.Tensor`; loading is `weights_only=True` and recursively rejects Python objects.
- **Integrity**: the writer stages into a temp directory, writes payloads and manifest hashes, then atomically renames. A missing `COMPLETE`, payload, rank shard, or a checksum mismatch fails before any stage runs.

## Identity model

Replay identity is two-layered:

- **Logical identity**: sample, semantic group, normalization cohort, actor step.
- **Physical provenance**: rollout partition, consumer batch, micro-batch, rank shard, weight lineage.

The training-step identity anchor is the **cohort an Actor update actually consumed**. `actor_step_id` is the `(rollout_id, step_id)` tuple (the `train_one_step` coordinate). The derived `accumulated_step_id` scalar is **not** a persisted identity: under dynamic batching and streaming schedules `num_steps_per_rollout` can vary per rollout.

Selectors fail closed when a membership mapping is missing instead of inferring a semantic group from a physical batch size — the root cause of PR #65.

## Stage contracts and the V1 capability matrix

The pipeline is split into independently versioned stages:

```text
sample -> reward.raw -> reward.post_process -> advantage.kl -> advantage.estimate -> loss.policy
```

Each stage declares a capability:

| Capability | Meaning |
| --- | --- |
| `recompute` | inputs and implementation are complete; can be recomputed and compared offline |
| `recorded-only` | only the production output is viewable; the producer cannot be reliably recomputed |
| `inspect-only` | partial inputs or summaries are viewable; contract incomplete |
| `unsupported` | the stage is not supported by capture or reader |

**Frozen V1 capability matrix**: only **GRPO with `CP=1`** declares `recompute` for `sample / reward.raw / reward.post_process / advantage.kl / advantage.estimate / loss.policy`. Every other topology (`CP>1`, PPO, SAPO/CISPO, OPD, agentic flattened) is `unsupported` and fails loudly in the runner — no best-effort guessing or silent downgrade.

The advantage/loss adapters **reuse the production kernels** (`compute_approx_kl` / `get_grpo_returns` / `compute_policy_loss` in `relax/utils/training/ppo_utils.py`) as the single source of truth. `reward.post_process` is reimplemented offline (the production function lives in a module that imports Ray at module scope) and is marked `implementation="reimplemented"`, pinned by the PR #65 fixture.

## CLI usage

```bash
# Show identity, capability, producer and payload summary (works on incomplete bundles).
python -m relax.tools.trajectory_replay inspect <bundle>

# Validate format, integrity, safety and dependency closure (no numerical replay).
python -m relax.tools.trajectory_replay validate <bundle>

# Recompute stages and compare against expected outputs; exit 0 on pass, 1 on divergence.
python -m relax.tools.trajectory_replay replay <bundle> [--stage all]

# Select a single sample / group / micro-batch (repeatable), or assert the step coordinate.
python -m relax.tools.trajectory_replay replay <bundle> \
    [--sample s-0] [--group g-1] [--batch mb-0007] [--step 120:0] [--rollout 120]
```

The `replay` report gives the first divergent stage, sample, field, token offset, expected/actual values and max absolute error. Skipped stages (`recorded-only` / `inspect-only` / `unsupported`) never count toward the first-divergence determination.

**Selection granularity**: `--sample` / `--group` / `--batch` may be combined. Selecting any sample or micro-batch expands to its full semantic-group closure (reward/advantage normalization is group-level) and fails closed on missing membership. Under a partial selection, per-sample/per-token stages (sample → advantage.estimate) recompute normally while the cohort-level `loss.policy` stage is skipped — a subset scalar cannot be compared to a full-cohort expected value. `--step ROLLOUT_ID:STEP_ID` accepts an exact `actor_step_id=(rollout_id, step_id)` match only; use `--rollout ROLLOUT_ID` for a rollout-level bundle. The two flags are mutually exclusive. If the path is a capture directory (several bundles or `rank-*` children) it picks the matching one. `--batch` selects by DataIterator micro-batch index (`mb-0000`, `mb-0001`, …), not by actor `step_id`.

Example (the PR #65 bug):

```text
bundle b-00001 — first divergent stage: reward.post_process
[   pass] sample
[   pass] reward.raw
[   fail] reward.post_process — normalized reward mismatch in 4 sample(s)
          reward s-0: expected=-5.5 actual=-1.0 abs_err=4.5
          ...
```

## Enabling production capture

Capture is off by default and does not change training numerics. Turn it on with
environment variables (no CLI argument-parsing change). The actor calls
`maybe_enable_for_actor` from `MegatronTrainRayActor.init`, which enables capture
only on last pipeline-stage ranks (they own the post-forward payloads). Other
ranks stay silent. A single producer writes `<DIR>/<bundle>/`; multiple producers
write `<DIR>/<bundle>/rank-<rank>/`. Each producer's writer thread
**try-finalizes** after publishing `COMPLETE.<rank>` (returns immediately if
peers are missing; writes the final `COMPLETE` once every producer has landed).
Each rank-local `COMPLETE` records `expected_ranks`. `validate` / `replay`
(and `BundleReader`) refuse a multi-rank `rank-*` path until the parent
cohort's final `COMPLETE` exists — passing `rank-0/` directly does not bypass
that check. The training thread never waits:

```text
<DIR>/
  replay-0-0/
    COMPLETE.0          # this rank's identity, owned payloads, checksums
    COMPLETE.1
    COMPLETE            # appears only after every expected rank has flushed
    rank-0/             # rank-local bundle; multi-rank replay needs parent COMPLETE
    rank-1/
```

```bash
export RELAX_REPLAY_CAPTURE=1
export RELAX_REPLAY_CAPTURE_DIR=/path/to/replay-bundles
# Optional: capture only the listed Actor steps / rollouts (comma-separated). Unset = all.
export RELAX_REPLAY_CAPTURE_STEPS=0:0
export RELAX_REPLAY_CAPTURE_ROLLOUTS=0
```

A training run then writes two bundle types: rollout-level (reward / advantage) and step-level (`loss.policy`). Replay them offline with the CLI above.

## Building a bundle programmatically

Bundles can still be built by hand with `BundleWriter` for offline validation (fixtures, injected faults, cross-version checks):

```python
import torch
from relax.utils.replay.bundle import BundleWriter
from relax.utils.replay.schema import (
    ActorStepId, BundleIndex, Identity, Manifest, ProducerInfo,
    RecomputeConfig, SampleRecord, StageCapability, StageContract, StageId,
)

index = BundleIndex(
    bundle_id="b-00001",
    identity=Identity(actor_step_id=ActorStepId(rollout_id=120, step_id=0), rank={"cp": 1}),
    samples=[SampleRecord(sample_id="s-0", group_index=0, response_length=2, total_length=3,
                          loss_mask=[1, 1], raw_reward=1.0, reward=1.0), ...],
    config=RecomputeConfig(advantage_estimator="grpo", n_samples_per_prompt=2),
)
manifest = Manifest(
    format_version="1.0.0", bundle_id="b-00001",
    producer=ProducerInfo(commit="...", torch_version=torch.__version__),
    stage_contracts={stage: StageContract(stage=stage, version="v1", capability=StageCapability.RECOMPUTE)
                     for stage in (StageId.SAMPLE, StageId.REWARD_RAW, StageId.REWARD_POST_PROCESS,
                                   StageId.ADVANTAGE_KL, StageId.ADVANTAGE_ESTIMATE, StageId.LOSS_POLICY)},
    payloads={}, comparison_policy=..., redaction={"prompt": "hash"},
)
expected = {"reward.raw": {"raw_rewards": [...]}, "reward.post_process": {"rewards": [...]},
            "loss.policy": {"loss": ...}}

writer = BundleWriter("<path>", manifest, index, expected)
writer.write_payload("old_log_probs", old_log_probs)
writer.write_payload("log_probs", log_probs)
writer.write_payload("entropy", entropy)
writer.write_payload("kl", kl)
writer.write_payload("advantages", advantages)
writer.finalize(ranks=[0])
```

## Security and redaction

- Public artifacts allow only JSON metadata + tensor payloads; tensors load with `weights_only=True`.
- prompt/response/label/tool output/paths/endpoints follow an explicit redaction policy (`hash` / `truncate` / `drop`).
- No full `Namespace`, environment variables, secrets, checkpoints or service credentials are stored.

## Current limitations

- Does **not** replay model forward/backward/gradient/optimizer; does not guarantee re-sampling the same trajectory; does not require bitwise identity across hardware/compilers.
- **V1 supports GRPO `CP=1` only**; `CP>1`, PPO value loss, OPD and agentic flattened are `unsupported`.
- Production capture is split into two bundle types: rollout-level (reward/advantage, identity `rollout_id`, instrumented in `train_actor`) and step-level (loss, identity `(rollout_id, step_id)`, instrumented in `train_one_step`); cross-bundle propagation of the "first divergent stage" is not yet unified. Enable with `RELAX_REPLAY_CAPTURE=1` and `RELAX_REPLAY_CAPTURE_DIR`.
- Remote RM/GenRM results are treated as `recorded-only` and are not recomputed offline.

See [Task 34 RFC #171](https://github.com/redai-studio/Relax/issues/171) for the design discussion.
