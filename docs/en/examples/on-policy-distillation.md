# On-Policy Distillation (OPD)

On-Policy Distillation (OPD) enables knowledge transfer from a large teacher model to a smaller student model by training the student on its own rollout data while matching the teacher's token-level log-probabilities. OPD is orthogonal to the advantage estimator—it acts as a KL penalty term that can be combined with any estimator, including PPO, GRPO, GSPO, SAPO, CISPO, and REINFORCE++.

## Key Parameters

| Parameter                 | Description                                                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `--use-opd`               | Enable On-Policy Distillation. Required flag when using OPD.                                                                               |
| `--opd-type`              | OPD type: `sglang` or `megatron`. Must be set when `--use-opd` is enabled.                                                                 |
| `--opd-token-selection`   | Token selection mode: `student_sampled` (default), `student_topk`, `teacher_topk`, `union`.                                               |
| `--opd-kl-coef`           | KL coefficient for advantage mode (default: 1.0). When set, `--opd-loss-coef` must be 0.                                                  |
| `--opd-loss-coef`         | KL coefficient for loss mode (default: 0.0). When set, `--opd-kl-coef` must be 0.                                                         |
| `--opd-kl-type`           | KL divergence type: `reverse_kl` (default), `forward_kl`, `low_var_kl`, `jsd`.                                                            |
| `--opd-jsd-alpha`         | JSD mixing coefficient (default: 0.5). 0.0 is equivalent to reverse_kl, 1.0 to forward_kl.                                                |
| `--opd-log-prob-top-k`    | Top-K candidate set size (set to `0` to disable, default: `0`).                                                                            |
| `--opd-norm-mode`         | Top-K tail handling: `tail` (default), `norm`, `trunc`.                                                                                   |
| `--opd-per-token-clip`    | Hard upper bound for per-token KL (optional).                                                                                              |
| `--opd-is-clip`           | Hard upper bound for importance sampling ratio (optional, loss mode only).                                                                |
| `--opd-teacher-load`      | Path to the teacher model. **Must** be set when `--opd-type=megatron`, **must not** be set when `--opd-type=sglang`.                       |
| `--opd-teacher-ckpt-step` | Optional checkpoint step for the teacher model.                                                                                            |
| `--opd-teacher-timeout-s` | Timeout (seconds) for OPD teacher HTTP requests in SGLang mode (default: 30).                                                              |
| `--opd-teacher-url`       | `/generate` URL of an SGLang teacher you run yourself. See [SGLang Teacher Deployment](#sglang-teacher-deployment).                        |
| `--teacher-hf-checkpoint` | HF checkpoint of a teacher that Relax launches and manages. Needs a `teacher` entry in `--resource`.                                       |
| `--teacher-num-gpus-per-engine` | GPUs per teacher engine (its TP size). Defaults to all teacher GPUs, i.e. one replica.                                               |
| `--opd-teacher-routes`    | JSON map from data source to teacher checkpoint, for several managed teachers (colocate only).                                             |
| `--opd-teacher-key`       | Sample metadata field that selects the teacher under `--opd-teacher-routes` (default: `data_source`).                                      |
| `--opd-teacher-defer`     | Ask a managed teacher after a batch has been generated instead of during generation (colocate only).                                       |
| `--opd-only-reward`       | Keep only the OPD reward signal (zero out base RL reward and use OPD KL term only). Requires `--use-opd`.                                 |

## How It Works

OPD injects distillation signals into training by computing token-level KL divergence between teacher and student. Relax supports two injection methods:

- **Advantage mode (adv)**: Subtract KL from advantage (via `--opd-kl-coef`)
- **Loss mode (loss)**: Add KL as an extra loss term (via `--opd-loss-coef`)

Only one mode can be active at a time. OPD is orthogonal to the advantage estimator and can be combined with any estimator, including PPO, GRPO, GSPO, SAPO, CISPO, and REINFORCE++.

## Token-Selection Modes

OPD supports four token-selection strategies that determine which tokens are used for KL computation:

| Mode | Student self top-K | Teacher self top-K | Teacher @ student top-K | Student @ teacher top-K | Description |
| --- | --- | --- | --- | --- | --- |
| `student_sampled` | — | — | — | — | Compute KL only on student-sampled 1D tokens, lowest overhead |
| `student_topk` | ✅ | — | ✅ | — | Compute KL on student top-K token set |
| `teacher_topk` | — | ✅ | — | ✅ | Compute KL on teacher top-K token set |
| `union` | ✅ | ✅ | ✅ | ✅ | Compute KL on the union of student and teacher top-K sets, most comprehensive |

Use `--opd-token-selection` to specify the mode and `--opd-log-prob-top-k` for the top-K size. For all modes except `student_sampled`, the environment variable `RELAX_OPD_PER_POS_TOKEN_IDS=1` must be set.

## Two Application Methods: adv and loss

### Advantage Mode (adv)

Enabled via `--opd-kl-coef` (with `--opd-loss-coef` set to 0). After advantage computation, the per-token KL is subtracted from the advantage:

$$\hat{A}_t = A_t - \lambda_{\text{opd}} \cdot D_{\text{KL}}(P_{\text{teacher}} \| P_{\text{student}})_t$$

Characteristics:

- KL term uses `.detach()`, **no gradient** is produced
- Only affects advantage estimation, does not change the loss function form
- Orthogonal to any advantage estimator (PPO, GRPO, GSPO, SAPO, CISPO, etc.)

Architecture flow:

```
Rollout Phase:
  Student Rollout → student top-K token IDs / log-probs
  Teacher Prefill → teacher log-probs / teacher top-K
  Student Prefill → student @ teacher top-K log-probs (adv mode only)
      ↓
  Assemble training data (opd_topk_token_ids, opd_topk_student_log_probs, opd_topk_teacher_log_probs)

Training Phase:
  compute_advantages_and_returns()
    → apply_opd_to_advantages()
    → modify advantage: adv = adv - opd_kl_coef * kl_term.detach()
```

### Loss Mode (loss)

Enabled via `--opd-loss-coef` (with `--opd-kl-coef` set to 0). During policy loss computation, the per-token KL is added as an extra loss term:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PG}} + \lambda_{\text{loss}} \cdot \mathbb{E}_t[D_{\text{KL}}(P_{\text{teacher}} \| P_{\text{student}})_t]$$

Characteristics:

- KL term **produces gradient**, directly affecting policy gradient direction
- Supports per-token clipping (`--opd-per-token-clip`) and importance ratio clipping (`--opd-is-clip`)
- Independent of the advantage estimator

Architecture flow:

```
Rollout Phase:
  Student Rollout → student top-K token IDs / log-probs
  Teacher Prefill → teacher log-probs / teacher top-K
      ↓
  Assemble training data (opd_topk_token_ids, opd_topk_teacher_log_probs)

Training Phase:
  policy_loss_function()
    → get_log_probs_and_entropy() (collect student top-K log-probs)
    → compute_policy_opd_loss()
    → compute KL → clipping → reduce
    → loss = loss + opd_loss_coef * opd_loss
```

> **Note**: adv and loss modes are mutually exclusive; only one can be enabled at a time.

## SGLang Teacher Deployment

With `--opd-type sglang` the teacher is an SGLang server that returns log-probs for the student's rollouts. You can point Relax at a server you run yourself, or let Relax launch the teacher.

**External teacher.** Pass `--opd-teacher-url http://teacher-host:30001/generate`. Relax only sends requests to it; starting, sizing and stopping the server is up to you.

**Managed teacher.** Pass `--teacher-hf-checkpoint` and add a `teacher` entry to `--resource`. Relax starts the teacher engines before the other services and fills in the teacher address itself. The teacher GPUs are divided into replicas of `--teacher-num-gpus-per-engine` GPUs each, and requests are spread over the replicas so that the samples of one prompt group stay on one replica.

```bash
python3 relax/entrypoints/train.py \
    --colocate \
    --resource '{"actor": [1, 8], "rollout": [1, 4], "teacher": [1, 4]}' \
    --rollout-num-gpus 4 \
    --use-opd --opd-type sglang \
    --teacher-hf-checkpoint /path/to/teacher \
    --teacher-num-gpus-per-engine 2 \
    ...
```

Under `--colocate` the teacher lives inside the actor placement group, on the bundles after the Rollout region, so `rollout GPUs + teacher GPUs` must equal the actor's GPU count. The teacher is offloaded while the Actor trains and loaded again together with the Rollout weights. Without `--colocate`, each teacher replica gets a placement group of its own. `--opd-teacher-routes` (one teacher per data source) is available under `--colocate` only; the teacher GPUs are split evenly between the teachers.

Before teacher engines start, Relax checks that the GPUs of each engine are on one node and have contiguous GPU ids. If they are not, the run stops with a "Physical placement mismatch" error that names the engine and shows where its bundles are.

### Deferred Teacher Scoring

By default the teacher is asked while Rollout generates, so both are in GPU memory at once and need separate bundles. `--opd-teacher-defer` moves the teacher to after generation: it sleeps while Rollout generates, and once a batch is complete Relax offloads Rollout, loads the teacher, asks it for the whole batch, and offloads it again. The batch is published for training only after the teacher's log-probs are written back; if every teacher request fails, the batch is not published and the run reports the error.

Because the two are no longer loaded at the same time, they may share bundles. Two colocate layouts are accepted:

| Layout | Condition                                         | Teacher bundles                |
| :----- | :------------------------------------------------ | :----------------------------- |
| Split  | `rollout GPUs + teacher GPUs == actor GPUs`       | After the Rollout region       |
| Shared | `rollout GPUs == teacher GPUs == actor GPUs`      | The same bundles as Rollout    |

The Shared layout is refused without `--opd-teacher-defer`. When the student must also be queried on the teacher's top-K tokens (`teacher_topk` or `union` in advantage mode), Relax loads Rollout again after the teacher is offloaded and runs that step then. Evaluation batches are not sent to the teacher.

`--opd-teacher-defer` needs a managed teacher under `--colocate`. It is rejected at startup with an external teacher, without `--colocate`, and with `--use-agentic-rollout`.

In advantage mode, deferred `teacher_topk` and `union` require fresh samples from the current student policy. Disable `--partial-rollout` and `--dynamic-sampling-filter-path`, and keep `--over-sampling-batch-size` equal to `--rollout-batch-size` (the default). Alternatively, enable both `--partial-rollout` and `--mask-offpolicy-in-partial-rollout` so carried response tokens do not contribute to training. Other combinations are rejected at startup because buffered samples can cross a weight update before student prefill and mix scores from different policies. This restriction does not apply to loss mode, `student_sampled`, or `student_topk`.

### Teacher Endpoints

A managed teacher is reachable through the `/teacher` route of the Ray Serve HTTP port, next to `/rollout` and `/genrm`. The route does not exist when no managed teacher is configured.

| Endpoint                                                          | Purpose                                                                             |
| :---------------------------------------------------------------- | :---------------------------------------------------------------------------------- |
| `GET /teacher/health`                                             | State of each teacher model (`ready`, `sleeping`, ...)                              |
| `GET /teacher/engines`                                            | Topology: every teacher, its replicas, their addresses and states                   |
| `GET /teacher/v1/models`                                          | The teacher models, in OpenAI model-list form                                       |
| `POST /teacher/generate`                                          | A native SGLang `/generate` payload, forwarded to a ready replica                   |
| `POST /teacher/v1/chat/completions`, `/teacher/chat/completions`  | OpenAI-style chat request, forwarded to a ready replica                             |

```bash
curl http://localhost:8000/teacher/health
```

With `--opd-teacher-routes`, name the teacher with `model` (or `route_key`) set to its data source; a single teacher needs neither. A request for a teacher that is offloaded gets `503` with a `Retry-After` header and does not wake it, and an unknown `model` gets `400` with the list of available teachers. Relax itself reads `/teacher/engines` to find the replicas, so a teacher engine that is rebuilt at another address is used without restarting the run.
