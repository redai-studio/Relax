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
