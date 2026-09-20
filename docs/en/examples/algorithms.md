# Algorithm Reference

Relax supports multiple policy gradient algorithms, all selected via the `--advantage-estimator` flag. This document covers PPO and the primary GRPO-family algorithms (for On-Policy Distillation, see the [dedicated page](./on-policy-distillation.md)).

GRPO, RLOO, CISPO, GSPO, SAPO, and M2PO share the same actor/rollout service topology, although RLOO is synchronous-only and enforces fixed batch invariants. PPO additionally requires a Critic model and an Advantages service; start from the [PPO training recipe](../guide/ppo-training.md) instead of only replacing `GRPO_ARGS`.

REINFORCE++ and REINFORCE++-baseline also reuse the GRPO service topology, but
their return, global normalization and KL contracts are algorithm-specific.
See [REINFORCE++ Training](../guide/reinforce-plus-plus.md) before enabling
either estimator.

---

## GRPO

GRPO (Group Relative Policy Optimization) is the default algorithm in Relax. It broadcasts the group-relative scalar reward to every token and uses a standard PPO-Clip objective.

Reference: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300).

### How It Works

The GRPO objective is the standard PPO-Clip:

$$J_\text{GRPO}(\theta) = \mathbb{E} \left[ \min\!\left( r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\varepsilon,\ 1+\varepsilon)\hat{A}_t \right) \right]$$

where $r_t(\theta) = \pi_\theta / \pi_{\theta_\text{old}}$, and $\hat{A}_t$ is the group-relative advantage (reward minus group mean, normalized by group standard deviation). Gradients are zeroed out when the ratio exceeds the clipping bounds.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage-estimator grpo` | default | Enable GRPO |
| `--eps-clip` | `0.2` | Clipping margin (ratio range = `[1-ε, 1+ε]`) |
| `--eps-clip-high` | same as `--eps-clip` | Upper clipping margin; can be set differently for asymmetric clipping |
| `--clip-grad` | — | Gradient clipping norm |

### Quick Start

GRPO is the default algorithm — no parameter changes needed. Just run the training script directly:

```bash
MODEL_DIR=/path/to/model \
DATA_DIR=/path/to/data \
EXP_DIR=/path/to/exp \
bash scripts/training/text/run-qwen3-4B-8xgpu.sh
```

---

## RLOO

RLOO (REINFORCE Leave-One-Out) uses the other samples for the same prompt as an unbiased baseline. Relax implements synchronous RLOO with an unclipped REINFORCE policy loss; it does not use PPO ratios or clipping.

Reference: [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740).

### How It Works

For a prompt with $G$ sampled responses and scalar rewards $r_i$, the leave-one-out baseline and advantage are:

$$b_i = \frac{1}{G-1}\sum_{j\ne i}r_j, \qquad A_i = r_i-b_i = \frac{G}{G-1}(r_i-\bar r)$$

Unlike GRPO, RLOO does not divide by the group standard deviation. The token loss is:

$$L_{i,t} = -\operatorname{stopgrad}(A_i)\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})$$

Each sample's scalar advantage is broadcast to its response tokens. Relax masks padding and normalizes the summed loss by the global number of valid response tokens $N_{\mathrm{eff}}$:

$$L_{\mathrm{RLOO}} = -\frac{1}{N_{\mathrm{eff}}}\sum_i\sum_t m_{i,t}\operatorname{stopgrad}(A_i)\log\pi_{i,t}$$

This global-token reduction does not apply a separate $1/T_i$ weight to each response. `train/pg_clipfrac` is always `0` because RLOO uses no clipping.

### Requirements and Parameters

| Parameter | Requirement | Description |
|-----------|-------------|-------------|
| `--advantage-estimator rloo` | required | Enable RLOO |
| `--n-samples-per-prompt` | at least `2` | Group size $G$; larger groups provide a more stable LOO baseline at higher rollout cost |
| `--rollout-batch-size × --n-samples-per-prompt` | equals `--global-batch-size` | Exactly one optimizer update per rollout |
| `--num-steps-per-rollout` | unset or `1` | Reusing the same rollout for multiple unclipped updates is rejected |
| `--calculate-per-token-loss` | enabled | Use global valid-token normalization; per-response token means would reweight unequal-length responses by $1/T_i$ |
| `--kl-coef` | `0` | Reward-side KL shaping is not implemented for RLOO; with a valid `--ref-load <checkpoint>`, use `--use-kl-loss --kl-loss-coef <value>` for the supported direct KL penalty |
| `--max-staleness` | `0` | Stale rollouts are rejected because the unclipped objective has no importance-ratio correction |
| reward normalization | enabled | RLOO's group transformation runs in the normalized-reward path |
| `--normalize-advantages` | disabled | Post-DP whitening would change RLOO semantics and make results partition-dependent |
| `--fully-async`, `--hybrid`, `--partial-rollout`, `--use-dynamic-global-batch-size` | disabled | RLOO currently requires synchronous, fixed-size rollout batches |

The batch sizes are hardware-tunable as long as their equality is preserved. For example, `ROLLOUT_BATCH_SIZE=4`, `N_SAMPLES=8`, and `GLOBAL_BATCH_SIZE=32` retain one update per rollout while reducing per-step memory relative to `16 × 8 = 128`.

### Diagnostics

Training rollout logs publish the following final metric names:

- `rollout/rloo/baseline_mean`: mean LOO baseline (equal to the mean group reward; retained as an explicit baseline trace)
- `rollout/rloo/adv_abs_mean`: mean absolute RLOO advantage
- `rollout/rloo/no_signal_frac`: fraction of effective loss tokens attached to zero-advantage samples
- `rollout/rloo/empty_response_frac`: fraction of samples with a literally empty response
- `rollout/rloo/zero_adv_group_frac`: fraction of complete groups with zero advantages throughout
- `rollout/rloo/dropped_group_frac`: fraction of observed groups omitted from diagnostics because their size is incomplete

These diagnostics are training-only, purely observational rollout statistics; they do not affect the training path. Evaluation uses its own sampling group size and does not emit misleading `eval/*/rloo/*` values. They are also omitted when a custom reward post-processor or agentic custom-advantage hook replaces the standard RLOO signal, because raw rewards cannot reconstruct the optimizer input in those modes.

### Quick Start

Use the dedicated Qwen3-0.6B GSM8K recipe. Its batch and rollout settings can be overridden with environment variables:

```bash
MODEL_DIR=/path/to/models \
DATA_DIR=/path/to/data \
NUM_ROLLOUT=60 \
ROLLOUT_BATCH_SIZE=4 \
N_SAMPLES=8 \
GLOBAL_BATCH_SIZE=32 \
bash examples/algorithms/run-qwen3-0.6B-1xgpu-rloo.sh
```

Set `ADVANTAGE_ESTIMATOR=grpo` to run a control arm with the same recipe and seeds. The recipe writes normalized GSM8K data to a writable artifact cache (override with `RLOO_DATA_CACHE_DIR`) and appends an instruction to emit the final answer as `\boxed{...}`, matching the `math` reward parser contract.

---

## PPO

PPO (Proximal Policy Optimization) is an actor-critic algorithm. Relax trains a separate Critic to predict token-level values, computes GAE advantages and returns, applies PPO-Clip to the Actor, and applies clipped value loss to the Critic.

Reference: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).

### How It Works

The temporal-difference residual and GAE recursion are:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

$$\hat{A}_t = \delta_t + \gamma\lambda\hat{A}_{t+1}, \qquad \hat{R}_t = \hat{A}_t + V(s_t)$$

The Actor then uses the same clipped policy objective shown for GRPO, but with Critic-derived token-level advantages. The Critic minimizes the maximum of clipped and unclipped squared value errors.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage-estimator ppo` | — | Enable PPO and the Critic service graph |
| `--gamma` | `1.0` | GAE discount factor |
| `--lambd` | `1.0` | GAE lambda |
| `--eps-clip` | `0.2` | Actor clipping margin |
| `--value-clip` | `0.2` | Critic value clipping range |
| `--num-critic-only-steps` | `0` | Initial Critic-only warmup steps |
| `--critic-lr` | same as `--lr` | Critic learning rate |

### Quick Start

PPO cannot be enabled by changing only the algorithm argument because its service graph requires `critic` and `advantages` resources. Fully-async PPO is not currently supported; use the dedicated synchronous colocate recipe:

```bash
MODEL_DIR=/path/to/models \
DATA_DIR=/path/to/data \
EXP_DIR=/path/to/experiments \
bash scripts/training/text/run-qwen35-9B-8xgpu-ppo.sh
```

See [PPO Training](../guide/ppo-training.md) for the resource topology, checkpoint rules, and KL constraints.

---

## CISPO

CISPO (Clipped Importance-ratio Soft Policy Optimization) preserves gradient signal for out-of-trust-region tokens instead of zeroing it out. It caps gradient magnitude via a stop-gradient'd coefficient while keeping the gradient direction alive.

Reference: [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585).

### How It Works

The CISPO objective is:

$$J_\text{CISPO}(\theta) = \mathbb{E}_{(q,a)\sim\mathcal{D},\ \{o_i\}_{i=1}^G \sim \pi_{\theta_\text{old}}(\cdot|q)} \left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \text{sg}\!\left(\hat{r}_{i,t}(\theta)\right) \hat{A}_{i,t} \log \pi_\theta(o_{i,t} \mid q, o_{i,<t}) \right]$$

where $\hat{r}_{i,t}(\theta)$ is the clipped importance-sampling weight:

$$\hat{r}_{i,t}(\theta) = \text{clip}\!\left(r_{i,t}(\theta),\ 1 - \varepsilon_\text{low}^\text{IS},\ 1 + \varepsilon_\text{high}^\text{IS}\right)$$

and $r_{i,t}(\theta) = \pi_\theta(o_{i,t} \mid q, o_{i,<t}) / \pi_{\theta_\text{old}}(o_{i,t} \mid q, o_{i,<t})$. Gradients flow **only** through $\log\pi_\theta$; both $\hat{r}_{i,t}$ and $\hat{A}_{i,t}$ are stop-gradient'd.

### Key Parameters

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `--advantage-estimator cispo` | — | — | Enable CISPO |
| `--eps-clip` | `0.2` | `0.2` | Lower clipping margin (ratio lower bound = `1 - eps_clip`) |
| `--eps-clip-high` | same as `--eps-clip` | `10` | Upper clipping margin (ratio upper bound = `1 + eps_clip_high`). Set to `10` to effectively unclamp the upper side |
| `--kl-loss-coef` | `0.0` | `0.001` | KL loss coefficient. Recommended: `0.001` to add a small KL penalty that constrains policy drift |
| `--use-kl-loss` | off | on | Enable KL loss computation (required for `--kl-loss-coef` to take effect) |
| `--use-tis` | off | on | Token Importance Sampling — recommended to enable with CISPO |
| `--clip-grad` | — | `1.0` | Gradient clipping norm |

### Quick Start

Use any existing GRPO training script and replace `GRPO_ARGS` with `CISPO_ARGS`:

```bash
CISPO_ARGS=(
   --advantage-estimator cispo
   --use-kl-loss
   --kl-loss-coef 0.001
   --eps-clip 0.2
   --eps-clip-high 10
   --use-tis
)
```

---

## GSPO

GSPO (Group-wise Sequence-level Policy Optimization) differs from GRPO in how KL divergence is computed: GSPO uses **sequence-level** KL instead of per-token KL. Every token in a sequence shares the same KL value (the mean over all tokens in that sequence), providing uniform constraint strength within a sequence.

### How It Works

GSPO uses the same PPO-Clip objective as GRPO, but the ratio is computed from sequence-level KL:

$$\text{KL}_\text{seq} = \frac{1}{|o|} \sum_{t=1}^{|o|} \left(\log\pi_{\theta_\text{old}}(o_t) - \log\pi_\theta(o_t)\right)$$

Every token's ratio is $r_t = \exp(-\text{KL}_\text{seq})$, rather than an independent per-token ratio.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage-estimator gspo` | — | Enable GSPO |
| `--eps-clip` | `0.2` | Clipping margin |
| `--eps-clip-high` | same as `--eps-clip` | Upper clipping margin |
| `--clip-grad` | — | Gradient clipping norm |

### Quick Start

```bash
GSPO_ARGS=(
   --advantage-estimator gspo
   --eps-clip 0.2
)
```

---

## SAPO

SAPO (Soft Adaptive Policy Optimization) replaces hard clipping with a smooth sigmoid gate. The gate's steepness is controlled by a temperature parameter, implementing a differentiable trust region constraint.

### How It Works

SAPO's core is a sigmoid gate centered at ratio=1:

$$f(r) = \frac{4}{\tau} \cdot \sigma\!\left(\tau(r - 1)\right)$$

where $\sigma$ is the sigmoid function, and $\tau$ is selected based on the advantage sign:

- $A > 0$: use $\tau_\text{pos}$ (default 1.0)
- $A \leq 0$: use $\tau_\text{neg}$ (default 1.05, stronger suppression for negative tokens)

SAPO objective: $J_\text{SAPO}(\theta) = f(r) \cdot A$

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--advantage-estimator sapo` | — | Enable SAPO |
| `--sapo-tau-pos` | `1.0` | Temperature for positive advantages |
| `--sapo-tau-neg` | `1.05` | Temperature for negative advantages (higher = stronger suppression) |
| `--clip-grad` | — | Gradient clipping norm |

### Quick Start

```bash
SAPO_ARGS=(
   --advantage-estimator sapo
   --sapo-tau-pos 1.0
   --sapo-tau-neg 1.05
)
```

---

## M2PO

M2PO (Second-Moment Trust Policy Optimization) uses the **second moment** of the log importance ratio over harmful tokens as its trust-region constraint: it tightens clipping only when that second moment exceeds a budget, and keeps the token otherwise. Compared to fixed clipping, it retains more useful gradient and mitigates entropy collapse in off-policy (stale-data) regimes, making it purpose-built for mini-batch reuse and asynchronous training.

Reference: [Prosperity before Collapse: How Far Can Off-Policy RL Reach with Reuse of Mini-Batch Data?](https://arxiv.org/abs/2510.01161) (NeurIPS 2025).

### How It Works

M2PO only constrains the "harmful" tokens that PPO would clip — those whose advantage sign aligns with the ratio's drift and would cause an over-update:

$$\mathcal{H} = \{t : \hat{A}_t > 0,\ r_t > 1\} \cup \{t : \hat{A}_t < 0,\ r_t < 1\}$$

where $r_t = \exp(-\text{KL}_t)$ and $\text{KL}_t = \log\pi_{\theta_\text{old}}(o_t) - \log\pi_\theta(o_t)$. The second moment of the log-ratio over these tokens is:

$$M_2 = \frac{1}{|\mathcal{H}|} \sum_{t \in \mathcal{H}} (\log r_t)^2$$

Relax solves this statistic independently on each Megatron microbatch's local
tokens. Logging preserves the existing aggregation: with
`--calculate-per-token-loss`, `train/ppo_kl_m2_before` and
`train/ppo_kl_m2_after` are weighted by the microbatch's loss-token count;
otherwise, each microbatch scalar is summed unchanged and the framework
divides by the sample count. The latter is not a sample-weighted mean. Neither
mode pools all harmful tokens in the global batch. These logged diagnostics
do not feed back into the loss.

- If $M_2 \le$ `kl2_budget`: no clipping, all tokens are kept;
- Otherwise, use water-filling to select the previous observed breakpoint. This gives a conservative trust-region radius $\tau$ whose capped sum does not exceed $|\mathcal{H}| \cdot \text{kl2\_budget}$, yielding the clip band $[e^{-\tau},\ e^{\tau}]$.

The final clipping margin is $\varepsilon = \max(\text{adaptive value},\ \text{miniclip})$, so it is never tighter than the configured floor. Choose the floors to match the GRPO margins when that is the desired lower bound. `ppo_kl_m2_after` is the legacy solver diagnostic for the selected breakpoint before that floor is applied; in the first-breakpoint edge case it retains the uncapped local mean. It is not a recomputation after the final policy clip. The policy loss reuses the PPO-Clip pessimistic form from the GRPO section, with only the clip bounds solved adaptively.

### Key Parameters

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `--advantage-estimator m2po` | — | — | Enable M2PO |
| `--m2po-kl2-budget` | `0.01` | `0.01`–`0.04` | Second-moment budget per harmful token. Smaller = tighter/more-frequent clipping, larger = more off-policy tolerance (the paper uses `0.04`) |
| `--m2po-miniclip-low` | `0.3` | `0.2` | Lower-side clip-margin floor (the margin is at least `miniclip_low`) |
| `--m2po-miniclip-high` | `0.5` | `0.28` | Upper clip-margin floor |
| `--use-rollout-logprobs` | off | on for stale async data | Use the behavior-policy log probabilities that generated each token as M2PO's old policy |
| `--use-tis` | off | off with rollout log probs | TIS is a post-loss correction and does not drive M2PO's adaptive threshold; it is mutually exclusive with `--use-rollout-logprobs` |

> M2PO derives its clip bounds adaptively, so it does **not** use `--eps-clip` / `--eps-clip-high`.

::: warning Existing implementation behavior
The registry refactor preserves M2PO's existing computation: reward processing
passes through raw sample rewards, and the threshold solver does not filter
tokens by the loss mask or aggregate its statistic across CP ranks. Its
breakpoint comparisons still use host scalars. Registration does not establish
equivalence to the paper. Because the clipping statistics are local, M2PO
declares `supports_context_parallel=False`: startup requires
`--context-parallel-size 1` and dynamic context parallelism disabled. This
validation leaves the algorithm's reward, solver and metric calculations
unchanged.
:::

### When to Use

M2PO's benefit grows with how off-policy the training data is, so reach for it **first** in these scenarios:

- **Asynchronous training with large staleness**: in fully-async mode the rollout weights lag noticeably behind the actor (`--max-staleness` of 32, 256, or higher), and stale samples inflate the importance ratio. Fixed clipping then either zeroes out many tokens (losing gradient) or lets them through (causing over-updates); M2PO uses the second moment to adaptively tighten only the genuinely "harmful" fraction, suppressing collapse while preserving the learning signal.
- **Mini-batch reuse / multi-step sampling**: when the same rollout batch is reused across several update steps, the later steps effectively train on off-policy data too, and M2PO extends the usable lifetime of that batch.
- **Late-training entropy collapse or reward stagnation**: when fixed clipping narrows the policy too quickly and starves exploration, M2PO's looser adaptive bounds help sustain entropy and delay collapse.

Conversely, under strictly on-policy synchronous training (`--max-staleness 0` with per-step weight sync), M2PO's gain over GRPO is limited — start from GRPO as a baseline there.

M2PO is valid in true-on-policy mode, but then its importance ratio is exactly
one and adaptive clipping does not engage. To evaluate its stale-data behavior,
make sure the run supplies old-policy log probabilities rather than
auto-enabling `--true-on-policy-mode`. For fully-async rollout staleness, pass
`--use-rollout-logprobs`: `--use-tis` is applied only after M2PO has already
chosen its clip bounds.

### Quick Start

Use any existing GRPO training script and replace `GRPO_ARGS` with `M2PO_ARGS`:

```bash
M2PO_ARGS=(
   --advantage-estimator m2po
   --m2po-kl2-budget 0.01
   --m2po-miniclip-low 0.2
   --m2po-miniclip-high 0.28
)
```

---

## Algorithm Comparison

| Algorithm | Advantage Computation | Policy Loss | KL Constraint |
|-----------|----------------------|-------------|---------------|
| **PPO** | Critic values + GAE | PPO-Clip (hard clip) | Disabled in the current synchronous topology |
| **GRPO** | Group-relative reward | PPO-Clip (hard clip) | Optional KL loss |
| **REINFORCE++** | Token KL-to-go return + global token normalization | PPO-Clip (hard clip) | k1 KL in shaped reward |
| **REINFORCE++-baseline** | Inclusive group mean + global token normalization | PPO-Clip (hard clip) | Separate k2 KL loss |
| **CISPO** | Group-relative reward | Stop-gradient coefficient | Recommended KL loss |
| **GSPO** | Group-relative reward | PPO-Clip + sequence-level KL | Sequence-level ratio |
| **SAPO** | Group-relative reward | Sigmoid gate | Temperature-controlled |
| **M2PO** | Raw sample reward broadcast to tokens | Adaptive second-moment clip | Optional KL loss (favor for large-staleness / off-policy) |
| **RLOO** | Leave-one-out baseline | Unclipped REINFORCE | Optional KL loss (same as GRPO) |

## Next Steps

- [PPO Training](../guide/ppo-training.md)
- [REINFORCE++ Training](../guide/reinforce-plus-plus.md)
- [Quick Start](../guide/quick-start.md)
- [On-Policy Distillation](./on-policy-distillation.md)
- [Generative Reward Model](./generative-reward-model.md)
