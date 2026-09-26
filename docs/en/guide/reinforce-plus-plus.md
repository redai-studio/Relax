# REINFORCE++ and REINFORCE++-baseline

Relax exposes two separate estimator names because the baseline variant is not
just a different recipe:

- `reinforce_plus_plus`
- `reinforce_plus_plus_baseline`

This page fixes their return, advantage, normalization, mask, KL and reduction
semantics. The implementation follows the main equations of
[REINFORCE++ arXiv v9](https://arxiv.org/abs/2501.03262) and the executable
normalization convention in OpenRLHF commit
[`bc71bb1`](https://github.com/OpenRLHF/OpenRLHF/tree/bc71bb19464aca306b33080b2d2bb45d154e2f49).

## Version and naming note

The paper changed between v1 and v9, and even v9's main method and Appendix B.2
do not describe exactly the same token placement. The following table makes the
version and implementation boundary explicit.

| Source | Return / baseline | Normalization and KL | Status in this implementation |
|---|---|---|---|
| paper v1 | token k1 KL-to-go advantage plus PPO clipping | describes reward normalization/clipping and batch z-score advantage normalization, but has no separately named group-baseline-plus-k2 variant | historical REINFORCE++ only; not the baseline definition |
| paper v9 main equations | token KL-to-go return; inclusive group-mean baseline variant | global normalization of advantage tokens | normative paper definition |
| paper v9 Appendix B.2 | zero advantage before the final token | sample-level reward normalization | documented conflict; not selected |
| OpenRLHF `bc71bb1` | inclusive group mean and global valid-token population normalization | its pinned baseline training script enables token KL shaping **and** a separate k2 loss | normalization reference only; its combined-KL baseline is intentionally not copied |
| Relax before this feature | partial helper names and return code | no registered pair, frozen validation, dedicated population moments, recipe, or complete numerical contract | compatibility baseline |

Relax selects the v9 main-equation interpretation. “OpenRLHF aligned” in this
document refers specifically to its executable inclusive-baseline and masked
population-normalization convention. The Relax baseline deliberately keeps KL
out of the advantage and applies only the independent k2 loss, as frozen in the
Task 29 Proposal; it is therefore not an exact reproduction of the pinned
OpenRLHF training script.

## Notation and mask contract

For response (i) and response position (t):

- (R_i) is the scalar terminal reward;
- (m_{i,t}\in\{0,1\}) is the response loss mask;
- (T_i) is the final valid response position;
- (L_i=\sum_t m_{i,t}) is the valid response length.

Prompt tokens, padding and response tokens with `mask=0` do not contribute to
reward shaping, return, normalization or loss. Production return and advantage
tensors are explicitly zero outside the mask. Selection is boolean rather than
multiplicative, so even `NaN` or `Inf` in a masked storage position cannot
contaminate a valid-token result.

## REINFORCE++

Let the signed k1 estimator be

$$
d_{i,t}=\log\pi_{old}(a_{i,t})-\log\pi_{ref}(a_{i,t}).
$$

The shaped token reward is

$$
r_{i,t}=m_{i,t}\left[-\beta d_{i,t}+\mathbf{1}(t=T_i)R_i\right].
$$

The terminal reward is added only to the final valid response token. Returns
are accumulated backwards:

$$
G_{i,t}=m_{i,t}\sum_{u=t}^{T_i}\gamma^{u-t}r_{i,u}.
$$

The formal recipe fixes `gamma=1`. The raw advantage is (G), followed by the
global masked normalization below. This variant uses token KL reward shaping
and does not add a second KL loss.

```text
--advantage-estimator reinforce_plus_plus
--normalize-advantages
--gamma 1.0
--kl-coef 0.01
--kl-loss-type k1
```

## REINFORCE++-baseline

For a prompt group (g) with (K) sampled responses,

$$
b_g=\frac{1}{K}\sum_{j\in g}R_j,\qquad C_i=R_i-b_g.
$$

The group mean includes the current response. It is not a leave-one-out
baseline. Relax does not divide (C_i) by a group standard deviation.

The raw token advantage is

$$
A^{raw}_{i,t}=m_{i,t}C_i.
$$

Token KL is not subtracted from this advantage. Reference regularization is a
separate k2 loss:

$$
D^{k2}_{i,t}=\frac{1}{2}
\left(\log\pi_\theta(a_{i,t})-\log\pi_{ref}(a_{i,t})\right)^2.
$$

```text
--advantage-estimator reinforce_plus_plus_baseline
--normalize-advantages
--n-samples-per-prompt 8
--kl-coef 0
--use-kl-loss
--kl-loss-type k2
--kl-loss-coef 0.01
```

The baseline variant requires more than one sample per prompt. The group mean
includes each sample itself, so `n_samples_per_prompt=1` would collapse every
raw advantage to zero and is rejected. Custom reward post-processing and
agentic custom-advantage hooks are also rejected for this estimator because
they would bypass the frozen inclusive group-mean semantics.

## Global masked normalization

The statistical population consists of every valid response token in the
closed synchronous global batch across all data-parallel ranks:

$$
S=\{(r,i,t)\mid m_{r,i,t}=1\},\qquad N=|S|.
$$

Relax uses population variance (`ddof=0`):

$$
\mu=\frac{1}{N}\sum_{S}A,
\qquad
\sigma^2=\frac{1}{N}\sum_{S}(A-\mu)^2.
$$

The normalized output is

$$
\hat A=m(A-\mu)\left[\max(\sigma^2,10^{-8})\right]^{-1/2}.
$$

The epsilon convention is a **variance floor**, matching the pinned OpenRLHF
implementation. It is different from both `sqrt(var) + epsilon` and Relax's
legacy `sqrt(unbiased_var + epsilon)` helper. The REINFORCE++ variants use a
dedicated helper; existing algorithms keep their current normalization.

Expected edge behavior:

- zero variance and a single valid token produce finite zero advantages;
- an all-zero baseline reward group produces finite zeros;
- an all-zero reward with nonzero KL can produce finite KL-shaped
  REINFORCE++ returns;
- a fully masked local response contributes a zero tensor and still reaches
  the data-parallel collective;
- a globally empty mask triggers a device-side asynchronous assertion on every
  participating rank without extracting a host scalar in the training hot path.

Because the baseline scalar is broadcast to each valid token, longer responses
have greater weight in these token-level global moments. This is intentional.

## PPO and KL reduction

Both variants use the ordinary token PPO clipped surrogate. Their formal scalar
objective is a response mean:

$$
L_{PG}=\frac{1}{B}\sum_i\frac{1}{L_i}
\sum_t m_{i,t}\max\left(
-\rho_{i,t}\hat A_{i,t},
-\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_{i,t}
\right).
$$

The baseline k2 loss uses the same response-mean reduction. The initial
implementation rejects `--calculate-per-token-loss` for these variants because
it changes the objective to a global token mean.

## Formula-level comparison

Let

$$
\rho_{i,t}=\exp(\log\pi_\theta(a_{i,t})-\log\pi_{old}(a_{i,t})),
$$

and let `clip-PPO` denote the token objective shown above. Relax's existing
group-relative algorithms first compute

$$
A_i^{grp}=R_i-\bar R_g,
$$

and, when `--grpo-std-normalization` is enabled, divide by
`torch.std({R_j:j in g}) + 1e-6`. That existing `torch.std` call uses Bessel's
sample correction (`ddof=1`), unlike the new global population variance.

| Algorithm | Raw advantage and statistical axes | Ratio / policy objective | Reference regularization |
|---|---|---|---|
| REINFORCE++ | token KL-to-go $G_{i,t}$; normalize over every valid token and DP rank with `ddof=0` | token $\rho_{i,t}$ and clip-PPO; response mean | k1 inside token reward |
| REINFORCE++-baseline | $R_i-\bar R_g$ broadcast to valid tokens, without group-std division; then the same global token/DP normalization | token $\rho_{i,t}$ and clip-PPO; response mean | separate k2 loss with response mean |
| GRPO | $A_i^{grp}$ with optional same-prompt sample-std scaling; broadcast within the response | token $\rho_{i,t}$ and clip-PPO | existing configurable Relax KL |
| GSPO | the same group advantage as GRPO | sequence ratio $\rho_i=\exp[L_i^{-1}\sum_t m_{i,t}(\log\pi_\theta-\log\pi_{old})]$ expanded to its tokens, then clip-PPO | existing configurable Relax KL |
| SAPO | the same group advantage as GRPO | token ratio with $f_\tau(\rho)=4\,\sigma[\tau(\rho-1)]/\tau$; loss $-f_\tau(\rho)A$, using separate positive/negative $\tau$ values | existing configurable Relax KL |

For all five paths, masks select contributing response tokens and the existing
Relax reducer computes a mean within each response followed by a mean across
responses. The new variants reject the alternative global-token reduction so
that this denominator cannot change silently.

This feature does not change GRPO, GSPO or SAPO defaults.

## Supported modes

The first implementation supports:

- synchronous colocate training;
- data-parallel normalization;
- `context_parallel_size=1`;
- response-mean loss reduction.

It rejects fully-async, hybrid, context parallelism greater than one, and
per-token global loss reduction. Fully-async does not currently define a closed
global batch over which these moments can be calculated, while CP greater than
one requires a separately verified unique-token ownership contract.

## Monitoring

The rollout metrics include:

- raw global advantage mean and standard deviation;
- normalized advantage mean and standard deviation;
- valid-token count;
- zero-variance indicator;
- ordinary reward, return and advantage summaries.

The three KL-related observables have deliberately different meanings:

- `train/ppo_kl` is the response-reduced old-policy/current-policy log-prob
  difference used to form the PPO importance ratio. It measures policy-update
  drift; it is not a reference-policy KL and does not show whether k1 or k2
  regularization is active.
- For REINFORCE++, reference-policy k1 shaping is already folded into
  `rollout/returns`. Comparing the same-step `rollout/returns` and
  `rollout/raw_reward` summaries exposes its empirical effect; there is no
  independent `train/kl_loss` for this variant.
- For REINFORCE++-baseline, `train/kl_loss` is the separately reduced k2
  reference-policy penalty. It is absent from the advantage and is added to
  the total loss with `--kl-loss-coef`.

Training metrics also continue to report policy loss and clip fraction.

## Testing

The numerical tests use an independent float64 reference that does not invoke
the production return, advantage, normalization or loss functions. Coverage
includes variable response lengths, padding, internal mask holes, finite and
non-finite sentinels outside the mask, zero rewards, zero variance, a single
valid token, a fully masked local rank, PPO clipping and response-reduced
policy/k2 losses. A Megatron-backend integration test also calls the production
`compute_advantages_and_returns` dispatcher. On a host without Megatron it
injects only the minimal `mpu` interface needed by that function, so the real
Relax dispatch and normalization code still execute rather than being mocked.

Distributed normalization is tested with two real Gloo processes and a real
`all_reduce`, including a case where one rank has no valid token. Its output is
compared with the independently concatenated global population.

See
`examples/algorithms/run-qwen3-0.6B-1xgpu-reinforce-plus-plus.sh` for the
parameterized Qwen3-0.6B recipe.

The equal-budget Qwen3-0.6B stability experiment, numerical evidence, curves,
and comparison with GRPO are documented in the
[training and numerical validation report](./reinforce-plus-plus-training-report.md).
