# Adding an Algorithm

Algorithms plug into Relax through the registry under `relax/algorithms/`. An
algorithm name is no longer scattered across `if/elif` chains — it is described
by one `AlgorithmSpec`, and each stage looks up what it needs.

## Registry Layout

```
relax/algorithms/
├── spec.py        AlgorithmSpec definition + the ALGORITHM_SPECS registry
├── rewards.py     reward normalization strategies + REWARD_NORMALIZERS
├── advantages.py  advantage estimators + ADVANTAGE_FNS
└── policy.py      policy loss adapters + POLICY_LOSS_FNS
```

Three hard constraints:

1. **No heavy top-level imports under `relax/algorithms/`** — not `megatron`,
   `ray`, `transfer_queue`, `tensordict`, `relax.components` or
   `relax.backends`. The registry is imported by argument parsing and by both
   worker processes; one heavy import drags the whole training stack into
   `--help` and into a CPU-only CI runner. Import inside the function when you
   genuinely need one.
2. **Spec fields hold string identifiers, not callables.** The advantage
   computation runs in the Ray Serve `Advantages` process while the policy loss
   runs in the Megatron worker, and those two import different module subsets.
   Only the algorithm name crosses the process boundary; each side resolves it
   against its own table.
3. **Do not hand-edit the `ALGOS` role table.** It is derived from the registry,
   so a new algorithm gets the standard RL role set automatically.

## How Much Does Adding One Cost

Honestly: **not "one dict entry".**

| Situation | Files to touch |
|---|---|
| Reuses existing reward normalization / advantage / policy loss, just combined differently | 1 (`spec.py`) |
| Needs new maths (a new advantage formula, say) | 2-3 (`spec.py` plus the implementation module) |
| Also needs new command-line options | 4-6 (the above, plus the option and its validation in `arguments.py`, plus an example and docs) |

What the registry removes is one algorithm name being interpreted in six
scattered if/elif chains — not the cost of adding an algorithm. An algorithm
that needs both new maths and new options lands in the last row.

The `ALGOS` role table is the one part that genuinely costs nothing: it derives
itself from the registry.

## Steps

### 1. Add a spec entry

Edit `ALGORITHM_SPECS` in `relax/algorithms/spec.py`:

```python
"my_algo": AlgorithmSpec(
    name="my_algo",
    reward_normalizer="group_mean_std",   # reuse an existing one, or see step 2
    requires_complete_reward_groups=True,
    advantage_fn="grpo_broadcast",
    policy_loss_fn="ppo_clip",
),
```

If your algorithm is identical to an existing one at some stage, reuse that
identifier. GRPO, GSPO, SAPO, CISPO, M2PO and RLOO all broadcast the scalar
reward at the advantage layer, so they share `"grpo_broadcast"`. Their reward
preprocessing differs: M2PO keeps `reward_normalizer="none"` to preserve its
existing behavior.

Capability fields:

| Field | Effect |
|-------|--------|
| `requires_complete_reward_groups` | Preserve complete prompt groups during debug subsampling when reward processing relies on group-level statistics; currently consumed by debug-data selection |
| `kl_level` | `"token"` or `"sequence"` (GSPO constrains the sequence) |
| `needs_full_log_probs` | Whether the loss needs CP-gathered full log probs |
| `supports_context_parallel` | Whether the algorithm's advantage and policy paths support CP-sharded responses; defaults to `True`. `False` rejects static CP sizes other than 1 and enabled dynamic CP at startup |
| `policy_scalar_metric_names` | Names, in return order, for extra scalar diagnostics produced by the policy adapter |
| `advantage_normalization` | What `--normalize-advantages` does: `"whiten"` (masked whitening) or `"token_global"` (REINFORCE++'s global token-level normalization, which also switches on the mask-safe loss reducer) |
| `needs_critic` | Whether a critic service is required; drives `args.use_critic` |
| `requires_normalize_advantages` | Demand `--normalize-advantages` |
| `forbids_normalize_advantages` | Reject `--normalize-advantages` (the estimator keeps the advantage's scale on purpose) |
| `requires_rewards_normalization` | Reject `--disable-rewards-normalization` |
| `min_group_size` | Floor on `--n-samples-per-prompt` |
| `forbids_reward_side_kl` | Demand `--kl-coef 0`; there is nowhere to put a reward-side KL term (`--use-kl-loss` is unaffected) |
| `requires_global_token_loss` | Demand `--calculate-per-token-loss`; the per-sample token-mean reducer would reweight responses by `1 / response_length` |
| `requires_on_policy_updates` | Rejects five knobs at once: `--fully-async` / `--hybrid`, `--max-staleness != 0`, `--num-steps-per-rollout != 1`, `rollout_batch_size * n_samples != global_batch_size`, and `--partial-rollout` / `--use-dynamic-global-batch-size`. For objectives with no importance-ratio correction |

M2PO and both REINFORCE++ variants declare `supports_context_parallel=False`.
This describes the whole algorithm, including advantage computation and policy
loss. It is independent of `requires_complete_reward_groups`, which concerns
samples sharing a prompt rather than tokens within a response. M2PO keeps
`requires_complete_reward_groups=False` because its reward stage performs no
group normalization.

The `validate_*` functions in `relax/utils/arguments.py` consume the startup
constraint fields. Runtime consumers read the remaining capabilities: reward
dispatch uses `reward_normalizer`, debug subsampling uses
`requires_complete_reward_groups`, and the policy path uses the KL,
normalization, full-log-probability and scalar-metric declarations.
Declaring an existing capability is enough; do not add an algorithm-name `if`
to those consumers. A genuinely new enum value or implementation still needs
one generic handler for that value.

### 2. Write pure functions for genuinely new maths

Only needed when your algorithm differs from every existing one at that stage.

**Reward normalization** (`relax/algorithms/rewards.py`), signature
`fn(args, samples, raw_rewards) -> list[float]`:

```python
def normalize_my_strategy(args, samples, raw_rewards):
    positions_by_group = group_positions(samples, args.n_samples_per_prompt)
    ...
    return normalized  # one scalar per sample

REWARD_NORMALIZERS["my_strategy"] = normalize_my_strategy
```

The output must be **one scalar per sample**. That constraint is what keeps the
TransferQueue schema fixed — an algorithm reading several reward components
collapses them to a scalar here.

**Advantage estimator** (`relax/algorithms/advantages.py`), signature
`fn(args, *, rewards, kl, loss_masks, response_lengths, total_lengths, values)`
returning `(advantages, returns)`, both `list[Tensor]`:

```python
def advantage_my_algo(args, *, rewards, kl, **_unused):
    ...
    return advantages, returns

ADVANTAGE_FNS["my_algo"] = advantage_my_algo
```

**Policy loss** (`relax/algorithms/policy.py`), signature
`fn(args, *, log_probs, ppo_kl, advantages) -> (pg_loss,
pg_clipfrac, *scalar_metrics)`. The underlying kernels take different argument
lists; the adapter normalizes them. Most adapters return only the first two
values. If yours returns scalar diagnostics, declare their names in
`policy_scalar_metric_names` in the same order. Each diagnostic must contain
exactly one value. Scalar logging preserves the existing M2PO convention:
sample mode contributes each microbatch scalar unchanged before the framework
divides by the sample count; token mode first multiplies it by the microbatch's
token count. This is not a sample-weighted mean of microbatch diagnostics.
Real diagnostics are normalized to float32 so one adapter cannot promote the
distributed logging vector; complex values are rejected.

### 3. Write unit tests

Tests under `tests/algorithms/` run on CPU with pytest, torch, NumPy and PyYAML.
Unrelated training dependencies are isolated in test fixtures; megatron, ray,
tensordict and transfer_queue are not required:

```bash
pytest tests/algorithms/ -v
```

Cover at least:

- Registration and dispatch: the name is in `ALGORITHM_SPECS`, capability fields
  match expectations, an unregistered name raises.
- Numerics: hand-compute a small case as the reference. Do not use all-zero or
  all-equal rewards — every formula returns 0 on those, so the test proves
  nothing.
- Degenerate cases: a group where all rewards are equal, boundary values of
  `n_samples_per_prompt`, missing fields, non-numeric input.
- **When changing an existing algorithm**: freeze the old implementation into
  the test file as a reference and compare bit-for-bit
  (`view(torch.int32).equal`). Do not use `allclose` — its default tolerance is
  wide enough to swallow the difference between a biased and an unbiased
  standard deviation. `tests/algorithms/test_reward_normalizers.py` is a
  worked example.

### 4. Add an example and documentation

- `examples/<algo>/`: a launch script, plus a custom reward function if needed.
- `docs/{zh,en}/examples/algorithms.md`: how it works, the parameter table, a
  quick start, and **known deviations** — write down where the implementation
  differs from the paper rather than leaving users to discover it.

## Arguments

Algorithm-specific options go in `add_algo_arguments` in
`relax/utils/arguments.py`. The `--advantage-estimator` choices come from
`list_algorithm_names()`, so registering is enough; there is no name list to
maintain.

Put cross-argument validation in `validate_algorithm_args`, and prefer
expressing it through a spec field over comparing algorithm names — the latter
is exactly what this registry exists to remove.

## References

- [Algorithm Reference](../examples/algorithms.md)
