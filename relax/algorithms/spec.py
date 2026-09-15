# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Declarative descriptions of the RL algorithms Relax supports.

A single ``--advantage-estimator`` value used to be interpreted independently by
six different files: role lookup, reward normalisation, the advantage formula,
the policy loss formula, and two rounds of argument validation.  Adding an
algorithm meant finding all of them.  ``AlgorithmSpec`` collects that metadata
in one place, so a new algorithm is one dict entry plus its pure functions.

Fields hold **string identifiers**, not callables.  The advantage formula runs
inside the Ray Serve ``Advantages`` deployment while the policy loss runs inside
the Megatron worker; those two processes import different module subsets, so we
ship the algorithm name and let each process resolve the identifier against its
own table.  That also keeps this module free of heavy imports, which is what
makes the registry testable on a CPU-only runner.
"""

from dataclasses import dataclass


# The two enum-like fields below are consumed by equality checks in
# `relax/backends/megatron/loss.py` -- `advantage_normalization == "token_global"`
# at 659 and 819, `kl_level == "sequence"` at 919. Anything that is not the
# awaited string takes the *other* branch, so a typo in a registry entry does
# not fail, it silently selects a different formula. These sets are what
# `__post_init__` checks against.
KL_LEVELS = frozenset({"token", "sequence"})
ADVANTAGE_NORMALIZATIONS = frozenset({"whiten", "token_global"})


@dataclass(frozen=True)
class AlgorithmSpec:
    """Everything the training pipeline needs to know about one algorithm."""

    name: str

    # --- reward stage (rollout side, CPU, scalar in / scalar out) ---
    reward_normalizer: str
    """Key into :data:`relax.algorithms.rewards.REWARD_NORMALIZERS`."""

    # --- advantage stage ---
    advantage_fn: str
    """Key into :data:`relax.algorithms.advantages.ADVANTAGE_FNS`."""

    # --- policy loss stage ---
    policy_loss_fn: str
    """Key into :data:`relax.algorithms.policy.POLICY_LOSS_FNS`."""

    requires_complete_reward_groups: bool = False
    """Whether rollout/debug batching must preserve whole prompt groups.

    This is explicit rather than inferred from ``reward_normalizer != "none"``:
    a future sample- or batch-scoped normalizer must not accidentally trigger
    group-aware subsampling merely because it performs some normalization.
    """

    policy_scalar_metric_names: tuple[str, ...] = ()
    """Names for scalar diagnostics returned after loss and clip fraction.

    Policy adapters return these values in the same order. The policy caller
    rejects complex values and normalizes real scalars to float32. Preserve
    main's logging convention: one raw scalar per microbatch in sample mode;
    multiply by the microbatch's token count in per-token mode before the
    framework aggregates the logging vector.
    """

    kl_level: str = "token"
    """``"token"`` or ``"sequence"``; GSPO constrains the sequence as a whole."""

    advantage_normalization: str = "whiten"
    """How ``--normalize-advantages`` normalises, read in
    ``relax/backends/megatron/loss.py``.

    ``"whiten"`` is the masked whitening every algorithm used before REINFORCE++
    arrived. ``"token_global"`` is REINFORCE++'s global token-level
    normalisation, which also requires the mask-safe loss reducer -- the two go
    together, which is why one field drives both call sites rather than two.

    Those two call sites were the last place a *maths* decision was still made
    by comparing algorithm names, and unlike the remaining ``== "ppo"`` checks
    this one is not expressible through an existing field: it is orthogonal to
    ``needs_critic``.
    """

    needs_full_log_probs: bool = False
    """Whether the loss needs CP-gathered full-response log probs."""

    supports_context_parallel: bool = True
    """Whether the algorithm implementation supports CP-sharded responses.

    False rejects both static CP sharding and dynamic CP at startup. This
    covers the advantage and policy stages, independently of reward grouping.
    """

    # --- orchestration and validation ---
    needs_critic: bool = False
    requires_normalize_advantages: bool = False
    forbids_normalize_advantages: bool = False
    """Set when the token-level ``--normalize-advantages`` pass would undo what
    the estimator deliberately kept.

    RLOO's advantage carries the reward scale on purpose; re-whitening it after
    DP sharding puts back the standard-deviation division RLOO removes, and
    makes the result depend on how the batch was partitioned.
    """

    requires_rewards_normalization: bool = False
    """Set when the algorithm's reward stage is load-bearing rather than
    optional, so ``--disable-rewards-normalization`` would silently skip it."""

    min_group_size: int = 1
    """Smallest ``--n-samples-per-prompt`` the reward stage is defined for."""

    forbids_reward_side_kl: bool = False
    """Set when the estimator has no place to put a reward-side KL penalty.

    Both members here compute their advantage from a completion-level signal
    that the ``--kl-coef`` shaping term never enters, so a nonzero coefficient
    would be silently ignored rather than applied. ``--use-kl-loss`` with
    ``--kl-loss-coef`` remains available: that one is a separate loss term, not
    a reward modification.
    """

    requires_global_token_loss: bool = False
    """Set when the objective is only correct under
    ``--calculate-per-token-loss``.

    The default per-sample token-mean reducer divides each sample by its own
    response length, which reweights unequal-length responses by
    ``1 / response_length``. An objective with no ratio correction has nothing
    to absorb that reweighting, so it must be normalised by the global number
    of valid response tokens instead.
    """

    requires_on_policy_updates: bool = False
    """Set when every optimizer step must consume exactly the rollout that
    produced it.

    An objective without an importance-ratio correction cannot account for the
    policy having moved, so *five* separate configuration knobs have to agree:
    no ``--fully-async`` / ``--hybrid``, ``--max-staleness 0``,
    ``--num-steps-per-rollout 1``, ``rollout_batch_size *
    n_samples_per_prompt == global_batch_size``, and neither
    ``--partial-rollout`` nor ``--use-dynamic-global-batch-size`` (both let the
    effective batch size drift at runtime).

    They are one field rather than five because they have one cause. RLOO is
    currently the only member, so the bundling is a guess about the next
    unclipped estimator; if one ever needs four of the five, split this field
    then rather than adding an exception to it.

    Note this is a *stronger* statement than "cannot run fully-async": an
    algorithm can be sync-only for unrelated reasons (PPO's critic topology,
    or an advantage that needs batch-level statistics the async deployment
    only sees a slice of). Those get their own field; do not fold them here.
    """

    def __post_init__(self) -> None:
        """Reject an unsupported enum value while the registry is being built.

        ``ALGORITHM_SPECS`` is a module-level literal, so this runs at import:
        a typo cannot reach a worker, let alone a training step. The
        implementation identifiers are already resolved eagerly for the same
        reason (``_assert_spec_implementations_resolve`` in ``arguments.py``);
        these two fields were the ones left unchecked, and they are the ones
        whose failure is silent rather than loud -- a bad ``advantage_fn``
        raises a KeyError, a bad ``kl_level`` just trains with token-level KL
        and never says so.
        """
        for field, value, allowed in (
            ("kl_level", self.kl_level, KL_LEVELS),
            ("advantage_normalization", self.advantage_normalization, ADVANTAGE_NORMALIZATIONS),
        ):
            if value not in allowed:
                raise ValueError(
                    f"AlgorithmSpec({self.name!r}) has {field}={value!r}, which no call site matches; "
                    f"the run would silently take the default branch instead. "
                    f"Expected one of {sorted(allowed)}."
                )

        if any(not name for name in self.policy_scalar_metric_names):
            raise ValueError(f"AlgorithmSpec({self.name!r}) has an empty policy scalar metric name.")
        if len(set(self.policy_scalar_metric_names)) != len(self.policy_scalar_metric_names):
            raise ValueError(f"AlgorithmSpec({self.name!r}) has duplicate policy scalar metric names.")

    @property
    def is_group_normalized(self) -> bool:
        """Backward-compatible alias for group-aware batching consumers."""
        return self.requires_complete_reward_groups


# NOTE(dev): explicit dict literal, deliberately not decorator-based registration.
# The advantage formula and the policy loss execute in two different processes
# whose import graphs differ; decorator registration depends on "was this module
# imported?" and silently loses an algorithm when one side misses the import.
ALGORITHM_SPECS: dict[str, AlgorithmSpec] = {
    "grpo": AlgorithmSpec(
        name="grpo",
        reward_normalizer="group_mean_std",
        requires_complete_reward_groups=True,
        advantage_fn="grpo_broadcast",
        policy_loss_fn="ppo_clip",
    ),
    "gspo": AlgorithmSpec(
        name="gspo",
        reward_normalizer="group_mean_std",
        requires_complete_reward_groups=True,
        advantage_fn="grpo_broadcast",
        policy_loss_fn="ppo_clip",
        kl_level="sequence",
        needs_full_log_probs=True,
    ),
    "sapo": AlgorithmSpec(
        name="sapo",
        reward_normalizer="group_mean_std",
        requires_complete_reward_groups=True,
        advantage_fn="grpo_broadcast",
        policy_loss_fn="sapo",
    ),
    "cispo": AlgorithmSpec(
        name="cispo",
        reward_normalizer="group_mean_std",
        requires_complete_reward_groups=True,
        advantage_fn="grpo_broadcast",
        policy_loss_fn="cispo",
    ),
    "m2po": AlgorithmSpec(
        name="m2po",
        # Preserve main's raw reward path; changing the reward baseline is
        # an algorithm change, independent of registering its policy loss.
        reward_normalizer="none",
        advantage_fn="grpo_broadcast",
        policy_loss_fn="m2po",
        supports_context_parallel=False,
        policy_scalar_metric_names=(
            "ppo_kl_m2_before",
            "ppo_kl_m2_after",
            "m2po_eps_low",
            "m2po_eps_high",
        ),
    ),
    "rloo": AlgorithmSpec(
        name="rloo",
        # REINFORCE leave-one-out (arXiv:2402.14740). The baseline is the mean
        # of the *other* completions in the prompt group, so the reward stage
        # differs from GRPO's while the advantage stage -- broadcast the scalar
        # over the response tokens -- is the same one.
        reward_normalizer="group_leave_one_out",
        requires_complete_reward_groups=True,
        advantage_fn="grpo_broadcast",
        policy_loss_fn="rloo",
        requires_rewards_normalization=True,
        forbids_normalize_advantages=True,
        min_group_size=2,
        forbids_reward_side_kl=True,
        requires_global_token_loss=True,
        requires_on_policy_updates=True,
    ),
    "ppo": AlgorithmSpec(
        name="ppo",
        reward_normalizer="none",
        advantage_fn="gae",
        policy_loss_fn="ppo_clip",
        needs_critic=True,
    ),
    "reinforce_plus_plus": AlgorithmSpec(
        name="reinforce_plus_plus",
        advantage_normalization="token_global",
        reward_normalizer="none",
        advantage_fn="reinforce_plus_plus",
        policy_loss_fn="ppo_clip",
        supports_context_parallel=False,
        requires_normalize_advantages=True,
    ),
    "reinforce_plus_plus_baseline": AlgorithmSpec(
        name="reinforce_plus_plus_baseline",
        advantage_normalization="token_global",
        reward_normalizer="group_mean",
        requires_complete_reward_groups=True,
        advantage_fn="reinforce_plus_plus_baseline",
        policy_loss_fn="ppo_clip",
        supports_context_parallel=False,
        requires_normalize_advantages=True,
        # `_validate_reinforce_plus_plus_args` in `relax/utils/arguments.py`
        # already enforces these by hand, and it is a frozen Task 29 contract
        # that owns the wording its tests match on. Declaring them here is
        # still not redundant: an undeclared field is not neutral, it asserts
        # the default -- leaving them out would have this spec state that the
        # estimator runs fine with `--disable-rewards-normalization` and a
        # group of one, both false. The frozen function runs first on both
        # validation paths, so its messages still win.
        requires_rewards_normalization=True,
        min_group_size=2,
        forbids_reward_side_kl=True,
    ),
}


def get_algorithm(name: str) -> AlgorithmSpec:
    """Look up an algorithm spec by its ``--advantage-estimator`` value."""
    try:
        return ALGORITHM_SPECS[name]
    except KeyError:
        available = ", ".join(ALGORITHM_SPECS)
        raise KeyError(f"Unknown advantage estimator {name!r}. Available: {available}") from None


def algorithm_needs_critic(config) -> bool:
    """Whether the configured algorithm runs a critic, read from the registry.

    The training pipeline asks this in five places -- role topology, the
    critic's placement group, the device hand-off of the critic's ``values``,
    the critic consumer's rollout fields, and the critic's own wait loop -- and
    each of them used to compare ``advantage_estimator`` against ``"ppo"``.
    That worked while PPO was the only value-based estimator, and silently
    stopped working the moment the registry could accept a second one:
    ``--advantage-estimator`` and ``ALGOS`` would take it, then none of the
    value plumbing would switch on.

    ``args.use_critic`` carries the same answer, but only after
    ``validate_algorithm_args`` has run; ``process_role`` and the controller's
    placement logic read a config that may not have been through it yet. This
    reads the spec directly so the answer does not depend on call order.

    Unknown or missing estimators answer False rather than raising: SFT and the
    debug-only role paths reach these call sites with no estimator at all, and
    an unknown name is rejected by argument parsing long before this matters.
    """
    spec = ALGORITHM_SPECS.get(getattr(config, "advantage_estimator", None))
    return spec is not None and spec.needs_critic


def list_algorithm_names() -> list[str]:
    """All registered algorithm names, in definition order."""
    return list(ALGORITHM_SPECS)
