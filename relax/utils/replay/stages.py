# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Stage registry and the frozen V1 capability matrix.

Task 34 ships with a frozen capability matrix for the first delivery: only the
GRPO CP=1 main path is declared recompute. Any other topology or algorithm is
explicitly unsupported and must fail loudly in the runner rather than silently
degrade. New capabilities are added only by bumping this matrix with fresh
parity evidence.
"""

from __future__ import annotations

from collections.abc import Callable

from relax.utils.replay.schema import StageCapability, StageId


# Frozen V1 matrix. Keys are StageId; the value is the capability for the
# only supported topology (GRPO, CP=1, colocated or fully-async). Every other
# topology is resolved through capability_for and yields UNSUPPORTED.
V1_STAGE_CAPABILITIES: dict[StageId, StageCapability] = {
    StageId.SAMPLE: StageCapability.RECOMPUTE,
    StageId.REWARD_RAW: StageCapability.RECOMPUTE,
    StageId.REWARD_POST_PROCESS: StageCapability.RECOMPUTE,
    StageId.ADVANTAGE_KL: StageCapability.RECOMPUTE,
    StageId.ADVANTAGE_ESTIMATE: StageCapability.RECOMPUTE,
    StageId.LOSS_POLICY: StageCapability.RECOMPUTE,
    StageId.LOSS_VALUE: StageCapability.UNSUPPORTED,  # PPO value head out of V1 scope
}

# Stage contracts shipped with the frozen matrix.
V1_STAGE_VERSIONS: dict[StageId, str] = {stage: "v1" for stage in V1_STAGE_CAPABILITIES}

# Stages that restate production math instead of calling the same kernel. The
# reward group-norm lives in relax.utils.utils.post_process_rewards, a
# module that imports Ray at module scope; to keep the offline runner Ray-free
# we restate the (tiny, stable) group-norm here and pin it with the PR #65
# parity fixture.
REIMPLEMENTED_STAGES: frozenset[StageId] = frozenset({StageId.REWARD_POST_PROCESS})

# Adapter callable shape: (index, bundle_inputs, expected) -> list[StageResult].
StageAdapter = Callable[..., object]

# Registered by register_adapter; read by the runner.
_ADAPTERS: dict[StageId, StageAdapter] = {}


def register_adapter(stage: StageId) -> Callable[[StageAdapter], StageAdapter]:
    """Decorator registering the replay implementation for stage."""

    def _register(func: StageAdapter) -> StageAdapter:
        if stage in _ADAPTERS:
            raise ValueError(f"Duplicate replay adapter for stage {stage.value!r}")
        _ADAPTERS[stage] = func
        return func

    return _register


def get_adapter(stage: StageId) -> StageAdapter:
    """Return the registered adapter for stage."""
    try:
        return _ADAPTERS[stage]
    except KeyError as exc:
        raise ValueError(f"No replay adapter registered for stage {stage.value!r}") from exc


def topology_supported(*, advantage_estimator: str, context_parallel: int = 1) -> bool:
    """True when the frozen V1 matrix can recompute this bundle's topology."""
    return advantage_estimator == "grpo" and context_parallel == 1


def capability_for(stage: StageId, *, advantage_estimator: str, context_parallel: int = 1) -> StageCapability:
    """Resolve the V1 capability for stage under a given topology.

    Only GRPO with CP=1 is supported. Everything else — including PPO,
    SAPO/CISPO and any CP>1 topology — is explicitly unsupported so a caller
    can never silently claim a recompute it has not proven.
    """
    if not topology_supported(advantage_estimator=advantage_estimator, context_parallel=context_parallel):
        return StageCapability.UNSUPPORTED
    return V1_STAGE_CAPABILITIES.get(stage, StageCapability.UNSUPPORTED)
