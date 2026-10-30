# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Everything the profiler knows about Megatron: the names of its timers and
what they measure, the blocks of its transformer layers, its parallel state,
its metrics rank and its interleaved schedule.

A Megatron upgrade that renames a timer, moves a block or changes one of the
functions used here should fail the contract tests
(``tests/utils/test_straggler_megatron_contract.py``), not a training run.
Megatron is imported only inside the functions that call it, and of the rest
of Relax only ``core`` is imported here.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from relax.utils.straggler.core.modules import ModuleSpec, ModuleTimers, module_names
from relax.utils.straggler.core.stats import FIELD_INDEX
from relax.utils.straggler.core.timers import EventSink, StragglerTimers


# The Megatron timers of ``dp_grad_sync``, in the order finalize_model_grads
# runs them; the position is the bracket's phase slot. Only ``all-grads-sync``
# waits for the DP reduction (hence first, see ``DP_REDUCTION_PHASE``); the
# others reduce over pipeline-stage or tensor-parallel peers.
GRAD_SYNC_TIMERS: tuple[str, ...] = (
    "all-grads-sync",
    "conditional-embedder-grads-all-reduce",
    "non-tensor-parallel-grads-all-reduce",
    "embedding-grads-all-reduce",
)

# Megatron timer name -> segment, for the training phase. Names come from
# megatron/core/pipeline_parallel/schedules.py, p2p_communication.py,
# distributed/finalize_model_grads.py, optimizer/optimizer.py and
# optimizer/distrib_optimizer.py.
MEGATRON_TIMER_SEGMENTS: Mapping[str, str] = {
    "forward-compute": "fwd",
    "backward-compute": "bwd",
    "forward-recv": "pp_recv",
    "backward-recv": "pp_recv",
    "forward-send-backward-recv": "pp_recv",
    "backward-send-forward-recv": "pp_recv",
    "forward-send-forward-recv": "pp_recv",
    "backward-send-backward-recv": "pp_recv",
    "forward-backward-send-forward-backward-recv": "pp_recv",
    "forward-send": "pp_send",
    "backward-send": "pp_send",
    **dict.fromkeys(GRAD_SYNC_TIMERS, "dp_grad_sync"),
    "params-all-gather": "dp_param_gather",
    "optimizer-copy-to-main-grad": "optim",
    "optimizer-unscale-and-check-inf": "optim",
    "optimizer-clip-main-grad": "optim",
    "optimizer-count-zeros": "optim",
    "optimizer-inner-step": "optim",
    "optimizer-copy-main-to-model-params": "optim",
}

# Timer names left untimed on purpose: ``forward-backward`` wraps a whole step
# and duplicates the segment timers (and Relax's own ``actor_train`` timer).
IGNORED_TIMER_NAMES: frozenset[str] = frozenset({"forward-backward"})

# Megatron timer name -> PP_PHASE_FIELDS entry, training phase only. In the
# non-interleaved 1F1B schedules (Megatron's and Relax's streaming one) the
# warmup only calls recv_forward, the steady state only the two combined
# send / recv calls and the cooldown only recv_backward, so the name gives the
# phase and the neighbour waited for: ``up`` waits for the previous stage's
# activations, ``down`` for the next stage's gradients. A balanced pipeline does
# not idle in the steady state, so ``up + down`` beyond the transfer time is
# extra bubble. The interleaved schedule's names count in ``pp_recv`` only.
PP_PHASE_TIMERS: Mapping[str, str] = {
    "forward-recv": "pp_wait_fill",
    "backward-send-forward-recv": "pp_wait_up",
    "forward-send-backward-recv": "pp_wait_down",
    "backward-recv": "pp_wait_drain",
}

# Megatron timer name -> phase slot of its brackets, training phase only: the
# PP_PHASE_FIELDS index of a PP wait, or a gradient sync's position in
# GRAD_SYNC_TIMERS. Other brackets carry -1.
TIMER_PHASES: Mapping[str, int] = {
    **{name: FIELD_INDEX[phase] for name, phase in PP_PHASE_TIMERS.items()},
    **{name: position for position, name in enumerate(GRAD_SYNC_TIMERS)},
}

# Same call sites during ``forward_only`` land in their own buckets so a slow
# log-prob pass does not pollute the training-step numbers.
FORWARD_ONLY_TIMER_SEGMENTS: Mapping[str, str] = {
    name: {"fwd": "lp_fwd", "pp_recv": "lp_pp_recv", "pp_send": "lp_pp_send"}.get(segment, segment)
    for name, segment in MEGATRON_TIMER_SEGMENTS.items()
    if segment in ("fwd", "pp_recv", "pp_send")
}


# The timers every step of a phase calls, whatever the parallel layout; the
# profiler reports any that its first rollout did not reach. The optimizer's is
# the one before the inf / nan check, which can skip the rest of its step.
REQUIRED_TIMER_NAMES: Mapping[str, frozenset[str]] = {
    "train": frozenset({"forward-compute", "backward-compute", "all-grads-sync", "optimizer-copy-to-main-grad"}),
    "forward_only": frozenset({"forward-compute"}),
}


def phase_timers(sink: EventSink) -> dict[str, StragglerTimers]:
    """The ``config.timers`` of each phase, feeding ``sink`` (the recorder).

    One object per phase, so that the same call sites land in different buckets
    during training and forward-only passes.
    """
    return {
        "train": StragglerTimers(sink, MEGATRON_TIMER_SEGMENTS, TIMER_PHASES, IGNORED_TIMER_NAMES),
        "forward_only": StragglerTimers(sink, FORWARD_ONLY_TIMER_SEGMENTS, ignored=IGNORED_TIMER_NAMES),
    }


MODULE_SPECS: tuple[ModuleSpec, ...] = (
    ModuleSpec("self_attention", "attn"),
    ModuleSpec("mlp", "moe", when="MoE"),
    ModuleSpec("mlp", "mlp"),
    # Inside a MoE layer: routing, the token exchange around the experts (the EP all-to-all) and the experts.
    ModuleSpec("mlp.router", "router"),
    ModuleSpec("mlp.token_dispatcher", "dispatch", method="token_dispatch", all_tensors=True),
    ModuleSpec("mlp.experts", "experts", all_tensors=True),
    ModuleSpec("mlp.token_dispatcher", "combine", method="token_combine"),
    ModuleSpec("mlp.shared_experts", "shared experts"),
)
MODULE_NAMES: tuple[str, ...] = module_names(MODULE_SPECS)

# Megatron's placeholders for blocks a layer does not have.
PLACEHOLDERS = frozenset({"IdentityOp", "IdentityFuncOp"})


def layer_number(module: Any) -> int | None:
    """A transformer layer's number, negated for a layer inside an MTP layer
    (those number from 1 again); ``None`` for any other module."""
    layer = getattr(module, "layer_number", None)
    if layer is None:
        return None
    return -int(layer) if getattr(module, "is_mtp_layer", False) else int(layer)


def module_timers(model_chunks: Iterable[Any], sink: EventSink) -> ModuleTimers:
    """The ``MODULE_SPECS`` blocks of every transformer layer in
    ``model_chunks``."""
    return ModuleTimers(model_chunks, sink, MODULE_SPECS, layer_number, PLACEHOLDERS)


def parallel_ranks() -> dict[str, int]:
    """This rank's position in Megatron's parallel groups (DP with CP)."""
    from megatron.core import mpu

    return {
        "dp": mpu.get_data_parallel_rank(with_context_parallel=True),
        "tp": mpu.get_tensor_model_parallel_rank(),
        "pp": mpu.get_pipeline_model_parallel_rank(),
        "cp": mpu.get_context_parallel_rank(),
        "ep": mpu.get_expert_model_parallel_rank(),
    }


def is_metrics_primary_rank() -> bool:
    """The rank whose ``log_perf_data`` logs the perf metrics, the same rule as
    ``relax.backends.megatron.data.log_perf_data``: TP 0, last PP stage, DP
    (with CP) 0."""
    from megatron.core import mpu

    return (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.is_pipeline_last_stage()
        and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
    )


def interleaved_schedule_table(
    model: Sequence[Any], num_microbatches: int | Sequence[int], forward_only: bool
) -> Sequence[tuple[int, int]]:
    """Megatron's interleaved schedule table for the forward-backward calls of
    ``num_microbatches`` (one count per call), joined in call order with each
    call's micro-batches numbered after the previous call's; empty for training
    under combined 1F1B, which merges forward and backward steps.

    The chunk count and config come from the model, as the schedule reads them,
    not from the launch arguments: Bridge does not copy every argument into the
    model config.
    """
    from megatron.core.pipeline_parallel.schedules import get_schedule_table
    from megatron.core.utils import get_model_config

    config = get_model_config(model[0])
    if config.overlap_moe_expert_parallel_comm and not forward_only:
        return ()
    calls = [num_microbatches] if isinstance(num_microbatches, int) else num_microbatches
    table: list[tuple[int, int]] = []
    offset = 0
    for count in calls:
        table += [
            (offset + micro_batch, chunk)
            for micro_batch, chunk in get_schedule_table(count, len(model), config.microbatch_group_size_per_vp_stage)
        ]
        offset += count
    return table
