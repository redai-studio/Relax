# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Coarse stage groups for the timer names Megatron actually emits.

Task 11 asks the tool to identify "forward / backward, communication, optimizer,
attention/MoE" coarse costs. Which names exist is not a design choice: it is a
property of the Megatron version Relax pins. The inventory below was taken from
the working stack on this machine
(``/root/autodl-tmp/megatron-stack/Megatron-LM``, Megatron core 0.19.0) by

    grep -rhoE "timers\\(['\\"][^'\\"]+['\\"]" --include=*.py megatron/ | sort -u

Its relevant properties, all measured rather than assumed:

* ``forward-backward`` and ``optimizer`` are the only level-1 phase timers;
* ``forward-compute``, ``backward-compute``, ``forward-step``, the send/recv
  pairs, ``params-all-gather``, ``all-grads-sync`` and the ``*-grads-all-reduce``
  names carry no explicit level, and :meth:`megatron.core.timers.Timers.__call__`
  defaults an absent level to the maximum (2), so this package captures them;
* **there is no attention or MoE timer name in this version** (no ``attention``,
  ``moe``, ``router``, ``dispatch`` or ``combine`` timer exists in the tree), so
  attention/MoE costs cannot be split out from ``forward-compute`` without a
  deep hook. That capability is declared here as *schema only* and is not
  claimed as supported — the mentor already agreed attention/MoE may be reserved
  behind a flag for the first phase.

Communication naming is deliberately conservative: a CUDA-event interval around a
send/recv or all-reduce timer measures the *observable interval*. With compute
and communication streams overlapping it is **not** the NCCL kernel execution
time, so the group is named ``communication`` and its intervals are reported as
``collective_interval`` rather than as communication cost.

Classification is closed and explicit: :func:`group_of` reads
:data:`STAGE_GROUPS` and nothing else, so an unrecognised name returns
``other`` instead of being guessed into a plausible-looking group from its
spelling. The 23 names Relax's core path actually emits are listed in
:data:`OBSERVED_TIMER_NAMES`; every one of them has an explicit entry. The
coarse group reaches a reader through the platform reporter, which logs the raw
timer name and its group together (see ``reporter.py``) without widening the
wire envelope.
"""

from typing import Any, Dict, Iterable, List, Mapping, Tuple


GROUP_FORWARD = "forward"
GROUP_BACKWARD = "backward"
GROUP_OPTIMIZER = "optimizer"
GROUP_COMMUNICATION = "communication"
GROUP_DATA = "data"
GROUP_SETUP = "setup"
GROUP_EVAL = "eval"
GROUP_OTHER = "other"

#: Groups whose members are measured directly by timers present in the pinned
#: Megatron version.
MEASURED_GROUPS = (
    GROUP_FORWARD,
    GROUP_BACKWARD,
    GROUP_OPTIMIZER,
    GROUP_COMMUNICATION,
    GROUP_DATA,
    GROUP_SETUP,
    GROUP_EVAL,
)

#: Capabilities declared in the envelope schema but *not* claimed as supported.
#: Attention and MoE are not instrumented by name in Megatron core 0.19.0; a
#: deep hook would be required and is deliberately out of the first phase.
SCHEMA_ONLY_GROUPS = ("attention", "moe")

#: Every timer name Relax's core training path passes to Megatron's
#: ``config.timers``, in the order the audit reported them. This is the set the
#: observer can actually record, so it is also the set
#: :func:`group_of` must classify; a name added here without an explicit
#: :data:`STAGE_GROUPS` entry fails the taxonomy test instead of silently
#: falling to ``other`` at runtime.
OBSERVED_TIMER_NAMES: Tuple[str, ...] = (
    "forward-backward",
    "forward-compute",
    "backward-compute",
    "forward-send",
    "forward-recv",
    "backward-send",
    "backward-recv",
    "forward-send-forward-recv",
    "forward-send-backward-recv",
    "backward-send-forward-recv",
    "backward-send-backward-recv",
    "forward-backward-send-forward-backward-recv",
    "all-grads-sync",
    "non-tensor-parallel-grads-all-reduce",
    "embedding-grads-all-reduce",
    "conditional-embedder-grads-all-reduce",
    "params-all-gather",
    "optimizer-inner-step",
    "optimizer-copy-to-main-grad",
    "optimizer-unscale-and-check-inf",
    "optimizer-copy-main-to-model-params",
    "optimizer-clip-main-grad",
    "optimizer-count-zeros",
)

#: The one place a timer name becomes a group. Only exact matches here are
#: honoured; there are deliberately no suffix/prefix rules, because a name that
#: merely *looks* like communication or optimizer is not evidence that it is.
STAGE_GROUPS: Dict[str, str] = {
    # Phase timers.
    "forward-backward": GROUP_FORWARD,
    "forward-compute": GROUP_FORWARD,
    "forward-step": GROUP_FORWARD,
    "backward-compute": GROUP_BACKWARD,
    # Optimizer: the phase timer plus the inner steps Relax and Megatron emit.
    "optimizer": GROUP_OPTIMIZER,
    "optimizer-inner-step": GROUP_OPTIMIZER,
    "optimizer-clip-main-grad": GROUP_OPTIMIZER,
    "optimizer-unscale-and-check-inf": GROUP_OPTIMIZER,
    "optimizer-copy-main-to-model-params": GROUP_OPTIMIZER,
    "optimizer-copy-to-main-grad": GROUP_OPTIMIZER,
    "optimizer-count-zeros": GROUP_OPTIMIZER,
    # Communication and overlap.
    "params-all-gather": GROUP_COMMUNICATION,
    "all-grads-sync": GROUP_COMMUNICATION,
    "non-tensor-parallel-grads-all-reduce": GROUP_COMMUNICATION,
    "embedding-grads-all-reduce": GROUP_COMMUNICATION,
    "conditional-embedder-grads-all-reduce": GROUP_COMMUNICATION,
    "forward-send": GROUP_COMMUNICATION,
    "forward-recv": GROUP_COMMUNICATION,
    "backward-send": GROUP_COMMUNICATION,
    "backward-recv": GROUP_COMMUNICATION,
    "forward-send-forward-recv": GROUP_COMMUNICATION,
    "forward-send-backward-recv": GROUP_COMMUNICATION,
    "backward-send-forward-recv": GROUP_COMMUNICATION,
    "backward-send-backward-recv": GROUP_COMMUNICATION,
    "forward-backward-send-forward-backward-recv": GROUP_COMMUNICATION,
    # Data pipeline.
    "batch-generator": GROUP_DATA,
    "train/valid/test-data-iterators-setup": GROUP_DATA,
    # Setup / load / evaluation.
    "model-and-optimizer-setup": GROUP_SETUP,
    "load-checkpoint": GROUP_SETUP,
    "load-pretrained-checkpoint": GROUP_SETUP,
    "tokenizer-setup": GROUP_SETUP,
    "gpu-sniff-test": GROUP_SETUP,
    "eval-time": GROUP_EVAL,
    "evaluate": GROUP_EVAL,
    "iteration-time": GROUP_EVAL,
    "interval-time": GROUP_EVAL,
}


def group_of(name: str) -> str:
    """Return the coarse group of one Megatron timer name.

    Only names explicitly listed in :data:`STAGE_GROUPS` are classified. Any
    unrecognised name - however plausible its spelling - returns ``other`` and
    is never guessed into a measured group: a wrong stage label is worse than
    an unclassified one. The function never raises, whatever the input.
    """
    if not isinstance(name, str):
        return GROUP_OTHER
    stripped = name.strip()
    if not stripped:
        return GROUP_OTHER
    return STAGE_GROUPS.get(stripped, GROUP_OTHER)


def with_stage_group(facts: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``facts`` with a coarse ``stage_group`` next to ``stage``.

    A straggler verdict already carries the raw Megatron timer name under
    ``facts["stage"]``. A consumer that wants to localise a measurement can
    pass those facts here and read the raw name and its coarse group together;
    the mapping is derived, so nothing has to travel on the wire. Facts without
    a string ``stage`` are copied through unchanged, and a name that is not in
    the taxonomy is labelled ``other`` rather than guessed.
    """
    annotated = dict(facts)
    stage = annotated.get("stage")
    if isinstance(stage, str) and stage:
        annotated["stage_group"] = group_of(stage)
    return annotated


def group_names(names: Iterable[str]) -> Dict[str, List[str]]:
    """Bucket timer names by group, preserving the input order."""
    buckets: Dict[str, List[str]] = {}
    for name in names:
        buckets.setdefault(group_of(name), []).append(name)
    return buckets


def coverage(names: Iterable[str]) -> Dict[str, object]:
    """Describe what a given instrumented name set can and cannot show.

    This is an offline audit helper, not a runtime publisher: the runtime
    reporter does not call it and it never travels on the wire. Tests and
    diagnostic tooling use it to check that every observed timer name maps to a
    measured group, that the schema-only groups stay unmeasured, and that
    nothing ends up unclassified.
    """
    buckets = group_names(names)
    return {
        "groups": {group: sorted(buckets.get(group, [])) for group in MEASURED_GROUPS if buckets.get(group)},
        "unclassified": sorted(buckets.get(GROUP_OTHER, [])),
        "measured_groups": sorted(group for group in MEASURED_GROUPS if buckets.get(group)),
        "schema_only_groups": list(SCHEMA_ONLY_GROUPS),
        "missing_groups": sorted(group for group in MEASURED_GROUPS if not buckets.get(group)),
    }


__all__ = [
    "GROUP_BACKWARD",
    "GROUP_COMMUNICATION",
    "GROUP_DATA",
    "GROUP_EVAL",
    "GROUP_FORWARD",
    "GROUP_OPTIMIZER",
    "GROUP_OTHER",
    "GROUP_SETUP",
    "MEASURED_GROUPS",
    "OBSERVED_TIMER_NAMES",
    "SCHEMA_ONLY_GROUPS",
    "STAGE_GROUPS",
    "coverage",
    "group_names",
    "group_of",
    "with_stage_group",
]
