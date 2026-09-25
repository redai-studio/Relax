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
"""

from typing import Dict, Iterable, List


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

_COMMUNICATION_SUFFIXES = ("-send", "-recv", "-all-reduce", "-all-gather", "-reduce-scatter", "-all-sync")


def group_of(name: str) -> str:
    """Return the coarse group of one Megatron timer name.

    Unknown names fall back to ``other`` instead of being guessed into a group:
    a wrong group is worse than an unclassified one.
    """
    stripped = name.strip()
    if not stripped:
        return GROUP_OTHER
    if stripped in STAGE_GROUPS:
        return STAGE_GROUPS[stripped]
    lowered = stripped.lower()
    if any(lowered.endswith(suffix) for suffix in _COMMUNICATION_SUFFIXES):
        return GROUP_COMMUNICATION
    if lowered.startswith("optimizer"):
        return GROUP_OPTIMIZER
    return GROUP_OTHER


def group_names(names: Iterable[str]) -> Dict[str, List[str]]:
    """Bucket timer names by group, preserving the input order."""
    buckets: Dict[str, List[str]] = {}
    for name in names:
        buckets.setdefault(group_of(name), []).append(name)
    return buckets


def coverage(names: Iterable[str]) -> Dict[str, object]:
    """Describe what the instrumented name set can and cannot show.

    Published alongside every measurement so a reader can see which of the
    requested coarse stages were actually observed and which were not.
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
    "SCHEMA_ONLY_GROUPS",
    "STAGE_GROUPS",
    "coverage",
    "group_names",
    "group_of",
]
