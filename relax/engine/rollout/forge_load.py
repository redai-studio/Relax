"""Replay a dumped rollout ``.pt`` instead of generating, so memory-test runs
keep sglang / router / weight-update and the colocate offload-onload dance
fully live while skipping real generation. Plug in via::

    --rollout-function-path relax.engine.rollout.forge_load.generate_rollout
    --load-forge-rollout-data /path/to/rollout_data/0.pt            (literal)
    --load-forge-rollout-data /path/to/rollout_data/{rollout_id}.pt (template)

Unlike ``--load-debug-rollout-data`` (which sets ``skip_sglang`` and only runs
training), this does NOT touch ``skip_sglang``: the real colocate memory profile
(train peak + weight sync + offload/onload transitions) is exercised. The dump
is the one produced by ``--save-debug-rollout-data`` / ``--dump-details``.
"""

import os
from argparse import Namespace
from collections import defaultdict
from typing import Any

import torch

from relax.engine.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from relax.utils.async_utils import run
from relax.utils.logging_utils import get_logger
from relax.utils.types import Sample
from relax.utils.utils import transfer_batch_to_data_system


logger = get_logger(__name__)


def _resolve_path(args: Namespace, rollout_id: int, evaluation: bool) -> str | None:
    """Resolve the dump path, mirroring the ``format(rollout_id=...)``
    convention of ``--save-debug-rollout-data``.

    A literal path (no ``{rollout_id}`` placeholder) is reused for every
    rollout; a template loads a per-rollout file. Literal mode cannot address
    eval dumps, so eval is a no-op there. The training path falls back to the
    ``0`` dump when a specific rollout_id file is missing (many memory tests
    keep only ``0.pt`` but want ``--num-rollout > 1``); eval has no such
    fallback.
    """
    tpl = getattr(args, "load_forge_rollout_data", None)
    if not tpl:
        raise RuntimeError(
            "--load-forge-rollout-data not set. Pass the dump path, e.g. "
            "/path/to/rollout_data/0.pt (literal) or "
            "/path/to/rollout_data/{rollout_id}.pt (template)."
        )
    if evaluation and "{rollout_id}" not in tpl:
        return None
    rid_str = ("eval_" if evaluation else "") + str(rollout_id)
    path = tpl.format(rollout_id=rid_str)
    if os.path.exists(path):
        return path
    if not evaluation:
        fallback = tpl.format(rollout_id="0")
        if os.path.exists(fallback):
            logger.info("forge_load: %s missing, falling back to %s", path, fallback)
            return fallback
    return None


def _load_samples(path: str) -> list[Sample]:
    blob = torch.load(path, weights_only=False)
    return [Sample.from_dict(s) for s in blob["samples"]]


def _group_samples(samples: list[Sample]) -> list[list[Sample]]:
    """Rebuild GRPO groups from the flat dump using ``group_index`` so the
    downstream grouping / advantage structure matches a real rollout.

    Samples without a ``group_index`` each become their own group.
    """
    groups: dict[Any, list[Sample]] = defaultdict(list)
    for i, sample in enumerate(samples):
        key = sample.group_index if sample.group_index is not None else f"_solo_{i}"
        groups[key].append(sample)
    return [groups[key] for key in groups]


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, data_system_client: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    path = _resolve_path(args, rollout_id, evaluation)

    if evaluation:
        # Eval replay is optional for a memory-test run; no dump -> no-op.
        if path is None:
            logger.info("forge_load: no eval dump found; returning empty eval result")
        else:
            logger.info("forge_load: eval replay not supported; returning empty eval result (dump=%s)", path)
        return RolloutFnEvalOutput(data={})

    if path is None:
        raise RuntimeError(
            f"forge_load: no dump found for rollout_id={rollout_id} "
            f"(--load-forge-rollout-data={getattr(args, 'load_forge_rollout_data', None)!r})"
        )

    samples = _load_samples(path)
    groups = _group_samples(samples)
    logger.info(
        "forge_load: replaying %d samples (%d groups) for rollout_id=%d from %s",
        len(samples),
        len(groups),
        rollout_id,
        os.path.basename(path),
    )
    # Push the replayed batch into the TransferQueue exactly like a real rollout;
    # is_last mirrors the single-flush semantics (streaming end-of-stream only in
    # fully-async mode, where each transfer closes the partition).
    run(
        transfer_batch_to_data_system(
            args,
            groups,
            len(samples),
            rollout_id,
            data_system_client,
            is_last=bool(getattr(args, "fully_async", False)),
        )
    )
    return RolloutFnTrainOutput(samples=groups)
