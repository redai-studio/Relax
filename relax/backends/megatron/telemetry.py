# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Opt-in glue for the pinned Megatron/Lens observability smoke recipe.

Relax owns the training loop, so Megatron's pretrain telemetry setup never
runs. The recipe calls ``initialize`` through the existing custom
initialization hook. Native Megatron decorators then use the same Lens provider
as these Relax spans.
"""

import atexit
import os
import uuid
from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)
_handle: Any = None


def initialize(args: Namespace) -> None:
    """Initialize Lens once per training process, after distributed setup."""
    global _handle
    if _handle is not None:
        return
    if os.environ.get("MEGATRON_OTEL_ENABLED", "0").lower() not in ("1", "true", "yes"):
        return

    from megatron.core.telemetry.span_groups import MegatronSpanGroup
    from megatron.training import global_vars
    from nemo.lens import NemoLensConfig, setup_telemetry

    config = NemoLensConfig.from_env(
        prefix="MEGATRON_OTEL", fallback_prefix="NEMO_LENS", span_group_cls=MegatronSpanGroup
    )
    if not config.run_id:
        raise ValueError("Set NEMO_LENS_RUN_ID to one shared run ID before submitting the Ray job.")
    config.service_name = os.environ.get("OTEL_SERVICE_NAME", "relax-megatron-smoke")
    run_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"relax/megatron-otel/{config.run_id}"))
    _handle = setup_telemetry(
        config,
        rank=args.rank,
        world_size=args.world_size,
        resource_attributes={
            "relax.run.id": config.run_id,
            "nemo.lens.job_uuid": run_uuid,
            "nemo.lens.run_uuid": run_uuid,
            "dl.tensor_parallel.size": args.tensor_model_parallel_size,
            "dl.pipeline_parallel.size": args.pipeline_model_parallel_size,
            "dl.context_parallel.size": args.context_parallel_size,
            "dl.expert_parallel.size": args.expert_model_parallel_size,
        },
    )
    # The pinned MCore exposes a getter but no setter. Native checkpoint and
    # schedule code consult this handle in addition to Lens's span-group state.
    global_vars._GLOBAL_TELEMETRY_HANDLE = _handle
    atexit.register(_handle.shutdown)
    logger.info(
        "Native Megatron OTel initialized: run=%s rank=%s exporting=%s exporter=%s groups=%s",
        config.run_id,
        args.rank,
        _handle.is_exporting,
        config.exporter,
        config.span_groups,
    )


def span(group: str, name: str, **attributes: Any) -> Any:
    """Use native Lens span groups without importing optional packages when
    off."""
    if _handle is None:
        return nullcontext()
    from nemo.lens.helpers import managed_span

    return managed_span(group, name, tracer=_handle.tracer, **attributes)


def flush() -> None:
    """Export completed step spans before a short Ray job kills its workers."""
    if _handle is None or not _handle.is_exporting:
        return
    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        try:
            if not provider.force_flush(timeout_millis=5000):
                logger.warning("Megatron OTel trace flush timed out")
        except Exception:
            logger.exception("Megatron OTel trace flush failed")


@contextmanager
def observe_step(rollout_id: int, step_id: int, role: str) -> Iterator[Any]:
    """Parent native spans to a Relax update and flush on success or
    failure."""
    try:
        with span(
            "step",
            "relax.train.step",
            **{"relax.rollout_id": rollout_id, "relax.step_id": step_id, "relax.role": role},
        ) as current:
            yield current
    finally:
        flush()
