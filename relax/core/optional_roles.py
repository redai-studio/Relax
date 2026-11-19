# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Optional role registration helpers for controller-side wiring."""

from argparse import Namespace

from relax.engine.sft.runtime import is_sft_mode


GENRM_ROLE = "genrm"


def register_genrm(config: Namespace, algo: dict) -> list[str]:
    """Conditionally register GenRM into the algo dict.

    Triggered by the normalized ``_genrm_instances_resolved`` (populated at
    arg-parse time from either --genrm-instances or the legacy
    --genrm-model-path), not directly by --genrm-model-path, so both single-
    and multi-instance configs go through the same gate.
    """
    if not getattr(config, "_genrm_instances_resolved", None):
        return []

    from relax.components.inference_gateway import InferenceGatewayService

    algo[GENRM_ROLE] = InferenceGatewayService
    return [GENRM_ROLE]


def register_sft_rollout(config: Namespace, algo: dict) -> list:
    """Conditionally register Rollout into the SFT algo dict.

    Only ``loss_type == "sft"`` supports generation prediction; other offline
    objectives reuse the producer and actor roles without a Rollout service.

    Rollout is only spun up when ``--sft-predict-interval`` is set — that is
    the sole consumer of generative eval under SFT today.
    """
    if not is_sft_mode(config):
        return []
    if getattr(config, "sft_predict_interval", None) is None:
        return []

    from relax.components.rollout import Rollout
    from relax.core.registry import ROLES

    algo[ROLES.rollout] = Rollout
    return [ROLES.rollout]


def register_extra_roles(config: Namespace, algo: dict) -> list:
    """Register optional roles and return them in controller iteration
    order."""
    extras = []
    extras.extend(register_genrm(config, algo))
    from relax.utils.opd.opd_utils import is_managed_opd_teacher_enabled

    if is_managed_opd_teacher_enabled(config):
        from relax.components.inference_gateway import InferenceGatewayService

        algo["teacher"] = InferenceGatewayService
        extras.append("teacher")
    extras.extend(register_sft_rollout(config, algo))
    return extras
