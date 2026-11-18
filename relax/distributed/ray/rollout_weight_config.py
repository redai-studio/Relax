# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Select rollout recipients for a training weight publication."""

from argparse import Namespace
from typing import Any

from relax.utils.quant_cast import resolve_receiver_quantization_config


def resolve_rollout_quantization_config(args: Namespace, hf_config: Any) -> dict | None:
    """Resolve one sender format for all engines receiving this publication."""
    default_path = getattr(args, "sglang_hf_checkpoint", None) or args.hf_checkpoint
    global_overrides = getattr(args, "sglang_json_model_override_args", None)
    global_quantization = getattr(args, "sglang_quantization", None)
    recipients = [("default", default_path, global_overrides, global_quantization)]
    if getattr(args, "sglang_config", None) is not None:
        from relax.distributed.ray.rollout import _resolve_sglang_config

        config = _resolve_sglang_config(args)
        # Tensor updates target the first model. DCS registers every YAML
        # model's engines under the shared rollout role.
        models = config.models if args.fully_async or args.hybrid else config.models[:1]
        recipients = []
        for model in models:
            model.resolve(args)
            for index, group in enumerate(model.engine_groups):
                if group.worker_type == "placeholder":
                    continue
                recipients.append(
                    (
                        f"{model.name}/{index}:{group.worker_type}",
                        group.overrides["model_path"],
                        group.overrides.get("json_model_override_args", global_overrides),
                        group.overrides.get("quantization", global_quantization),
                    )
                )

    return resolve_receiver_quantization_config(args, hf_config, recipients)
