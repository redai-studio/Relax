# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from .configuration import register_hf_config


register_hf_config()

__all__ = ["DeepSeekV41Bridge", "DeepSeekV41ModelProvider"]


def __getattr__(name: str) -> type:
    # Checkpoint metadata validation is also used by inference-only processes.
    if name == "DeepSeekV41Bridge":
        from .bridge import DeepSeekV41Bridge

        return DeepSeekV41Bridge
    if name == "DeepSeekV41ModelProvider":
        from .provider import DeepSeekV41ModelProvider

        return DeepSeekV41ModelProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
