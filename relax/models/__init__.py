# Copyright (c) 2026 Relax Authors. All Rights Reserved.

try:
    from megatron.bridge.models.qwen_omni import (  # type: ignore[attr-defined]  # noqa: F401
        Qwen3OmniModelProvider,
        Qwen3OmniMoEBridge,
        Qwen3OmniMoeModel,
    )
except (ImportError, AttributeError):
    from relax.models.qwen_omni.modeling_qwen3_omni.model import Qwen3OmniMoeModel  # noqa: F811
    from relax.models.qwen_omni.qwen3_omni_bridge import Qwen3OmniMoEBridge  # noqa: F811
    from relax.models.qwen_omni.qwen3_omni_provider import Qwen3OmniModelProvider  # noqa: F811

# Own try/except so a failure above still lets @register_bridge run -- otherwise
# AutoBridge silently falls back to the generic MLA bridge.
try:
    from relax.models import glm_moe_dsa  # noqa: F401
except Exception as _e:
    from relax.utils.logging_utils import get_logger

    get_logger(__name__).warning("Failed to import relax.models.glm_moe_dsa: %s", _e)

# Register DotsOCR2 bridge. Importing the module triggers its
# @MegatronModelBridge.register_bridge decorator.
try:
    from relax.models.dots_ocr import megatron as dots_ocr_megatron  # noqa: F401
except Exception as _e:
    import logging as _logging

    _logging.getLogger(__name__).warning("Failed to import relax.models.dots_ocr.megatron: %s", _e)


# Register the gemma-4 bridge, overriding upstream's entry so gemma-4 builds
# with Relax's packed-safe core attention.
try:
    from relax.models import gemma4  # noqa: F401
except Exception as _e:
    import logging as _logging

    _logging.getLogger(__name__).warning("Failed to import relax.models.gemma4: %s", _e)


__all__ = [
    "Qwen3OmniMoEBridge",
    "Qwen3OmniMoeModel",
    "Qwen3OmniModelProvider",
]
