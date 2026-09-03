# Copyright (c) 2026 Relax Authors. All Rights Reserved.

_OMNI_NAMES = ("Qwen3OmniMoEBridge", "Qwen3OmniMoeModel", "Qwen3OmniModelProvider")
# Set when the Qwen3-Omni symbols could not be bound, so __getattr__ can name the
# real cause instead of a bare "cannot import name" far from it.
_OMNI_IMPORT_ERROR: Exception | None = None

try:
    from megatron.bridge.models.qwen_omni import (  # type: ignore[attr-defined]  # noqa: F401
        Qwen3OmniModelProvider,
        Qwen3OmniMoEBridge,
        Qwen3OmniMoeModel,
    )
except (ImportError, AttributeError):
    # Keep optional Omni import failures from disabling unrelated model paths.
    try:
        from relax.models.qwen_omni.modeling_qwen3_omni.model import Qwen3OmniMoeModel  # noqa: F401,F811
        from relax.models.qwen_omni.qwen3_omni_bridge import Qwen3OmniMoEBridge  # noqa: F401,F811
        from relax.models.qwen_omni.qwen3_omni_provider import Qwen3OmniModelProvider  # noqa: F401,F811
    except Exception as _e:
        from relax.utils.logging_utils import get_logger

        _OMNI_IMPORT_ERROR = _e
        get_logger(__name__).warning("Failed to import relax.models.qwen_omni (Omni path disabled): %s", _e)

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


# Only advertise what actually got bound: exporting an unbound name here makes
# `from relax.models import *` die with an unrelated AttributeError.
__all__ = [_name for _name in _OMNI_NAMES if _name in globals()]


def __getattr__(name: str):
    """Explain a missing Omni symbol instead of raising far from the cause."""
    if name in _OMNI_NAMES:
        raise ImportError(
            f"relax.models.{name} is unavailable because the Qwen3-Omni model path failed to "
            f"import: {_OMNI_IMPORT_ERROR!r}. Fix that import (it is usually a torch/torchao or "
            "megatron.bridge version skew) or use a model that does not need Qwen3-Omni."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
