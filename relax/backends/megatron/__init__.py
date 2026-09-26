# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib
import logging

from relax.utils.env import Envs


try:
    import mindspeed.megatron_adaptor  # noqa
except ImportError:
    pass

try:
    import relax.models  # noqa
except BaseException as e:
    print(f"failed to import relax.models, error={e}")

# Extension hook (analogue of ``--custom-generate-function-path``): every
# Megatron actor / driver loads this module, so any module names listed in
# ``RELAX_EXTRA_MODULES`` (comma-separated) are imported here for their
# side effects — typically downstream packages registering Megatron-Bridge
# converters, model providers, or family-token tables.
for _mod in filter(None, (m.strip() for m in Envs.RELAX_EXTRA_MODULES.split(","))):
    try:
        importlib.import_module(_mod)
    except BaseException as e:
        print(f"failed to import RELAX_EXTRA_MODULES entry {_mod!r}, error={e}")

from relax.utils.device import device_module  # noqa


try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(False)
        old_init(self, *args, **kwargs)
        device_module.synchronize()
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")


def patch_rotary_embedding(cls):
    _original_forward = cls.forward

    def _patched_forward(self, *args, packed_seq_params=None, **kwargs):
        return _original_forward(self, *args, **kwargs)

    cls.forward = _patched_forward


try:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
        Qwen3VLMoETextRotaryEmbedding,
        Qwen3VLTextRotaryEmbedding,
    )

    patch_rotary_embedding(Qwen3VLTextRotaryEmbedding)
    patch_rotary_embedding(Qwen3VLMoETextRotaryEmbedding)
except ImportError:
    pass

try:
    from megatron.bridge.models.qwen_omni.modelling_qwen3_omni.text_model import Qwen3OmniMoeThinkerTextRotaryEmbedding

    patch_rotary_embedding(Qwen3OmniMoeThinkerTextRotaryEmbedding)
except ImportError:
    pass

logging.getLogger("megatron").setLevel(logging.WARNING)

from . import megatron_patch  # noqa: F401, E402
