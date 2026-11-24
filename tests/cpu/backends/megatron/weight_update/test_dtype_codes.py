# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for the dtype<->code table used to encode NCCL weight-sync metadata.

Qwen3.6 QAT sync broadcasts FP8 tensors, so ``_DTYPE_TO_CODE`` gained
``float8_e4m3fn`` / ``float8_e5m2`` entries. The table is encoded into an int
metadata tensor on the source rank and decoded on receivers, so it must be a
stable bijection. We mirror the megatron-stubbing used by the sibling weight-
sync test so the module imports without a GPU.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


torch = pytest.importorskip("torch")

_MEGATRON_MODULES = [
    "megatron",
    "megatron.core",
    "megatron.core.mpu",
    "megatron.core.transformer",
    "megatron.core.transformer.transformer_layer",
    "megatron.core.tensor_parallel",
    "megatron.bridge",
    "megatron.bridge.models",
    "megatron.bridge.peft",
    "megatron.bridge.peft.lora",
]

_MISSING = object()
_saved = {}
for _mod in _MEGATRON_MODULES:
    _saved[_mod] = sys.modules.get(_mod, _MISSING)
    sys.modules[_mod] = ModuleType(_mod)
    if _mod in {"megatron", "megatron.core", "megatron.core.transformer", "megatron.bridge", "megatron.bridge.peft"}:
        sys.modules[_mod].__path__ = []
sys.modules["megatron.core"].mpu = sys.modules["megatron.core.mpu"]
sys.modules["megatron.core.transformer.transformer_layer"].get_transformer_layer_offset = MagicMock(return_value=0)
sys.modules["megatron.bridge.peft.lora"].LoRAMerge = MagicMock

try:
    pytest.importorskip("triton")

    from relax.backends.megatron.weight_update.hf_weight_iterator_bridge import (  # noqa: E402
        _CODE_TO_DTYPE,
        _DTYPE_TO_CODE,
    )
finally:
    for _mod, _orig in _saved.items():
        if _orig is _MISSING:
            sys.modules.pop(_mod, None)
        else:
            sys.modules[_mod] = _orig


class TestDtypeCodes:
    def test_fp8_dtypes_registered(self):
        assert _DTYPE_TO_CODE[torch.float8_e4m3fn] == 7
        assert _DTYPE_TO_CODE[torch.float8_e5m2] == 8

    def test_reverse_map_resolves_fp8(self):
        assert _CODE_TO_DTYPE[7] is torch.float8_e4m3fn
        assert _CODE_TO_DTYPE[8] is torch.float8_e5m2

    def test_table_is_a_bijection(self):
        # every forward entry round-trips, and codes are unique
        assert len(_CODE_TO_DTYPE) == len(_DTYPE_TO_CODE)
        for dtype, code in _DTYPE_TO_CODE.items():
            assert _CODE_TO_DTYPE[code] is dtype

    def test_preexisting_codes_unchanged(self):
        # the FP8 additions must not have renumbered existing dtypes
        assert _DTYPE_TO_CODE[torch.float32] == 0
        assert _DTYPE_TO_CODE[torch.bfloat16] == 2
        assert _DTYPE_TO_CODE[torch.uint8] == 6
