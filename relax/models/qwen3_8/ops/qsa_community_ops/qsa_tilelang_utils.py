# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Vendored community source; original notices follow.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lazy TileLang import shim for Qwen3.8-Flash-Next kernels."""

from __future__ import annotations

import importlib.util
from functools import wraps
from typing import Any

import torch


HAS_TILELANG = importlib.util.find_spec("tilelang") is not None


def fit_qsa_forward_num_stages(
    device: torch.device,
    *,
    padded_heads: int,
    head_dim: int,
    block_size: int,
    requested_stages: int,
) -> int:
    """Fit the QSA forward software pipeline to the GPU shared-memory limit."""
    stages = max(int(requested_stages), 1)
    properties = torch.cuda.get_device_properties(device)
    shared_memory_limit = properties.shared_memory_per_block_optin

    def required_bytes(stage_count: int) -> int:
        element_bytes = 2  # BF16
        query_bytes = padded_heads * head_dim * element_bytes
        key_value_bytes = 2 * stage_count * block_size * head_dim * element_bytes
        probability_bytes = padded_heads * block_size * element_bytes
        return query_bytes + key_value_bytes + probability_bytes

    while stages > 1 and required_bytes(stages) > shared_memory_limit:
        stages -= 1
    if required_bytes(stages) > shared_memory_limit:
        raise RuntimeError(
            "QSA TileLang forward kernel exceeds the device dynamic shared-memory limit even "
            f"with one pipeline stage: required={required_bytes(stages)}, "
            f"limit={shared_memory_limit}."
        )
    return stages


def _load_tilelang() -> tuple[Any, Any]:
    try:
        import tilelang as real_tilelang
        from tilelang import language as real_language
    except ImportError as exc:
        raise ImportError(f"tilelang is required for Qwen3.8-Flash-Next TileLang kernels: {exc}") from exc
    return real_tilelang, real_language


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _resolve_pass_configs(real_tilelang: Any, pass_configs: dict[Any, Any] | None) -> dict[Any, Any] | None:
    if pass_configs is None:
        return None
    return {
        getattr(real_tilelang.PassConfigKey, key) if isinstance(key, str) else key: value
        for key, value in pass_configs.items()
    }


def _resolve_jit_kwargs(real_tilelang: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(kwargs)
    if "pass_configs" in resolved:
        resolved["pass_configs"] = _resolve_pass_configs(real_tilelang, resolved["pass_configs"])
    return resolved


def _lazy_jit(fn, jit_args: tuple[Any, ...], jit_kwargs: dict[str, Any]):
    compiled = None

    @wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal compiled
        if compiled is None:
            real_tilelang, _ = _load_tilelang()
            resolved_kwargs = _resolve_jit_kwargs(real_tilelang, jit_kwargs)
            compiled = real_tilelang.jit(*jit_args, **resolved_kwargs)(fn)
        return compiled(*args, **kwargs)

    return wrapper


def _phony_jit(*args: Any, **kwargs: Any):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return _lazy_jit(args[0], (), {})

    def decorate(fn):
        return _lazy_jit(fn, args, kwargs)

    return decorate


class _PassConfigKey:
    TL_DISABLE_TMA_LOWER = "TL_DISABLE_TMA_LOWER"
    TL_DISABLE_THREAD_STORAGE_SYNC = "TL_DISABLE_THREAD_STORAGE_SYNC"
    TL_DISABLE_WARP_SPECIALIZED = "TL_DISABLE_WARP_SPECIALIZED"
    TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE = "TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE"
    TL_ENABLE_FAST_MATH = "TL_ENABLE_FAST_MATH"


class _Math:
    next_power_of_2 = staticmethod(_next_power_of_2)

    def __getattr__(self, name: str) -> Any:
        real_tilelang, _ = _load_tilelang()
        return getattr(real_tilelang.math, name)


class _PhonyTileLang:
    PassConfigKey = _PassConfigKey
    cdiv = staticmethod(_cdiv)
    jit = staticmethod(_phony_jit)
    math = _Math()

    def __getattr__(self, name: str) -> Any:
        real_tilelang, _ = _load_tilelang()
        return getattr(real_tilelang, name)


class _PhonyLanguage:
    bfloat16 = "bfloat16"
    float = "float"
    float32 = "float32"
    int32 = "int32"

    @staticmethod
    def prim_func(fn):
        _, real_language = _load_tilelang()
        return real_language.prim_func(fn)

    def __getattr__(self, name: str) -> Any:
        _, real_language = _load_tilelang()
        return getattr(real_language, name)


tilelang = _PhonyTileLang()
T = _PhonyLanguage()
