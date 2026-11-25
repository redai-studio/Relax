# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Explicit validity for lossless weight layout operations.

Values and validity travel together through the original converter/loader.
Arithmetic and dtype conversion require a FULL tensor instead. This is not a
general sparse arithmetic tensor: it deliberately supports only layout changes
and same-dtype writes to explicitly owned destination storage.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils._pytree import tree_map


class FullTensorRequired(ValueError):
    """The converter or loader needs the complete value for this input."""


def storage_identity(tensor: torch.Tensor) -> tuple[str, int]:
    return str(tensor.device), tensor.untyped_storage()._cdata


class SparseTensor(torch.Tensor):
    """Read-only dense workspace with a separate boolean validity tensor.

    A receiver supplies its live storage identities as ``destinations``. A
    converter supplies none, so writes through the input are always rejected.
    No global torch methods are patched and ordinary tensors are unaffected.
    """

    @staticmethod
    def __new__(
        cls,
        value: torch.Tensor,
        valid: torch.Tensor,
        destinations: frozenset[tuple[str, int]] = frozenset(),
    ) -> SparseTensor:
        if value.dtype not in (torch.bfloat16, torch.float32):
            raise FullTensorRequired("Sparse layout requires BF16 or FP32 values")
        if valid.dtype != torch.bool or valid.shape != value.shape or valid.device != value.device:
            raise ValueError("Sparse validity must be boolean and match the values' shape and device")
        if storage_identity(value) in destinations or storage_identity(valid) in destinations:
            raise FullTensorRequired("Immutable sparse input storage cannot also be a live destination")
        return torch.Tensor._make_wrapper_subclass(
            cls,
            value.shape,
            strides=value.stride(),
            storage_offset=value.storage_offset(),
            dtype=value.dtype,
            layout=value.layout,
            device=value.device,
            requires_grad=False,
        )

    def __init__(
        self,
        value: torch.Tensor,
        valid: torch.Tensor,
        destinations: frozenset[tuple[str, int]] = frozenset(),
    ) -> None:
        self.value = value
        self.valid = valid
        self.destinations = destinations

    @classmethod
    def __torch_dispatch__(cls, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
        kwargs = kwargs or {}
        if func == torch.ops.aten.copy_.default:
            destination, source = args[:2]
            if isinstance(destination, SparseTensor):
                raise FullTensorRequired("A loader cannot mutate an immutable sparse input")
            if not isinstance(source, SparseTensor) or storage_identity(destination) not in source.destinations:
                raise FullTensorRequired("Sparse writes require an explicitly owned live destination")
            if destination.shape != source.shape or destination.dtype != source.dtype:
                raise FullTensorRequired("Sparse writes require equal shapes and storage dtypes")
            words = torch.int16 if source.element_size() == 2 else torch.int32
            # Select integer words, preserving all NaN payloads and signed zero.
            destination.view(words).copy_(torch.where(source.valid, source.value.view(words), destination.view(words)))
            return destination

        if any(
            argument.alias_info is not None and argument.alias_info.is_write for argument in func._schema.arguments
        ):
            raise FullTensorRequired("Sparse layout operations cannot mutate an input or out= destination")

        layout_ops = {
            "aten::alias",
            "aten::cat",
            "aten::clone",
            "aten::detach",
            "aten::expand",
            "aten::permute",
            "aten::reshape",
            "aten::select",
            "aten::slice",
            "aten::split",
            "aten::split_with_sizes",
            "aten::squeeze",
            "aten::stack",
            "aten::t",
            "aten::transpose",
            "aten::unbind",
            "aten::unsqueeze",
            "aten::view",
            "aten::_unsafe_view",
            "aten::_to_copy",
        }
        if func._schema.name not in layout_ops or func == torch.ops.aten.view.dtype:
            raise FullTensorRequired(f"Sparse layout does not support {func}; export this conversion group as FULL")

        inputs: list[SparseTensor] = []
        tensor_inputs: list[torch.Tensor] = []

        def values(tensor: Any) -> Any:
            if isinstance(tensor, torch.Tensor):
                tensor_inputs.append(tensor)
            if isinstance(tensor, SparseTensor):
                inputs.append(tensor)
                return tensor.value
            return tensor

        value_args = tree_map(values, args)
        value_kwargs = tree_map(values, kwargs)
        destinations = inputs[0].destinations
        if any(tensor.dtype != inputs[0].dtype for tensor in tensor_inputs):
            raise FullTensorRequired("Sparse layout cannot implicitly promote different storage dtypes")
        if any(item.destinations != destinations for item in inputs):
            raise FullTensorRequired("Sparse layout inputs have different destination owners")
        if func._schema.name == "aten::_to_copy" and kwargs.get("dtype", inputs[0].dtype) != inputs[0].dtype:
            raise FullTensorRequired("Sparse dtype conversion requires FULL input")

        def validity(tensor: Any) -> Any:
            if isinstance(tensor, SparseTensor):
                return tensor.valid
            if isinstance(tensor, torch.Tensor):
                # An ordinary tensor in a concatenation is a complete input.
                return torch.ones_like(tensor, dtype=torch.bool)
            return tensor

        mask_kwargs = tree_map(validity, kwargs)
        if "dtype" in mask_kwargs:
            mask_kwargs["dtype"] = torch.bool
        value = func(*value_args, **value_kwargs)
        valid = func(*tree_map(validity, args), **mask_kwargs)
        if isinstance(value, (list, tuple)):
            if any(tensor.dtype != inputs[0].dtype for tensor in value):
                raise FullTensorRequired("Sparse layout changed its storage dtype")
            return type(value)(cls(v, m, destinations) for v, m in zip(value, valid, strict=True))
        if value.dtype != inputs[0].dtype:
            raise FullTensorRequired("Sparse layout changed its storage dtype")
        return cls(value, valid, destinations)
